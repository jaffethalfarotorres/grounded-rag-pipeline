"""The pipeline: ingest with quarantine, then question -> defense -> retrieve -> relevance
gate -> generate -> grounding gate -> typed answer.

Order matters, and every early exit saves a model call:

1. the question is scanned for prompt injection before anything else runs;
2. if the retrieved text does not cover what was asked, the pipeline refuses
   without calling the model at all;
3. an LLM answer is released only if it passes the grounding gate; if it fails,
   the pipeline falls back to an extractive answer quoted from the sources (or
   blocks, if fallback is turned off). It never retries the model blindly.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from grounded_rag.cache import QueryCache, cache_key
from grounded_rag.chunker import DEFAULT_MAX_CHARS, chunk_document
from grounded_rag.defense import scan
from grounded_rag.embedder import Embedder, HashingEmbedder
from grounded_rag.envelopes import (
    AnswerStatus,
    Citation,
    DefenseReport,
    Document,
    GroundingReport,
    Hit,
    IngestReport,
    QuarantinedPassage,
    RAGAnswer,
)
from grounded_rag.generation import ExtractiveGenerator, LLMGenerator, is_refusal
from grounded_rag.grounding import DEFAULT_MIN_SUPPORT, cited_ids, verify
from grounded_rag.metrics import Metrics
from grounded_rag.providers.base import BackendError
from grounded_rag.retrieval import HybridRetriever
from grounded_rag.text import content_tokens

_EXCERPT_CHARS = 240


@dataclass(frozen=True)
class PipelineConfig:
    top_k: int = 4
    max_chars: int = DEFAULT_MAX_CHARS
    min_relevance: float = 0.5
    min_support: float = DEFAULT_MIN_SUPPORT
    fallback_to_extractive: bool = True
    cache_size: int = 256
    cache_ttl_seconds: float = 600.0


class RAGPipeline:
    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        generator: LLMGenerator | None = None,
        config: PipelineConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or PipelineConfig()
        self.retriever = HybridRetriever(embedder or HashingEmbedder())
        self.llm = generator
        self.extractive = ExtractiveGenerator()
        self.metrics = Metrics()
        self.clock = clock
        self.cache = QueryCache(
            max_entries=self.config.cache_size,
            ttl_seconds=self.config.cache_ttl_seconds,
            clock=lambda: self.clock(),
        )
        self.index_version = 0
        self.documents: dict[str, Document] = {}

    @property
    def generator_name(self) -> str:
        return f"llm:{self.llm.backend.name}" if self.llm else "extractive"

    # --- ingest --------------------------------------------------------------------------

    def ingest(self, documents: Sequence[Document]) -> IngestReport:
        quarantined: list[QuarantinedPassage] = []
        chunks = []
        for document in documents:
            kept = []
            # Scan per paragraph, not per chunk: an injected sentence is dropped while the
            # legitimate text around it stays searchable. Every defense rule is confined to
            # a line or paragraph, so no match can span two paragraphs.
            for number, paragraph in enumerate(document.text.split("\n\n"), start=1):
                # Size is not an attack in a document (a PDF page can be one long paragraph).
                report = scan(paragraph, max_chars=max(len(paragraph), 1))
                if report.blocked:
                    quarantined.append(
                        self._quarantine(document.doc_id, f"paragraph {number}", report, paragraph)
                    )
                else:
                    kept.append(paragraph)
            clean = document.model_copy(update={"text": "\n\n".join(kept)})
            chunks.extend(chunk_document(clean, max_chars=self.config.max_chars))
            self.retriever.remove_document(document.doc_id)
            self.documents[document.doc_id] = document

        self.retriever.add(chunks)
        self.index_version += 1
        self.cache.clear()
        self.metrics.inc("rag_chunks_indexed_total", len(chunks))
        self.metrics.inc("rag_passages_quarantined_total", len(quarantined))
        return IngestReport(
            documents=len(documents),
            chunks_indexed=len(chunks),
            quarantined=tuple(quarantined),
            index_version=self.index_version,
        )

    @staticmethod
    def _quarantine(
        doc_id: str, location: str, report: DefenseReport, text: str
    ) -> QuarantinedPassage:
        return QuarantinedPassage(
            doc_id=doc_id,
            location=location,
            rules=tuple(finding.rule for finding in report.findings),
            excerpt=" ".join(text.split())[:80],
        )

    # --- ask -----------------------------------------------------------------------------

    def relevance(self, question: str, hits: Sequence[Hit]) -> float:
        """Best share of the question's corpus-known words found in a single hit.

        Words the corpus never mentions are left out: they say nothing about whether a
        retrieved chunk is on topic, and they would make every natural question
        ("how long does it take...") look irrelevant. No known word at all scores 0.
        """
        known = {word for word in content_tokens(question) if self.retriever.knows(word)}
        if not known or not hits:
            return 0.0
        return max(len(known & set(content_tokens(hit.chunk.text))) / len(known) for hit in hits)

    def ask(
        self, question: str, *, top_k: int | None = None, filters: Mapping[str, str] | None = None
    ) -> RAGAnswer:
        started = self.clock()
        top_k = top_k or self.config.top_k
        filters = dict(filters or {})
        key = cache_key(
            question,
            top_k=top_k,
            filters=filters,
            index_version=self.index_version,
            generator=self.generator_name,
        )
        if cached := self.cache.get(key):
            self.metrics.inc("rag_cache_hits_total")
            return self._finish(cached.model_copy(update={"cached": True}), started)

        answer, cacheable = self._answer(question, top_k, filters)
        if cacheable:
            self.cache.put(key, answer)
        return self._finish(answer, started)

    def _finish(self, answer: RAGAnswer, started: float) -> RAGAnswer:
        self.metrics.inc("rag_queries_total", status=answer.status.value)
        self.metrics.observe("rag_query_seconds", max(0.0, self.clock() - started))
        return answer

    def _answer(self, question: str, top_k: int, filters: dict[str, str]) -> tuple[RAGAnswer, bool]:
        defense = scan(question)
        if defense.blocked:
            rules = ", ".join(finding.rule for finding in defense.findings)
            return self._result(
                question, AnswerStatus.BLOCKED, defense, trace=[f"defense: blocked ({rules})"]
            ), True
        trace = ["defense: clean"]

        hits = self.retriever.search(question, top_k=top_k, filters=filters)
        relevance = self.relevance(question, hits)
        trace.append(f"retrieval: {len(hits)} hits, relevance {relevance:.2f}")
        if relevance < self.config.min_relevance:
            trace.append(
                f"relevance: {relevance:.2f} < {self.config.min_relevance:.2f}; "
                "refused without calling the model"
            )
            return self._result(question, AnswerStatus.REFUSED, defense, trace=trace), True

        rejected: GroundingReport | None = None
        cacheable = True
        if self.llm is not None:
            try:
                text = self.llm.generate(question, hits)
                self.metrics.inc("rag_llm_calls_total", outcome="ok")
            except BackendError as exc:
                self.metrics.inc("rag_llm_calls_total", outcome="error")
                trace.append(f"llm: {exc}; using the extractive answer")
                text, cacheable = None, False
            if text is not None:
                if is_refusal(text):
                    trace.append("llm: the model said the sources do not answer the question")
                    return self._result(question, AnswerStatus.REFUSED, defense, trace=trace), True
                report = verify(text, hits, min_support=self.config.min_support)
                if report.passed:
                    trace.append("grounding: llm answer passed")
                    return self._answered(question, text, "llm", hits, report, defense, trace), True
                rejected = report
                rules = sorted({finding.rule for finding in report.findings})
                self.metrics.inc("rag_grounding_rejections_total")
                trace.append(f"grounding: llm answer rejected ({', '.join(rules)})")
                if not self.config.fallback_to_extractive:
                    return self._result(
                        question, AnswerStatus.BLOCKED, defense, grounding=report, trace=trace
                    ), True

        text = self.extractive.generate(question, hits)
        if is_refusal(text):
            trace.append("extractive: no source sentence matches the question")
            return self._result(
                question, AnswerStatus.REFUSED, defense, rejected=rejected, trace=trace
            ), cacheable
        report = verify(text, hits, min_support=self.config.min_support)
        trace.append(f"grounding: extractive answer {'passed' if report.passed else 'rejected'}")
        if not report.passed:
            return self._result(
                question,
                AnswerStatus.BLOCKED,
                defense,
                grounding=report,
                rejected=rejected,
                trace=trace,
            ), cacheable
        answer = self._answered(question, text, "extractive", hits, report, defense, trace)
        return answer.model_copy(update={"rejected": rejected}), cacheable

    @staticmethod
    def _result(
        question: str,
        status: AnswerStatus,
        defense: DefenseReport,
        *,
        grounding: GroundingReport | None = None,
        rejected: GroundingReport | None = None,
        trace: Sequence[str] = (),
    ) -> RAGAnswer:
        return RAGAnswer(
            status=status,
            question=question,
            answer=None,
            generator="none",
            grounding=grounding,
            rejected=rejected,
            defense=defense,
            trace=tuple(trace),
        )

    @staticmethod
    def _answered(
        question: str,
        text: str,
        generator: str,
        hits: Sequence[Hit],
        report: GroundingReport,
        defense: DefenseReport,
        trace: Sequence[str],
    ) -> RAGAnswer:
        by_label = {hit.source_id: hit for hit in hits}
        citations = tuple(
            Citation(
                source_id=label,
                chunk_id=by_label[label].chunk.chunk_id,
                doc_id=by_label[label].chunk.doc_id,
                title=by_label[label].chunk.title,
                excerpt=" ".join(by_label[label].chunk.text.split())[:_EXCERPT_CHARS],
            )
            for label in cited_ids(text)
            if label in by_label
        )
        return RAGAnswer(
            status=AnswerStatus.ANSWERED,
            question=question,
            answer=text,
            generator=generator,
            citations=citations,
            grounding=report,
            defense=defense,
            trace=tuple(trace),
        )
