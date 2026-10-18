"""Eval harness: four versioned YAML suites with gates fixed before results were seen.

    python -m grounded_rag.eval                         # offline: extractive answers
    python -m grounded_rag.eval --backend groq --k 3    # a real model, 3 runs per QA case

| suite     | measures                                   | gate              |
|-----------|--------------------------------------------|-------------------|
| retrieval | relevant document in the top 4             | recall@4 >= 0.90  |
| grounding | the gate blocks/passes hand-written answers | pass^k == 1.0     |
| injection | quarantine and question blocking           | pass^k == 1.0     |
| qa        | end-to-end status, facts and citations     | pass@k >= 0.80    |

Cases marked ``known_limitation`` run and are reported, but are not gated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from grounded_rag.bootstrap import (
    DEFAULT_CORPUS_DIR,
    EMBEDDER_CHOICES,
    backend_spec,
    build_pipeline,
)
from grounded_rag.embedder import EmbedderUnavailable
from grounded_rag.envelopes import Document, Hit, RAGAnswer
from grounded_rag.grounding import verify
from grounded_rag.pipeline import PipelineConfig, RAGPipeline
from grounded_rag.providers import BackendUnavailable

DEFAULT_CASES_DIR = Path(__file__).resolve().parents[2] / "evals"
RETRIEVAL_K = 4

Case = dict[str, Any]


@dataclass
class Suite:
    name: str
    path: Path
    metric: str
    threshold: float


SUITES = [
    Suite("retrieval", DEFAULT_CASES_DIR / "retrieval.yaml", "recall@4", 0.90),
    Suite("grounding", DEFAULT_CASES_DIR / "grounding.yaml", "pass^k", 1.0),
    Suite("injection", DEFAULT_CASES_DIR / "injection.yaml", "pass^k", 1.0),
    Suite("qa", DEFAULT_CASES_DIR / "qa.yaml", "pass@k", 0.80),
]


@dataclass
class SuiteReport:
    name: str
    metric: str
    threshold: float
    ran: int
    score: float
    passed: bool
    pass_at_k: float
    pass_hat_k: float
    extra: dict[str, float] = field(default_factory=dict)
    failures: dict[str, list[str]] = field(default_factory=dict)
    limitations: dict[str, str] = field(default_factory=dict)
    rejections: dict[str, list[str]] = field(default_factory=dict)


def load_cases(path: Path) -> list[Case]:
    return list(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or [])


def eval_pipeline(
    corpus_dir: Path,
    *,
    backend: str,
    env: Mapping[str, str] | None,
    embedder: str = "hashing",
) -> RAGPipeline:
    """The bundled corpus, with the answer cache off so repeated runs really re-run."""
    pipeline, _ = build_pipeline(
        corpus_dir=corpus_dir,
        backend=backend,
        embedder=embedder,
        env=env,
        config=PipelineConfig(cache_size=0),
    )
    return pipeline


def pass_rates(outcomes: Mapping[str, Sequence[bool]]) -> tuple[float, float]:
    """(pass@k, pass^k): share of cases passing in at least one run / in every run."""
    if not outcomes:
        return 0.0, 0.0
    runs = list(outcomes.values())
    return sum(any(r) for r in runs) / len(runs), sum(all(r) for r in runs) / len(runs)


# --- per-suite case runners: each returns the list of failures (empty means pass) -------------


def run_retrieval_case(
    case: Case, rag: RAGPipeline, k: int = RETRIEVAL_K
) -> tuple[int | None, list[str]]:
    hits = rag.retriever.search(case["question"], top_k=k)
    ranked = [hit.chunk.doc_id for hit in hits]
    rank = next((i for i, doc in enumerate(ranked, 1) if doc in case["relevant"]), None)
    failures = [] if rank else [f"none of {case['relevant']} in top {k}: {ranked}"]
    return rank, failures


def run_grounding_case(case: Case, rag: RAGPipeline) -> list[str]:
    chunks = [chunk for doc_id in case["sources"] for chunk in rag.retriever.chunks_of(doc_id)]
    hits = [
        Hit(source_id=f"S{i}", chunk=chunk, score=0.0) for i, chunk in enumerate(chunks, start=1)
    ]
    report = verify(case["answer"], hits, min_support=rag.config.min_support)
    found = {finding.rule for finding in report.findings}
    failures = []
    if report.passed != (case["expect"] == "pass"):
        failures.append(
            f"expected {case['expect']}, gate "
            f"{'passed' if report.passed else 'blocked'} ({sorted(found)})"
        )
    if missing := set(case.get("rules", [])) - found:
        failures.append(f"missing findings {sorted(missing)}, got {sorted(found)}")
    return failures


def run_injection_case(case: Case, rag: RAGPipeline) -> list[str]:
    if "document" in case:
        doc = case["document"]
        fresh = RAGPipeline()
        report = fresh.ingest(
            [Document(doc_id=doc["doc_id"], title=doc["doc_id"], text=doc["text"])]
        )
        got = bool(report.quarantined)
        want = case["expect_quarantined"]
        return [] if got == want else [f"quarantined={got}, expected {want}"]
    blocked = rag.ask(case["question"]).defense.blocked
    want = case["expect_blocked"]
    return [] if blocked == want else [f"blocked={blocked}, expected {want}"]


def check_answer(expect: Mapping[str, Any], answer: RAGAnswer) -> list[str]:
    failures = []
    if "status" in expect:
        allowed = expect["status"] if isinstance(expect["status"], list) else [expect["status"]]
        if answer.status.value not in allowed:
            failures.append(f"status {answer.status.value}, expected {allowed}")
    text = (answer.answer or "").lower()
    failures += [f"missing {s!r}" for s in expect.get("include", []) if s.lower() not in text]
    failures += [f"contains {s!r}" for s in expect.get("exclude", []) if s.lower() in text]
    cited = {citation.doc_id for citation in answer.citations}
    failures += [f"does not cite {doc}" for doc in expect.get("cite", []) if doc not in cited]
    return failures


# --- suite runners ----------------------------------------------------------------------------


def _run_suite(suite: Suite, rag: RAGPipeline, k: int) -> SuiteReport:
    outcomes: dict[str, list[bool]] = {}
    failures: dict[str, list[str]] = {}
    limitations: dict[str, str] = {}
    ranks: list[int | None] = []
    semantic_ranks: list[int | None] = []
    generators: Counter[str] = Counter()
    rejections: dict[str, list[str]] = {}
    rejected = 0
    for case in load_cases(suite.path):
        runs = k if suite.name == "qa" else 1
        results = []
        for _ in range(runs):
            if suite.name == "retrieval":
                rank, problems = run_retrieval_case(case, rag)
                if case.get("semantic"):
                    # Paraphrases with no shared keywords: reported, never gated, so a
                    # lexical embedder is not failed for what it cannot do by design.
                    semantic_ranks.append(rank)
                    break
                ranks.append(rank)
            elif suite.name == "grounding":
                problems = run_grounding_case(case, rag)
            elif suite.name == "injection":
                problems = run_injection_case(case, rag)
            else:
                answer = rag.ask(case["question"])
                if answer.answer is not None:
                    generators[answer.generator] += 1
                if answer.rejected is not None:
                    rejected += 1
                    rejections.setdefault(case["id"], []).extend(
                        f"{f.rule}: {f.sentence!r}" for f in answer.rejected.findings
                    )
                problems = check_answer(case["expect"], answer)
                if problems:  # keep what the model said: a flaky failure is otherwise lost
                    problems.append(f"{answer.generator} answer: {answer.answer!r}")
            results.append(not problems)
            if problems:
                failures.setdefault(case["id"], problems)
        if not results:  # a semantic case: reported on its own, outside the gate
            continue
        if "known_limitation" in case:
            verdict = "still missed" if not all(results) else "now handled"
            limitations[case["id"]] = f"{verdict}: {' '.join(case['known_limitation'].split())}"
            failures.pop(case["id"], None)
            continue
        outcomes[case["id"]] = results

    pass_at_k, pass_hat_k = pass_rates(outcomes)
    extra = {}
    if suite.name == "retrieval":
        extra["mrr"] = round(sum(1 / r for r in ranks if r) / len(ranks), 3) if ranks else 0.0
        if semantic_ranks:
            found = sum(rank is not None for rank in semantic_ranks)
            extra["semantic_cases"] = len(semantic_ranks)
            extra["semantic_recall"] = round(found / len(semantic_ranks), 3)
            reciprocal = sum(1 / r for r in semantic_ranks if r)
            extra["semantic_mrr"] = round(reciprocal / len(semantic_ranks), 3)
    if suite.name == "qa":
        extra |= {f"answered_by_{name}": count for name, count in sorted(generators.items())}
        extra["llm_answers_rejected"] = rejected
    score = pass_hat_k if suite.metric == "pass^k" else pass_at_k
    return SuiteReport(
        name=suite.name,
        metric=suite.metric,
        threshold=suite.threshold,
        ran=len(outcomes),
        score=round(score, 3),
        passed=bool(outcomes) and score >= suite.threshold,
        pass_at_k=round(pass_at_k, 3),
        pass_hat_k=round(pass_hat_k, 3),
        extra=extra,
        failures=failures,
        limitations=limitations,
        rejections=rejections,
    )


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m grounded_rag.eval")
    parser.add_argument("--backend", type=backend_spec, default="none", metavar="BACKEND")
    parser.add_argument("--embedder", choices=EMBEDDER_CHOICES, default="hashing")
    parser.add_argument("--k", type=int, default=1, help="runs per QA case")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--json", type=Path, help="also write the report to this file")
    args = parser.parse_args(argv)

    try:
        rag = eval_pipeline(args.corpus, backend=args.backend, embedder=args.embedder, env=env)
    except (BackendUnavailable, EmbedderUnavailable) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    k = args.k if args.backend != "none" else 1

    reports = [_run_suite(suite, rag, k) for suite in SUITES]
    mode = "extractive answers, no LLM" if args.backend == "none" else f"LLM via {args.backend}"
    print(f"backend: {args.backend} ({mode})  embedder: {args.embedder}  k={k}")
    print(f"{'suite':<10} {'ran':>4} {'pass@k':>7} {'pass^k':>7}  gate")
    for report in reports:
        extra = "".join(f"  {name}={value}" for name, value in report.extra.items())
        verdict = "PASS" if report.passed else "FAIL"
        print(
            f"{report.name:<10} {report.ran:>4} {report.pass_at_k:>7.2f} {report.pass_hat_k:>7.2f}"
            f"  {report.metric} >= {report.threshold:.2f}  {verdict}{extra}"
        )
        for case_id, problems in report.failures.items():
            print(f"  FAIL {report.name}/{case_id}: {'; '.join(problems)}")
        for case_id, note in report.limitations.items():
            print(f"  KNOWN LIMITATION {report.name}/{case_id}: {note}")
        for case_id, findings in report.rejections.items():
            print(f"  GROUNDING REJECTED {report.name}/{case_id}: {'; '.join(findings)}")

    if args.json:
        payload = {
            "backend": args.backend,
            "embedder": args.embedder,
            "k": k,
            "suites": [r.__dict__ for r in reports],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if all(report.passed for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
