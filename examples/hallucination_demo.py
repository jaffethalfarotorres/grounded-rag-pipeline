"""A confident, wrong model meets the grounding gate (runs offline, no API key).

python examples/hallucination_demo.py
"""

from grounded_rag.bootstrap import DEFAULT_CORPUS_DIR
from grounded_rag.generation import LLMGenerator
from grounded_rag.loaders import load_directory
from grounded_rag.pipeline import RAGPipeline
from grounded_rag.providers import MockBackend

# The scripted "model" answers with a number the source does not contain.
model = MockBackend(["Backups are kept for 90 days and restores are free [S1]."])
rag = RAGPipeline(generator=LLMGenerator(model))
documents, _ = load_directory(DEFAULT_CORPUS_DIR)
rag.ingest(documents)

result = rag.ask("How long are backups kept?")
print("status:   ", result.status.value)
print("released: ", result.answer, f"({result.generator})")
print("rejected: ", [(f.rule, f.detail) for f in result.rejected.findings])
