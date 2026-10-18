import sys
from types import SimpleNamespace

import pytest

from grounded_rag.generation import LLMGenerator
from grounded_rag.pipeline import PipelineConfig, RAGPipeline
from grounded_rag.providers import (
    BackendError,
    BackendUnavailable,
    FallbackBackend,
    MockBackend,
    build_chain,
)
from tests.corpus import DOCS

PROMPT, SYSTEM = "p", "s"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def chain(*members, **options) -> FallbackBackend:
    return FallbackBackend(list(members), clock=options.pop("clock", Clock()), **options)


def test_the_first_healthy_provider_answers_and_later_ones_are_untouched():
    first, second = MockBackend(["from first"]), MockBackend(["from second"])
    backend = chain(("groq", first), ("ollama", second))
    assert backend.complete(PROMPT, system=SYSTEM) == "from first"
    assert backend.last_provider == "groq"
    assert second.calls == []


def test_a_failing_provider_hands_over_to_the_next_one():
    backend = chain(
        ("groq", MockBackend(fail_with="429 rate limited")),
        ("ollama", MockBackend(["from ollama"])),
    )
    assert backend.complete(PROMPT, system=SYSTEM) == "from ollama"
    assert backend.last_provider == "ollama"


def test_when_every_provider_fails_the_error_names_each_one():
    backend = chain(
        ("groq", MockBackend(fail_with="429 rate limited")),
        ("ollama", MockBackend(fail_with="connection refused")),
    )
    with pytest.raises(BackendError) as error:
        backend.complete(PROMPT, system=SYSTEM)
    message = str(error.value)
    assert "groq" in message
    assert "429" in message
    assert "ollama" in message
    assert "connection refused" in message


def test_the_tier_is_passed_to_the_provider_that_answers():
    large = MockBackend(["ok"])
    chain(("groq", large)).complete(PROMPT, system=SYSTEM, tier="large")
    assert large.calls[0].model == "mock-large"


def test_a_provider_that_keeps_failing_is_skipped_during_its_cooldown():
    clock = Clock()
    flaky = MockBackend(fail_with="503")
    steady = MockBackend(lambda call: "steady")
    backend = chain(
        ("groq", flaky), ("ollama", steady), failure_threshold=2, cooldown=60, clock=clock
    )

    for _ in range(2):  # two consecutive failures open the circuit
        backend.complete(PROMPT, system=SYSTEM)
    assert len(flaky.calls) == 2

    backend.complete(PROMPT, system=SYSTEM)
    assert len(flaky.calls) == 2  # skipped without a call
    assert backend.status()["groq"]["state"] == "cooling_down"

    clock.now = 61  # the cooldown is over: one probe is allowed
    backend.complete(PROMPT, system=SYSTEM)
    assert len(flaky.calls) == 3


class Switch:
    """A provider that can be broken and repaired between calls."""

    def __init__(self) -> None:
        self.healthy = False

    def complete(self, prompt, *, system, tier="small"):
        if not self.healthy:
            raise BackendError("switch: blip")
        return "switch is back"


def test_one_success_resets_the_failure_count():
    switch = Switch()
    backend = chain(
        ("groq", switch), ("ollama", MockBackend(lambda call: "ollama")), failure_threshold=2
    )
    assert backend.complete(PROMPT, system=SYSTEM) == "ollama"  # groq fails once
    assert backend.status()["groq"]["consecutive_failures"] == 1
    switch.healthy = True
    assert backend.complete(PROMPT, system=SYSTEM) == "switch is back"
    assert backend.status()["groq"]["consecutive_failures"] == 0


def test_a_chain_where_every_provider_is_cooling_down_fails_fast_without_calling_anyone():
    clock = Clock()
    only = MockBackend(fail_with="down")
    backend = chain(("groq", only), failure_threshold=1, cooldown=60, clock=clock)
    with pytest.raises(BackendError):
        backend.complete(PROMPT, system=SYSTEM)
    with pytest.raises(BackendError, match="cooling down"):
        backend.complete(PROMPT, system=SYSTEM)
    assert len(only.calls) == 1


def test_an_empty_chain_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        FallbackBackend([])


# --- building a chain from the environment ---------------------------------------------------


def test_build_chain_skips_providers_without_credentials_and_says_so():
    env = {"HF_TOKEN": "hf_test"}
    fake_sdk = SimpleNamespace(InferenceClient=lambda *, token: object())
    sys.modules["huggingface_hub"] = fake_sdk
    try:
        backend = build_chain(["groq", "huggingface"], env)
    finally:
        del sys.modules["huggingface_hub"]
    assert [name for name, _ in backend.members] == ["huggingface"]
    assert "groq" in backend.skipped
    assert "GROQ_API_KEY" in backend.skipped["groq"]


def test_build_chain_with_nothing_usable_lists_every_reason():
    with pytest.raises(BackendUnavailable) as error:
        build_chain(["groq", "huggingface"], {})
    assert "GROQ_API_KEY" in str(error.value)
    assert "HF_TOKEN" in str(error.value)


def test_build_chain_rejects_unknown_provider_names():
    with pytest.raises(ValueError, match="unknown backend"):
        build_chain(["groq", "word2vec"], {})


# --- inside the pipeline ---------------------------------------------------------------------


def test_the_pipeline_answers_through_the_chain_and_records_the_provider():
    backend = chain(
        ("groq", MockBackend(fail_with="429")),
        ("ollama", MockBackend(["Refunds are issued within 14 days of a written request [S1]."])),
    )
    rag = RAGPipeline(generator=LLMGenerator(backend), config=PipelineConfig(cache_size=0))
    rag.ingest(DOCS)
    result = rag.ask("How long do refunds take?")
    assert result.generator == "llm"
    assert any("answered by ollama" in step for step in result.trace)
    assert 'rag_llm_calls_total{outcome="ok",provider="ollama"} 1' in rag.metrics.render()
