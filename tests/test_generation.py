from grounded_rag.defense import DEFENSE_HEADER
from grounded_rag.envelopes import Chunk, Hit
from grounded_rag.generation import REFUSAL, ExtractiveGenerator, LLMGenerator, is_refusal
from grounded_rag.grounding import verify
from grounded_rag.providers import MockBackend


def hit(source_id: str, doc_id: str, text: str) -> Hit:
    chunk = Chunk(chunk_id=f"{doc_id}#0", doc_id=doc_id, title=doc_id.title(), index=0, text=text)
    return Hit(source_id=source_id, chunk=chunk, score=0.03)


HITS = [
    hit("S1", "refunds", "# Refund policy\n\nRefunds are issued within 14 days of a request."),
    hit("S2", "sla", "Monthly uptime target is 99.9%.\n\n- Credits of 10% apply below target."),
]


def test_extractive_answer_quotes_the_best_matching_sentence_with_its_label():
    answer = ExtractiveGenerator().generate("How long do refunds take?", HITS)
    assert answer == "Refunds are issued within 14 days of a request [S1]."


def test_extractive_answers_always_pass_the_grounding_gate():
    for question in ["refund request", "uptime target", "credits below target"]:
        answer = ExtractiveGenerator().generate(question, HITS)
        assert verify(answer, HITS, min_support=1.0).passed, answer


def test_extractive_generator_strips_list_bullets_and_skips_headings():
    answer = ExtractiveGenerator().generate("credits below target", HITS)
    assert answer.startswith("Credits of 10% apply below target")
    assert "#" not in answer


def test_extractive_generator_refuses_when_nothing_matches():
    assert ExtractiveGenerator().generate("quantum entanglement", HITS) == REFUSAL


def test_llm_prompt_fences_question_and_labelled_sources_under_the_defense_header():
    backend = MockBackend(["Refunds are issued within 14 days of a request [S1]."])
    LLMGenerator(backend).generate("How long do refunds take?", HITS)
    call = backend.calls[0]
    assert call.system.startswith(DEFENSE_HEADER)
    assert "[S1] Refunds" in call.prompt
    assert call.prompt.count("<untrusted_input>") == 2
    assert REFUSAL in call.system


def test_llm_generator_uses_the_requested_tier():
    backend = MockBackend(["x"])
    LLMGenerator(backend, tier="large").generate("q", HITS)
    assert backend.calls[0].model == "mock-large"


def test_refusals_are_recognized_with_or_without_final_punctuation():
    assert is_refusal(REFUSAL)
    assert is_refusal(REFUSAL.rstrip(".") + "\n")
    assert not is_refusal("Refunds take 14 days [S1].")
