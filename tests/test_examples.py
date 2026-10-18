import runpy
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_hallucination_demo_matches_the_readme(capsys):
    runpy.run_path(str(EXAMPLES / "hallucination_demo.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "status:    answered" in out
    assert "Backups are kept for 35 days [S1]." in out
    assert "('number_not_in_source', 'not in cited sources: 90')" in out
