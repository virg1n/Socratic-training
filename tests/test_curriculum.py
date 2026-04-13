from pathlib import Path

from socratic_training.curriculum import load_curriculum


def test_load_curriculum_smoke() -> None:
    cur = load_curriculum(Path("curriculum.txt"))
    assert cur.topics
    idx = cur.topic_index()
    assert "loops" in idx
    assert "easy" in idx["loops"].difficulties

