"""Tests for interactive reviewer."""

import json
from pathlib import Path

from skill_miner.skill_generator import find_latest_queue, load_queue


def test_find_latest_queue(tmp_path: Path) -> None:
    (tmp_path / "2026-03-10_skill_candidates.json").write_text("{}")
    (tmp_path / "2026-03-17_skill_candidates.json").write_text("{}")
    result = find_latest_queue(tmp_path)
    assert result is not None
    assert "2026-03-17" in result.name


def test_find_latest_queue_empty(tmp_path: Path) -> None:
    assert find_latest_queue(tmp_path) is None


def test_load_queue(tmp_path: Path) -> None:
    data = {"candidates": [
        {"proposed_name": "test_skill", "summary": "A test", "decision": "accept", "overlap": "none",
         "evidence": [{"session_id": "s1", "source": "test", "date": "2026-03-15T00:00:00+00:00",
                        "user_intent": "Test", "tools_used": ["Bash"]}],
         "privacy_flags": []}
    ]}
    queue_file = tmp_path / "2026-03-17_skill_candidates.json"
    queue_file.write_text(json.dumps(data))
    candidates = load_queue(queue_file)
    assert len(candidates) == 1
    assert candidates[0].proposed_name == "test_skill"
