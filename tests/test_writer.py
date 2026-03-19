"""Tests for output writer."""

import json
from datetime import datetime, timezone

from skill_miner.candidate_extractor import _write_output as write_output
from skill_miner.config import Candidate, Decision, Evidence, RejectReason


def test_write_json(tmp_path) -> None:
    candidates = [Candidate(proposed_name="test_skill", summary="A test", decision=Decision.ACCEPT,
                            evidence=[Evidence(session_id="s1", source="claude_code",
                                               date=datetime(2026, 3, 15, tzinfo=timezone.utc),
                                               user_intent="Test", tools_used=["Bash"])])]
    paths = write_output(candidates, tmp_path)
    assert len(paths) == 2  # json + markdown
    data = json.loads(paths[0].read_text())
    assert data["summary"]["accepted"] == 1
    assert data["candidates"][0]["proposed_name"] == "test_skill"


def test_write_markdown(tmp_path) -> None:
    candidates = [
        Candidate(proposed_name="accepted", summary="Good", decision=Decision.ACCEPT,
                  evidence=[Evidence(session_id="s1", source="test",
                                     date=datetime(2026, 3, 15, tzinfo=timezone.utc),
                                     user_intent="Do something", tools_used=["Read"])]),
        Candidate(proposed_name="rejected", summary="Bad", decision=Decision.REJECT,
                  reject_reason=RejectReason.ONE_OFF),
    ]
    paths = write_output(candidates, tmp_path)
    md = paths[1].read_text()
    assert "## Accept" in md
    assert "accepted" in md
    assert "## Reject" in md
    assert "rejected" in md


def test_write_empty(tmp_path) -> None:
    paths = write_output([], tmp_path)
    data = json.loads(paths[0].read_text())
    assert data["summary"]["total"] == 0
