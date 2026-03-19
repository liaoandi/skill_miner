"""Integration test for the pipeline with mocked LLM."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from skill_miner.candidate_extractor import run
from skill_miner.config import Config, Session, SessionEntry, SourceConfig

_MOCK_LLM_RESPONSE = json.dumps({"candidates": [
    {"proposed_name": "project_cleanup", "summary": "Scan project for stale files",
     "evidence": [
         {"session_id": f"mock-{i}", "source": "claude_code",
          "date": f"2026-03-{10+i}T00:00:00+00:00",
          "user_intent": "Clean up", "key_quotes": [], "tools_used": ["Bash", "Read"], "outcome": "worked"}
         for i in range(3)
     ], "overlap": "none", "privacy_flags": []}
]})


def _mock_sessions() -> list[Session]:
    return [Session(source="claude_code", session_id=f"mock-{i}", project="/mock",
                    created=datetime(2026, 3, 10 + i, tzinfo=timezone.utc),
                    modified=datetime(2026, 3, 10 + i, tzinfo=timezone.utc),
                    entries=[SessionEntry(timestamp=datetime(2026, 3, 10 + i, tzinfo=timezone.utc),
                                          role="user", content=f"Step {i}")],
                    tools_used=["Bash", "Read", "Edit"])
            for i in range(4)]


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    config = Config(
        sources={"claude_code": SourceConfig(enabled=True, session_dir=tmp_path / "sessions")},
        min_sessions=3, min_tools=2,
        output_dir=tmp_path / "output", state_dir=tmp_path / "state",
    )

    class MockMessages:
        def create(self, **kwargs):
            class R:
                content = [type("B", (), {"text": _MOCK_LLM_RESPONSE})()]
            return R()

    class MockClient:
        messages = MockMessages()

    with (
        patch("skill_miner.session_reader.load_all_sessions", return_value=_mock_sessions()),
        patch("skill_miner.candidate_extractor.create_llm_client", return_value=MockClient()),
    ):
        result = run(config)  # noqa: F841 used in assertions below

    assert result.summary is not None
    assert result.summary.accepted == 1
    assert len(result.output_paths) == 2
    data = json.loads(list((tmp_path / "output").glob("*.json"))[0].read_text())
    assert data["candidates"][0]["proposed_name"] == "project_cleanup"
