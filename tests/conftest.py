"""Shared test fixtures."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from skill_miner.config import Candidate, Evidence, RunState


@pytest.fixture
def sample_evidence() -> list[Evidence]:
    return [
        Evidence(session_id="sess-001", source="claude_code", date=datetime(2026, 3, 10, tzinfo=timezone.utc),
                 user_intent="Set up cross-agent review loop", key_quotes=["review the PR changes"],
                 tools_used=["Bash", "Read", "Edit"], outcome="worked"),
        Evidence(session_id="sess-002", source="codex", date=datetime(2026, 3, 12, tzinfo=timezone.utc),
                 user_intent="Run multi-agent code review", key_quotes=["spawn review agent"],
                 tools_used=["Bash", "Read"], outcome="worked"),
        Evidence(session_id="sess-003", source="claude_code", date=datetime(2026, 3, 14, tzinfo=timezone.utc),
                 user_intent="Cross-agent review for Star Office UI",
                 tools_used=["Bash", "Read", "Write"], outcome="worked"),
    ]


@pytest.fixture
def sample_candidate(sample_evidence: list[Evidence]) -> Candidate:
    return Candidate(
        proposed_name="cross_agent_review",
        summary="Multi-agent write-review-revise loop for code changes",
        evidence=sample_evidence, overlap="none",
        first_seen=datetime(2026, 3, 10, tzinfo=timezone.utc),
        last_evidence_date=datetime(2026, 3, 14, tzinfo=timezone.utc),
    )


@pytest.fixture
def weak_candidate() -> Candidate:
    return Candidate(
        proposed_name="api_key_rotation",
        summary="Rotate and manage API keys across services",
        evidence=[Evidence(session_id="sess-010", source="claude_code",
                           date=datetime(2026, 3, 11, tzinfo=timezone.utc),
                           user_intent="Rotate API keys", tools_used=["Bash"], outcome="worked")],
        first_seen=datetime(2026, 3, 11, tzinfo=timezone.utc),
    )


@pytest.fixture
def duplicate_candidate() -> Candidate:
    return Candidate(
        proposed_name="code_review_basic", summary="Review code changes for issues",
        evidence=[
            Evidence(session_id="sess-020", source="claude_code",
                     date=datetime(2026, 3, 12, tzinfo=timezone.utc),
                     user_intent="Review my code", tools_used=["Read", "Bash"]),
            Evidence(session_id="sess-021", source="codex",
                     date=datetime(2026, 3, 13, tzinfo=timezone.utc),
                     user_intent="Check code quality", tools_used=["Read"]),
        ],
        overlap="duplicate:review_changes",
    )


@pytest.fixture
def empty_run_state() -> RunState:
    return RunState()
