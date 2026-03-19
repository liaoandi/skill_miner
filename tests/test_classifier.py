"""Tests for classification logic."""

from datetime import datetime, timezone

from skill_miner.candidate_extractor import _classify as classify
from skill_miner.config import Candidate, Config, Decision, Evidence, RejectReason


def test_accept_meets_thresholds(sample_candidate: Candidate) -> None:
    result = classify([sample_candidate], Config())
    assert result[0].decision == Decision.ACCEPT


def test_reject_one_off(weak_candidate: Candidate) -> None:
    result = classify([weak_candidate], Config())
    assert result[0].decision == Decision.REJECT
    assert result[0].reject_reason == RejectReason.ONE_OFF


def test_reject_duplicate(duplicate_candidate: Candidate) -> None:
    result = classify([duplicate_candidate], Config())
    assert result[0].decision == Decision.REJECT
    assert result[0].reject_reason == RejectReason.DUPLICATE


def test_observe_insufficient_evidence(weak_candidate: Candidate) -> None:
    weak_candidate.evidence.append(
        Evidence(session_id="sess-011", source="codex",
                 date=datetime(2026, 3, 12, tzinfo=timezone.utc),
                 user_intent="Manage API keys", tools_used=["Bash"]))
    result = classify([weak_candidate], Config())
    assert result[0].decision == Decision.OBSERVE


def test_stale_observe_rejected() -> None:
    c = Candidate(proposed_name="stale", summary="old", observation_weeks=5)
    result = classify([c], Config(observe_ttl_days=28))
    assert result[0].decision == Decision.REJECT
    assert result[0].reject_reason == RejectReason.STALE


def test_mixed_batch(sample_candidate, weak_candidate, duplicate_candidate) -> None:
    result = classify([sample_candidate, weak_candidate, duplicate_candidate], Config())
    assert result[0].decision == Decision.ACCEPT
    assert result[1].decision == Decision.REJECT
    assert result[2].decision == Decision.REJECT
