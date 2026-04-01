"""Tests for observation carry-forward logic."""

from datetime import datetime, timezone

from skill_miner.candidate_extractor import _merge_with_prior as merge
from skill_miner.candidate_extractor import _save_state as save_state
from skill_miner.candidate_extractor import load_state
from skill_miner.config import Candidate, Config, Decision, Evidence, RejectReason, RunState


def test_merge_new_candidate_no_prior() -> None:
    c = Candidate(proposed_name="new_skill", summary="A new pattern")
    result = merge([c], RunState(), Config())
    assert len(result) == 1


def test_merge_accumulates_evidence() -> None:
    prior = RunState(
        last_run=datetime(2026, 3, 10, tzinfo=timezone.utc),
        observe_candidates=[Candidate(
            proposed_name="recurring_pattern", summary="Some pattern",
            evidence=[Evidence(session_id="old-001", source="claude_code",
                               date=datetime(2026, 3, 5, tzinfo=timezone.utc),
                               user_intent="Old evidence", tools_used=["Bash"])],
            first_seen=datetime(2026, 3, 5, tzinfo=timezone.utc), observation_weeks=2,
        )],
    )
    new = Candidate(
        proposed_name="recurring_pattern", summary="Same pattern",
        evidence=[Evidence(session_id="new-001", source="codex",
                           date=datetime(2026, 3, 15, tzinfo=timezone.utc),
                           user_intent="New evidence", tools_used=["Read"])],
    )
    result = merge([new], prior, Config())
    matching = [c for c in result if c.proposed_name == "recurring_pattern"]
    assert len(matching) == 1
    assert len(matching[0].evidence) == 2
    assert matching[0].observation_weeks == 0


def test_merge_expires_stale_observe() -> None:
    prior = RunState(observe_candidates=[
        Candidate(proposed_name="stale", summary="Old", observation_weeks=4)])
    result = merge([], prior, Config(observe_ttl_days=28))
    assert result[0].decision == Decision.REJECT
    assert result[0].reject_reason == RejectReason.STALE


def test_merge_keeps_within_ttl() -> None:
    prior = RunState(observe_candidates=[
        Candidate(proposed_name="young", summary="Recent", observation_weeks=1)])
    result = merge([], prior, Config(observe_ttl_days=28))
    assert result[0].decision == Decision.OBSERVE
    assert result[0].observation_weeks == 2


def test_save_and_load_state(tmp_path) -> None:
    candidates = [
        Candidate(proposed_name="obs", summary="observed", decision=Decision.OBSERVE,
                  evidence=[Evidence(session_id="s1", source="test",
                                     date=datetime(2026, 3, 15, tzinfo=timezone.utc),
                                     user_intent="test", tools_used=["Bash"])],
                  first_seen=datetime(2026, 3, 10, tzinfo=timezone.utc), observation_weeks=1),
        Candidate(proposed_name="acc", summary="accepted", decision=Decision.ACCEPT),
    ]
    save_state(tmp_path, candidates)
    loaded = load_state(tmp_path)
    assert len(loaded.observe_candidates) == 1
    assert loaded.observe_candidates[0].proposed_name == "obs"
    assert loaded.last_run is not None
    assert loaded.run_history[0].accepted == 1
