"""Tests for privacy redaction."""

from datetime import datetime, timezone

from skill_miner.candidate_extractor import _redact as redact
from skill_miner.config import Candidate, Evidence


def test_redact_api_keys() -> None:
    c = Candidate(proposed_name="test", summary="Using key sk-abc123456789012345678901 in production",
                  evidence=[Evidence(session_id="s1", source="test", date=datetime.now(timezone.utc),
                                     user_intent="Set ANTHROPIC_API_KEY=sk-abc123456789012345678901", tools_used=[])])
    result = redact([c])
    assert "sk-abc" not in result[0].summary
    assert "[API_KEY]" in result[0].summary
    assert "sk-abc" not in result[0].evidence[0].user_intent


def test_redact_email() -> None:
    c = Candidate(proposed_name="test", summary="Contact alice@company.com for access")
    result = redact([c])
    assert "[EMAIL]" in result[0].summary


def test_redact_absolute_paths() -> None:
    c = Candidate(proposed_name="test", summary="Edit /Users/antonio/.config/file.yaml")
    result = redact([c])
    assert "/Users/antonio" not in result[0].summary
    assert "/Users/[USER]" in result[0].summary


def test_redact_key_quotes() -> None:
    c = Candidate(proposed_name="test", summary="clean",
                  evidence=[Evidence(session_id="s1", source="test", date=datetime.now(timezone.utc),
                                     user_intent="clean",
                                     key_quotes=["Check ghp_abcdefghijklmnopqrstuvwxyz1234567890 token"],
                                     tools_used=[])])
    result = redact([c])
    assert "[GITHUB_TOKEN]" in result[0].evidence[0].key_quotes[0]
