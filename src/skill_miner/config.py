"""Configuration and data models."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    ACCEPT = "accept"
    OBSERVE = "observe"
    REJECT = "reject"


class RejectReason(str, Enum):
    SPECIFIC = "too_specific"
    GENERIC = "too_generic"
    DUPLICATE = "duplicate"
    ONE_OFF = "one_off"
    STALE = "stale"


@dataclass
class SessionEntry:
    timestamp: datetime
    role: str
    content: str
    tool_name: str | None = None


@dataclass
class Session:
    source: str
    session_id: str
    project: str
    created: datetime
    modified: datetime
    entries: list[SessionEntry] = field(default_factory=list)
    summary: str | None = None
    first_prompt: str | None = None
    tools_used: list[str] = field(default_factory=list)


@dataclass
class Evidence:
    session_id: str
    source: str
    date: datetime
    user_intent: str
    key_quotes: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    outcome: str = "unknown"


@dataclass
class Candidate:
    proposed_name: str
    summary: str
    evidence: list[Evidence] = field(default_factory=list)
    overlap: str = "none"
    extends: str | None = None
    decision: Decision = Decision.OBSERVE
    reject_reason: RejectReason | None = None
    privacy_flags: list[str] = field(default_factory=list)
    first_seen: datetime | None = None
    last_evidence_date: datetime | None = None
    observation_weeks: int = 0
    reviewer_note: str | None = None

    @property
    def session_count(self) -> int:
        return len(self.evidence)

    @property
    def source_count(self) -> int:
        return len({e.source for e in self.evidence})

    @property
    def unique_tools(self) -> set[str]:
        tools: set[str] = set()
        for e in self.evidence:
            tools.update(e.tools_used)
        return tools


@dataclass
class RunSummary:
    date: datetime
    accepted: int = 0
    observed: int = 0
    rejected: int = 0
    sources_scanned: dict[str, int] = field(default_factory=dict)


@dataclass
class RunState:
    last_run: datetime | None = None
    observe_candidates: list[Candidate] = field(default_factory=list)
    run_history: list[RunSummary] = field(default_factory=list)


@dataclass
class PipelineResult:
    candidates: list[Candidate]
    output_paths: list[Path] = field(default_factory=list)
    summary: RunSummary | None = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SourceConfig:
    enabled: bool = True
    session_dir: Path = Path()


@dataclass
class Config:
    sources: dict[str, SourceConfig] = field(default_factory=dict)
    # User-facing options (CLI flags)
    days: int = 14
    model: str = "claude-sonnet-4-20250514"
    # Advanced (config file only)
    existing_skills_dir: Path | None = None
    min_sessions: int = 3
    min_tools: int = 2
    observe_ttl_days: int = 28
    # Internal (not exposed)
    output_dir: Path = field(default_factory=lambda: _resolve("~/.config/skill_miner/review_queue/"))
    state_dir: Path = field(default_factory=lambda: _resolve("~/.config/skill_miner/state/"))


def _resolve(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))


_DEFAULT_SOURCES = {
    "claude_code": "~/.claude/projects/",
    "codex": "~/.codex/sessions/",
    "openclaw": "~/.openclaw/agents/main/sessions/",
    "gemini": "~/.gemini/history/",
}


def load_config(path: Path | None = None, days: int | None = None, model: str | None = None) -> Config:
    """Load config. Auto-detects agents, reads API key from env. All options are optional."""
    if path is None:
        for candidate in [Path("./skill_miner.yaml"), Path("~/.config/skill_miner/config.yaml")]:
            resolved = _resolve(str(candidate))
            if resolved.is_file():
                path = resolved
                break

    raw: dict = {}
    if path is not None:
        resolved = _resolve(str(path))
        if resolved.is_file():
            raw = yaml.safe_load(resolved.read_text()) or {}

    # Sources: auto-detect by default
    sources: dict[str, SourceConfig] = {}
    raw_sources = raw.get("sources", {})
    for name, default_dir in _DEFAULT_SOURCES.items():
        if name in raw_sources:
            src = raw_sources[name]
            sources[name] = SourceConfig(
                enabled=src.get("enabled", True),
                session_dir=_resolve(src.get("session_dir", default_dir)),
            )
        else:
            resolved_dir = _resolve(default_dir)
            sources[name] = SourceConfig(enabled=resolved_dir.is_dir(), session_dir=resolved_dir)

    raw_skills = raw.get("existing_skills_dir")

    return Config(
        sources=sources,
        days=days or raw.get("days", 14),
        model=model or raw.get("model", "claude-sonnet-4-20250514"),
        existing_skills_dir=_resolve(raw_skills) if raw_skills else None,
        min_sessions=raw.get("min_sessions", 3),
        min_tools=raw.get("min_tools", 2),
        observe_ttl_days=raw.get("observe_ttl_days", 28),
    )


def generate_default_config() -> str:
    return """\
# skill_miner configuration (all fields optional)
# Without this file, skill-miner auto-detects installed agents.

# Override source detection
# sources:
#   claude_code:
#     enabled: false
#   codex:
#     session_dir: ~/custom/path/

# Existing skills directory for overlap detection
# existing_skills_dir: ~/.config/skillshare/skills/
"""
