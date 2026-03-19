"""Step 2: Extract candidates from sessions, classify, redact, and output."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from skill_miner.config import (
    Candidate,
    Config,
    Decision,
    Evidence,
    PipelineResult,
    RejectReason,
    RunState,
    RunSummary,
    Session,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

_API_KEY_ENV = "ANTHROPIC_API_KEY"


def create_llm_client(config: Config):
    api_key = os.environ.get(_API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(f"Set {_API_KEY_ENV} environment variable.")
    try:
        import anthropic
    except ImportError:
        raise ImportError("Install anthropic: pip install skill-miner[anthropic]")
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a skill mining assistant. Your job is to analyze AI agent session histories \
and identify reusable workflow patterns that could be codified as "skills" — \
structured instructions that any AI coding agent can execute.

A good skill candidate:
- Appears across multiple sessions (ideally 3+)
- Involves multiple tools or steps
- Is generalizable beyond a single project or repo
- Can be described concisely enough for another LLM to execute
- Is not too generic (e.g., "write code" is not a skill)
- Is not too specific (e.g., tied to one repo's internal API)

For each candidate, provide:
- proposed_name: snake_case name
- summary: 1-2 sentence description of what the skill does
- evidence: list of sessions where this pattern appeared
- overlap: "none" or "extend:<existing_skill>" or "duplicate:<existing_skill>"
- privacy_flags: any sensitive data patterns noticed (hostnames, emails, keys, etc.)

Output ONLY valid JSON matching this schema:
{
  "candidates": [
    {
      "proposed_name": "string",
      "summary": "string",
      "evidence": [
        {
          "session_id": "string",
          "source": "string",
          "date": "ISO8601",
          "user_intent": "string",
          "key_quotes": ["string"],
          "tools_used": ["string"],
          "outcome": "worked|failed|partial"
        }
      ],
      "overlap": "none|extend:<name>|duplicate:<name>",
      "privacy_flags": ["string"]
    }
  ]
}
"""


def _extract_via_llm(sessions: list[Session], existing_skills: list[str], config: Config) -> list[Candidate]:
    if not sessions:
        return []

    client = create_llm_client(config)
    user_prompt = _build_user_prompt(sessions, existing_skills)

    logger.info("Sending %d sessions to LLM for pattern extraction...", len(sessions))
    response = client.messages.create(
        model=config.model,
        max_tokens=8192,
        system=_EXTRACTION_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "\n".join(b.text for b in response.content if hasattr(b, "text"))
    return _parse_llm_response(text)


def _build_user_prompt(sessions: list[Session], existing_skills: list[str]) -> str:
    parts: list[str] = []
    if existing_skills:
        parts.append("## Existing skills (check for overlap)")
        parts.append(", ".join(existing_skills))
        parts.append("")

    by_source: dict[str, list[Session]] = {}
    for s in sessions:
        by_source.setdefault(s.source, []).append(s)

    parts.append(f"## Session data ({len(sessions)} sessions total)")
    parts.append("")
    for source, ss in sorted(by_source.items()):
        parts.append(f"### {source} ({len(ss)} sessions)")
        for s in ss:
            parts.append(f"\n--- Session {s.session_id} ---")
            parts.append(f"Project: {s.project}")
            parts.append(f"Date: {s.modified.isoformat()}")
            if s.first_prompt:
                parts.append(f"First prompt: {s.first_prompt[:300]}")
            if s.tools_used:
                parts.append(f"Tools: {', '.join(s.tools_used)}")
            if s.summary:
                parts.append(f"Summary: {s.summary[:500]}")
            for entry in [e for e in s.entries if e.role == "user"][:5]:
                parts.append(f"  User: {entry.content[:300]}")
        parts.append("")

    parts.append("## Instructions")
    parts.append(
        "Analyze the sessions above. Identify reusable workflow patterns that appear "
        "across multiple sessions. Focus on patterns involving tool use and multi-step "
        "workflows. Output JSON only."
    )
    return "\n".join(parts)


def _parse_llm_response(response: str) -> list[Candidate]:
    json_str = response.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        json_str = "\n".join(lines)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON: %s", e)
        return []

    candidates: list[Candidate] = []
    for raw in data.get("candidates", []):
        evidence = []
        for ev in raw.get("evidence", []):
            date_str = ev.get("date", "")
            try:
                date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                date = datetime.now(timezone.utc)
            evidence.append(Evidence(
                session_id=ev.get("session_id", ""), source=ev.get("source", ""), date=date,
                user_intent=ev.get("user_intent", ""), key_quotes=ev.get("key_quotes", []),
                tools_used=ev.get("tools_used", []), outcome=ev.get("outcome", "unknown"),
            ))

        overlap = raw.get("overlap", "none")
        candidates.append(Candidate(
            proposed_name=raw.get("proposed_name", "unnamed"),
            summary=raw.get("summary", ""),
            evidence=evidence, overlap=overlap,
            extends=overlap.split(":", 1)[1] if overlap.startswith("extend:") else None,
            privacy_flags=raw.get("privacy_flags", []),
            first_seen=datetime.now(timezone.utc),
            last_evidence_date=max((e.date for e in evidence), default=datetime.now(timezone.utc)),
        ))

    logger.info("Extracted %d candidates from LLM response", len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------

def _classify(candidates: list[Candidate], config: Config) -> list[Candidate]:
    for c in candidates:
        if c.reviewer_note is not None:
            continue
        if c.session_count >= config.min_sessions and len(c.unique_tools) >= config.min_tools:
            c.decision = Decision.ACCEPT
        elif c.observation_weeks * 7 > config.observe_ttl_days:
            c.decision = Decision.REJECT
            c.reject_reason = RejectReason.STALE
        elif c.session_count <= 1:
            c.decision = Decision.REJECT
            c.reject_reason = RejectReason.ONE_OFF
        elif c.overlap.startswith("duplicate:"):
            c.decision = Decision.REJECT
            c.reject_reason = RejectReason.DUPLICATE
        else:
            c.decision = Decision.OBSERVE
    return candidates


# ---------------------------------------------------------------------------
# Redact
# ---------------------------------------------------------------------------

_REDACTION_RULES = [(re.compile(p), r) for p, r in [
    (r"\bsk-[A-Za-z0-9]{20,}\b", "[API_KEY]"),
    (r"\bAIza[A-Za-z0-9_-]{35}\b", "[API_KEY]"),
    (r"\bgsk_[A-Za-z0-9]{20,}\b", "[API_KEY]"),
    (r"\bghp_[A-Za-z0-9]{36}\b", "[GITHUB_TOKEN]"),
    (r"\bxoxb-[A-Za-z0-9-]+\b", "[SLACK_TOKEN]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP_ADDR]"),
    (r"/Users/[A-Za-z0-9._-]+", "/Users/[USER]"),
    (r"/home/[A-Za-z0-9._-]+", "/home/[USER]"),
]]


def _redact(candidates: list[Candidate]) -> list[Candidate]:
    def scrub(text: str) -> str:
        for pattern, replacement in _REDACTION_RULES:
            text = pattern.sub(replacement, text)
        return text

    for c in candidates:
        c.summary = scrub(c.summary)
        for ev in c.evidence:
            ev.user_intent = scrub(ev.user_intent)
            ev.key_quotes = [scrub(q) for q in ev.key_quotes]
    return candidates


# ---------------------------------------------------------------------------
# Carry-forward state
# ---------------------------------------------------------------------------

def load_state(state_dir: Path) -> RunState:
    state_file = state_dir / "observations.json"
    if not state_file.is_file():
        return RunState()
    try:
        raw = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        return RunState()

    last_run = None
    if raw.get("last_run"):
        try:
            last_run = datetime.fromisoformat(raw["last_run"])
        except ValueError:
            pass

    return RunState(
        last_run=last_run,
        observe_candidates=[_candidate_from_dict(c) for c in raw.get("observe_candidates", [])],
        run_history=[_summary_from_dict(s) for s in raw.get("run_history", [])],
    )


def _merge_with_prior(new_candidates: list[Candidate], prior: RunState, config: Config) -> list[Candidate]:
    new_by_name = {c.proposed_name: c for c in new_candidates}
    merged = list(new_candidates)

    for prev in prior.observe_candidates:
        if prev.proposed_name in new_by_name:
            cur = new_by_name[prev.proposed_name]
            existing_ids = {e.session_id for e in cur.evidence}
            for ev in prev.evidence:
                if ev.session_id not in existing_ids:
                    cur.evidence.append(ev)
            cur.first_seen = prev.first_seen
            cur.observation_weeks = prev.observation_weeks + 1
        else:
            prev.observation_weeks += 1
            if prev.observation_weeks * 7 > config.observe_ttl_days:
                prev.decision = Decision.REJECT
                prev.reject_reason = RejectReason.STALE
            else:
                prev.decision = Decision.OBSERVE
            merged.append(prev)

    return merged


def _save_state(state_dir: Path, candidates: list[Candidate]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "observations.json"
    existing = load_state(state_dir)

    observe = [c for c in candidates if c.decision == Decision.OBSERVE]
    summary = RunSummary(
        date=datetime.now(timezone.utc),
        accepted=sum(1 for c in candidates if c.decision == Decision.ACCEPT),
        observed=len(observe),
        rejected=sum(1 for c in candidates if c.decision == Decision.REJECT),
    )
    history = existing.run_history[-19:] + [summary]

    state_file.write_text(json.dumps({
        "last_run": datetime.now(timezone.utc).isoformat(),
        "observe_candidates": [_candidate_to_dict(c) for c in observe],
        "run_history": [_summary_to_dict(s) for s in history],
    }, indent=2, default=str))
    logger.info("Saved state: %d observe, %d history", len(observe), len(history))


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

def _write_output(candidates: list[Candidate], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    paths: list[Path] = []

    accepted = [c for c in candidates if c.decision == Decision.ACCEPT]
    observed = [c for c in candidates if c.decision == Decision.OBSERVE]
    rejected = [c for c in candidates if c.decision == Decision.REJECT]

    # JSON
    json_path = output_dir / f"{date_str}_skill_candidates.json"
    json_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": len(candidates), "accepted": len(accepted),
            "observed": len(observed), "rejected": len(rejected),
        },
        "candidates": [_candidate_to_dict(c) for c in candidates],
    }, indent=2, ensure_ascii=False, default=str))
    paths.append(json_path)

    # Markdown
    md_path = output_dir / f"{date_str}_skill_candidates.md"
    lines = [f"# Skill Candidates — {date_str}", ""]
    lines += [f"- Accept: {len(accepted)}", f"- Observe: {len(observed)}", f"- Reject: {len(rejected)}", ""]

    for section, items in [("Accept", accepted), ("Observe", observed)]:
        if items:
            lines.append(f"## {section}\n")
            for c in items:
                lines.append(f"### {c.proposed_name}\n")
                lines.append(f"**Summary:** {c.summary}\n")
                if c.extends:
                    lines.append(f"**Extends:** {c.extends}\n")
                if c.privacy_flags:
                    lines.append(f"**Privacy flags:** {', '.join(c.privacy_flags)}\n")
                lines.append(f"**Evidence:** {c.session_count} sessions, {c.source_count} sources\n")
                for ev in c.evidence:
                    lines.append(f"- **{ev.source}** ({ev.date.strftime('%Y-%m-%d')}): {ev.user_intent}")
                    if ev.tools_used:
                        lines.append(f"  - Tools: {', '.join(ev.tools_used)}")
                    for q in ev.key_quotes[:2]:
                        lines.append(f"  - > {q[:200]}")
                lines.append("")

    if rejected:
        lines.append("## Reject\n")
        for c in rejected:
            reason = c.reject_reason.value if c.reject_reason else "unspecified"
            lines.append(f"- **{c.proposed_name}**: {reason} — {c.summary}")
        lines.append("")

    md_path.write_text("\n".join(lines))
    paths.append(md_path)
    logger.info("Wrote output: %s", ", ".join(str(p) for p in paths))
    return paths


# ---------------------------------------------------------------------------
# Main entry point: run the full extraction
# ---------------------------------------------------------------------------

def run(config: Config) -> PipelineResult:
    from skill_miner.session_reader import load_all_sessions, load_existing_skills

    logger.info("Loading sessions...")
    sessions = load_all_sessions(config)
    if not sessions:
        logger.warning("No sessions found.")
        return PipelineResult(candidates=[], summary=RunSummary(date=datetime.now(timezone.utc)))

    prior = load_state(config.state_dir)
    existing_skills = load_existing_skills(config.existing_skills_dir)

    logger.info("Extracting patterns via LLM...")
    raw = _extract_via_llm(sessions, existing_skills, config)

    logger.info("Merging with prior observations...")
    merged = _merge_with_prior(raw, prior, config)

    logger.info("Classifying...")
    classified = _classify(merged, config)

    logger.info("Redacting...")
    redacted = _redact(classified)

    logger.info("Writing output...")
    paths = _write_output(redacted, config.output_dir)

    logger.info("Saving state...")
    _save_state(config.state_dir, redacted)

    summary = RunSummary(
        date=datetime.now(timezone.utc),
        accepted=sum(1 for c in redacted if c.decision == Decision.ACCEPT),
        observed=sum(1 for c in redacted if c.decision == Decision.OBSERVE),
        rejected=sum(1 for c in redacted if c.decision == Decision.REJECT),
    )
    logger.info("Done: %dA / %dO / %dR", summary.accepted, summary.observed, summary.rejected)
    return PipelineResult(candidates=redacted, output_paths=paths, summary=summary)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _candidate_to_dict(c: Candidate) -> dict:
    return {
        "proposed_name": c.proposed_name, "summary": c.summary,
        "decision": c.decision.value,
        "reject_reason": c.reject_reason.value if c.reject_reason else None,
        "overlap": c.overlap, "extends": c.extends,
        "privacy_flags": c.privacy_flags,
        "session_count": c.session_count, "source_count": c.source_count,
        "unique_tools": sorted(c.unique_tools),
        "observation_weeks": c.observation_weeks,
        "first_seen": c.first_seen.isoformat() if c.first_seen else None,
        "last_evidence_date": c.last_evidence_date.isoformat() if c.last_evidence_date else None,
        "evidence": [
            {"session_id": e.session_id, "source": e.source, "date": e.date.isoformat(),
             "user_intent": e.user_intent, "key_quotes": e.key_quotes,
             "tools_used": e.tools_used, "outcome": e.outcome}
            for e in c.evidence
        ],
    }


def _candidate_from_dict(d: dict) -> Candidate:
    evidence = []
    for ev in d.get("evidence", []):
        try:
            date = datetime.fromisoformat(ev["date"])
        except (ValueError, KeyError):
            date = datetime.now(timezone.utc)
        evidence.append(Evidence(
            session_id=ev.get("session_id", ""), source=ev.get("source", ""), date=date,
            user_intent=ev.get("user_intent", ""), key_quotes=ev.get("key_quotes", []),
            tools_used=ev.get("tools_used", []), outcome=ev.get("outcome", "unknown"),
        ))

    def _parse_dt(key: str) -> datetime | None:
        val = d.get(key)
        if val:
            try:
                return datetime.fromisoformat(val)
            except ValueError:
                pass
        return None

    return Candidate(
        proposed_name=d.get("proposed_name", ""), summary=d.get("summary", ""),
        evidence=evidence, overlap=d.get("overlap", "none"), extends=d.get("extends"),
        decision=Decision(d.get("decision", "observe")),
        privacy_flags=d.get("privacy_flags", []),
        first_seen=_parse_dt("first_seen"), last_evidence_date=_parse_dt("last_evidence_date"),
        observation_weeks=d.get("observation_weeks", 0),
    )


def _summary_to_dict(s: RunSummary) -> dict:
    return {"date": s.date.isoformat(), "accepted": s.accepted, "observed": s.observed, "rejected": s.rejected}


def _summary_from_dict(d: dict) -> RunSummary:
    try:
        date = datetime.fromisoformat(d["date"])
    except (ValueError, KeyError):
        date = datetime.now(timezone.utc)
    return RunSummary(
        date=date, accepted=d.get("accepted", 0), observed=d.get("observed", 0), rejected=d.get("rejected", 0)
    )
