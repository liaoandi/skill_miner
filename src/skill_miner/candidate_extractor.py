"""Step 2: Extract candidates from sessions, classify, redact, and output."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
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

_ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
_PROXY_API_KEY_ENV = "TAP_LLM_PROXY_API_KEY"
_PROXY_BASE_URL_ENV = "TAP_LLM_PROXY_BASE_URL"
_CENTRAL_ENV_FILE = Path("~/.config/api-keys.env").expanduser()
_DEFAULT_PROXY_BASE_URL = "https://llm-proxy.tapsvc.com/v1"
_DEFAULT_PROXY_TIMEOUT_SECONDS = 180.0
_MAX_PROMPT_CHARS = 24000
_MAX_PROMPT_SESSIONS = 24
_MAX_PROMPT_SESSIONS_PER_SOURCE = 8
_MAX_PROMPT_USER_ENTRIES_PER_SESSION = 3
_MODEL_ALIASES = {
    "claude-sonnet-4-20250514": {
        "proxy": "claude-sonnet-4-6",
        "anthropic": "claude-sonnet-4-20250514",
    },
    "claude-sonnet-4-6": {
        "proxy": "claude-sonnet-4-6",
        "anthropic": "claude-sonnet-4-20250514",
    },
}


def _read_central_env_var(key: str) -> str | None:
    if not _CENTRAL_ENV_FILE.exists():
        return None

    for raw_line in _CENTRAL_ENV_FILE.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        candidate_key, raw_value = line.split("=", 1)
        if candidate_key.strip() != key:
            continue
        return os.path.expandvars(raw_value.strip().strip('"').strip("'"))
    return None


def _get_env_var(key: str) -> str | None:
    return os.environ.get(key) or _read_central_env_var(key)


def _resolve_model_name(model: str, backend: str) -> str:
    alias = _MODEL_ALIASES.get(model)
    if not alias:
        return model
    return alias.get(backend, model)


def _flatten_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


@dataclass
class _CompatTextBlock:
    text: str


@dataclass
class _CompatMessageResponse:
    content: list[_CompatTextBlock]


class _ProxyMessages:
    def __init__(self, client, model_resolver):
        self._client = client
        self._model_resolver = model_resolver

    def create(self, *, model: str, max_tokens: int, messages: list[dict], system: str | None = None, **kwargs):
        proxy_messages: list[dict[str, str]] = []
        if system:
            proxy_messages.append({"role": "system", "content": system})
        for message in messages:
            proxy_messages.append({
                "role": str(message.get("role", "user")),
                "content": _flatten_message_content(message.get("content", "")),
            })

        response = self._client.chat.completions.create(
            model=self._model_resolver(model),
            messages=proxy_messages,
            max_tokens=max_tokens,
            **kwargs,
        )
        text = response.choices[0].message.content or ""
        return _CompatMessageResponse(content=[_CompatTextBlock(text=text)])


class _ProxyAnthropicCompatClient:
    def __init__(self, *, api_key: str, base_url: str, timeout: float = _DEFAULT_PROXY_TIMEOUT_SECONDS):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai support: pip install skill-miner")
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.messages = _ProxyMessages(
            self._client,
            lambda model: _resolve_model_name(model, "proxy"),
        )


class _AnthropicMessages:
    def __init__(self, client):
        self._client = client

    def create(self, *, model: str, **kwargs):
        return self._client.messages.create(
            model=_resolve_model_name(model, "anthropic"),
            **kwargs,
        )


class _AnthropicCompatClient:
    def __init__(self, *, api_key: str):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic fallback support: pip install skill-miner[anthropic]")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.messages = _AnthropicMessages(self._client)


def create_llm_client(config: Config):
    proxy_api_key = _get_env_var(_PROXY_API_KEY_ENV)
    if proxy_api_key:
        proxy_base_url = (_get_env_var(_PROXY_BASE_URL_ENV) or _DEFAULT_PROXY_BASE_URL).rstrip("/")
        return _ProxyAnthropicCompatClient(api_key=proxy_api_key, base_url=proxy_base_url)

    api_key = _get_env_var(_ANTHROPIC_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Set {_PROXY_API_KEY_ENV} for the internal proxy or {_ANTHROPIC_API_KEY_ENV} for direct Anthropic."
        )
    return _AnthropicCompatClient(api_key=api_key)


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
- summary: 用简体中文写 1-2 句话，说明这个 skill 做什么、适用在什么场景
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

_SUMMARY_TRANSLATION_PROMPT = """\
You translate skill candidate summaries into Simplified Chinese.

Rules:
- Keep `proposed_name` unchanged.
- Translate only the `summary` field.
- Use concise, natural Simplified Chinese.
- Preserve technical terms, filenames, commands, and product names when appropriate.
- Do not add new facts.
- Output valid JSON only in this schema:
{
  "items": [
    {
      "proposed_name": "string",
      "summary": "string"
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
        max_tokens=4096,
        system=_EXTRACTION_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "\n".join(b.text for b in response.content if hasattr(b, "text"))
    return _parse_llm_response(text)


def _select_sessions_for_prompt(sessions: list[Session]) -> list[Session]:
    selected: list[Session] = []
    per_source: dict[str, int] = {}
    for session in sorted(sessions, key=lambda s: s.modified, reverse=True):
        if len(selected) >= _MAX_PROMPT_SESSIONS:
            break
        if per_source.get(session.source, 0) >= _MAX_PROMPT_SESSIONS_PER_SOURCE:
            continue
        per_source[session.source] = per_source.get(session.source, 0) + 1
        selected.append(session)
    return selected


def _render_session_for_prompt(session: Session) -> str:
    lines = [
        f"--- Session {session.session_id} ---",
        f"Project: {session.project}",
        f"Date: {session.modified.isoformat()}",
    ]
    if session.first_prompt:
        lines.append(f"First prompt: {session.first_prompt[:180]}")
    if session.tools_used:
        lines.append(f"Tools: {', '.join(session.tools_used[:8])}")
    if session.summary:
        lines.append(f"Summary: {session.summary[:280]}")
    for entry in [e for e in session.entries if e.role == "user"][:_MAX_PROMPT_USER_ENTRIES_PER_SESSION]:
        lines.append(f"  User: {entry.content[:180]}")
    return "\n".join(lines)


def _build_user_prompt(sessions: list[Session], existing_skills: list[str]) -> str:
    sampled_sessions = _select_sessions_for_prompt(sessions)
    if len(sampled_sessions) < len(sessions):
        logger.info(
            "Prompt sampling enabled: using %d of %d sessions",
            len(sampled_sessions),
            len(sessions),
        )

    parts: list[str] = []
    if existing_skills:
        parts.append("## Existing skills (check for overlap)")
        parts.append(", ".join(existing_skills))
        parts.append("")

    by_source: dict[str, list[Session]] = {}
    for s in sampled_sessions:
        by_source.setdefault(s.source, []).append(s)

    parts.append(
        f"## Session data ({len(sessions)} sessions total, showing {len(sampled_sessions)} recent representative sessions)"
    )
    parts.append("")
    total_chars = sum(len(part) + 1 for part in parts)
    for source, ss in sorted(by_source.items()):
        section_header = f"### {source} ({len(ss)} sampled sessions)"
        if total_chars + len(section_header) + 1 > _MAX_PROMPT_CHARS:
            break
        parts.append(section_header)
        total_chars += len(section_header) + 1
        for s in ss:
            block = _render_session_for_prompt(s)
            if total_chars + len(block) + 2 > _MAX_PROMPT_CHARS:
                logger.info("Prompt char budget reached at %d characters", total_chars)
                break
            parts.append(f"\n{block}")
            total_chars += len(block) + 2
        if total_chars + 1 > _MAX_PROMPT_CHARS:
            break
        parts.append("")
        total_chars += 1

    parts.append("## Instructions")
    parts.append(
        "Analyze the sampled sessions above. Identify reusable workflow patterns that appear "
        "across multiple sessions. Focus on patterns involving tool use and multi-step "
        "workflows. Keep `proposed_name` in snake_case English, but write every candidate "
        "`summary` in Simplified Chinese. Output JSON only."
    )
    return "\n".join(parts)


def _parse_llm_response(response: str) -> list[Candidate]:
    json_str = _strip_code_fences(response)

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


def _strip_code_fences(text: str) -> str:
    json_str = text.strip()
    if json_str.startswith("```"):
        lines = json_str.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        json_str = "\n".join(lines)
    return json_str


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _candidate_evidence_keys(candidate: Candidate) -> set[str]:
    return {f"{e.source}:{e.session_id}" for e in candidate.evidence if e.session_id}


def _candidate_name_tokens(candidate: Candidate) -> set[str]:
    return {token for token in candidate.proposed_name.lower().split("_") if token}


def _candidate_text_features(candidate: Candidate) -> set[str]:
    text = " ".join([
        candidate.proposed_name.replace("_", " "),
        candidate.summary,
        candidate.extends or "",
        " ".join(sorted(candidate.unique_tools)),
    ]).lower()
    features = set(re.findall(r"[a-z0-9]+", text))
    cjk_chars = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    for n in (2, 3):
        for index in range(len(cjk_chars) - n + 1):
            features.add(cjk_chars[index:index + n])
    return features


def _is_semantic_duplicate(left: Candidate, right: Candidate) -> bool:
    if left.proposed_name == right.proposed_name:
        return True

    evidence_similarity = _jaccard_similarity(
        _candidate_evidence_keys(left),
        _candidate_evidence_keys(right),
    )
    name_similarity = _jaccard_similarity(
        _candidate_name_tokens(left),
        _candidate_name_tokens(right),
    )
    text_similarity = _jaccard_similarity(
        _candidate_text_features(left),
        _candidate_text_features(right),
    )
    shared_extension = bool(left.extends and left.extends == right.extends)
    shared_overlap = bool(left.overlap != "none" and left.overlap == right.overlap)

    if evidence_similarity >= 0.8 and text_similarity >= 0.18:
        return True
    if evidence_similarity >= 0.5 and text_similarity >= 0.18 and (shared_extension or shared_overlap):
        return True
    if evidence_similarity >= 0.5 and name_similarity >= 0.6:
        return True
    if (shared_extension or shared_overlap) and (name_similarity >= 0.6 or text_similarity >= 0.32):
        return True
    return False


def _candidate_sort_key(candidate: Candidate) -> tuple[int, int, int, int, str]:
    return (
        candidate.session_count,
        candidate.source_count,
        len(candidate.unique_tools),
        -candidate.observation_weeks,
        candidate.proposed_name,
    )


def _choose_preferred_candidate(left: Candidate, right: Candidate) -> tuple[Candidate, Candidate]:
    if _candidate_sort_key(left) > _candidate_sort_key(right):
        return left, right
    if _candidate_sort_key(right) > _candidate_sort_key(left):
        return right, left
    return (left, right) if left.proposed_name <= right.proposed_name else (right, left)


def _merge_candidate_pair(left: Candidate, right: Candidate) -> Candidate:
    primary, secondary = _choose_preferred_candidate(left, right)

    evidence_by_key: dict[str, Evidence] = {}
    for evidence in primary.evidence + secondary.evidence:
        key = f"{evidence.source}:{evidence.session_id}" if evidence.session_id else f"missing:{id(evidence)}"
        if key not in evidence_by_key:
            evidence_by_key[key] = evidence
    primary.evidence = sorted(
        evidence_by_key.values(),
        key=lambda evidence: evidence.date,
        reverse=True,
    )

    if (not re.search(r"[\u4e00-\u9fff]", primary.summary) and re.search(r"[\u4e00-\u9fff]", secondary.summary)):
        primary.summary = secondary.summary
    elif len(secondary.summary) > len(primary.summary) + 24:
        primary.summary = secondary.summary

    if primary.overlap == "none" and secondary.overlap != "none":
        primary.overlap = secondary.overlap
    if not primary.extends and secondary.extends:
        primary.extends = secondary.extends
    primary.privacy_flags = sorted(set(primary.privacy_flags) | set(secondary.privacy_flags))

    first_seen_candidates = [value for value in (primary.first_seen, secondary.first_seen) if value is not None]
    primary.first_seen = min(first_seen_candidates) if first_seen_candidates else None
    last_seen_candidates = [value for value in (primary.last_evidence_date, secondary.last_evidence_date) if value is not None]
    if last_seen_candidates:
        primary.last_evidence_date = max(last_seen_candidates)

    primary.observation_weeks = min(primary.observation_weeks, secondary.observation_weeks)
    primary.reviewer_note = primary.reviewer_note or secondary.reviewer_note
    if primary.reject_reason is None:
        primary.reject_reason = secondary.reject_reason
    return primary


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    deduped: list[Candidate] = []
    for candidate in candidates:
        match_index = next(
            (index for index, existing in enumerate(deduped) if _is_semantic_duplicate(existing, candidate)),
            None,
        )
        if match_index is None:
            deduped.append(candidate)
            continue
        deduped[match_index] = _merge_candidate_pair(deduped[match_index], candidate)
    return deduped


def _needs_chinese_summary(text: str) -> bool:
    return bool(text and not re.search(r"[\u4e00-\u9fff]", text))


def _translate_summaries_to_chinese(candidates: list[Candidate], config: Config) -> list[Candidate]:
    pending = [
        {"proposed_name": c.proposed_name, "summary": c.summary}
        for c in candidates
        if _needs_chinese_summary(c.summary)
    ]
    if not pending:
        return candidates

    logger.info("Translating %d candidate summaries to Chinese...", len(pending))
    client = create_llm_client(config)
    payload = json.dumps({"items": pending}, ensure_ascii=False, indent=2)
    response = client.messages.create(
        model=config.model,
        max_tokens=4096,
        system=_SUMMARY_TRANSLATION_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                "Translate the following candidate summaries into Simplified Chinese. "
                "Return JSON only.\n\n"
                f"{payload}"
            ),
        }],
    )
    text = "\n".join(b.text for b in response.content if hasattr(b, "text"))
    try:
        json_str = _strip_code_fences(text)
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("Summary translation JSON parse failed: %s", exc)
        return candidates
    translated = {
        item.get("proposed_name"): item.get("summary", "")
        for item in data.get("items", [])
        if item.get("proposed_name")
    }
    for candidate in candidates:
        new_summary = translated.get(candidate.proposed_name)
        if new_summary:
            candidate.summary = new_summary
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
            c.reject_reason = None
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
            c.reject_reason = None
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
            cur.observation_weeks = 0
        else:
            prev.observation_weeks += 1
            if prev.observation_weeks * 7 > config.observe_ttl_days:
                prev.decision = Decision.REJECT
                prev.reject_reason = RejectReason.STALE
            else:
                prev.decision = Decision.OBSERVE
            merged.append(prev)

    return _dedupe_candidates(merged)


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
    generated_at = datetime.now(timezone.utc)
    date_str = generated_at.strftime("%Y-%m-%d")
    paths: list[Path] = []

    accepted = [c for c in candidates if c.decision == Decision.ACCEPT]
    observed = [c for c in candidates if c.decision == Decision.OBSERVE]
    rejected = [c for c in candidates if c.decision == Decision.REJECT]

    # JSON
    json_path = output_dir / f"{date_str}_skill_candidates.json"
    json_payload = {
        "generated_at": generated_at.isoformat(),
        "summary": {
            "total": len(candidates), "accepted": len(accepted),
            "observed": len(observed), "rejected": len(rejected),
        },
        "candidates": [_candidate_to_dict(c) for c in candidates],
    }
    json_text = json.dumps(json_payload, indent=2, ensure_ascii=False, default=str)
    json_path.write_text(json_text)
    paths.append(json_path)

    # Markdown
    md_path = output_dir / f"{date_str}_skill_candidates.md"
    lines = [
        f"# 每周 Skill 候选提取 — {date_str}",
        "",
        "> 需要人工 review。当前还没有对 `~/.config/skillshare/skills/` 做任何修改。",
        "",
        "## 摘要",
        "",
        f"- 接受：{len(accepted)}",
        f"- 观察：{len(observed)}",
        f"- 拒绝：{len(rejected)}",
        "",
        "## Review 入口",
        "",
        f"- 当天 Markdown 队列：`{md_path}`",
        f"- 当天 JSON 队列：`{json_path}`",
        "- 固定 inbox：`~/.config/skillshare/review_queue/latest_review_inbox.md`",
        "- 最新 Markdown 队列：`~/.config/skillshare/review_queue/latest_skill_candidates.md`",
        "- 最新 JSON 队列：`~/.config/skillshare/review_queue/latest_skill_candidates.json`",
        "- 下一步：`uv run --project ~/Desktop/projects/skill_miner python -m skill_miner review`",
        "",
    ]

    for section, items in [("接受", accepted), ("观察", observed)]:
        if items:
            lines.append(f"## {section}\n")
            for c in items:
                lines.append(f"### {c.proposed_name}\n")
                lines.append(f"**摘要：** {c.summary}\n")
                if c.extends:
                    lines.append(f"**建议扩展到：** {c.extends}\n")
                if c.privacy_flags:
                    lines.append(f"**隐私标记：** {', '.join(c.privacy_flags)}\n")
                lines.append(f"**证据：** {c.session_count} 次会话，{c.source_count} 个来源\n")
                for ev in c.evidence:
                    lines.append(f"- **{ev.source}** ({ev.date.strftime('%Y-%m-%d')}): {ev.user_intent}")
                    if ev.tools_used:
                        lines.append(f"  - 工具：{', '.join(ev.tools_used)}")
                    for q in ev.key_quotes[:2]:
                        lines.append(f"  - > {q[:200]}")
                lines.append("")

    if rejected:
        lines.append("## 拒绝\n")
        for c in rejected:
            reason = c.reject_reason.value if c.reject_reason else "未说明"
            lines.append(f"- **{c.proposed_name}**: {reason} — {c.summary}")
        lines.append("")

    md_text = "\n".join(lines)
    md_path.write_text(md_text)
    paths.append(md_path)
    (output_dir / "latest_skill_candidates.json").write_text(json_text)
    (output_dir / "latest_skill_candidates.md").write_text(md_text)
    (output_dir / "latest_review_inbox.md").write_text("\n".join([
        "# 每周 Skill Review 收件箱",
        "",
        f"- 生成时间：{generated_at.isoformat()}",
        f"- 接受：{len(accepted)}",
        f"- 观察：{len(observed)}",
        f"- 拒绝：{len(rejected)}",
        "",
        "## 文件",
        "",
        f"- 当天 Markdown 队列：`{md_path}`",
        f"- 当天 JSON 队列：`{json_path}`",
        "- 最新 Markdown 队列：`~/.config/skillshare/review_queue/latest_skill_candidates.md`",
        "- 最新 JSON 队列：`~/.config/skillshare/review_queue/latest_skill_candidates.json`",
        "",
        "## 必要的人工步骤",
        "",
        "- 任何 skill 变更前，先 review 最新队列。",
        "- 运行 `uv run --project ~/Desktop/projects/skill_miner python -m skill_miner review` 记录决策。",
        "- 被接受的候选只会在 `~/.config/skillshare/review_queue/drafts/` 下生成 draft。",
        "- 当前还没有对 `~/.config/skillshare/skills/` 做任何修改。",
        "- 在 review 完成且明确批准 apply 前，不要执行 `skillshare sync`。",
        "",
    ]))
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

    logger.info("Normalizing summaries to Chinese...")
    redacted = _translate_summaries_to_chinese(redacted, config)

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
