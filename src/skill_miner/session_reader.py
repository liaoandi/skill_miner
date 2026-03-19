"""Session source adapters for Claude Code, Codex, OpenClaw, and Gemini CLI."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from skill_miner.config import Config, Session, SessionEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_content(content_blocks: list) -> str:
    parts: list[str] = []
    for block in content_blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text", "")
            if block.get("type") in ("text", "input_text") and text:
                parts.append(text)
    return "\n".join(parts)


def _collect_tools(content_blocks: list) -> set[str]:
    tools: set[str] = set()
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") in ("tool_use", "function_call"):
            name = block.get("name", "")
            if name:
                tools.add(name)
    return tools


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

def _load_claude_code(session_dir: Path, cutoff: datetime) -> list[Session]:
    sessions: list[Session] = []
    for index_file in session_dir.rglob("sessions-index.json"):
        try:
            data = json.loads(index_file.read_text())
            entries = data.get("entries", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            continue

        for entry in entries:
            modified = _parse_ts(entry.get("modified", ""))
            if not modified or modified < cutoff:
                continue
            session_path = entry.get("fullPath", "")
            if not session_path or not Path(session_path).is_file():
                continue

            msg_entries: list[SessionEntry] = []
            tools: set[str] = set()
            try:
                for line in Path(session_path).read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = _parse_ts(obj.get("timestamp", ""))
                    if not ts or obj.get("type") not in ("user", "assistant"):
                        continue
                    msg = obj.get("message", {})
                    content = _extract_content(msg.get("content", []))
                    if msg.get("role") == "assistant":
                        tools |= _collect_tools(msg.get("content", []))
                    if content:
                        role = msg.get("role", obj["type"])
                        msg_entries.append(SessionEntry(timestamp=ts, role=role, content=content[:2000]))
            except OSError:
                continue

            sessions.append(Session(
                source="claude_code",
                session_id=entry.get("sessionId", ""),
                project=entry.get("projectPath", "unknown"),
                created=_parse_ts(entry.get("created", "")) or modified,
                modified=modified,
                entries=msg_entries,
                summary=entry.get("summary"),
                first_prompt=entry.get("firstPrompt"),
                tools_used=sorted(tools),
            ))

    logger.info("claude_code: loaded %d sessions", len(sessions))
    return sessions


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------

def _load_codex(session_dir: Path, cutoff: datetime) -> list[Session]:
    sessions: list[Session] = []
    for path in sorted(session_dir.rglob("rollout-*.jsonl")):
        try:
            if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) < cutoff:
                continue
        except OSError:
            continue

        entries: list[SessionEntry] = []
        tools: set[str] = set()
        session_id = path.stem
        project = "unknown"
        created: datetime | None = None
        modified: datetime | None = None

        try:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = _parse_ts(obj.get("timestamp", ""))
                t = obj.get("type", "")
                if t == "session_meta":
                    p = obj.get("payload", {})
                    session_id = p.get("id", session_id)
                    project = p.get("cwd", project)
                    if ts:
                        created = ts
                elif t == "response_item":
                    p = obj.get("payload", {})
                    role = p.get("role", "")
                    if role in ("user", "assistant", "developer"):
                        content = _extract_content(p.get("content", []))
                        if ts and content:
                            r = "user" if role == "developer" else role
                        entries.append(SessionEntry(timestamp=ts, role=r, content=content[:2000]))
                        if role == "assistant":
                            tools |= _collect_tools(p.get("content", []))
                if ts:
                    modified = ts
        except OSError:
            continue

        now = datetime.now(timezone.utc)
        sessions.append(Session(
            source="codex", session_id=session_id, project=project,
            created=created or now, modified=modified or now, entries=entries,
            first_prompt=entries[0].content[:200] if entries else None,
            tools_used=sorted(tools),
        ))

    logger.info("codex: loaded %d sessions", len(sessions))
    return sessions


# ---------------------------------------------------------------------------
# OpenClaw
# ---------------------------------------------------------------------------

def _load_openclaw(session_dir: Path, cutoff: datetime) -> list[Session]:
    sessions: list[Session] = []
    for path in sorted(session_dir.glob("*.jsonl*")):
        if ".deleted." in path.name:
            continue
        try:
            if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) < cutoff:
                continue
        except OSError:
            continue

        entries: list[SessionEntry] = []
        tools: set[str] = set()
        session_id = path.stem.split(".")[0]
        project = "unknown"
        created: datetime | None = None
        modified: datetime | None = None

        try:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = _parse_ts(obj.get("timestamp", ""))
                t = obj.get("type", "")
                if t == "session":
                    session_id = obj.get("id", session_id)
                    project = obj.get("cwd", project)
                    if ts:
                        created = ts
                elif t == "message":
                    msg = obj.get("message", {})
                    role = msg.get("role", "")
                    if role in ("user", "assistant"):
                        content = _extract_content(msg.get("content", []))
                        if ts and content:
                            entries.append(SessionEntry(timestamp=ts, role=role, content=content[:2000]))
                        if role == "assistant":
                            tools |= _collect_tools(msg.get("content", []))
                if ts:
                    modified = ts
        except OSError:
            continue

        now = datetime.now(timezone.utc)
        sessions.append(Session(
            source="openclaw", session_id=session_id, project=project,
            created=created or now, modified=modified or now, entries=entries,
            first_prompt=entries[0].content[:200] if entries else None,
            tools_used=sorted(tools),
        ))

    logger.info("openclaw: loaded %d sessions", len(sessions))
    return sessions


# ---------------------------------------------------------------------------
# Gemini (placeholder — minimal data available)
# ---------------------------------------------------------------------------

def _load_gemini(session_dir: Path, cutoff: datetime) -> list[Session]:
    count = 0
    for d in session_dir.iterdir():
        if d.is_dir():
            try:
                if datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc) >= cutoff:
                    count += 1
            except OSError:
                pass
    if count:
        logger.info("gemini: found %d session dirs but no extractable content yet", count)
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LOADERS = {
    "claude_code": _load_claude_code,
    "codex": _load_codex,
    "openclaw": _load_openclaw,
    "gemini": _load_gemini,
}


def load_all_sessions(config: Config) -> list[Session]:
    all_sessions: list[Session] = []
    for name, src in config.sources.items():
        if not src.enabled or not src.session_dir.is_dir():
            continue
        loader = _LOADERS.get(name)
        if loader is None:
            logger.warning("No adapter for source: %s", name)
            continue
        cutoff = datetime.now(timezone.utc) - timedelta(days=config.days)
        all_sessions.extend(loader(src.session_dir, cutoff))
    logger.info("Total sessions loaded: %d", len(all_sessions))
    return all_sessions


def get_source_status(config: Config) -> list[dict]:
    statuses: list[dict] = []
    for name, src in config.sources.items():
        available = src.session_dir.is_dir()
        count = 0
        if available and src.enabled:
            loader = _LOADERS.get(name)
            if loader:
                cutoff = datetime.now(timezone.utc) - timedelta(days=config.days)
                count = len(loader(src.session_dir, cutoff))
        statuses.append({
            "name": name, "enabled": src.enabled, "session_dir": str(src.session_dir),
            "available": available, "session_count": count,
        })
    return statuses


def load_existing_skills(skills_dir: Path | None) -> list[str]:
    if skills_dir is None or not skills_dir.is_dir():
        return []
    return sorted(d.name for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
