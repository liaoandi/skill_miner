"""Interactive review + SKILL.md generation for accepted candidates."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import click

from skill_miner.candidate_extractor import _candidate_from_dict
from skill_miner.config import Candidate, Config, Decision

logger = logging.getLogger(__name__)


def find_latest_queue(output_dir: Path) -> Path | None:
    json_files = sorted(output_dir.glob("*_skill_candidates.json"), reverse=True)
    return json_files[0] if json_files else None


def load_queue(queue_path: Path) -> list[Candidate]:
    try:
        data = json.loads(queue_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load queue %s: %s", queue_path, e)
        return []
    return [_candidate_from_dict(c) for c in data.get("candidates", [])]


def run_review(candidates: list[Candidate], queue_path: Path, config: Config) -> None:
    if not candidates:
        click.echo("No candidates to review.")
        return

    order = {Decision.ACCEPT: 0, Decision.OBSERVE: 1, Decision.REJECT: 2}
    candidates.sort(key=lambda c: order.get(c.decision, 9))

    click.echo(f"\nReviewing {len(candidates)} candidates from {queue_path.name}")
    click.echo("Commands: a=accept, o=observe, r=reject, s=skip, q=quit\n")
    click.echo("-" * 60)

    reviewed = 0
    accepted_candidates: list[Candidate] = []
    decisions: dict[str, str] = {}

    for i, c in enumerate(candidates, 1):
        click.echo(f"\n[{i}/{len(candidates)}] {c.proposed_name}")
        click.echo(f"  Recommendation: {c.decision.value}")
        click.echo(f"  Summary: {c.summary}")
        click.echo(f"  Evidence: {c.session_count} sessions, {c.source_count} sources")
        if c.extends:
            click.echo(f"  Extends: {c.extends}")
        if c.privacy_flags:
            click.echo(f"  Privacy flags: {', '.join(c.privacy_flags)}")
        if c.evidence:
            for ev in c.evidence[:3]:
                click.echo(f"    [{ev.source}] {ev.date.strftime('%Y-%m-%d')}: {ev.user_intent}")

        click.echo()
        choices = click.Choice(["a", "o", "r", "s", "q"], case_sensitive=False)
        choice = click.prompt("  Decision", type=choices, default="s")

        if choice == "q":
            break
        elif choice == "s":
            continue
        elif choice == "a":
            c.decision = Decision.ACCEPT
            accepted_candidates.append(c)
            decisions[c.proposed_name] = "accept"
            reviewed += 1
        elif choice == "o":
            c.decision = Decision.OBSERVE
            decisions[c.proposed_name] = "observe"
            reviewed += 1
        elif choice == "r":
            c.decision = Decision.REJECT
            decisions[c.proposed_name] = "reject"
            reviewed += 1

    # Save decisions
    if decisions:
        decision_path = queue_path.with_name(
            queue_path.stem.replace("_skill_candidates", "_review_decisions") + ".json"
        )
        decision_path.write_text(json.dumps({
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "decisions": decisions,
        }, indent=2))

    # Generate SKILL.md for accepted candidates
    if accepted_candidates:
        click.echo(f"\nGenerating SKILL.md for {len(accepted_candidates)} accepted candidates...")
        for c in accepted_candidates:
            _generate_skill(c, config)

    click.echo(f"\n{'=' * 60}")
    a = sum(1 for d in decisions.values() if d == "accept")
    o = sum(1 for d in decisions.values() if d == "observe")
    r = sum(1 for d in decisions.values() if d == "reject")
    click.echo(f"Done: {a} accepted, {o} observed, {r} rejected")


def _generate_skill(candidate: Candidate, config: Config) -> None:
    """Generate SKILL.md for an accepted candidate using LLM."""
    from skill_miner.candidate_extractor import create_llm_client

    evidence_text = "\n".join(
        f"- [{ev.source}] {ev.user_intent}" + (f" (tools: {', '.join(ev.tools_used)})" if ev.tools_used else "")
        for ev in candidate.evidence
    )

    prompt = f"""Generate a SKILL.md file for the following skill candidate.

Name: {candidate.proposed_name}
Summary: {candidate.summary}
Evidence from session history:
{evidence_text}

Follow the Agent Skills open standard. Output ONLY the SKILL.md content (including frontmatter).

Requirements:
- YAML frontmatter with `name` and `description` fields
- `description` should be a clear sentence describing when to use this skill
- Content should be actionable step-by-step instructions that an AI agent can execute
- Keep it concise (under 100 lines)
- Do not include any sensitive data (API keys, emails, hostnames, paths)
"""

    try:
        client = create_llm_client(config)
        response = client.messages.create(
            model=config.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        content = "\n".join(b.text for b in response.content if hasattr(b, "text"))
    except Exception as e:
        click.echo(f"  Failed to generate {candidate.proposed_name}: {e}")
        return

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        content = "\n".join(lines)

    # Write to output directory
    skill_dir = config.output_dir / "skills" / candidate.proposed_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(content.strip() + "\n")
    click.echo(f"  Generated: {skill_path}")
