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
    json_files = sorted(
        (path for path in output_dir.glob("*_skill_candidates.json") if not path.name.startswith("latest_")),
        reverse=True,
    )
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
        click.echo("当前没有可 review 的候选。")
        return

    order = {Decision.ACCEPT: 0, Decision.OBSERVE: 1, Decision.REJECT: 2}
    candidates.sort(key=lambda c: order.get(c.decision, 9))

    click.echo(f"\n正在 review {len(candidates)} 个候选，来源文件：{queue_path.name}")
    click.echo("这是人工 review 闸门：这一步只记录决策，并按需生成 draft。")
    click.echo("review 过程中不会修改 ~/.config/skillshare/skills/。")
    click.echo("输入说明：a=接受，o=继续观察，r=拒绝，s=跳过，q=退出\n")
    click.echo("-" * 60)

    reviewed = 0
    accepted_candidates: list[Candidate] = []
    decisions: dict[str, str] = {}

    for i, c in enumerate(candidates, 1):
        click.echo(f"\n[{i}/{len(candidates)}] {c.proposed_name}")
        click.echo(f"  推荐决策：{c.decision.value}")
        click.echo(f"  摘要：{c.summary}")
        click.echo(f"  证据：{c.session_count} 次会话，{c.source_count} 个来源")
        if c.extends:
            click.echo(f"  建议扩展：{c.extends}")
        if c.privacy_flags:
            click.echo(f"  隐私标记：{', '.join(c.privacy_flags)}")
        if c.evidence:
            for ev in c.evidence[:3]:
                click.echo(f"    [{ev.source}] {ev.date.strftime('%Y-%m-%d')}: {ev.user_intent}")

        click.echo()
        choices = click.Choice(["a", "o", "r", "s", "q"], case_sensitive=False)
        choice = click.prompt("  你的决策", type=choices, default="s")

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
        accepted = sorted(name for name, decision in decisions.items() if decision == "accept")
        observed = sorted(name for name, decision in decisions.items() if decision == "observe")
        rejected = sorted(name for name, decision in decisions.items() if decision == "reject")
        decision_text = json.dumps({
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "queue_file": str(queue_path),
            "status": "pending_apply" if accepted else "review_complete",
            "requires_apply": bool(accepted),
            "note": "仅完成 review 记录；尚未对 ~/.config/skillshare/skills/ 做任何修改。",
            "accepted": accepted,
            "observed": observed,
            "rejected": rejected,
            "decisions": decisions,
        }, indent=2)
        decision_path.write_text(decision_text)
        (config.output_dir / "latest_review_decisions.json").write_text(decision_text)
        click.echo(f"已保存 review 决策：{decision_path}")

    # Generate SKILL.md for accepted candidates
    if accepted_candidates:
        click.echo(f"\n正在为 {len(accepted_candidates)} 个已接受候选生成 draft SKILL.md...")
        for c in accepted_candidates:
            _generate_skill(c, config)

    click.echo(f"\n{'=' * 60}")
    a = sum(1 for d in decisions.values() if d == "accept")
    o = sum(1 for d in decisions.values() if d == "observe")
    r = sum(1 for d in decisions.values() if d == "reject")
    click.echo(f"完成：{a} 个接受，{o} 个观察，{r} 个拒绝")


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
        click.echo(f"  生成 {candidate.proposed_name} 失败：{e}")
        return

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        content = "\n".join(lines)

    # Write to review drafts directory. Apply/sync remains a separate manual stage.
    draft_dir = config.output_dir / "drafts" / candidate.proposed_name
    draft_dir.mkdir(parents=True, exist_ok=True)
    skill_path = draft_dir / "SKILL.md"
    skill_path.write_text(content.strip() + "\n")
    click.echo(f"  已生成 draft：{skill_path}")
