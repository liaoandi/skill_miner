"""CLI interface."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from skill_miner import __version__


@click.group()
@click.version_option(__version__, prog_name="skill-miner")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def main(ctx: click.Context, verbose: bool, config_path: str | None) -> None:
    """Mine reusable skills from your AI agent session history."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = Path(config_path) if config_path else None


@main.command()
def init() -> None:
    """Generate default configuration file."""
    from skill_miner.config import generate_default_config

    config_dir = Path("~/.config/skill_miner").expanduser()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    if config_file.exists() and not click.confirm(f"{config_file} already exists. Overwrite?"):
        return
    config_file.write_text(generate_default_config())
    click.echo(f"Config written to: {config_file}")


@main.command()
@click.option("--days", type=int, help="How many days of history to scan (default: 14)")
@click.option("--model", type=str, help="LLM model to use (default: claude-sonnet-4-20250514)")
@click.pass_context
def scan(ctx: click.Context, days: int | None, model: str | None) -> None:
    """Scan session history and extract skill candidates."""
    from skill_miner.candidate_extractor import run
    from skill_miner.config import load_config

    config = load_config(ctx.obj["config_path"], days=days, model=model)
    result = run(config)

    if result.summary:
        click.echo(f"\nAccepted: {result.summary.accepted}")
        click.echo(f"Observed: {result.summary.observed}")
        click.echo(f"Rejected: {result.summary.rejected}")
    for p in result.output_paths:
        click.echo(f"  {p}")
    click.echo("\nRun `skill-miner review` to review candidates.")


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show detected agents, observation state, and run history."""
    from skill_miner.candidate_extractor import load_state
    from skill_miner.config import load_config
    from skill_miner.session_reader import get_source_status

    config = load_config(ctx.obj["config_path"])

    # Sources
    click.echo("Agents:")
    for s in get_source_status(config):
        icon = "+" if s["available"] and s["enabled"] else "-"
        click.echo(f"  [{icon}] {s['name']}: {s['session_count']} sessions")

    # State
    state = load_state(config.state_dir)
    if not state.last_run:
        click.echo("\nNo previous runs.")
        return

    click.echo(f"\nLast run: {state.last_run.strftime('%Y-%m-%d %H:%M UTC')}")
    if state.observe_candidates:
        click.echo(f"Observing: {len(state.observe_candidates)}")
        for c in state.observe_candidates:
            click.echo(f"  - {c.proposed_name} (week {c.observation_weeks}, {c.session_count} sessions)")
    if state.run_history:
        click.echo("\nHistory:")
        for s in state.run_history[-5:]:
            click.echo(f"  {s.date.strftime('%Y-%m-%d')}: {s.accepted}A / {s.observed}O / {s.rejected}R")


@main.command()
@click.option("--queue", "queue_path", type=click.Path(exists=True), help="Queue file to review")
@click.pass_context
def review(ctx: click.Context, queue_path: str | None) -> None:
    """Review candidates and generate SKILL.md for accepted ones."""
    from skill_miner.config import load_config
    from skill_miner.skill_generator import find_latest_queue, load_queue, run_review

    config = load_config(ctx.obj["config_path"])
    if queue_path:
        path = Path(queue_path)
    else:
        path = find_latest_queue(config.output_dir)
        if path is None:
            click.echo(f"No queue files in {config.output_dir}\nRun `skill-miner scan` first.")
            return
    run_review(load_queue(path), path, config)
