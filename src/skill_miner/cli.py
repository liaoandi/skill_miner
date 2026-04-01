"""CLI interface."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from skill_miner import __version__


@click.group()
@click.version_option(__version__, prog_name="skill-miner")
@click.option("-v", "--verbose", is_flag=True, help="启用详细日志")
@click.option("--config", "config_path", type=click.Path(exists=True), help="配置文件路径")
@click.pass_context
def main(ctx: click.Context, verbose: bool, config_path: str | None) -> None:
    """从 AI agent 会话历史中挖掘可复用 skill。"""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = Path(config_path) if config_path else None


@main.command()
def init() -> None:
    """生成默认配置文件。"""
    from skill_miner.config import generate_default_config

    config_dir = Path("~/.config/skill_miner").expanduser()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    if config_file.exists() and not click.confirm(f"{config_file} 已存在，是否覆盖？"):
        return
    config_file.write_text(generate_default_config())
    click.echo(f"配置已写入：{config_file}")


@main.command()
@click.option("--days", type=int, help="扫描最近多少天的历史（默认：14）")
@click.option("--model", type=str, help="使用的 LLM 模型（默认：claude-sonnet-4-6）")
@click.pass_context
def scan(ctx: click.Context, days: int | None, model: str | None) -> None:
    """扫描会话历史并提取 skill 候选。"""
    from skill_miner.candidate_extractor import run
    from skill_miner.config import load_config

    config = load_config(ctx.obj["config_path"], days=days, model=model)
    result = run(config)

    if result.summary:
        click.echo(f"\naccept：{result.summary.accepted}")
        click.echo(f"observe：{result.summary.observed}")
        click.echo(f"reject：{result.summary.rejected}")
    for p in result.output_paths:
        click.echo(f"  {p}")
    click.echo("\n下一步运行 `skill-miner review` 进入人工 review。")


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """查看已检测 agent、观察状态和历史记录。"""
    from skill_miner.candidate_extractor import load_state
    from skill_miner.config import load_config
    from skill_miner.session_reader import get_source_status

    config = load_config(ctx.obj["config_path"])

    # Sources
    click.echo("Agents：")
    for s in get_source_status(config):
        icon = "+" if s["available"] and s["enabled"] else "-"
        click.echo(f"  [{icon}] {s['name']}：{s['session_count']} 次会话")

    # State
    state = load_state(config.state_dir)
    if not state.last_run:
        click.echo("\n还没有历史运行记录。")
        return

    click.echo(f"\n最近一次运行：{state.last_run.strftime('%Y-%m-%d %H:%M UTC')}")
    if state.observe_candidates:
        click.echo(f"正在观察：{len(state.observe_candidates)}")
        for c in state.observe_candidates:
            click.echo(f"  - {c.proposed_name}（第 {c.observation_weeks} 周，{c.session_count} 次会话）")
    if state.run_history:
        click.echo("\n历史：")
        for s in state.run_history[-5:]:
            click.echo(f"  {s.date.strftime('%Y-%m-%d')}: {s.accepted}A / {s.observed}O / {s.rejected}R")


@main.command()
@click.option("--queue", "queue_path", type=click.Path(exists=True), help="要 review 的 queue 文件")
@click.pass_context
def review(ctx: click.Context, queue_path: str | None) -> None:
    """人工 review 候选，并为 accepted 项生成 draft SKILL.md。"""
    from skill_miner.config import load_config
    from skill_miner.skill_generator import find_latest_queue, load_queue, run_review

    config = load_config(ctx.obj["config_path"])
    if queue_path:
        path = Path(queue_path)
    else:
        path = find_latest_queue(config.output_dir)
        if path is None:
            click.echo(f"{config.output_dir} 下还没有 queue 文件。\n请先运行 `skill-miner scan`。")
            return
    run_review(load_queue(path), path, config)
