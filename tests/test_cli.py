"""Tests for CLI commands."""

from click.testing import CliRunner

from skill_miner.cli import main


def test_version() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_init_generates_config() -> None:
    from skill_miner.config import generate_default_config
    content = generate_default_config()
    assert "sources" in content


def test_status_command() -> None:
    result = CliRunner().invoke(main, ["status"])
    assert result.exit_code == 0
    assert "Agents:" in result.output
