"""Tests for session loading utilities."""

from pathlib import Path

from skill_miner.session_reader import load_existing_skills


def test_load_existing_skills(tmp_path: Path) -> None:
    (tmp_path / "review_changes").mkdir()
    (tmp_path / "cleanup_project").mkdir()
    (tmp_path / ".hidden").mkdir()
    assert load_existing_skills(tmp_path) == ["cleanup_project", "review_changes"]


def test_load_existing_skills_none() -> None:
    assert load_existing_skills(None) == []
