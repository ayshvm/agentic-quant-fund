from pathlib import Path

from hedge_fund.paths import resolve_user_dir


def test_default_user_dir(monkeypatch):
    monkeypatch.delenv("AGENTIC_QUANT_HOME", raising=False)
    assert resolve_user_dir() == Path.home() / ".hedge-fund"


def test_configurable_user_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_QUANT_HOME", str(tmp_path / "aqf"))
    assert resolve_user_dir() == tmp_path / "aqf"


def test_expands_tilde(monkeypatch):
    monkeypatch.setenv("AGENTIC_QUANT_HOME", "~/aqf-data")
    assert resolve_user_dir() == Path.home() / "aqf-data"
