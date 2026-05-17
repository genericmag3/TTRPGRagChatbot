"""Tests for the build-mode flag (local single-user vs remote multi-user)."""
import importlib

import src.app_config as app_config


def _reload():
    return importlib.reload(app_config)


class TestModeResolution:
    def test_defaults_to_local_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("TTRPG_APP_MODE", raising=False)
        cfg = _reload()
        assert cfg.get_mode() == cfg.LOCAL
        assert cfg.is_local() is True
        assert cfg.is_remote() is False

    def test_remote_when_env_is_remote(self, monkeypatch):
        monkeypatch.setenv("TTRPG_APP_MODE", "remote")
        cfg = _reload()
        assert cfg.get_mode() == cfg.REMOTE
        assert cfg.is_remote() is True
        assert cfg.is_local() is False

    def test_env_value_is_case_insensitive_and_trimmed(self, monkeypatch):
        monkeypatch.setenv("TTRPG_APP_MODE", "  ReMoTe  ")
        cfg = _reload()
        assert cfg.is_remote() is True

    def test_unknown_value_falls_back_to_local(self, monkeypatch):
        monkeypatch.setenv("TTRPG_APP_MODE", "banana")
        cfg = _reload()
        assert cfg.is_local() is True

    def test_mode_is_evaluated_dynamically(self, monkeypatch):
        """get_mode must read the environment at call time, not import time."""
        monkeypatch.delenv("TTRPG_APP_MODE", raising=False)
        cfg = _reload()
        assert cfg.is_local() is True
        monkeypatch.setenv("TTRPG_APP_MODE", "remote")
        assert cfg.is_remote() is True
