"""Tests for per-user data path resolution.

Local mode must preserve the existing global ``data/`` layout exactly.
Remote mode must scope every artifact under ``data/users/<user_id>/``.
"""
import os

import pytest

import src.utils.paths as paths


class TestLocalMode:
    @pytest.fixture(autouse=True)
    def _local(self, monkeypatch):
        monkeypatch.delenv("TTRPG_APP_MODE", raising=False)

    def test_data_root_is_global(self):
        assert paths.data_root() == "data"

    def test_user_id_is_ignored_in_local_mode(self):
        assert paths.data_root("some-user") == "data"

    def test_artifact_paths_match_legacy_layout(self):
        assert paths.user_data_file() == os.path.join("data", "user_data.json")
        assert paths.database_dir() == os.path.join("data", "chrome_langchain_db")
        assert paths.summary_file() == os.path.join("data", "campaign_summary.json")
        assert paths.raw_notes_file() == os.path.join("data", "raw_notes.json")
        assert paths.editor_notes_file() == os.path.join("data", "editor_notes.txt")
        assert paths.editor_config_file() == os.path.join("data", "editor_config.json")


class TestRemoteMode:
    @pytest.fixture(autouse=True)
    def _remote(self, monkeypatch):
        monkeypatch.setenv("TTRPG_APP_MODE", "remote")

    def test_data_root_is_scoped_to_user(self):
        root = paths.data_root("user-123")
        assert root == os.path.join("data", "users", "user-123")

    def test_requires_user_id(self):
        with pytest.raises(ValueError):
            paths.data_root()
        with pytest.raises(ValueError):
            paths.data_root(None)

    def test_two_users_are_isolated(self):
        a = paths.database_dir("alice")
        b = paths.database_dir("bob")
        assert a != b
        assert "alice" in a and "bob" in b

    def test_user_id_is_sanitized_against_traversal(self):
        root = paths.data_root("../../etc/passwd")
        # No path-traversal segments survive sanitisation.
        assert ".." not in root
        normalized = os.path.normpath(root)
        assert normalized.startswith(os.path.join("data", "users"))

    def test_all_artifacts_live_under_user_root(self):
        uid = "abc-def"
        root = paths.data_root(uid)
        for p in (
            paths.user_data_file(uid),
            paths.database_dir(uid),
            paths.summary_file(uid),
            paths.raw_notes_file(uid),
            paths.editor_notes_file(uid),
            paths.editor_config_file(uid),
        ):
            assert os.path.normpath(p).startswith(os.path.normpath(root))


class TestEnsureDataRoot:
    def test_creates_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TTRPG_APP_MODE", "remote")
        monkeypatch.setattr(paths, "DATA_DIR", str(tmp_path / "data"))
        root = paths.ensure_data_root("user-xyz")
        assert os.path.isdir(root)
