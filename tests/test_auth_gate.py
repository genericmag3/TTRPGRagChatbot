"""Integration tests for the auth gate wired into streamlit_app.py.

Local mode must boot straight into the app with no auth surface.
Remote mode must (a) bootstrap an admin on first run, (b) require
login when an admin exists, and (c) let a valid session through.
"""
import os

import pytest
from streamlit.testing.v1 import AppTest

from src.auth import gate
from src.auth.store import AuthStore

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "streamlit_app.py")
TIMEOUT = 30


@pytest.fixture
def remote_env(tmp_path, monkeypatch):
    """Remote mode with auth DB + per-user data isolated under tmp_path."""
    db = str(tmp_path / "auth" / "auth.db")
    monkeypatch.setenv("TTRPG_APP_MODE", "remote")
    monkeypatch.setenv("TTRPG_AUTH_DB", db)
    monkeypatch.setenv("TTRPG_DATA_DIR", str(tmp_path / "data"))
    return tmp_path, db


def _boot(**session):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for k, v in session.items():
        at.session_state[k] = v
    at.run()
    return at


def _all_text(at):
    chunks = []
    for attr in ("title", "header", "subheader", "markdown", "info", "text", "caption"):
        for el in getattr(at, attr):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks).lower()


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------

class TestInviteTokenFromQuery:
    def test_extracts_token(self):
        assert gate.invite_token_from_query({"invite": "abc123"}) == "abc123"

    def test_handles_list_values(self):
        assert gate.invite_token_from_query({"invite": ["abc123"]}) == "abc123"

    def test_missing_returns_none(self):
        assert gate.invite_token_from_query({}) is None
        assert gate.invite_token_from_query({"other": "x"}) is None


class TestAuthDbPath:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("TTRPG_AUTH_DB", "/tmp/custom/auth.db")
        assert gate.auth_db_path() == "/tmp/custom/auth.db"

    def test_default_under_data(self, monkeypatch):
        monkeypatch.delenv("TTRPG_AUTH_DB", raising=False)
        assert gate.auth_db_path().endswith(os.path.join("auth", "auth.db"))


# ---------------------------------------------------------------------------
# Local mode — no auth surface
# ---------------------------------------------------------------------------

class TestLocalModeUnaffected:
    def test_local_mode_boots_into_app_without_login(self, monkeypatch):
        monkeypatch.delenv("TTRPG_APP_MODE", raising=False)
        at = _boot()
        assert not at.exception
        text = _all_text(at)
        assert "ttrpg" in text
        assert "sign in" not in text
        assert "administrator account" not in text


# ---------------------------------------------------------------------------
# Remote mode — bootstrap, login gate, session pass-through
# ---------------------------------------------------------------------------

class TestRemoteBootstrap:
    def test_first_run_prompts_admin_creation(self, remote_env):
        at = _boot()
        assert not at.exception
        text = _all_text(at)
        assert "administrator" in text or "admin account" in text
        # The protected app must NOT be visible yet.
        assert "journal q&a chatbot" not in text


class TestRemoteLoginGate:
    def test_login_required_when_admin_exists(self, remote_env):
        _, db = remote_env
        AuthStore(db).create_user("dungeonmaster", "dungeonpass1", is_admin=True)
        at = _boot()
        assert not at.exception
        text = _all_text(at)
        assert "sign in" in text or "log in" in text
        assert "journal q&a chatbot" not in text

    def test_unauthenticated_cannot_see_chatbot(self, remote_env):
        _, db = remote_env
        AuthStore(db).create_user("dungeonmaster", "dungeonpass1", is_admin=True)
        at = _boot()
        # chat input must not be reachable without auth
        assert len(at.chat_input) == 0


class TestRemoteSessionPassthrough:
    def test_valid_session_reaches_app(self, remote_env):
        _, db = remote_env
        store = AuthStore(db)
        user = store.create_user("player", "playerpass1", is_admin=False)
        token = store.create_session(user["id"])
        at = _boot(auth_session_token=token)
        assert not at.exception
        text = _all_text(at)
        assert "journal q&a chatbot" in text or "ttrpg" in text
        assert "sign in" not in text

    def test_invalid_session_does_not_reach_app(self, remote_env):
        _, db = remote_env
        AuthStore(db).create_user("dungeonmaster", "dungeonpass1", is_admin=True)
        at = _boot(auth_session_token="bogus-token")
        assert not at.exception
        text = _all_text(at)
        assert "sign in" in text or "log in" in text
