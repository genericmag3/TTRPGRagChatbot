"""Tests for the self-contained SQLite auth store.

Covers password hashing, account creation, credential verification,
login rate-limiting, single-use expiring invites, the shared
registration-password gate, and server-side session lifecycle.
"""
import pytest

from src.auth.store import AuthStore
from src.auth.errors import (
    DuplicateUserError,
    InvalidInputError,
    InvalidInviteError,
    RateLimitedError,
)


class Clock:
    """Mutable fake clock so time-based behaviour is deterministic."""

    def __init__(self, start=1_000_000.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return AuthStore(
        str(tmp_path / "auth.db"),
        max_failed_attempts=3,
        lockout_seconds=300,
        session_ttl=3600,
        invite_ttl=600,
        now_fn=clock,
    )


# ---------------------------------------------------------------------------
# Schema / bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_fresh_store_has_no_users(self, store):
        assert store.user_count() == 0
        assert store.admin_exists() is False

    def test_creating_admin_is_detected(self, store):
        store.create_user("dungeonmaster", "correct horse battery", is_admin=True)
        assert store.admin_exists() is True
        assert store.user_count() == 1


# ---------------------------------------------------------------------------
# User creation + validation
# ---------------------------------------------------------------------------

class TestUserCreation:
    def test_create_user_returns_safe_dict(self, store):
        user = store.create_user("alice", "supersecret1")
        assert user["username"] == "alice"
        assert user["is_admin"] is False
        assert "id" in user and user["id"]
        assert "password" not in user
        assert "password_hash" not in user

    def test_password_is_hashed_not_stored_plaintext(self, store):
        store.create_user("bob", "plaintextpw123")
        row = store._debug_raw_user("bob")
        assert row["password_hash"] != "plaintextpw123"
        assert "plaintextpw123" not in row["password_hash"]

    def test_duplicate_username_rejected_case_insensitive(self, store):
        store.create_user("Carol", "password123")
        with pytest.raises(DuplicateUserError):
            store.create_user("carol", "password456")

    @pytest.mark.parametrize("bad", ["", "ab", "x" * 64, "has space", "bad/slash"])
    def test_invalid_username_rejected(self, store, bad):
        with pytest.raises(InvalidInputError):
            store.create_user(bad, "password123")

    @pytest.mark.parametrize("bad", ["", "short", "1234567"])
    def test_short_password_rejected(self, store, bad):
        with pytest.raises(InvalidInputError):
            store.create_user("validname", bad)

    def test_long_password_is_accepted(self, store):
        # bcrypt truncates at 72 bytes; the store must pre-hash so long
        # passphrases remain fully significant.
        long_pw = "a" * 200
        store.create_user("longpw", long_pw)
        assert store.verify_credentials("longpw", long_pw) is not None
        assert store.verify_credentials("longpw", "a" * 72) is None


# ---------------------------------------------------------------------------
# Credential verification + rate limiting
# ---------------------------------------------------------------------------

class TestVerifyCredentials:
    def test_correct_password_succeeds(self, store):
        store.create_user("dave", "rightpassword")
        u = store.verify_credentials("dave", "rightpassword")
        assert u is not None and u["username"] == "dave"

    def test_wrong_password_returns_none(self, store):
        store.create_user("erin", "rightpassword")
        assert store.verify_credentials("erin", "wrongpassword") is None

    def test_unknown_user_returns_none(self, store):
        assert store.verify_credentials("ghost", "whatever12") is None

    def test_lockout_after_max_failed_attempts(self, store):
        store.create_user("frank", "rightpassword")
        for _ in range(3):
            assert store.verify_credentials("frank", "bad") is None
        # 4th attempt is locked even with the *correct* password.
        with pytest.raises(RateLimitedError):
            store.verify_credentials("frank", "rightpassword")

    def test_lockout_expires_after_window(self, store, clock):
        store.create_user("grace", "rightpassword")
        for _ in range(3):
            store.verify_credentials("grace", "bad")
        with pytest.raises(RateLimitedError):
            store.verify_credentials("grace", "rightpassword")
        clock.advance(301)
        assert store.verify_credentials("grace", "rightpassword") is not None

    def test_successful_login_clears_failure_counter(self, store):
        store.create_user("heidi", "rightpassword")
        store.verify_credentials("heidi", "bad")
        store.verify_credentials("heidi", "bad")
        assert store.verify_credentials("heidi", "rightpassword") is not None
        # Counter reset: two fresh failures must not trip the lock.
        store.verify_credentials("heidi", "bad")
        store.verify_credentials("heidi", "bad")
        assert store.verify_credentials("heidi", "rightpassword") is not None


# ---------------------------------------------------------------------------
# Invites
# ---------------------------------------------------------------------------

class TestInvites:
    def test_invite_token_is_opaque_and_not_stored_raw(self, store):
        admin = store.create_user("admin", "adminpassword", is_admin=True)
        token = store.create_invite(created_by=admin["id"])
        assert isinstance(token, str) and len(token) >= 20
        invites = store.list_invites()
        assert all(token not in str(row.values()) for row in invites)

    def test_consume_invite_creates_user(self, store):
        admin = store.create_user("admin", "adminpassword", is_admin=True)
        token = store.create_invite(created_by=admin["id"])
        user = store.consume_invite(token, "newplayer", "newpassword1")
        assert user["username"] == "newplayer"
        assert user["is_admin"] is False
        assert store.verify_credentials("newplayer", "newpassword1") is not None

    def test_invite_is_single_use(self, store):
        admin = store.create_user("admin", "adminpassword", is_admin=True)
        token = store.create_invite(created_by=admin["id"])
        store.consume_invite(token, "first", "password111")
        with pytest.raises(InvalidInviteError):
            store.consume_invite(token, "second", "password222")

    def test_expired_invite_rejected(self, store, clock):
        admin = store.create_user("admin", "adminpassword", is_admin=True)
        token = store.create_invite(created_by=admin["id"])
        clock.advance(601)
        with pytest.raises(InvalidInviteError):
            store.consume_invite(token, "late", "password333")

    def test_unknown_invite_rejected(self, store):
        with pytest.raises(InvalidInviteError):
            store.consume_invite("not-a-real-token", "validuser", "password444")

    def test_consume_invite_validates_inputs(self, store):
        admin = store.create_user("admin", "adminpassword", is_admin=True)
        token = store.create_invite(created_by=admin["id"])
        with pytest.raises(InvalidInputError):
            store.consume_invite(token, "okuser", "short")
        # Invite not consumed by the failed attempt — still usable.
        assert store.consume_invite(token, "okuser", "goodpassword") is not None


# ---------------------------------------------------------------------------
# Shared registration password
# ---------------------------------------------------------------------------

class TestRegistrationPassword:
    def test_disabled_by_default(self, store):
        assert store.registration_password_enabled() is False

    def test_register_with_password_when_disabled_is_rejected(self, store):
        with pytest.raises(InvalidInviteError):
            store.register_with_password("anything", "user", "password123")

    def test_enable_and_register(self, store):
        store.set_registration_password("the-tavern-key", enabled=True)
        assert store.registration_password_enabled() is True
        user = store.register_with_password("the-tavern-key", "newbie", "password123")
        assert user["username"] == "newbie"
        assert user["is_admin"] is False

    def test_wrong_registration_password_rejected(self, store):
        store.set_registration_password("the-tavern-key", enabled=True)
        with pytest.raises(InvalidInviteError):
            store.register_with_password("guess", "newbie", "password123")

    def test_can_be_disabled_again(self, store):
        store.set_registration_password("key", enabled=True)
        store.set_registration_password(None, enabled=False)
        assert store.registration_password_enabled() is False
        with pytest.raises(InvalidInviteError):
            store.register_with_password("key", "newbie", "password123")

    def test_registration_password_is_hashed(self, store):
        store.set_registration_password("plain-reg-pw", enabled=True)
        assert store._debug_setting("registration_password_hash") != "plain-reg-pw"


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class TestSessions:
    def test_create_and_validate_session(self, store):
        user = store.create_user("ivan", "password123")
        token = store.create_session(user["id"])
        validated = store.validate_session(token)
        assert validated is not None
        assert validated["id"] == user["id"]
        assert validated["username"] == "ivan"

    def test_session_token_is_opaque(self, store):
        user = store.create_user("judy", "password123")
        token = store.create_session(user["id"])
        assert store.validate_session("garbage") is None
        assert store.validate_session(token + "x") is None

    def test_session_expires(self, store, clock):
        user = store.create_user("ken", "password123")
        token = store.create_session(user["id"])
        clock.advance(3601)
        assert store.validate_session(token) is None

    def test_destroy_session_invalidates_it(self, store):
        user = store.create_user("laura", "password123")
        token = store.create_session(user["id"])
        store.destroy_session(token)
        assert store.validate_session(token) is None

    def test_deleting_user_revokes_sessions_and_login(self, store):
        user = store.create_user("mike", "password123")
        token = store.create_session(user["id"])
        store.delete_user(user["id"])
        assert store.validate_session(token) is None
        assert store.verify_credentials("mike", "password123") is None
        assert store.get_user(user["id"]) is None

    def test_purge_expired_removes_dead_sessions(self, store, clock):
        user = store.create_user("nora", "password123")
        token = store.create_session(user["id"])
        clock.advance(3601)
        store.purge_expired()
        assert store.validate_session(token) is None


# ---------------------------------------------------------------------------
# User listing / admin management
# ---------------------------------------------------------------------------

class TestUserManagement:
    def test_list_users_excludes_secrets(self, store):
        store.create_user("admin", "adminpassword", is_admin=True)
        store.create_user("player1", "password123")
        users = store.list_users()
        assert {u["username"] for u in users} == {"admin", "player1"}
        for u in users:
            assert "password_hash" not in u

    def test_get_user_by_username_is_case_insensitive(self, store):
        created = store.create_user("MixedCase", "password123")
        found = store.get_user_by_username("mixedcase")
        assert found is not None and found["id"] == created["id"]

    def test_persistence_across_instances(self, tmp_path, clock):
        db = str(tmp_path / "auth.db")
        s1 = AuthStore(db, now_fn=clock)
        s1.create_user("persist", "password123", is_admin=True)
        s2 = AuthStore(db, now_fn=clock)
        assert s2.get_user_by_username("persist") is not None
        assert s2.admin_exists() is True
