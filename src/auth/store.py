"""SQLite-backed credential / invite / session store.

Design notes
------------
* Passwords and the shared registration password are bcrypt-hashed. The
  secret is SHA-256 pre-hashed first so passphrases longer than bcrypt's
  72-byte limit remain fully significant and NUL bytes can't truncate.
* Invite and session tokens are random 256-bit URL-safe strings; only
  their SHA-256 digest is persisted, so a database leak does not expose
  usable tokens.
* Login throttling is persisted (not in-process) so it survives the
  store being reconstructed per request and can't be reset by a reload.
* Each method uses its own short-lived connection; WAL mode keeps
  concurrent readers/writers from blocking each other on a small,
  single-host deployment.
"""
import hashlib
import os
import re
import secrets
import sqlite3
import time
import uuid
from datetime import datetime, timezone

import bcrypt

from .errors import (
    DuplicateUserError,
    InvalidInputError,
    InvalidInviteError,
    RateLimitedError,
)

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,32}$")
_MIN_PASSWORD_LEN = 8
_MAX_PASSWORD_LEN = 1024

_REG_PW_HASH_KEY = "registration_password_hash"
_REG_PW_ENABLED_KEY = "registration_password_enabled"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _hash_secret(secret: str) -> str:
    pre = hashlib.sha256(secret.encode("utf-8")).hexdigest().encode("ascii")
    return bcrypt.hashpw(pre, bcrypt.gensalt()).decode("ascii")


def _verify_secret(secret: str, hashed: str) -> bool:
    pre = hashlib.sha256(secret.encode("utf-8")).hexdigest().encode("ascii")
    try:
        return bcrypt.checkpw(pre, hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


def _validate_username(username: str) -> None:
    if not isinstance(username, str) or not _USERNAME_RE.match(username):
        raise InvalidInputError(
            "Username must be 3-32 characters: letters, digits, '-' or '_'."
        )


def _validate_password(password: str) -> None:
    if not isinstance(password, str) or not (
        _MIN_PASSWORD_LEN <= len(password) <= _MAX_PASSWORD_LEN
    ):
        raise InvalidInputError(
            f"Password must be at least {_MIN_PASSWORD_LEN} characters."
        )


class AuthStore:
    def __init__(
        self,
        db_path: str,
        *,
        max_failed_attempts: int = 5,
        lockout_seconds: int = 300,
        session_ttl: int = 86_400,
        invite_ttl: int = 604_800,
        now_fn=time.time,
    ):
        self.db_path = db_path
        self.max_failed_attempts = max_failed_attempts
        self.lockout_seconds = lockout_seconds
        self.session_ttl = session_ttl
        self.invite_ttl = invite_ttl
        self._now = now_fn

        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection / schema
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id            TEXT PRIMARY KEY,
                    username      TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_admin      INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username
                    ON users(username COLLATE NOCASE);

                CREATE TABLE IF NOT EXISTS invites (
                    token_hash TEXT PRIMARY KEY,
                    created_by TEXT,
                    created_at TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    used_by    TEXT,
                    used_at    TEXT
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id    TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS login_attempts (
                    username TEXT NOT NULL,
                    ts       REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_login_attempts_username
                    ON login_attempts(username);

                CREATE TABLE IF NOT EXISTS settings (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );
                """
            )

    @staticmethod
    def _public_user(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "username": row["username"],
            "is_admin": bool(row["is_admin"]),
            "created_at": row["created_at"],
        }

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def user_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def admin_exists(self) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE is_admin = 1 LIMIT 1"
            ).fetchone()
        return row is not None

    def create_user(self, username: str, password: str, is_admin: bool = False) -> dict:
        _validate_username(username)
        _validate_password(password)
        user_id = uuid.uuid4().hex
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users (id, username, password_hash, is_admin, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (user_id, username, _hash_secret(password), int(is_admin), _now_iso()),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateUserError(f"Username '{username}' is taken.") from exc
        return {
            "id": user_id,
            "username": username,
            "is_admin": is_admin,
            "created_at": _now_iso(),
        }

    def get_user(self, user_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return self._public_user(row) if row else None

    def get_user_by_username(self, username: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
            ).fetchone()
        return self._public_user(row) if row else None

    def list_users(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM users ORDER BY created_at"
            ).fetchall()
        return [self._public_user(r) for r in rows]

    def delete_user(self, user_id: str) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            if row:
                conn.execute(
                    "DELETE FROM login_attempts WHERE username = ? COLLATE NOCASE",
                    (row["username"],),
                )

    # ------------------------------------------------------------------
    # Credential verification + login throttling
    # ------------------------------------------------------------------

    def _recent_failures(self, conn: sqlite3.Connection, username: str) -> int:
        cutoff = self._now() - self.lockout_seconds
        return conn.execute(
            "SELECT COUNT(*) FROM login_attempts"
            " WHERE username = ? COLLATE NOCASE AND ts >= ?",
            (username, cutoff),
        ).fetchone()[0]

    def verify_credentials(self, username: str, password: str) -> dict | None:
        with self._connect() as conn:
            if self._recent_failures(conn, username) >= self.max_failed_attempts:
                raise RateLimitedError(
                    "Too many failed attempts. Please wait before trying again."
                )
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
            ).fetchone()

            if row is not None and _verify_secret(password, row["password_hash"]):
                conn.execute(
                    "DELETE FROM login_attempts WHERE username = ? COLLATE NOCASE",
                    (username,),
                )
                return self._public_user(row)

            conn.execute(
                "INSERT INTO login_attempts (username, ts) VALUES (?, ?)",
                (username, self._now()),
            )
            return None

    # ------------------------------------------------------------------
    # Invites
    # ------------------------------------------------------------------

    def create_invite(self, created_by: str | None = None) -> str:
        token = secrets.token_urlsafe(32)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO invites (token_hash, created_by, created_at, expires_at)"
                " VALUES (?, ?, ?, ?)",
                (_digest(token), created_by, _now_iso(), self._now() + self.invite_ttl),
            )
        return token

    def list_invites(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT created_by, created_at, expires_at, used_by, used_at"
                " FROM invites ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def consume_invite(self, token: str, username: str, password: str) -> dict:
        _validate_username(username)
        _validate_password(password)
        token_hash = _digest(token or "")
        user_id = uuid.uuid4().hex
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM invites WHERE token_hash = ?", (token_hash,)
            ).fetchone()
            if row is None:
                raise InvalidInviteError("This invite link is not valid.")
            if row["used_at"] is not None:
                raise InvalidInviteError("This invite link has already been used.")
            if self._now() > row["expires_at"]:
                raise InvalidInviteError("This invite link has expired.")
            try:
                conn.execute(
                    "INSERT INTO users (id, username, password_hash, is_admin, created_at)"
                    " VALUES (?, ?, ?, 0, ?)",
                    (user_id, username, _hash_secret(password), _now_iso()),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateUserError(f"Username '{username}' is taken.") from exc
            conn.execute(
                "UPDATE invites SET used_by = ?, used_at = ? WHERE token_hash = ?",
                (user_id, _now_iso(), token_hash),
            )
        return {
            "id": user_id,
            "username": username,
            "is_admin": False,
            "created_at": _now_iso(),
        }

    # ------------------------------------------------------------------
    # Shared registration password
    # ------------------------------------------------------------------

    def _get_setting(self, conn: sqlite3.Connection, key: str) -> str | None:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def _set_setting(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def set_registration_password(self, password: str | None, enabled: bool) -> None:
        with self._connect() as conn:
            if password:
                self._set_setting(conn, _REG_PW_HASH_KEY, _hash_secret(password))
            if not enabled:
                self._set_setting(conn, _REG_PW_ENABLED_KEY, "0")
            else:
                if not password and self._get_setting(conn, _REG_PW_HASH_KEY) is None:
                    raise InvalidInputError(
                        "Set a registration password before enabling it."
                    )
                self._set_setting(conn, _REG_PW_ENABLED_KEY, "1")

    def registration_password_enabled(self) -> bool:
        with self._connect() as conn:
            return self._get_setting(conn, _REG_PW_ENABLED_KEY) == "1"

    def register_with_password(
        self, reg_password: str, username: str, password: str
    ) -> dict:
        with self._connect() as conn:
            enabled = self._get_setting(conn, _REG_PW_ENABLED_KEY) == "1"
            stored = self._get_setting(conn, _REG_PW_HASH_KEY)
        if not enabled or not stored:
            raise InvalidInviteError("Self-registration is not currently open.")
        if not _verify_secret(reg_password or "", stored):
            raise InvalidInviteError("The registration password is incorrect.")
        return self.create_user(username, password, is_admin=False)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at)"
                " VALUES (?, ?, ?, ?)",
                (_digest(token), user_id, _now_iso(), self._now() + self.session_ttl),
            )
        return token

    def validate_session(self, token: str) -> dict | None:
        if not token:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT s.expires_at, u.* FROM sessions s"
                " JOIN users u ON u.id = s.user_id"
                " WHERE s.token_hash = ?",
                (_digest(token),),
            ).fetchone()
            if row is None:
                return None
            if self._now() > row["expires_at"]:
                conn.execute(
                    "DELETE FROM sessions WHERE token_hash = ?", (_digest(token),)
                )
                return None
        return self._public_user(row)

    def destroy_session(self, token: str) -> None:
        if not token:
            return
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM sessions WHERE token_hash = ?", (_digest(token),)
            )

    def destroy_user_sessions(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def purge_expired(self) -> None:
        now = self._now()
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            conn.execute("DELETE FROM invites WHERE expires_at < ? AND used_at IS NULL", (now,))
            conn.execute(
                "DELETE FROM login_attempts WHERE ts < ?",
                (now - self.lockout_seconds,),
            )

    # ------------------------------------------------------------------
    # Test-only introspection
    # ------------------------------------------------------------------

    def _debug_raw_user(self, username: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
            ).fetchone()
        return dict(row) if row else None

    def _debug_setting(self, key: str) -> str | None:
        with self._connect() as conn:
            return self._get_setting(conn, key)
