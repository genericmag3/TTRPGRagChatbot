"""Resolves where on-disk artifacts live for the current build mode.

Local mode keeps the historical flat layout under ``data/`` so existing
behaviour (and existing tests) are byte-for-byte unchanged. Remote mode
scopes every per-user artifact under ``data/users/<user_id>/`` so one
host can serve many players without their campaigns colliding.

The auth database itself is host-level, not per-user, and is resolved
separately in :mod:`src.auth.gate`.
"""
import os

from .. import app_config

# Base directory for all app data. ``TTRPG_DATA_DIR`` lets a launcher (or a
# test) relocate the whole tree without touching the production folder.
DATA_DIR = "data"
USERS_SUBDIR = "users"


def _base_dir() -> str:
    return os.environ.get("TTRPG_DATA_DIR", DATA_DIR)


def _sanitize_user_id(user_id: str) -> str:
    """Keep only filesystem-safe characters so a user id can never escape
    its own directory (defence in depth — ids are server-generated)."""
    cleaned = "".join(c for c in str(user_id) if c.isalnum() or c in ("-", "_"))
    if not cleaned:
        raise ValueError("user_id did not contain any usable characters")
    return cleaned


def data_root(user_id: str | None = None) -> str:
    """Return the data directory for the active context.

    In local mode this is always the shared base directory. In remote
    mode a ``user_id`` is required and the path is scoped to that user.
    """
    base = _base_dir()
    if app_config.is_remote():
        if not user_id:
            raise ValueError("user_id is required to resolve the data root in remote mode")
        return os.path.join(base, USERS_SUBDIR, _sanitize_user_id(user_id))
    return base


def ensure_data_root(user_id: str | None = None) -> str:
    root = data_root(user_id)
    os.makedirs(root, exist_ok=True)
    return root


def user_data_file(user_id: str | None = None) -> str:
    return os.path.join(data_root(user_id), "user_data.json")


def database_dir(user_id: str | None = None) -> str:
    return os.path.join(data_root(user_id), "chrome_langchain_db")


def summary_file(user_id: str | None = None) -> str:
    return os.path.join(data_root(user_id), "campaign_summary.json")


def raw_notes_file(user_id: str | None = None) -> str:
    return os.path.join(data_root(user_id), "raw_notes.json")


def editor_notes_file(user_id: str | None = None) -> str:
    return os.path.join(data_root(user_id), "editor_notes.txt")


def editor_config_file(user_id: str | None = None) -> str:
    return os.path.join(data_root(user_id), "editor_config.json")
