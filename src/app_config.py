"""Build-mode flag.

The same codebase ships as two distributables:

* ``local``  — single-user, no authentication, global ``data/`` directory
               (the historical behaviour; this is the default).
* ``remote`` — multi-user host: authentication required, every user gets
               an isolated data directory.

The mode is selected at runtime via the ``TTRPG_APP_MODE`` environment
variable so a single image can be driven by either launcher. It is read
on every call (not cached at import) so tests and launchers can set it
before the app builds its pages.
"""
import os

LOCAL = "local"
REMOTE = "remote"

_ENV_VAR = "TTRPG_APP_MODE"


def get_mode() -> str:
    value = os.environ.get(_ENV_VAR, "").strip().lower()
    return REMOTE if value == REMOTE else LOCAL


def is_remote() -> bool:
    return get_mode() == REMOTE


def is_local() -> bool:
    return not is_remote()
