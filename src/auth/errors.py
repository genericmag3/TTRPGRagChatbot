"""Auth error types.

Kept distinct so the UI can show a precise message (and so a wrong
password is never conflated with a locked-out account)."""


class AuthError(Exception):
    """Base class for all authentication failures."""


class InvalidInputError(AuthError):
    """Username/password did not meet the required format."""


class DuplicateUserError(AuthError):
    """A user with that username already exists."""


class InvalidInviteError(AuthError):
    """Invite token / registration password was missing, wrong, used, or expired."""


class RateLimitedError(AuthError):
    """Too many failed logins for this account; try again later."""
