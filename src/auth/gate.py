"""Streamlit auth gate for the remote build.

``require_auth`` is called once at the top of ``streamlit_app.py`` before
any page is built. It:

* bootstraps the first admin account on a fresh host,
* validates the server-side session token held in ``st.session_state``,
* otherwise renders the login / register surface and halts the script
  with ``st.stop()`` so no protected page is ever constructed.

Sessions are server-side: the random token lives only in Streamlit's
per-connection ``session_state`` (never in the URL or client storage)
and is validated against the SQLite store on every run.
"""
import os

import streamlit as st

from .errors import AuthError, InvalidInputError
from .store import AuthStore

_SESSION_KEY = "auth_session_token"
_USER_ID_KEY = "auth_user_id"
_USER_KEY = "auth_user"


def auth_db_path() -> str:
    """Host-level auth DB (shared by all users, never per-user)."""
    override = os.environ.get("TTRPG_AUTH_DB")
    if override:
        return override
    base = os.environ.get("TTRPG_DATA_DIR", "data")
    return os.path.join(base, "auth", "auth.db")


def get_store() -> AuthStore:
    return AuthStore(auth_db_path())


def invite_token_from_query(query_params) -> str | None:
    """Pull the ``invite`` token out of query params.

    Streamlit may hand back either a scalar or a list depending on
    version, so both are handled.
    """
    if not query_params:
        return None
    value = query_params.get("invite")
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return str(value)


def current_user(store: AuthStore) -> dict | None:
    token = st.session_state.get(_SESSION_KEY)
    if not token:
        return None
    return store.validate_session(token)


def _establish_session(store: AuthStore, user: dict) -> None:
    token = store.create_session(user["id"])
    st.session_state[_SESSION_KEY] = token
    st.session_state[_USER_ID_KEY] = user["id"]
    st.session_state[_USER_KEY] = user
    st.rerun()


def logout(store: AuthStore) -> None:
    token = st.session_state.pop(_SESSION_KEY, None)
    if token:
        store.destroy_session(token)
    st.session_state.pop(_USER_ID_KEY, None)
    st.session_state.pop(_USER_KEY, None)
    st.rerun()


# ---------------------------------------------------------------------------
# UI surfaces
# ---------------------------------------------------------------------------

def _render_bootstrap(store: AuthStore) -> None:
    st.title("🔐 First-time setup")
    st.info(
        "No accounts exist yet. Create the **administrator account** that "
        "will manage invites and access for this host."
    )
    with st.form("bootstrap_admin"):
        username = st.text_input("Admin username")
        pw1 = st.text_input("Password", type="password")
        pw2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create administrator account", type="primary")
    if submitted:
        if pw1 != pw2:
            st.error("Passwords do not match.")
            return
        try:
            user = store.create_user(username, pw1, is_admin=True)
        except AuthError as exc:
            st.error(str(exc))
            return
        _establish_session(store, user)


def _render_login(store: AuthStore) -> None:
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        try:
            user = store.verify_credentials(username, password)
        except AuthError as exc:
            st.error(str(exc))
            return
        if user is None:
            st.error("Invalid username or password.")
            return
        _establish_session(store, user)


def _render_register(store: AuthStore, invite_token: str | None) -> None:
    reg_open = store.registration_password_enabled()
    if not invite_token and not reg_open:
        st.info(
            "Registration is invite-only. Ask the host for an invite link, "
            "or for the registration password if they have enabled one."
        )
        return

    with st.form("register_form"):
        if invite_token:
            st.caption("You are registering with an invite link.")
            reg_password = None
        else:
            reg_password = st.text_input("Registration password", type="password")
        username = st.text_input("Choose a username")
        pw1 = st.text_input("Choose a password", type="password")
        pw2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account", type="primary")

    if submitted:
        if pw1 != pw2:
            st.error("Passwords do not match.")
            return
        try:
            if invite_token:
                user = store.consume_invite(invite_token, username, pw1)
            else:
                user = store.register_with_password(reg_password, username, pw1)
        except AuthError as exc:
            st.error(str(exc))
            return
        _establish_session(store, user)


def _render_auth_ui(store: AuthStore, invite_token: str | None) -> None:
    st.title("🧙‍♂️ TTRPG Campaign Assistant")
    st.caption("Sign in or register to access your private campaign workspace.")
    login_tab, register_tab = st.tabs(["Sign in", "Register"])
    with login_tab:
        _render_login(store)
    with register_tab:
        _render_register(store, invite_token)


def require_auth() -> dict:
    """Gate the app. Returns the authenticated user or halts the script."""
    store = get_store()
    store.purge_expired()

    user = current_user(store)
    if user is not None:
        st.session_state[_USER_ID_KEY] = user["id"]
        st.session_state[_USER_KEY] = user
        return user

    if not store.admin_exists():
        _render_bootstrap(store)
        st.stop()

    invite_token = invite_token_from_query(dict(st.query_params))
    _render_auth_ui(store, invite_token)
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar + account/admin pages
# ---------------------------------------------------------------------------

def render_logout_sidebar(user: dict) -> None:
    store = get_store()
    with st.sidebar:
        st.divider()
        st.caption(f"Signed in as **{user['username']}**" + (" (admin)" if user["is_admin"] else ""))
        if st.button("Sign out", use_container_width=True):
            logout(store)


def render_account_page() -> None:
    store = get_store()
    user = st.session_state.get(_USER_KEY) or {}
    st.title("👤 Account")
    st.write(f"**Username:** {user.get('username', '')}")
    st.write(f"**Role:** {'Administrator' if user.get('is_admin') else 'Player'}")
    st.caption("Your campaign notes and data are private to this account.")

    st.divider()
    st.subheader("Sign out everywhere")
    st.caption("Invalidate all of this account's active sessions.")
    if st.button("Sign out of all sessions"):
        store.destroy_user_sessions(user.get("id", ""))
        st.session_state.pop(_SESSION_KEY, None)
        st.rerun()


def render_admin_page() -> None:
    store = get_store()
    user = st.session_state.get(_USER_KEY) or {}
    st.title("🛡️ Admin")
    if not user.get("is_admin"):
        st.error("Administrator access required.")
        st.stop()

    st.subheader("Invite links")
    st.caption(
        "Generate a single-use invite. Send the recipient your server URL "
        "with `?invite=<token>` appended."
    )
    if st.button("Generate invite link", type="primary"):
        token = store.create_invite(created_by=user.get("id"))
        st.session_state["_last_invite"] = token
    last = st.session_state.get("_last_invite")
    if last:
        st.success("Invite created. Share this query string with the player:")
        st.code(f"?invite={last}", language="text")

    st.divider()
    st.subheader("Shared registration password")
    enabled = store.registration_password_enabled()
    st.caption(
        "When enabled, anyone with this password can self-register their "
        "own private account. Leave disabled for invite-only access."
    )
    with st.form("reg_pw_form"):
        new_pw = st.text_input(
            "Set / change registration password", type="password",
            help="Leave blank to keep the current password.",
        )
        want_enabled = st.checkbox("Enable self-registration", value=enabled)
        if st.form_submit_button("Save"):
            try:
                store.set_registration_password(new_pw or None, want_enabled)
                st.success("Registration settings saved.")
            except InvalidInputError as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("Users")
    for u in store.list_users():
        cols = st.columns([3, 2, 2])
        cols[0].write(u["username"])
        cols[1].write("admin" if u["is_admin"] else "player")
        if u["id"] != user.get("id"):
            if cols[2].button("Remove", key=f"del_{u['id']}"):
                store.delete_user(u["id"])
                st.rerun()
        else:
            cols[2].write("— you —")
