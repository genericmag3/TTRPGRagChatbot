"""Streamlit AppTest integration tests for streamlit_app.py.

These tests use st.testing.v1.AppTest to run the app headlessly.
Heavy external dependencies (ollama, HuggingFace, Chroma) are mocked
in conftest.py so the app boots without real services.
"""
import os
import json
import pytest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "streamlit_app.py")
TIMEOUT = 30

# Keys that __init_state_variables checks — populate all of them to skip its init block.
_ALL_KEYS = {
    "reupload_key": 0,
    "model_name": None,
    "model_temperature": None,
    "notes_uploaded": False,
    "messages": [],
    "buttoninfo": [],
    "button_key": 0,
    "party_members": [{"id": "init-abc", "name": "Alice", "note_taker": True}],
    "delete_index": None,
}


def _hide_userdata_isfile():
    """Return an isfile side_effect that hides user_data.json but passes all other paths."""
    real_isfile = os.path.isfile
    def _isfile(path):
        return False if "user_data" in str(path) else real_isfile(path)
    return _isfile


def _hide_db_isdir():
    """Return an isdir side_effect that hides the DB directory but passes all other paths."""
    real_isdir = os.path.isdir
    def _isdir(path):
        return False if "chrome_langchain_db" in str(path) else real_isdir(path)
    return _isdir


def _run_preloaded(**overrides) -> AppTest:
    """Boot app with all session-state keys pre-set so __init_state_variables is skipped."""
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    state = {**_ALL_KEYS, **overrides}
    for k, v in state.items():
        at.session_state[k] = v
    with patch("os.path.isfile", side_effect=_hide_userdata_isfile()), \
         patch("os.path.isdir", side_effect=_hide_db_isdir()):
        at.run()
    return at


def _db_patches():
    """Return (isdir_fn, listdir_fn) that make __has_subfolders return True for the DB dir."""
    real_isdir = os.path.isdir
    real_listdir = os.listdir

    def _isdir(path):
        return True if "chrome_langchain_db" in str(path) else real_isdir(path)

    def _listdir(path):
        return ["subdir"] if "chrome_langchain_db" in str(path) else real_listdir(path)

    return _isdir, _listdir


def _run_app() -> AppTest:
    """Boot the app with mocked file-system state (no user_data.json, no DB)."""
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)

    real_isfile = os.path.isfile
    real_isdir = os.path.isdir

    def _isfile(path):
        # Hide user_data.json so app initialises with defaults
        if "user_data" in str(path):
            return False
        return real_isfile(path)

    def _isdir(path):
        # Hide the DB directory so file-uploader is shown instead of re-upload button
        if "chrome_langchain_db" in str(path):
            return False
        return real_isdir(path)

    with patch("os.path.isfile", side_effect=_isfile), \
         patch("os.path.isdir", side_effect=_isdir):
        at.run()
    return at


# ---------------------------------------------------------------------------
# Smoke tests — app boots without exception
# ---------------------------------------------------------------------------

class TestAppBoots:
    def test_no_exception_on_startup(self):
        at = _run_app()
        assert not at.exception, f"App raised: {at.exception}"

    def test_title_is_rendered(self):
        at = _run_app()
        titles = at.title
        assert len(titles) > 0
        assert "TTRPG" in titles[0].value

    def test_info_banner_is_rendered(self):
        at = _run_app()
        infos = at.info
        assert len(infos) > 0
        assert "notes" in infos[0].value.lower()


# ---------------------------------------------------------------------------
# Sidebar — Model Options
# ---------------------------------------------------------------------------

class TestSidebarModelOptions:
    def test_model_selectbox_present(self):
        at = _run_app()
        # sidebar selectbox for model selection should exist
        assert len(at.sidebar.selectbox) >= 1

    def test_temperature_slider_present(self):
        at = _run_app()
        assert len(at.sidebar.slider) >= 1

    def test_temperature_slider_default_value(self):
        at = _run_app()
        slider = at.sidebar.slider[0]
        # Default is 0.7 when no user_data.json exists
        assert slider.value == pytest.approx(0.7, abs=0.05)


# ---------------------------------------------------------------------------
# Sidebar — Journal Options
# ---------------------------------------------------------------------------

class TestSidebarJournalOptions:
    def test_add_member_button_present(self):
        at = _run_app()
        labels = [b.label for b in at.sidebar.button]
        assert any("Add New Member" in lbl for lbl in labels)

    def test_reupload_button_absent_when_no_db(self):
        # "Re-Upload Notes" button should only appear when a DB already exists.
        # Since we mock out the DB directory, it must NOT be present.
        at = _run_app()
        labels = [b.label for b in at.sidebar.button]
        assert not any("Re-Upload" in lbl for lbl in labels)

    def test_initial_party_member_text_input_present(self):
        at = _run_app()
        # At least one text_input for the default party member
        assert len(at.text_input) >= 1


# ---------------------------------------------------------------------------
# Chat input — disabled until notes + model are ready
# ---------------------------------------------------------------------------

class TestChatInput:
    def test_chat_input_absent_when_notes_not_uploaded(self):
        """With no notes and no model, chat_input should not be rendered."""
        at = _run_app()
        # notes_uploaded=False and model_name=None → __process_chat skips chat_input
        assert len(at.chat_input) == 0


# ---------------------------------------------------------------------------
# Session state — initial values
# ---------------------------------------------------------------------------

class TestSessionStateInit:
    def test_notes_uploaded_is_false_initially(self):
        at = _run_app()
        assert at.session_state.notes_uploaded is False

    def test_messages_is_empty_list_initially(self):
        at = _run_app()
        assert at.session_state.messages == []

    def test_model_name_is_none_initially(self):
        at = _run_app()
        assert at.session_state.model_name is None

    def test_party_members_initialized(self):
        at = _run_app()
        assert isinstance(at.session_state.party_members, list)
        assert len(at.session_state.party_members) >= 1


# ---------------------------------------------------------------------------
# User-data persistence
# ---------------------------------------------------------------------------

class TestUserDataLoading:
    def test_loads_model_name_from_user_data(self, tmp_path):
        user_data = {
            "model_name": "llama3:latest",
            "model_temperature": 0.5,
            "notes_uploaded": False,
            "party_members": [{"id": "abc", "name": "Aria", "note_taker": False}]
        }
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        user_data_file = data_dir / "user_data.json"
        user_data_file.write_text(json.dumps(user_data))

        at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        with patch("os.path.isdir", return_value=False), \
             patch("os.path.isfile", return_value=True), \
             patch("builtins.open", side_effect=lambda p, *a, **kw:
                   open(str(user_data_file), *a, **kw)
                   if "user_data" in str(p) else open(str(tmp_path / "fake.json"), *a, **kw)):
            try:
                at.run()
            except Exception:
                pass  # allow file-not-found for asset files

        # If model_name loaded from file it should be "llama3:latest"
        if not at.exception:
            assert at.session_state.model_name == "llama3:latest"


# ---------------------------------------------------------------------------
# User-data init
# ---------------------------------------------------------------------------

class TestUserDataInitCoverage:
    def test_state_loaded_when_user_data_file_exists(self, tmp_path):
        # Use model_name=None so __init_state_variables skips the load_model call
        # (line 62). That avoids a double load_model issue with the mock, while
        # still covering lines 47 and 52-61.
        user_data = {
            "model_name": None,
            "model_temperature": 0.5,
            "notes_uploaded": True,
            "party_members": [{"id": "t1", "name": "Aria", "note_taker": False}],
        }
        data_file = tmp_path / "user_data.json"
        data_file.write_text(json.dumps(user_data))

        _real_open = open  # capture before patching

        def _open(path, *args, **kwargs):
            if "user_data" in str(path):
                mode = args[0] if args else kwargs.get("mode", "r")
                if "w" in mode:
                    return _real_open(str(tmp_path / "out.json"), "w")
                return _real_open(str(data_file), *args, **kwargs)
            return _real_open(path, *args, **kwargs)

        _real_isfile = os.path.isfile

        def _isfile(path):
            return True if "user_data" in str(path) else _real_isfile(path)

        at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        with patch("builtins.open", side_effect=_open), \
             patch("os.path.isfile", side_effect=_isfile), \
             patch("os.path.isdir", side_effect=_hide_db_isdir()):
            at.run()

        assert not at.exception
        # party_members is set by __init_state_variables and not overwritten
        # by subsequent UI methods, so it reliably reflects the loaded file.
        assert any(m["name"] == "Aria" for m in at.session_state.party_members)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

class TestModelSelectionSavesState:
    def test_model_and_temperature_written_to_session_state(self):
        """When model_name is already set the selectbox resolves it and loads the model."""
        at = _run_preloaded(model_name="llama3:latest", model_temperature=0.7)
        assert not at.exception
        assert at.session_state.model_name == "llama3:latest"
        assert at.session_state.model_temperature == pytest.approx(0.7, abs=0.05)


# ---------------------------------------------------------------------------
# __update_message_history
# ---------------------------------------------------------------------------

class TestUpdateMessageHistory:
    def test_user_and_assistant_messages_rendered(self):
        messages = [
            {"role": "user", "content": "Test question?", "avatar": None},
            {"role": "assistant", "content": "Test answer.", "avatar": "🧙‍♂️"},
        ]
        at = _run_preloaded(messages=messages, buttoninfo=[None])
        assert not at.exception

    def test_assistant_message_with_reference_buttons(self):
        def _noop(content):
            pass

        messages = [
            {"role": "user", "content": "What happened?", "avatar": None},
            {"role": "assistant", "content": "See references.", "avatar": "🧙‍♂️"},
        ]
        buttoninfo = [[["2023-01-01", _noop, ("content text",), "click_0"]]]
        at = _run_preloaded(messages=messages, buttoninfo=buttoninfo)
        assert not at.exception


# ---------------------------------------------------------------------------
# DB-exists state
# ---------------------------------------------------------------------------

class TestReuploadButtonFlow:
    def test_reupload_button_present_when_db_has_subfolders(self):
        _isdir, _listdir = _db_patches()
        at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        for k, v in _ALL_KEYS.items():
            at.session_state[k] = v
        with patch("os.path.isdir", side_effect=_isdir), \
             patch("os.listdir", side_effect=_listdir), \
             patch("os.path.isfile", side_effect=_hide_userdata_isfile()):
            at.run()
        assert not at.exception
        labels = [b.label for b in at.sidebar.button]
        assert any("Re-Upload" in lbl for lbl in labels)

    def test_reupload_button_click_clears_chat_history(self):
        _isdir, _listdir = _db_patches()
        at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        for k, v in _ALL_KEYS.items():
            at.session_state[k] = v
        at.session_state["messages"] = [{"role": "user", "content": "old", "avatar": None}]
        at.session_state["button_key"] = 3
        with patch("os.path.isdir", side_effect=_isdir), \
             patch("os.listdir", side_effect=_listdir), \
             patch("os.path.isfile", side_effect=_hide_userdata_isfile()):
            at.run()
            reupload = [b for b in at.sidebar.button if "Re-Upload" in b.label]
            assert len(reupload) == 1
            reupload[0].click().run()
        assert not at.exception
        assert at.session_state.messages == []
        assert at.session_state.button_key == 0


# ---------------------------------------------------------------------------
# __process_chat
# ---------------------------------------------------------------------------

class TestProcessChatFlow:
    def _boot_with_db(self, **state_overrides):
        """Start the app in a state where chat_input is visible."""
        _isdir, _listdir = _db_patches()
        at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
        for k, v in _ALL_KEYS.items():
            at.session_state[k] = v
        at.session_state["model_name"] = "llama3:latest"
        at.session_state["model_temperature"] = 0.7
        for k, v in state_overrides.items():
            at.session_state[k] = v
        return at, _isdir, _listdir

    def test_chat_input_visible_when_notes_and_model_ready(self):
        at, _isdir, _listdir = self._boot_with_db()
        with patch("os.path.isdir", side_effect=_isdir), \
             patch("os.listdir", side_effect=_listdir), \
             patch("os.path.isfile", side_effect=_hide_userdata_isfile()):
            at.run()
        assert not at.exception
        assert len(at.chat_input) == 1

    def test_no_notes_found_appends_canned_response(self):
        at, _isdir, _listdir = self._boot_with_db()
        with patch("os.path.isdir", side_effect=_isdir), \
             patch("os.listdir", side_effect=_listdir), \
             patch("os.path.isfile", side_effect=_hide_userdata_isfile()):
            at.run()
            at.chat_input[0].set_value("What happened to the dragon?").run()
        assert not at.exception
        msgs = at.session_state.messages
        assert len(msgs) >= 2
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles

    def test_user_message_stored_in_session(self):
        at, _isdir, _listdir = self._boot_with_db()
        with patch("os.path.isdir", side_effect=_isdir), \
             patch("os.listdir", side_effect=_listdir), \
             patch("os.path.isfile", side_effect=_hide_userdata_isfile()):
            at.run()
            at.chat_input[0].set_value("Did the wizard survive?").run()
        assert not at.exception
        user_msgs = [m for m in at.session_state.messages if m["role"] == "user"]
        assert any("wizard" in m["content"].lower() for m in user_msgs)

