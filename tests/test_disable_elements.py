"""Tests verifying is_processing flag management for the disable-elements feature."""

import pytest
from unittest.mock import MagicMock, mock_open, patch

from src.app.TTRPGChatBot import TTRPGChatbot
from src.app.NoteEditor import NoteEditor
from src.app.CampaignSummarizer import CampaignSummarizer


class _SS(dict):
    """Minimal session_state stand-in supporting both attr and item access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot():
    bot = TTRPGChatbot.__new__(TTRPGChatbot)
    bot._DATABASEDIR = "test_db"
    bot._USERDATAFILE = "data/user_data.json"
    bot._PROMPTEMPLATE = MagicMock()
    bot.databasehandler = MagicMock()
    bot.llmhandler = MagicMock()
    bot.summaryhandler = MagicMock()
    return bot


def _make_editor():
    editor = NoteEditor.__new__(NoteEditor)
    editor.databasehandler = MagicMock()
    editor._editor = MagicMock()
    return editor


def _make_summarizer():
    cs = CampaignSummarizer.__new__(CampaignSummarizer)
    cs.llm_handler = MagicMock()
    cs.summary_handler = MagicMock()
    return cs


# ---------------------------------------------------------------------------
# TTRPGChatbot: is_processing initialisation
# ---------------------------------------------------------------------------

class TestChatbotIsProcessingInit:
    def test_initialized_to_false_on_fresh_start(self, tmp_path):
        bot = _make_bot()
        bot._USERDATAFILE = str(tmp_path / "missing.json")
        bot.summaryhandler.summary_exists.return_value = False
        ss = _SS()
        with patch("streamlit.session_state", ss):
            bot._TTRPGChatbot__init_state_variables()
        assert ss.get("is_processing") is False

    def test_not_overwritten_when_already_present(self, tmp_path):
        bot = _make_bot()
        bot._USERDATAFILE = str(tmp_path / "missing.json")
        bot.summaryhandler.summary_exists.return_value = False
        ss = _SS(
            is_processing=True,
            reupload_key=0, model_name=None, model_temperature=None,
            notes_uploaded=False, messages=[], buttoninfo=[], button_key=0,
            party_members=[], delete_index=None, summary_generated=False,
        )
        with patch("streamlit.session_state", ss):
            bot._TTRPGChatbot__init_state_variables()
        assert ss["is_processing"] is True


# ---------------------------------------------------------------------------
# TTRPGChatbot: chat Phase 1 — capture question, set flags, call rerun
# ---------------------------------------------------------------------------

class TestChatPhase1:
    def _make_ss(self):
        return _SS(
            notes_uploaded=True,
            model_name="llama3:latest",
            messages=[],
            buttoninfo=[],
            button_key=0,
            party_members=[{"id": "p1", "name": "Aria", "note_taker": True}],
            is_processing=False,
        )

    def _run(self, bot, ss, question):
        mock_rerun = MagicMock()
        with patch("streamlit.session_state", ss), \
             patch("streamlit.chat_input", return_value=question), \
             patch("streamlit.rerun", mock_rerun):
            try:
                bot._TTRPGChatbot__process_chat()
            except Exception:
                pass
        return mock_rerun

    def test_saves_question_to_pending_chat(self):
        bot = _make_bot()
        ss = self._make_ss()
        self._run(bot, ss, "What happened last session?")
        assert ss.get("_pending_chat") == "What happened last session?"

    def test_sets_is_processing_true(self):
        bot = _make_bot()
        ss = self._make_ss()
        self._run(bot, ss, "What happened last session?")
        assert ss.get("is_processing") is True

    def test_calls_rerun(self):
        bot = _make_bot()
        ss = self._make_ss()
        mock_rerun = self._run(bot, ss, "What happened last session?")
        mock_rerun.assert_called_once()

    def test_no_action_when_input_is_none(self):
        bot = _make_bot()
        ss = self._make_ss()
        self._run(bot, ss, None)
        assert "_pending_chat" not in ss
        assert ss.get("is_processing") is False


# ---------------------------------------------------------------------------
# TTRPGChatbot: chat Phase 2 — process pending question, clear flags
# ---------------------------------------------------------------------------

class TestChatPhase2:
    def _make_ss(self, question="What happened?"):
        return _SS(
            notes_uploaded=True,
            model_name="llama3:latest",
            messages=[],
            buttoninfo=[],
            button_key=0,
            party_members=[{"id": "p1", "name": "Aria", "note_taker": True}],
            is_processing=True,
            _pending_chat=question,
        )

    def _run_phase2(self, bot, ss, notes=None, llm_response="Answer."):
        if notes is None:
            notes = []
        bot.databasehandler.retrieve_notes.return_value = notes
        bot.llmhandler.invoke_model.return_value = llm_response
        with patch("streamlit.session_state", ss), \
             patch("streamlit.chat_input", return_value=None), \
             patch("streamlit.chat_message", return_value=MagicMock()), \
             patch("streamlit.markdown"), \
             patch("streamlit.write_stream"), \
             patch("streamlit.button"), \
             patch("streamlit.empty", return_value=MagicMock()), \
             patch("streamlit.rerun"), \
             patch("time.sleep"), \
             patch("builtins.open", mock_open(read_data="{}")), \
             patch("json.load", return_value={}):
            try:
                bot._TTRPGChatbot__process_chat()
            except Exception:
                pass

    def test_clears_pending_chat(self):
        bot = _make_bot()
        ss = self._make_ss()
        self._run_phase2(bot, ss)
        assert "_pending_chat" not in ss

    def test_sets_is_processing_false(self):
        bot = _make_bot()
        ss = self._make_ss()
        self._run_phase2(bot, ss)
        assert ss.get("is_processing") is False

    def test_appends_user_message(self):
        bot = _make_bot()
        ss = self._make_ss("Did anything happen?")
        self._run_phase2(bot, ss)
        user_msgs = [m for m in ss["messages"] if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "Did anything happen?"

    def test_invokes_llm_when_notes_found(self):
        bot = _make_bot()
        ss = self._make_ss()
        note = MagicMock()
        note.page_content = "The party fought goblins."
        note.metadata = {"Date": "2023-10-27"}
        self._run_phase2(bot, ss, notes=[note])
        bot.llmhandler.invoke_model.assert_called_once()


# ---------------------------------------------------------------------------
# TTRPGChatbot: file upload Phase 1 — set _processing_upload and is_processing
# ---------------------------------------------------------------------------

class _RerunSignal(Exception):
    """Raised by the mocked st.rerun to stop script execution, just like the real one."""


class TestFileUploadPhase1:
    def _make_ss(self):
        return _SS(
            notes_uploaded=False,
            model_name="llama3:latest",
            model_temperature=0.7,
            # name="" matches what the mocked st.text_input returns, avoiding
            # an early st.rerun() from the name-change auto-save check.
            party_members=[{"id": "p1", "name": "", "note_taker": True}],
            is_processing=False,
            reupload_key=False,
            summary_generated=False,
            delete_index=None,
        )

    def _run(self, bot, ss, uploaded_file):
        mock_rerun = MagicMock(side_effect=_RerunSignal())
        mock_container = MagicMock()
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock(return_value=False)
        mock_empty = MagicMock()
        mock_empty.container.return_value = mock_container
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=False)

        with patch("streamlit.session_state", ss), \
             patch("streamlit.file_uploader", return_value=uploaded_file), \
             patch("streamlit.empty", return_value=mock_empty), \
             patch("streamlit.rerun", mock_rerun), \
             patch("streamlit.info"), \
             patch("streamlit.header"), \
             patch("streamlit.subheader"), \
             patch("streamlit.button", return_value=False), \
             patch("streamlit.text_input", return_value=""), \
             patch("streamlit.checkbox", return_value=False), \
             patch("streamlit.columns", return_value=[mock_col, mock_col, mock_col]), \
             patch("streamlit.sidebar", MagicMock()):
            try:
                bot._TTRPGChatbot__process_journal_options()
            except (_RerunSignal, Exception):
                pass
        return mock_rerun

    def test_sets_processing_upload_flag(self):
        bot = _make_bot()
        bot.summaryhandler.raw_notes_exist.return_value = False
        ss = self._make_ss()
        self._run(bot, ss, MagicMock())
        assert ss.get("_processing_upload") is True

    def test_sets_is_processing_true(self):
        bot = _make_bot()
        bot.summaryhandler.raw_notes_exist.return_value = False
        ss = self._make_ss()
        self._run(bot, ss, MagicMock())
        assert ss.get("is_processing") is True

    def test_calls_rerun(self):
        bot = _make_bot()
        bot.summaryhandler.raw_notes_exist.return_value = False
        ss = self._make_ss()
        mock_rerun = self._run(bot, ss, MagicMock())
        mock_rerun.assert_called()

    def test_no_action_when_no_file(self):
        bot = _make_bot()
        bot.summaryhandler.raw_notes_exist.return_value = False
        ss = self._make_ss()
        self._run(bot, ss, None)
        assert "_processing_upload" not in ss
        assert ss.get("is_processing") is False


# ---------------------------------------------------------------------------
# TTRPGChatbot: file upload Phase 2 — process file, clear flags, call rerun
# ---------------------------------------------------------------------------

class TestFileUploadPhase2:
    def _make_ss(self):
        return _SS(
            notes_uploaded=False,
            model_name="llama3:latest",
            model_temperature=0.7,
            party_members=[{"id": "p1", "name": "", "note_taker": True}],
            is_processing=True,
            reupload_key=False,
            summary_generated=False,
            delete_index=None,
            _processing_upload=True,
        )

    def _run(self, bot, ss, uploaded_file):
        mock_rerun = MagicMock(side_effect=_RerunSignal())
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=False)
        mock_empty = MagicMock()

        with patch("streamlit.session_state", ss), \
             patch("streamlit.file_uploader", return_value=uploaded_file), \
             patch("streamlit.empty", return_value=mock_empty), \
             patch("streamlit.rerun", mock_rerun), \
             patch("streamlit.header"), \
             patch("streamlit.button", return_value=False), \
             patch("streamlit.text_input", return_value=""), \
             patch("streamlit.checkbox", return_value=False), \
             patch("streamlit.columns", return_value=[mock_col, mock_col, mock_col]), \
             patch("streamlit.sidebar", MagicMock()), \
             patch("streamlit.toast"), \
             patch("streamlit.info"), \
             patch("builtins.open", mock_open(read_data="{}")), \
             patch("os.path.isfile", return_value=False), \
             patch("os.makedirs"), \
             patch.object(bot, "_TTRPGChatbot__create_database_handler", return_value=True):
            try:
                bot._TTRPGChatbot__process_journal_options()
            except (_RerunSignal, Exception):
                pass
        return mock_rerun

    def test_calls_rerun_after_processing(self):
        bot = _make_bot()
        bot.summaryhandler.raw_notes_exist.return_value = False
        ss = self._make_ss()
        mock_rerun = self._run(bot, ss, MagicMock())
        mock_rerun.assert_called()

    def test_clears_is_processing(self):
        bot = _make_bot()
        bot.summaryhandler.raw_notes_exist.return_value = False
        ss = self._make_ss()
        self._run(bot, ss, MagicMock())
        assert ss.get("is_processing") is False

    def test_clears_processing_upload_flag(self):
        bot = _make_bot()
        bot.summaryhandler.raw_notes_exist.return_value = False
        ss = self._make_ss()
        self._run(bot, ss, MagicMock())
        assert "_processing_upload" not in ss


# ---------------------------------------------------------------------------
# NoteEditor: is_processing initialisation
# ---------------------------------------------------------------------------

class TestNoteEditorIsProcessingInit:
    def test_initialized_to_false(self):
        editor = _make_editor()
        ss = _SS(
            editor_content="",
            editor_key=0,
            editor_font_family="Georgia",
            editor_font_size=16,
        )
        with patch("streamlit.session_state", ss):
            editor._NoteEditor__init_state_variables()
        assert ss.get("is_processing") is False

    def test_not_overwritten_when_already_present(self):
        editor = _make_editor()
        ss = _SS(
            editor_content="",
            editor_key=0,
            editor_font_family="Georgia",
            editor_font_size=16,
            is_processing=True,
        )
        with patch("streamlit.session_state", ss):
            editor._NoteEditor__init_state_variables()
        assert ss.get("is_processing") is True


# ---------------------------------------------------------------------------
# NoteEditor: __vectorize_notes clears is_processing on completion
# ---------------------------------------------------------------------------

class TestNoteEditorVectorizeIsProcessing:
    def _make_gen(self, return_code=True):
        """Generator that yields one progress value then returns return_code."""
        def _gen():
            yield 50
            return return_code
        return _gen()

    def _run_vectorize(self, editor, ss, return_code=True):
        editor.databasehandler.generate_database.return_value = self._make_gen(return_code)
        editor.databasehandler.last_processed_df = None
        mock_slot = MagicMock()
        mock_slot.container.return_value.__enter__ = MagicMock(return_value=None)
        mock_slot.container.return_value.__exit__ = MagicMock(return_value=False)
        mock_slot.progress.return_value = MagicMock()

        with patch("streamlit.session_state", ss), \
             patch("streamlit.empty", return_value=mock_slot), \
             patch("streamlit.progress", return_value=MagicMock()), \
             patch("streamlit.error"), \
             patch("streamlit.toast"), \
             patch("builtins.open", mock_open(read_data="{}")), \
             patch("json.load", return_value={}), \
             patch("os.path.isfile", return_value=False), \
             patch("os.remove"), \
             patch("os.makedirs"):
            editor._NoteEditor__vectorize_notes()

    def test_is_processing_false_after_success(self):
        editor = _make_editor()
        ss = _SS(editor_content="2023-01-01\nSome notes", is_processing=True)
        self._run_vectorize(editor, ss, return_code=True)
        assert ss.get("is_processing") is False

    def test_is_processing_false_after_failure(self):
        editor = _make_editor()
        ss = _SS(editor_content="2023-01-01\nSome notes", is_processing=True)
        self._run_vectorize(editor, ss, return_code=False)
        assert ss.get("is_processing") is False


# ---------------------------------------------------------------------------
# CampaignSummarizer: is_processing initialisation
# ---------------------------------------------------------------------------

class TestCampaignSummarizerIsProcessingInit:
    def test_initialized_to_false(self, tmp_path):
        cs = _make_summarizer()
        cs._USERDATAFILE = str(tmp_path / "missing.json")
        ss = _SS()
        with patch("streamlit.session_state", ss):
            cs._CampaignSummarizer__init_state_variables()
        assert ss.get("is_processing") is False

    def test_not_overwritten_when_already_present(self, tmp_path):
        cs = _make_summarizer()
        cs._USERDATAFILE = str(tmp_path / "missing.json")
        ss = _SS(
            summary_model_name="llama3",
            summary_model_temperature=0.7,
            party_members=[],
            is_processing=True,
        )
        with patch("streamlit.session_state", ss):
            cs._CampaignSummarizer__init_state_variables()
        assert ss.get("is_processing") is True
