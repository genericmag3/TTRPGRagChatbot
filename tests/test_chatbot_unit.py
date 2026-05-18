"""Pure unit tests for TTRPGChatbot class methods — no AppTest, no Streamlit runtime."""

import json
import os
import pytest
from unittest.mock import MagicMock, mock_open, patch

from src.app.TTRPGChatBot import TTRPGChatbot
from src.utils.DatabaseHandler import DATABASE_DIR


class _SS(dict):
    """Minimal session_state stand-in that supports both attr and item access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        del self[name]


def _make_bot():
    """Instantiate TTRPGChatbot without running __init__ to avoid Streamlit and I/O."""
    bot = TTRPGChatbot.__new__(TTRPGChatbot)
    bot._DATABASEDIR = DATABASE_DIR
    bot._USERDATAFILE = "data//user_data.json"
    bot._PROMPTEMPLATE = MagicMock()
    bot.databasehandler = MagicMock()
    bot.llmhandler = MagicMock()
    return bot


# ---------------------------------------------------------------------------
# __save_user_data — must preserve fields written by other pages
# ---------------------------------------------------------------------------

class TestSaveUserData:
    """__save_user_data must not erase summary_model fields set by CampaignSummarizer."""

    def test_preserves_summary_model_fields_when_file_has_them(self, tmp_path):
        bot = _make_bot()
        data_file = tmp_path / "user_data.json"
        data_file.write_text(json.dumps({
            "summary_model_name": "mistral",
            "summary_model_temperature": 0.3,
        }))
        bot._USERDATAFILE = str(data_file)
        ss = _SS(
            model_name="llama3:latest",
            model_temperature=0.7,
            notes_uploaded=True,
            party_members=[{"id": "1", "name": "Aria", "note_taker": True}],
        )
        with patch("streamlit.session_state", ss):
            bot._TTRPGChatbot__save_user_data()
        saved = json.loads(data_file.read_text())
        assert saved["summary_model_name"] == "mistral"
        assert saved["summary_model_temperature"] == 0.3
        assert saved["model_name"] == "llama3:latest"

    def test_writes_qa_fields_when_no_prior_file(self, tmp_path):
        bot = _make_bot()
        data_file = tmp_path / "user_data.json"
        bot._USERDATAFILE = str(data_file)
        ss = _SS(
            model_name="llama3:latest",
            model_temperature=0.5,
            notes_uploaded=False,
            party_members=[],
        )
        with patch("streamlit.session_state", ss):
            bot._TTRPGChatbot__save_user_data()
        saved = json.loads(data_file.read_text())
        assert saved["model_name"] == "llama3:latest"
        assert saved["notes_uploaded"] is False


# ---------------------------------------------------------------------------
# __stream_data
# ---------------------------------------------------------------------------

class TestStreamData:
    def test_yields_space_appended_words(self):
        bot = _make_bot()
        with patch("time.sleep"):
            result = list(bot._TTRPGChatbot__stream_data("hello world"))
        assert result == ["hello ", "world "]

    def test_single_word(self):
        bot = _make_bot()
        with patch("time.sleep"):
            result = list(bot._TTRPGChatbot__stream_data("single"))
        assert result == ["single "]


# ---------------------------------------------------------------------------
# __reset_chat_history
# ---------------------------------------------------------------------------

class TestResetChatHistory:
    def test_clears_messages_buttoninfo_and_key(self):
        bot = _make_bot()
        ss = _SS(messages=["m1"], buttoninfo=["b1"], button_key=5)
        with patch("streamlit.session_state", ss):
            bot._TTRPGChatbot__reset_chat_history()
        assert ss["messages"] == []
        assert ss["buttoninfo"] == []
        assert ss["button_key"] == 0


# ---------------------------------------------------------------------------
# __delete_member
# ---------------------------------------------------------------------------

class TestDeleteMember:
    def test_removes_the_specified_member(self):
        bot = _make_bot()
        ss = _SS(party_members=[
            {"id": "aaa", "name": "Alice", "note_taker": False},
            {"id": "bbb", "name": "Bob", "note_taker": False},
        ])
        with patch("streamlit.session_state", ss):
            bot._TTRPGChatbot__delete_member("aaa")
        assert [m["id"] for m in ss["party_members"]] == ["bbb"]

    def test_no_error_when_id_absent(self):
        bot = _make_bot()
        ss = _SS(party_members=[{"id": "aaa", "name": "Alice", "note_taker": False}])
        with patch("streamlit.session_state", ss):
            bot._TTRPGChatbot__delete_member("zzz")
        assert len(ss["party_members"]) == 1


# ---------------------------------------------------------------------------
# __toggle_note_taker
# ---------------------------------------------------------------------------

class TestToggleNoteTaker:
    def test_sets_note_taker_true_on_matching_member(self):
        bot = _make_bot()
        mid = "m1"
        ss = _SS(party_members=[{"id": mid, "name": "Alice", "note_taker": False}])
        ss[f"note_taker_{mid}"] = True
        with patch("streamlit.session_state", ss):
            bot._TTRPGChatbot__toggle_note_taker(mid)
        assert ss["party_members"][0]["note_taker"] is True

    def test_sets_note_taker_false_on_matching_member(self):
        bot = _make_bot()
        mid = "m1"
        ss = _SS(party_members=[{"id": mid, "name": "Alice", "note_taker": True}])
        ss[f"note_taker_{mid}"] = False
        with patch("streamlit.session_state", ss):
            bot._TTRPGChatbot__toggle_note_taker(mid)
        assert ss["party_members"][0]["note_taker"] is False

    def test_does_not_change_non_matching_members(self):
        bot = _make_bot()
        mid = "m1"
        other = "m2"
        ss = _SS(party_members=[
            {"id": mid, "name": "Alice", "note_taker": False},
            {"id": other, "name": "Bob", "note_taker": False},
        ])
        ss[f"note_taker_{mid}"] = True
        with patch("streamlit.session_state", ss):
            bot._TTRPGChatbot__toggle_note_taker(mid)
        assert ss["party_members"][1]["note_taker"] is False


# ---------------------------------------------------------------------------
# __process_chat — happy path (notes found, LLM invoked, response stored)
# ---------------------------------------------------------------------------

class TestProcessChatHappyPath:
    """Unit tests for __process_chat when retrieve_notes returns documents."""

    def _make_ready_bot(self, party_members=None):
        if party_members is None:
            party_members = [{"id": "p1", "name": "Aria", "note_taker": True}]
        bot = _make_bot()
        ss = _SS(
            notes_uploaded=True,
            model_name="llama3:latest",
            messages=[],
            buttoninfo=[],
            button_key=0,
            party_members=party_members,
            is_processing=True,
        )
        return bot, ss

    def _mock_note(self, date="2023-10-27", content="The party defeated the dragon."):
        note = MagicMock()
        note.page_content = content
        note.metadata = {"Date": date}
        return note

    def _run_chat(self, bot, ss, question, notes, llm_response="Answer."):
        # Simulate Phase 2: a question was already captured in Phase 1
        ss["_pending_chat"] = question
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
            bot._TTRPGChatbot__process_chat()

    def test_llm_response_stored_as_assistant_message(self):
        bot, ss = self._make_ready_bot()
        self._run_chat(bot, ss, "What happened?", [self._mock_note()], "Aria defeated the dragon.")
        assistant_msgs = [m for m in ss["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert "Aria defeated the dragon." in assistant_msgs[0]["content"]

    def test_user_message_stored_before_response(self):
        bot, ss = self._make_ready_bot()
        self._run_chat(bot, ss, "What happened?", [self._mock_note()])
        roles = [m["role"] for m in ss["messages"]]
        assert roles == ["user", "assistant"]
        assert ss["messages"][0]["content"] == "What happened?"

    def test_reference_buttoninfo_populated_per_note(self):
        bot, ss = self._make_ready_bot()
        notes = [self._mock_note("2023-10-27"), self._mock_note("2023-10-28")]
        self._run_chat(bot, ss, "Tell me about the campaign.", notes)
        assert len(ss["buttoninfo"]) == 1
        btn_entries = ss["buttoninfo"][0]
        assert btn_entries is not None
        assert len(btn_entries) == 2
        dates = [e[0] for e in btn_entries]
        assert "2023-10-27" in dates
        assert "2023-10-28" in dates

    def test_invoke_model_receives_correct_mappings(self):
        bot, ss = self._make_ready_bot()
        self._run_chat(bot, ss, "Did the wizard help?", [self._mock_note()])
        # bot.llmhandler is a MagicMock so call_args.args has no implicit self
        # args[0] = prompt_template, args[1] = mappings dict
        mappings = bot.llmhandler.invoke_model.call_args.args[1]
        assert mappings["question"] == "Did the wizard help?"
        assert "Aria" in mappings["partymembers"]
        assert mappings["notetaker"] == "Aria"

    def test_multiple_party_members_formatted_with_and(self):
        members = [
            {"id": "p1", "name": "Aria", "note_taker": True},
            {"id": "p2", "name": "Brom", "note_taker": False},
            {"id": "p3", "name": "Cael", "note_taker": False},
        ]
        bot, ss = self._make_ready_bot(party_members=members)
        self._run_chat(bot, ss, "Any question?", [self._mock_note()])
        members_str = bot.llmhandler.invoke_model.call_args.args[1]["partymembers"]
        assert "Aria" in members_str
        assert "Brom" in members_str
        assert "Cael" in members_str
        assert "and" in members_str

    def test_button_key_incremented_once_per_retrieved_note(self):
        bot, ss = self._make_ready_bot()
        notes = [
            self._mock_note("2023-11-01"),
            self._mock_note("2023-11-02"),
            self._mock_note("2023-11-03"),
        ]
        self._run_chat(bot, ss, "What happened?", notes)
        assert ss["button_key"] == 3
