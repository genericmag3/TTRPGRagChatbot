"""Unit tests for CampaignSummarizer — Streamlit calls are mocked."""

import json
import pytest
from unittest.mock import MagicMock, patch, mock_open

from src.app.CampaignSummarizer import CampaignSummarizer


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
    def get(self, key, default=None):
        return self[key] if key in self else default
    def pop(self, key, *args):
        return dict.pop(self, key, *args)


def _make_summarizer():
    """Instantiate CampaignSummarizer without running __init__."""
    cs = CampaignSummarizer.__new__(CampaignSummarizer)
    cs.llm_handler = MagicMock()
    cs.summary_handler = MagicMock()
    return cs


# ---------------------------------------------------------------------------
# __init_state_variables — party_members persistence
# ---------------------------------------------------------------------------

class TestInitStateVariables:
    """__init_state_variables must load party_members from user data when absent from session."""

    def test_loads_party_members_from_file_when_absent_from_session(self, tmp_path):
        cs = _make_summarizer()
        saved_members = [{"id": "1", "name": "Aria", "note_taker": True}]
        data_file = tmp_path / "user_data.json"
        data_file.write_text(json.dumps({
            "summary_model_name": "llama3:latest",
            "summary_model_temperature": 0.5,
            "party_members": saved_members,
        }))
        cs._USERDATAFILE = str(data_file)
        ss = _SS()
        with patch("streamlit.session_state", ss):
            cs._CampaignSummarizer__init_state_variables()
        assert ss.get("party_members") == saved_members

    def test_party_members_defaults_to_empty_list_when_file_absent(self, tmp_path):
        cs = _make_summarizer()
        cs._USERDATAFILE = str(tmp_path / "nonexistent.json")
        ss = _SS()
        with patch("streamlit.session_state", ss):
            cs._CampaignSummarizer__init_state_variables()
        assert ss.get("party_members") == []

    def test_does_not_overwrite_party_members_already_in_session(self, tmp_path):
        cs = _make_summarizer()
        existing = [{"id": "99", "name": "Veteran", "note_taker": False}]
        data_file = tmp_path / "user_data.json"
        data_file.write_text(json.dumps({"party_members": [{"id": "1", "name": "Aria", "note_taker": True}]}))
        cs._USERDATAFILE = str(data_file)
        ss = _SS(party_members=existing)
        with patch("streamlit.session_state", ss):
            cs._CampaignSummarizer__init_state_variables()
        assert ss.get("party_members") == existing


# ---------------------------------------------------------------------------
# __extract_headers
# ---------------------------------------------------------------------------

class TestExtractHeaders:
    def test_extracts_h1_header(self):
        cs = _make_summarizer()
        result = cs._CampaignSummarizer__extract_headers("# Title\nSome text")
        assert result == [(1, "Title")]

    def test_extracts_multiple_levels(self):
        cs = _make_summarizer()
        text = "# H1\n## H2\n### H3"
        result = cs._CampaignSummarizer__extract_headers(text)
        assert result == [(1, "H1"), (2, "H2"), (3, "H3")]

    def test_returns_empty_for_no_headers(self):
        cs = _make_summarizer()
        assert cs._CampaignSummarizer__extract_headers("no headers here") == []

    def test_ignores_h4_and_deeper(self):
        cs = _make_summarizer()
        result = cs._CampaignSummarizer__extract_headers("#### Too deep\n##### Even deeper")
        assert result == []

    def test_strips_trailing_whitespace_from_title(self):
        cs = _make_summarizer()
        result = cs._CampaignSummarizer__extract_headers("# Title   ")
        assert result == [(1, "Title")]

    def test_multiple_headers_in_order(self):
        cs = _make_summarizer()
        text = "# Intro\nsome text\n## Section One\nmore text\n## Section Two"
        result = cs._CampaignSummarizer__extract_headers(text)
        assert result == [(1, "Intro"), (2, "Section One"), (2, "Section Two")]


# ---------------------------------------------------------------------------
# run() — party member gate
# ---------------------------------------------------------------------------

class TestRunPartyMemberGate:
    """run() must stop before offering generation when no named party members exist."""

    def _run_to_stop(self, cs, ss):
        """Run cs.run() expecting it to hit st.stop(); capture what st.info was called with."""
        info_calls = []
        with patch("streamlit.session_state", ss), \
             patch.object(cs, "_CampaignSummarizer__init_state_variables"), \
             patch.object(cs, "_CampaignSummarizer__process_model_options"), \
             patch.object(cs, "_notes_in_database", return_value=True), \
             patch("streamlit.title"), \
             patch("streamlit.info", side_effect=lambda msg: info_calls.append(msg)), \
             patch("streamlit.page_link"), \
             patch("streamlit.stop", side_effect=StopIteration), \
             patch("streamlit.button", return_value=False), \
             patch("streamlit.warning"):
            try:
                cs.run()
            except StopIteration:
                pass
        return info_calls

    def test_stops_when_party_members_empty_list(self):
        cs = _make_summarizer()
        cs.summary_handler.raw_notes_exist.return_value = True
        cs.summary_handler.get_saved_summary.return_value = None
        ss = _SS(summary_model_name="llama3:latest", party_members=[])
        info_calls = self._run_to_stop(cs, ss)
        assert any("party member" in msg.lower() for msg in info_calls)

    def test_stops_when_all_members_have_blank_names(self):
        cs = _make_summarizer()
        cs.summary_handler.raw_notes_exist.return_value = True
        cs.summary_handler.get_saved_summary.return_value = None
        ss = _SS(summary_model_name="llama3:latest", party_members=[
            {"id": "1", "name": "", "note_taker": False},
            {"id": "2", "name": "   ", "note_taker": False},
        ])
        info_calls = self._run_to_stop(cs, ss)
        assert any("party member" in msg.lower() for msg in info_calls)

    def test_stops_when_party_members_key_absent(self):
        cs = _make_summarizer()
        cs.summary_handler.raw_notes_exist.return_value = True
        cs.summary_handler.get_saved_summary.return_value = None
        ss = _SS(summary_model_name="llama3:latest")
        info_calls = self._run_to_stop(cs, ss)
        assert any("party member" in msg.lower() for msg in info_calls)

    def test_proceeds_when_at_least_one_named_member(self):
        cs = _make_summarizer()
        cs.summary_handler.raw_notes_exist.return_value = True
        cs.summary_handler.get_saved_summary.return_value = None
        ss = _SS(
            summary_model_name="llama3:latest",
            party_members=[{"id": "1", "name": "Aria", "note_taker": True}],
            is_processing=False,
        )
        button_called = []
        warning_called = []
        with patch("streamlit.session_state", ss), \
             patch.object(cs, "_CampaignSummarizer__init_state_variables"), \
             patch.object(cs, "_CampaignSummarizer__process_model_options"), \
             patch.object(cs, "_notes_in_database", return_value=True), \
             patch("streamlit.title"), \
             patch("streamlit.info"), \
             patch("streamlit.warning", side_effect=lambda msg: warning_called.append(msg)), \
             patch("streamlit.button", side_effect=lambda *a, **kw: button_called.append(True) or False), \
             patch("streamlit.stop", side_effect=StopIteration), \
             patch("streamlit.page_link"):
            try:
                cs.run()
            except StopIteration:
                pass
        # Button should have been rendered (generate button), not stopped at gate
        assert button_called


# ---------------------------------------------------------------------------
# run() — time warning displayed before generate button
# ---------------------------------------------------------------------------

class TestRunTimeWarning:
    def test_warning_shown_when_no_existing_summary(self):
        cs = _make_summarizer()
        cs.summary_handler.raw_notes_exist.return_value = True
        cs.summary_handler.get_saved_summary.return_value = None
        ss = _SS(
            summary_model_name="llama3:latest",
            party_members=[{"id": "1", "name": "Aria", "note_taker": True}],
            is_processing=False,
        )
        warning_calls = []
        with patch("streamlit.session_state", ss), \
             patch.object(cs, "_CampaignSummarizer__init_state_variables"), \
             patch.object(cs, "_CampaignSummarizer__process_model_options"), \
             patch.object(cs, "_notes_in_database", return_value=True), \
             patch("streamlit.title"), \
             patch("streamlit.info"), \
             patch("streamlit.warning", side_effect=lambda msg: warning_calls.append(msg)), \
             patch("streamlit.button", return_value=False), \
             patch("streamlit.stop", side_effect=StopIteration), \
             patch("streamlit.page_link"):
            try:
                cs.run()
            except StopIteration:
                pass
        assert any("minute" in msg.lower() for msg in warning_calls)


# ---------------------------------------------------------------------------
# __generate_and_display — passes party_members to generate_summary_streaming
# ---------------------------------------------------------------------------

class TestGenerateAndDisplay:
    def _run_generate(self, cs, ss, streaming_results):
        cs.summary_handler.generate_summary_streaming.return_value = iter(streaming_results)
        cs.summary_handler.get_saved_summary.return_value = {
            "summary": "Final text.",
            "model": "llama3:latest",
            "generated_at": "2026-01-01T00:00:00",
        }
        mock_slot = MagicMock()
        with patch("streamlit.session_state", ss), \
             patch("streamlit.empty", return_value=mock_slot), \
             patch("streamlit.progress", return_value=MagicMock()), \
             patch("streamlit.success"), \
             patch("streamlit.error"), \
             patch("streamlit.stop", side_effect=StopIteration), \
             patch("streamlit.markdown"), \
             patch("streamlit.sidebar", MagicMock()), \
             patch("streamlit.caption"), \
             patch("builtins.open", mock_open(read_data="{}")), \
             patch("json.load", return_value={}):
            try:
                cs._CampaignSummarizer__generate_and_display()
            except StopIteration:
                pass

    def test_passes_party_members_to_streaming(self):
        cs = _make_summarizer()
        cs.llm_handler.load_model.return_value = None
        party = [{"id": "1", "name": "Aria", "note_taker": True}]
        ss = _SS(
            summary_model_name="llama3:latest",
            summary_model_temperature=0.7,
            party_members=party,
        )
        self._run_generate(cs, ss, [(False, 50, "Working..."), (True, 100, "Done")])
        cs.summary_handler.generate_summary_streaming.assert_called_once_with(
            "llama3:latest", party
        )

    def test_sets_summary_generated_true_on_success(self):
        cs = _make_summarizer()
        cs.llm_handler.load_model.return_value = None
        ss = _SS(
            summary_model_name="llama3:latest",
            summary_model_temperature=0.7,
            party_members=[{"id": "1", "name": "Aria", "note_taker": True}],
            summary_generated=False,
        )
        self._run_generate(cs, ss, [(False, 50, "Working..."), (True, 100, "Done")])
        assert ss.get("summary_generated") is True


# ---------------------------------------------------------------------------
# Persistence — existing summary bypasses all generation prerequisites
# ---------------------------------------------------------------------------

_SAVED_SUMMARY = {
    "summary": "The campaign so far.",
    "model": "llama3:latest",
    "generated_at": "2026-01-01T00:00:00",
}


class TestSummaryPersistence:
    """A saved summary must be displayed without requiring notes, raw notes, or party members."""

    def _run_expecting_render(self, cs, ss):
        """Return True if __render_existing_summary was called."""
        render_called = []
        with patch("streamlit.session_state", ss), \
             patch.object(cs, "_CampaignSummarizer__init_state_variables"), \
             patch.object(cs, "_CampaignSummarizer__process_model_options"), \
             patch.object(cs, "_CampaignSummarizer__render_existing_summary",
                          side_effect=lambda s: render_called.append(s)), \
             patch("streamlit.stop", side_effect=StopIteration):
            try:
                cs.run()
            except StopIteration:
                pass
        return bool(render_called)

    def test_shows_summary_when_no_model_selected(self):
        cs = _make_summarizer()
        cs.summary_handler.get_saved_summary.return_value = _SAVED_SUMMARY
        ss = _SS(summary_model_name=None, party_members=[])
        assert self._run_expecting_render(cs, ss)

    def test_shows_summary_when_notes_not_in_database(self):
        cs = _make_summarizer()
        cs.summary_handler.get_saved_summary.return_value = _SAVED_SUMMARY
        with patch.object(cs, "_notes_in_database", return_value=False):
            ss = _SS(summary_model_name="llama3:latest", party_members=[])
            assert self._run_expecting_render(cs, ss)

    def test_shows_summary_when_raw_notes_missing(self):
        cs = _make_summarizer()
        cs.summary_handler.get_saved_summary.return_value = _SAVED_SUMMARY
        cs.summary_handler.raw_notes_exist.return_value = False
        with patch.object(cs, "_notes_in_database", return_value=True):
            ss = _SS(summary_model_name="llama3:latest", party_members=[])
            assert self._run_expecting_render(cs, ss)

    def test_shows_summary_when_no_named_party_members(self):
        cs = _make_summarizer()
        cs.summary_handler.get_saved_summary.return_value = _SAVED_SUMMARY
        cs.summary_handler.raw_notes_exist.return_value = True
        with patch.object(cs, "_notes_in_database", return_value=True):
            ss = _SS(summary_model_name="llama3:latest", party_members=[])
            assert self._run_expecting_render(cs, ss)
