"""Unit tests for SummaryHandler — file I/O and LLM calls are mocked."""
import json
import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open

from src.utils.SummaryHandler import SummaryHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_handler(tmp_path):
    """Return a SummaryHandler with paths redirected to tmp_path."""
    mock_llm = MagicMock()
    handler = SummaryHandler(mock_llm)
    handler.SUMMARY_FILE = str(tmp_path / "campaign_summary.json")
    handler.RAW_NOTES_FILE = str(tmp_path / "raw_notes.json")
    return handler


def _write_raw_notes(tmp_path, rows=None):
    """Write a minimal raw_notes.json fixture to tmp_path."""
    if rows is None:
        rows = [
            {"Date": "2023-01-01", "Contents": "The party entered the dungeon."},
            {"Date": "2023-01-08", "Contents": "They defeated the goblin king."},
        ]
    df = pd.DataFrame(rows)
    df.to_json(str(tmp_path / "raw_notes.json"))


# ---------------------------------------------------------------------------
# summary_exists / raw_notes_exist / get_saved_summary
# ---------------------------------------------------------------------------

class TestFileChecks:
    def test_summary_exists_false_when_no_file(self, tmp_path):
        h = _make_handler(tmp_path)
        assert h.summary_exists() is False

    def test_summary_exists_true_when_file_present(self, tmp_path):
        h = _make_handler(tmp_path)
        (tmp_path / "campaign_summary.json").write_text(json.dumps({"summary": "x"}))
        assert h.summary_exists() is True

    def test_raw_notes_exist_false_when_no_file(self, tmp_path):
        h = _make_handler(tmp_path)
        assert h.raw_notes_exist() is False

    def test_raw_notes_exist_true_when_file_present(self, tmp_path):
        h = _make_handler(tmp_path)
        _write_raw_notes(tmp_path)
        assert h.raw_notes_exist() is True

    def test_get_saved_summary_returns_none_when_missing(self, tmp_path):
        h = _make_handler(tmp_path)
        assert h.get_saved_summary() is None

    def test_get_saved_summary_returns_dict_when_present(self, tmp_path):
        h = _make_handler(tmp_path)
        data = {"summary": "Campaign started.", "model": "llama3:latest"}
        (tmp_path / "campaign_summary.json").write_text(json.dumps(data))
        result = h.get_saved_summary()
        assert result["summary"] == "Campaign started."
        assert result["model"] == "llama3:latest"


# ---------------------------------------------------------------------------
# _format_party_members
# ---------------------------------------------------------------------------

class TestFormatPartyMembers:
    def setup_method(self):
        self.h = SummaryHandler(MagicMock())

    def test_none_returns_fallback(self):
        assert self.h._format_party_members(None) == "unknown party members"

    def test_empty_list_returns_fallback(self):
        assert self.h._format_party_members([]) == "unknown party members"

    def test_all_unnamed_returns_fallback(self):
        members = [{"id": "1", "name": "", "note_taker": False}]
        assert self.h._format_party_members(members) == "unknown party members"

    def test_single_named_member(self):
        members = [{"id": "1", "name": "Aria", "note_taker": True}]
        assert self.h._format_party_members(members) == "Aria"

    def test_two_named_members_uses_and(self):
        members = [
            {"id": "1", "name": "Aria", "note_taker": True},
            {"id": "2", "name": "Brom", "note_taker": False},
        ]
        result = self.h._format_party_members(members)
        assert "Aria" in result
        assert "Brom" in result
        assert "and" in result

    def test_three_members_comma_separated(self):
        members = [
            {"id": "1", "name": "Aria", "note_taker": True},
            {"id": "2", "name": "Brom", "note_taker": False},
            {"id": "3", "name": "Cael", "note_taker": False},
        ]
        result = self.h._format_party_members(members)
        assert "Aria" in result
        assert "Brom" in result
        assert "Cael" in result
        assert "and" in result

    def test_whitespace_only_name_excluded(self):
        members = [
            {"id": "1", "name": "   ", "note_taker": False},
            {"id": "2", "name": "Brom", "note_taker": False},
        ]
        result = self.h._format_party_members(members)
        assert result == "Brom"


# ---------------------------------------------------------------------------
# _split_into_chunks
# ---------------------------------------------------------------------------

class TestSplitIntoChunks:
    def setup_method(self):
        self.h = SummaryHandler(MagicMock())

    def test_single_chunk_when_text_fits(self):
        text = "Short text."
        chunks = self.h._split_into_chunks(text, chunk_size=1000)
        assert chunks == ["Short text."]

    def test_splits_into_multiple_chunks(self):
        text = "aaaa bbbb cccc dddd eeee"
        chunks = self.h._split_into_chunks(text, chunk_size=10)
        assert len(chunks) > 1

    def test_all_text_represented_across_chunks(self):
        text = "word " * 200
        chunks = self.h._split_into_chunks(text, chunk_size=100)
        combined = "".join(chunks)
        assert "word" in combined

    def test_no_empty_chunks(self):
        text = "a " * 300
        chunks = self.h._split_into_chunks(text, chunk_size=50)
        for c in chunks:
            assert c.strip() != ""

    def test_exact_fit_returns_one_chunk(self):
        text = "x" * 500
        chunks = self.h._split_into_chunks(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text


# ---------------------------------------------------------------------------
# _sort_chronologically
# ---------------------------------------------------------------------------

class TestSortChronologically:
    def setup_method(self):
        self.h = SummaryHandler(MagicMock())

    def test_sorts_iso_dates_ascending(self):
        df = pd.DataFrame({
            "Date": ["2023-03-15", "2023-01-01", "2023-06-30"],
            "Contents": ["C", "A", "B"],
        })
        result = self.h._sort_chronologically(df)
        assert list(result["Contents"]) == ["A", "C", "B"]

    def test_unparseable_dates_do_not_raise(self):
        df = pd.DataFrame({
            "Date": ["not-a-date", "also-bad"],
            "Contents": ["X", "Y"],
        })
        result = self.h._sort_chronologically(df)
        assert len(result) == 2

    def test_returns_reset_index(self):
        df = pd.DataFrame(
            {"Date": ["2023-02-01", "2023-01-01"], "Contents": ["B", "A"]},
            index=[10, 20],
        )
        result = self.h._sort_chronologically(df)
        assert list(result.index) == [0, 1]


# ---------------------------------------------------------------------------
# _get_chunk_char_size
# ---------------------------------------------------------------------------

class TestGetChunkCharSize:
    def setup_method(self):
        self.h = SummaryHandler(MagicMock())

    def test_returns_int(self):
        size = self.h._get_chunk_char_size("llama3:latest")
        assert isinstance(size, int)

    def test_uses_default_when_ollama_show_raises(self):
        import ollama as mock_ollama_mod
        mock_ollama_mod.show.side_effect = Exception("connection refused")
        size = self.h._get_chunk_char_size("llama3:latest")
        assert size == 8192
        mock_ollama_mod.show.side_effect = None

    def test_uses_context_length_from_modelinfo(self):
        import ollama as mock_ollama_mod
        mock_info = MagicMock()
        mock_info.modelinfo = {"llama.context_length": 8192}
        mock_ollama_mod.show.return_value = mock_info
        size = self.h._get_chunk_char_size("llama3:latest")
        assert size == 16384

    def test_falls_back_to_default_when_modelinfo_missing_key(self):
        import ollama as mock_ollama_mod
        mock_info = MagicMock()
        mock_info.modelinfo = {}
        mock_ollama_mod.show.return_value = mock_info
        size = self.h._get_chunk_char_size("llama3:latest")
        assert size == 8192


# ---------------------------------------------------------------------------
# generate_summary_streaming
# ---------------------------------------------------------------------------

class TestGenerateSummaryStreaming:
    def test_raises_when_raw_notes_missing(self, tmp_path):
        h = _make_handler(tmp_path)
        with pytest.raises(FileNotFoundError):
            list(h.generate_summary_streaming("llama3:latest"))

    def test_yields_only_false_then_true(self, tmp_path):
        _write_raw_notes(tmp_path)
        h = _make_handler(tmp_path)
        h.llm_handler.invoke_model.return_value = "The campaign summary."

        with patch.object(h, "_get_chunk_char_size", return_value=100_000), \
             patch.object(h, "_sort_chronologically", side_effect=lambda df: df):
            results = list(h.generate_summary_streaming("llama3:latest"))

        done_results = [r for r in results if r[0] is True]
        progress_results = [r for r in results if r[0] is False]
        assert len(done_results) == 1
        assert len(progress_results) >= 1

    def test_final_yield_contains_summary_text(self, tmp_path):
        _write_raw_notes(tmp_path)
        h = _make_handler(tmp_path)
        h.llm_handler.invoke_model.return_value = "Narrative summary text."

        with patch.object(h, "_get_chunk_char_size", return_value=100_000):
            results = list(h.generate_summary_streaming("llama3:latest"))

        is_done, progress, text = results[-1]
        assert is_done is True
        assert progress == 100
        assert "Narrative summary text." in text

    def test_saves_summary_to_disk(self, tmp_path):
        _write_raw_notes(tmp_path)
        h = _make_handler(tmp_path)
        h.llm_handler.invoke_model.return_value = "Saved summary."

        with patch.object(h, "_get_chunk_char_size", return_value=100_000):
            list(h.generate_summary_streaming("llama3:latest"))

        assert os.path.isfile(h.SUMMARY_FILE)
        with open(h.SUMMARY_FILE) as f:
            data = json.load(f)
        assert data["summary"] == "Saved summary."
        assert data["model"] == "llama3:latest"
        assert "generated_at" in data

    def test_multi_chunk_calls_invoke_multiple_times(self, tmp_path):
        _write_raw_notes(tmp_path, rows=[
            {"Date": "2023-01-01", "Contents": "A " * 500},
            {"Date": "2023-01-02", "Contents": "B " * 500},
        ])
        h = _make_handler(tmp_path)
        h.llm_handler.invoke_model.return_value = "chunk summary"

        with patch.object(h, "_get_chunk_char_size", return_value=200):
            list(h.generate_summary_streaming("llama3:latest"))

        assert h.llm_handler.invoke_model.call_count > 1

    def test_progress_values_are_within_range(self, tmp_path):
        _write_raw_notes(tmp_path)
        h = _make_handler(tmp_path)
        h.llm_handler.invoke_model.return_value = "ok"

        with patch.object(h, "_get_chunk_char_size", return_value=100_000):
            results = list(h.generate_summary_streaming("llama3:latest"))

        for _, progress, _ in results:
            assert 0 <= progress <= 100

    def test_party_members_included_in_final_summary_prompt(self, tmp_path):
        _write_raw_notes(tmp_path)
        h = _make_handler(tmp_path)
        h.llm_handler.invoke_model.return_value = "Summary with party."

        party = [
            {"id": "1", "name": "Aria", "note_taker": True},
            {"id": "2", "name": "Brom", "note_taker": False},
        ]

        with patch.object(h, "_get_chunk_char_size", return_value=100_000):
            list(h.generate_summary_streaming("llama3:latest", party_members=party))

        # The final invoke_model call should include party_members in its input dict
        last_call_kwargs = h.llm_handler.invoke_model.call_args
        input_dict = last_call_kwargs.args[1]
        assert "party_members" in input_dict
        assert "Aria" in input_dict["party_members"]
        assert "Brom" in input_dict["party_members"]

    def test_generate_without_party_members_does_not_raise(self, tmp_path):
        _write_raw_notes(tmp_path)
        h = _make_handler(tmp_path)
        h.llm_handler.invoke_model.return_value = "Summary."

        with patch.object(h, "_get_chunk_char_size", return_value=100_000):
            results = list(h.generate_summary_streaming("llama3:latest", party_members=None))

        assert results[-1][0] is True

    def test_generate_with_empty_party_members_does_not_raise(self, tmp_path):
        _write_raw_notes(tmp_path)
        h = _make_handler(tmp_path)
        h.llm_handler.invoke_model.return_value = "Summary."

        with patch.object(h, "_get_chunk_char_size", return_value=100_000):
            results = list(h.generate_summary_streaming("llama3:latest", party_members=[]))

        assert results[-1][0] is True
