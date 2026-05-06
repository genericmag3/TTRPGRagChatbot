"""Unit tests for LLMHandler — ollama and OllamaLLM are mocked in conftest."""
import pytest
from unittest.mock import MagicMock, patch

from src.utils.LLMHandler import LLMHandler


def _make_model(name: str) -> MagicMock:
    m = MagicMock()
    m.model = name
    m.__getitem__ = lambda self, key: name if key == "model" else None
    return m


class TestGetAvailableModels:
    def test_returns_list_from_ollama(self):
        handler = LLMHandler()
        models = handler.get_available_models()
        # conftest sets one model: "llama3:latest"
        assert len(models) == 1
        assert models[0].model == "llama3:latest"


class TestLoadModel:
    def test_raises_value_error_for_unknown_model(self):
        handler = LLMHandler()
        with pytest.raises(ValueError, match="not found"):
            handler.load_model("nonexistent:model", 0.7)

    def test_loads_known_model_successfully(self):
        handler = LLMHandler()
        # "llama3:latest" is the mocked model from conftest
        handler.load_model("llama3:latest", 0.5)
        assert handler.currnet_model is not None

    def test_current_model_is_none_before_load(self):
        handler = LLMHandler()
        assert handler.currnet_model is None

    def test_loading_sets_current_model_to_non_none(self):
        handler = LLMHandler()
        assert handler.currnet_model is None
        handler.load_model("llama3:latest", 0.7)
        assert handler.currnet_model is not None


class TestInvokeModel:
    def test_raises_when_no_model_loaded(self):
        handler = LLMHandler()
        mock_prompt = MagicMock()
        with pytest.raises(ValueError, match="No model loaded"):
            handler.invoke_model(mock_prompt, {})

    def test_invokes_chain_with_mappings(self):
        handler = LLMHandler()
        handler.load_model("llama3:latest", 0.7)

        mock_prompt = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "The dragon appeared at dawn."

        # Chain is built as: prompt | model | StrOutputParser()
        # Patch __or__ (__ror__) so prompt | model | parser → mock_chain
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        result = handler.invoke_model(mock_prompt, {"question": "Where is the dragon?"})
        assert result == "The dragon appeared at dawn."
        mock_chain.invoke.assert_called_once_with({"question": "Where is the dragon?"})
