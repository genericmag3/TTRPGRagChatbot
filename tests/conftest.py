"""
Patch heavy/external dependencies before any project modules are imported.
AppTest runs in the same process, so sys.modules patches apply to the app too.
"""
import sys
import os
from unittest.mock import MagicMock

# Ensure project root is on path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- ollama ---
mock_ollama = MagicMock()
mock_model = MagicMock()
mock_model.model = "llama3:latest"
# LLMHandler uses both item['model'] and item.model access patterns
mock_model.__getitem__ = MagicMock(side_effect=lambda key: "llama3:latest" if key == "model" else None)
mock_ollama.list.return_value.models = [mock_model]
sys.modules["ollama"] = mock_ollama

# --- langchain_ollama ---
mock_langchain_ollama = MagicMock()
sys.modules["langchain_ollama"] = mock_langchain_ollama

# --- HuggingFace embeddings ---
mock_hf = MagicMock()
sys.modules["langchain_huggingface"] = mock_hf

# --- Chroma vector store ---
mock_chroma_mod = MagicMock()
sys.modules["langchain_chroma"] = mock_chroma_mod

# --- SemanticChunker ---
mock_exp = MagicMock()
mock_exp_splitter = MagicMock()
sys.modules["langchain_experimental"] = mock_exp
sys.modules["langchain_experimental.text_splitter"] = mock_exp_splitter

# --- streamlit_lottie ---
sys.modules["streamlit_lottie"] = MagicMock()

# --- torch / transformers (pulled in transitively) ---
for heavy in ("torch", "torchvision", "transformers", "sentence_transformers"):
    if heavy not in sys.modules:
        sys.modules[heavy] = MagicMock()
