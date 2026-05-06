import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)
# LocalAIAgent.spec  —  PyInstaller build configuration
#
# Build with:
#   python3 -m PyInstaller LocalAIAgent.spec
#
# Output: dist/LocalAIAgent/LocalAIAgent.exe  (plus supporting files)
#
# NOTE: The first build will be slow because PyInstaller analyses every
# import. Subsequent builds reuse the cache and are faster.
#
# SIZE WARNING: This app bundles PyTorch (torch) which is several GB.
# The final dist/ folder will be large. This is expected.

from PyInstaller.utils.hooks import collect_all, collect_data_files

# ── Streamlit static assets (JS, CSS, icons, etc.) ────────────────────────
st_datas, st_binaries, st_hiddenimports = collect_all("streamlit")

# ── Altair — Streamlit's chart rendering dependency ───────────────────────
alt_datas, alt_binaries, alt_hiddenimports = collect_all("altair")

# ── pyarrow — required by Streamlit's data serialisation ──────────────────
arrow_datas, arrow_binaries, arrow_hiddenimports = collect_all("pyarrow")

# ── sentence_transformers — used by HuggingFace embeddings ────────────────
st_datas2, st_binaries2, st_hiddenimports2 = collect_all("sentence_transformers")

# ── transformers — sentence_transformers dependency ───────────────────────
tf_datas, tf_binaries, tf_hiddenimports = collect_all("transformers")

# ── torch — PyTorch; uses complex DLL loading that PyInstaller misses ─────
torch_datas, torch_binaries, torch_hiddenimports = collect_all("torch")

# ── tokenizers — HuggingFace fast tokenizer (Rust extension) ──────────────
tok_datas, tok_binaries, tok_hiddenimports = collect_all("tokenizers")

# ── huggingface_hub — model/cache management ──────────────────────────────
hfhub_datas, hfhub_binaries, hfhub_hiddenimports = collect_all("huggingface_hub")

# ── langchain_huggingface — wraps sentence_transformers for LangChain ─────
lchf_datas, lchf_binaries, lchf_hiddenimports = collect_all("langchain_huggingface")

# ── chromadb — uses dynamic imports for telemetry and segment backends ────
chroma_datas, chroma_binaries, chroma_hiddenimports = collect_all("chromadb")

# ── Additional hidden imports that PyInstaller's static analyser misses ───
extra_hiddenimports = [
    # ---- project source ----
    "src",
    "src.app",
    "src.app.TTRPGChatBot",
    "src.utils",
    "src.utils.DatabaseHandler",
    "src.utils.LLMHandler",
    # ---- langchain stack ----
    "langchain_core",
    "langchain_core.prompts",
    "langchain_core.prompts.chat",
    "langchain_ollama",
    "langchain_chroma",
    "langchain_huggingface",
    "langchain_experimental",
    "langchain_experimental.text_splitter",
    "langchain.schema.output_parser",
    "langchain.docstore.document",
    # ---- vector store ----
    "chromadb",
    "chromadb.db.impl",
    "chromadb.db.impl.sqlite",
    "chromadb.api.segment",
    "chromadb.telemetry.product.posthog",
    "chromadb.telemetry.product",
    "chromadb.segment.impl.manager.local",
    "chromadb.segment.impl.metadata.sqlite",
    "chromadb.segment.impl.vector.local_persistent_hnsw",
    "chromadb.segment.impl.vector.local_hnsw",
    # ---- embeddings ----
    "sentence_transformers",
    "huggingface_hub",
    "tokenizers",
    # ---- document parsing ----
    "docx",
    "pdfplumber",
    "pdfminer",
    "pdfminer.high_level",
    # ---- data ----
    "pandas",
    "numpy",
    # ---- Streamlit extras ----
    "streamlit_lottie",
    # ---- LLM client ----
    "ollama",
    # ---- stdlib ----
    "uuid",
    "json",
    "threading",
    "webbrowser",
]

a = Analysis(
    ["launcher.py"],
    pathex=["."],           # project root on the module search path
    binaries=[
        *st_binaries,
        *alt_binaries,
        *arrow_binaries,
        *st_binaries2,
        *tf_binaries,
        *torch_binaries,
        *tok_binaries,
        *hfhub_binaries,
        *lchf_binaries,
        *chroma_binaries,
    ],
    datas=[
        # ── Streamlit / Altair / Arrow bundled assets ──
        *st_datas,
        *alt_datas,
        *arrow_datas,
        # ── sentence_transformers + transformers + torch ──
        *st_datas2,
        *tf_datas,
        *torch_datas,
        *tok_datas,
        *hfhub_datas,
        *lchf_datas,
        *chroma_datas,
        # ── App source files ──────────────────────────
        # These are placed in the root of dist/LocalAIAgent/ so that
        # relative imports and file paths work out of the box.
        ("streamlit_app.py", "."),
        ("src",              "src"),
        ("assets",           "assets"),
        # ── Streamlit lottie package data ─────────────
        *collect_data_files("streamlit_lottie"),
    ],
    hiddenimports=[
        *st_hiddenimports,
        *alt_hiddenimports,
        *arrow_hiddenimports,
        *st_hiddenimports2,
        *tf_hiddenimports,
        *torch_hiddenimports,
        *tok_hiddenimports,
        *hfhub_hiddenimports,
        *lchf_hiddenimports,
        *chroma_hiddenimports,
        *extra_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "_pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LocalAIAgent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,               # compress binaries (requires UPX to be installed)
    # console=True  → keep the terminal open so startup errors are visible.
    # Change to False once the app is stable for a cleaner end-user experience.
    console=True,
    icon=None,              # set to e.g. "assets/icon.ico" if you add one
)

# --onedir layout: all files land in dist/LocalAIAgent/
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="LocalAIAgent",
)
