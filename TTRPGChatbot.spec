import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)
# TTRPGChatbot.spec  —  PyInstaller build configuration
#
# Build with:
#   python3 -m PyInstaller TTRPGChatbot.spec
#
# Output: dist/TTRPGChatbot/TTRPGChatbot.exe  (plus supporting files)

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# ── Streamlit static assets (JS, CSS, icons, etc.) ────────────────────────
st_datas, st_binaries, st_hiddenimports = collect_all("streamlit")

# ── Altair — Streamlit's chart rendering dependency ───────────────────────
alt_datas, alt_binaries, alt_hiddenimports = collect_all("altair")

# ── pyarrow — required by Streamlit's data serialisation ──────────────────
# Filter out arrow_flight (gRPC network transport) — not needed locally.
arrow_datas, arrow_binaries_all, arrow_hiddenimports = collect_all("pyarrow")
arrow_binaries = [
    (src, dst) for src, dst in arrow_binaries_all
    if "flight" not in src.lower()
]

# ── fastembed — ONNX-based embeddings, no PyTorch required ────────────────
fe_datas, fe_binaries, fe_hiddenimports = collect_all("fastembed")

# ── langchain_community — only collect data files; importing the full
#    package via collect_all pulls in every optional integration (incl. torch).
#    FastEmbedEmbeddings is declared as a hidden import instead.
lcc_datas = collect_data_files("langchain_community")

# ── chromadb — uses dynamic imports for telemetry and segment backends ────
chroma_datas, chroma_binaries, chroma_hiddenimports = collect_all("chromadb")

# ── Additional hidden imports that PyInstaller's static analyser misses ───
extra_hiddenimports = [
    # ---- project source ----
    "src",
    "src.app",
    "src.app.TTRPGChatBot",
    "src.app.NoteEditor",
    "src.utils",
    "src.utils.DatabaseHandler",
    "src.utils.LLMHandler",
    "src.utils.TextEditorHandler",
    # ---- langchain stack ----
    "langchain_core",
    "langchain_core.prompts",
    "langchain_core.prompts.chat",
    "langchain_ollama",
    "langchain_chroma",
    "langchain_experimental",
    "langchain_experimental.text_splitter",
    "langchain.schema.output_parser",
    "langchain.docstore.document",
    # ---- langchain_community (only the embedding we actually use) ----
    "langchain_community.embeddings",
    "langchain_community.embeddings.fastembed",
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
    "fastembed",
    # ---- document parsing (CSV, DOCX, TXT only — no PDF) ----
    "docx",
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
    pathex=["."],
    binaries=[
        *st_binaries,
        *alt_binaries,
        *arrow_binaries,        # flight-filtered
        *fe_binaries,
        *chroma_binaries,
        # lcc_binaries omitted — langchain_community has no meaningful binaries
    ],
    datas=[
        *st_datas,
        *alt_datas,
        *arrow_datas,
        *fe_datas,
        *lcc_datas,
        *chroma_datas,
        ("streamlit_app.py", "."),
        ("pages",             "pages"),
        ("src",               "src"),
        ("assets",            "assets"),
        *collect_data_files("streamlit_lottie"),
    ],
    hiddenimports=[
        *st_hiddenimports,
        *alt_hiddenimports,
        *arrow_hiddenimports,
        *fe_hiddenimports,
        *chroma_hiddenimports,
        *extra_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Hard-block packages that are no longer used but may be
        # discovered transitively by PyInstaller's static analysis.
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "sentence_transformers",
        "langchain_huggingface",
        # PDF libraries — app only handles CSV, DOCX, TXT
        "pdfplumber",
        "pdfminer",
        "pypdfium2",
        # Test frameworks
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
    name="TTRPGChatbot",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon="assets/icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TTRPGChatbot",
)
