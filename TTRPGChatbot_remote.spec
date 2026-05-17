import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)
# TTRPGChatbot_remote.spec  —  PyInstaller build for the REMOTE (multi-user) host
#
# Build with:
#   python3 -m PyInstaller TTRPGChatbot_remote.spec
#
# Output: dist/TTRPGChatbot_remote/TTRPGChatbot_remote.exe (plus supporting files)
#
# This mirrors TTRPGChatbot.spec but bundles launcher_remote.py and the
# authentication stack (bcrypt + cryptography + sqlite3).

from PyInstaller.utils.hooks import collect_all, collect_data_files

st_datas, st_binaries, st_hiddenimports = collect_all("streamlit")
alt_datas, alt_binaries, alt_hiddenimports = collect_all("altair")

arrow_datas, arrow_binaries_all, arrow_hiddenimports = collect_all("pyarrow")
arrow_binaries = [
    (src, dst) for src, dst in arrow_binaries_all
    if "flight" not in src.lower()
]

fe_datas, fe_binaries, fe_hiddenimports = collect_all("fastembed")
lcc_datas = collect_data_files("langchain_community")
chroma_datas, chroma_binaries, chroma_hiddenimports = collect_all("chromadb")
crypto_datas, crypto_binaries, crypto_hiddenimports = collect_all("cryptography")

extra_hiddenimports = [
    # ---- project source ----
    "src",
    "src.app_config",
    "src.app",
    "src.app.TTRPGChatBot",
    "src.app.CampaignSummarizer",
    "src.app.NoteEditor",
    "src.utils",
    "src.utils.DatabaseHandler",
    "src.utils.LLMHandler",
    "src.utils.SummaryHandler",
    "src.utils.TextEditorHandler",
    "src.utils.paths",
    # ---- auth stack ----
    "src.auth",
    "src.auth.store",
    "src.auth.gate",
    "src.auth.certs",
    "src.auth.errors",
    "bcrypt",
    "cryptography",
    "sqlite3",
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
    # ---- document parsing ----
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
    "secrets",
    "hashlib",
    "threading",
    "webbrowser",
]

a = Analysis(
    ["launcher_remote.py"],
    pathex=["."],
    binaries=[
        *st_binaries,
        *alt_binaries,
        *arrow_binaries,
        *fe_binaries,
        *chroma_binaries,
        *crypto_binaries,
    ],
    datas=[
        *st_datas,
        *alt_datas,
        *arrow_datas,
        *fe_datas,
        *lcc_datas,
        *chroma_datas,
        *crypto_datas,
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
        *crypto_hiddenimports,
        *extra_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "sentence_transformers",
        "langchain_huggingface",
        "pdfplumber",
        "pdfminer",
        "pypdfium2",
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
    name="TTRPGChatbot_remote",
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
    name="TTRPGChatbot_remote",
)
