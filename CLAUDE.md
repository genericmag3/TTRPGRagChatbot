# CLAUDE.md

## Project Overview

**DandDRagChatbot** is a campaign Q&A chatbot for tabletop RPG players. It uses RAG (Retrieval-Augmented Generation) to answer questions about a D&D campaign based on uploaded campaign notes. The intended users are players (and DMs) who want to query campaign information conversationally.

## Tech Stack

- **Language:** Python (entirely)
- **UI:** Streamlit (`streamlit_app.py` in project root)
- **RAG pipeline:** LangChain with FastEmbed embeddings (`BAAI/bge-base-en-v1.5`)
- **LLM:** Local models via Ollama
- **Testing:** pytest (run via `run_tests.bat`)

## Project Structure

```
streamlit_app.py          # App entrypoint
pages/
  1_Campaign_Summary.py   # Campaign summary Streamlit page
src/
  app/
    TTRPGRagChatbot       # Top-level chatbot class
  utils/
    DatabaseHandler       # Vector DB / document storage
    LLMHandler            # Ollama LLM interface
    SummaryHandler        # Hierarchical map-reduce campaign summarizer
data/                     # Generated at runtime by the app — do not edit directly
assets/                   # Static assets — do not modify
requirements.txt
run_tests.bat
```

## Setup & Running

**Install dependencies:**
```
pip install -r requirements.txt
```
Use the project-local virtual environment (`.venv`).

**Run the app:**
```
python -m streamlit run streamlit_app.py
```

**Run tests:**
```
run_tests.bat
```

## Feature Implementation Workflow

When implementing a new feature, follow this sequence:

1. **Branch** — check out a new branch from `main` with a descriptive name for the feature.
2. **Tests first** — write unit tests for the feature before implementing it.
3. **Implement** — build the feature.
4. **Smoke test** — run the app and verify there are no obvious runtime exceptions.
5. **Run tests** — execute `run_tests.bat`.
6. **If tests fail** — first check whether the *implementation* is wrong before modifying the tests. Only update tests if the implementation is correct and the test expectation is the problem.
7. **Wait for review** — once the app runs cleanly and tests pass, stop. Do not push or open PRs; the user handles that.

## Coding Conventions

- Match the existing style of the codebase as closely as possible (naming, formatting, structure).
- Do not add comments that describe *what* the code does — only add them when the *why* is non-obvious.

## Off-Limits

- **`assets/`** — do not modify anything in this directory.
- **`data/`** — do not edit files here directly. This folder is managed by the running app; manipulating it via the app is fine.
