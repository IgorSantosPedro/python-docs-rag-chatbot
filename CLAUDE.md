# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies
.venv\Scripts\pip install sentence-transformers numpy groq fastapi uvicorn pydantic httpx beautifulsoup4 python-dotenv

# Start the chatbot (recommended — waits for server and opens browser automatically)
.venv\Scripts\python launch.py

# Start the server directly
.venv\Scripts\uvicorn llm:app --reload

# Start the RAG API demo (fastrag.py) separately
.venv\Scripts\uvicorn fastrag:app --reload
```

## Architecture

This project contains two independent RAG (Retrieval-Augmented Generation) implementations that share the same concept but differ in scope and interface.

### `llm.py` — Full chatbot server (primary file)

The main application. Serves both the API and the frontend from a single process.

**Startup flow (lifespan):**
1. Loads `SentenceTransformer("all-MiniLM-L6-v2")` for embeddings
2. Checks for `.cache/chunks.json` + `.cache/embeddings.npy`
   - **Cache hit:** loads from disk in seconds
   - **Cache miss:** runs `crawl_all_docs()` — BFS crawler starting at `https://docs.python.org/pt-br/3/` that discovers and chunks all pages in a single pass, then saves cache to `.cache/`
3. First crawl takes 10–20 minutes; subsequent starts are instant

**Request flow (`POST /chat`):**
1. `retrieve()` — encodes question, dot-product against `doc_embeddings`, returns top 5 chunks
2. Injects chunks as context into the user message
3. Appends to in-memory `history` list (persists for the session)
4. Calls Groq API (`llama-3.3-70b-versatile`) with full history
5. Returns `{ answer, sources }`

**Other endpoints:**
- `GET /` — serves `index.html`
- `POST /reset` — clears conversation history

### `fastrag.py` — Minimal RAG API demo

Simpler version using 4 hardcoded `DOC_URLS` instead of a full crawl. No cache, no frontend serving, no conversation history. Useful as a reference for the RAG pattern without the crawler complexity.

### `index.html` — Chat frontend

Single-file vanilla JS/HTML. Calls `http://localhost:8000/chat` and `/reset`. No build step.

### `launch.py` — Process launcher

Starts uvicorn as a subprocess, polls `http://localhost:8000` until the server responds, then opens the browser. Called by `start.bat`.

## Key files

| File | Purpose |
|---|---|
| `.env` | `GROQ_API_KEY=...` — required for generation |
| `.cache/` | Auto-generated; delete to force re-crawl |
| `start.bat` | Double-click launcher on Windows |

## Cache invalidation

To force a full re-crawl (e.g., after the Python docs update):
```bash
rm -rf .cache/
```

## LLM / embedding config

- **Embeddings:** `all-MiniLM-L6-v2` (384 dims, normalized, cosine via dot product)
- **Generation:** Groq `llama-3.3-70b-versatile`, `max_tokens=512`
- **Retrieval:** top-k=5 chunks per query
- **Chunking:** `<p>` and `<li>` tags, min 80 chars, prefixed with `[Page Title]`
