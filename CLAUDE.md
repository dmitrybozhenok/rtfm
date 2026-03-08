# RTFM For Me

## Quick Start
```bash
docker compose up redis -d
ollama pull qwen2.5:14b
pip install -e .
python -m rtfm ingest ./docs/
python -m rtfm ask "How do I install RTFM?"
```

## Architecture
- **Stack:** Python + FastAPI, Ollama (local LLM), Redis Stack, Typer CLI
- **LLM:** Ollama with `qwen2.5:14b` via OpenAI-compatible API (localhost:11434)
- **RAG pipeline:** `rag.py` is the central orchestrator tying together search, cache, session, memory, and LLM
- **Embedding model:** `all-MiniLM-L6-v2` (384 dims, local) — used for ingestion, search, and caching
- **Redis handles:** vector search, semantic caching, session storage, long-term memory

## Project Layout
- `src/rtfm/` — main package
- `src/rtfm/ingest/` — document loading, chunking, embedding, storage
- `src/rtfm/retrieval/` — vector search and RAG orchestration
- `src/rtfm/cache/` — semantic caching via RedisVL
- `src/rtfm/memory/` — session history and long-term memory
- `src/rtfm/api/` — FastAPI endpoints
- `src/rtfm/cli/` — Typer CLI
- `schemas/` — RedisVL index schemas (YAML)

## Commands
- `python -m rtfm ingest <path>` — ingest documents
- `python -m rtfm ask "<question>"` — single question
- `python -m rtfm chat` — interactive chat session
- `python -m rtfm clear-cache` — flush semantic cache
- `uvicorn rtfm.api.routes:app` — start API server

## Key Design Decisions
- Same embedding model must be used everywhere (configured in `config.py`)
- Chunk IDs are deterministic hashes for idempotent re-ingestion
- Cache is flushed on re-ingestion to prevent stale answers
- Graceful degradation: cache/memory failures don't break core RAG
- Token budget: truncate oldest session messages first, then reduce top_k
