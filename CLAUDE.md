# RTFM For Me

## Quick Start
```bash
docker compose up redis -d
ollama pull qwen2.5:1.5b
pip install -e .
python -m rtfm ingest ./docs/
python -m rtfm ask "How do I install RTFM?"
```

## Architecture
- **Stack:** Python + FastAPI, Ollama (local LLM), Redis Stack, Typer CLI
- **LLM:** Ollama with `qwen2.5:1.5b` via OpenAI-compatible API (localhost:11434)
- **RAG pipeline:** `rag.py` is the central orchestrator tying together search, cache, session, memory, and LLM
- **Embedding model:** `all-MiniLM-L6-v2` (384 dims, local) — used for ingestion, search, and caching
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (local cross-encoder for hybrid search)
- **Redis handles:** vector search, semantic caching, session storage, long-term memory

## Project Layout
- `src/rtfm/` — main package
- `src/rtfm/ingest/` — document loading, chunking, embedding, storage (files + web)
- `src/rtfm/retrieval/` — hybrid search (vector + BM25 + reranker) and RAG orchestration
- `src/rtfm/cache/` — semantic caching via RedisVL
- `src/rtfm/memory/` — session history and long-term memory
- `src/rtfm/api/` — FastAPI endpoints
- `src/rtfm/guardrails/` — input validation, PII redaction, output filtering
- `src/rtfm/observability/` — structured logging, OpenTelemetry tracing, Prometheus metrics
- `src/rtfm/cli/` — Typer CLI
- `schemas/` — RedisVL index schemas (YAML)
- `evals/` — benchmark framework (Pro Git Q&A, PDF vs Web comparison, LLM judge, multi-model)

## Commands
- `python -m rtfm ingest <path>` — ingest documents from file/directory
- `python -m rtfm ingest <url>` — ingest from URL (add `--recursive` for full site crawl)
- `python -m rtfm ask "<question>"` — single question
- `python -m rtfm chat` — interactive chat session
- `python -m rtfm clear-cache` — flush semantic cache
- `uvicorn rtfm.api.routes:app` — start API server

## Evals
- `python evals/run_benchmark.py` — run benchmark (default: Pro Git)
- `python evals/run_benchmark.py --source progit.pdf` — filter by source file
- `python evals/run_benchmark.py --llm-judge` — enable LLM-as-judge scoring
- `python evals/run_benchmark.py --llm-judge --judge-model qwen2.5:1.5b` — specify judge model
- `python evals/run_benchmark.py --models qwen2.5:1.5b,qwen2.5:3b` — multi-model comparison
- `python evals/compare_models.py evals/model_comparison.json` — analyze saved comparison results
- `pytest tests/e2e/ -m e2e` — run end-to-end browser tests (requires running server)
- `pytest tests/e2e/ -m e2e --headed` — run with visible browser

## Key Design Decisions
- Same embedding model must be used everywhere (configured in `config.py`)
- Chunk IDs are deterministic hashes for idempotent re-ingestion
- Cache is flushed on re-ingestion to prevent stale answers
- Graceful degradation: cache/memory failures don't break core RAG
- Token budget: truncate oldest session messages first, then reduce top_k
- Web ingestion: HTML→markdown conversion preserving headings, code, tables; rate-limited crawling
- Guardrails: opt-in (`GUARDRAILS_ENABLED=true`), prompt injection detection, PII redaction, off-topic blocking
- Observability: structured JSON logging (`LOG_FORMAT=json|text`), OpenTelemetry tracing (`TRACING_ENABLED=true`), Prometheus metrics at `/metrics/prometheus`
