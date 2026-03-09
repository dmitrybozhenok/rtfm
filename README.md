# RTFM For Me

A self-hosted, privacy-first AI documentation assistant. Ingest your documents (PDF, Markdown, plain text), and get accurate, cited answers using local LLMs — no data leaves your infrastructure.

## Features

- **Fully local** — all processing (embedding, search, LLM inference) runs on your machine
- **Grounded answers** — responses cite specific sources; refuses when context is insufficient
- **Hybrid search** — vector similarity + BM25 full-text search, combined via Reciprocal Rank Fusion, with cross-encoder reranking
- **Semantic caching** — similar questions return cached answers instantly (~5-20ms vs ~4-7s)
- **Session memory** — conversation history with automatic summarization and cross-session long-term memory
- **Web UI** — 3-panel layout with source browser, streaming chat, and document viewer
- **Guardrails** — opt-in prompt injection detection, PII redaction, off-topic blocking
- **Observability** — structured JSON logging, OpenTelemetry tracing, Prometheus metrics
- **Evaluation framework** — benchmarks with keyword matching, semantic similarity, LLM-as-judge, faithfulness scoring, and multi-model comparison

## Quick Start

```bash
# Start Redis
docker compose up redis -d

# Pull a local LLM
ollama pull qwen2.5:1.5b

# Install
pip install -e .

# Ingest documents
python -m rtfm ingest ./docs/

# Ask a question
python -m rtfm ask "How do I get started?"

# Or launch the web UI
uvicorn rtfm.api.routes:app
# Open http://localhost:8000
```

## Architecture

```
                    +-----------+
                    |  Web UI   |  (Single-page HTML/JS app)
                    +-----+-----+
                          |
                    +-----v-----+
                    | FastAPI   |  (REST + SSE streaming)
                    +-----+-----+
                          |
              +-----------+-----------+
              |           |           |
        +-----v---+ +----v----+ +----v----+
        | Ingest  | |  RAG    | | Source  |
        | Pipeline| | Pipeline| | Mgmt   |
        +---------+ +----+----+ +---------+
                         |
           +-------------+-------------+
           |             |             |
     +-----v---+  +-----v---+  +------v-----+
     | Hybrid  |  | Ollama  |  | Semantic   |
     | Search  |  | (LLM)   |  | Cache      |
     +---------+  +---------+  +------------+
           |                         |
     +-----v-------------------------v-----+
     |           Redis Stack               |
     +-------------------------------------+
```

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11+ / FastAPI |
| LLM | Ollama (qwen2.5:1.5b default) |
| Embeddings | nomic-ai/nomic-embed-text-v1.5 (768d, local) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Database | Redis Stack (vectors, full-text, cache, sessions) |
| CLI | Typer |
| Frontend | Vanilla HTML/JS/CSS (single file) |
| Deployment | Docker Compose |

## Project Layout

```
src/rtfm/
├── ingest/        # Document loading, chunking, embedding, storage
├── retrieval/     # Hybrid search + RAG orchestration
├── cache/         # Semantic caching via RedisVL
├── memory/        # Session history + long-term memory
├── api/           # FastAPI endpoints + web UI
├── guardrails/    # Input validation, PII redaction, output filtering
├── observability/ # Logging, tracing, Prometheus metrics
└── cli/           # Typer CLI
schemas/           # RedisVL index schemas (YAML)
evals/             # Benchmark framework
tests/             # Unit + E2E (Playwright) tests
```

## CLI Usage

```bash
# Ingest
python -m rtfm ingest <path>           # File or directory
python -m rtfm ingest ./books/ai.pdf   # Single PDF

# Query
python -m rtfm ask "What is RAG?"      # Single question
python -m rtfm chat                    # Interactive chat session
python -m rtfm search "embeddings"     # Search without LLM

# Manage
python -m rtfm list-sources            # List ingested sources
python -m rtfm clear-cache             # Flush semantic cache
```

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Web chat UI |
| POST | `/ingest` | Upload and ingest file |
| POST | `/ingest/path` | Ingest from local path |
| POST | `/ask` | Single question |
| POST | `/chat` | Chat with session history (supports SSE streaming) |
| GET | `/health` | System health check |
| GET | `/sources` | List ingested sources |
| DELETE | `/sources/{source}` | Delete a source |
| GET | `/metrics/prometheus` | Prometheus metrics |

## Docker Deployment

```bash
# All services
docker compose up -d

# With Jaeger tracing
docker compose --profile tracing up -d
```

| Service | Port | Purpose |
|---------|------|---------|
| app | 8000 | RTFM API + Web UI |
| redis | 6379, 8001 | Vector DB + RedisInsight |
| ollama | 11434 | Local LLM inference |
| redis-agent-memory | 9200 | Long-term memory |
| jaeger (optional) | 16686 | Trace visualization |

## Configuration

All settings via environment variables or `.env` file:

| Setting | Default | Description |
|---------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API |
| `LLM_MODEL` | `qwen2.5:1.5b` | LLM model |
| `CHUNK_SIZE` | `500` | Chunk size (tokens) |
| `TOP_K` | `5` | Search results count |
| `BM25_WEIGHT` | `0.5` | BM25 weight in hybrid search |
| `CACHE_TTL` | `3600` | Cache TTL (seconds) |
| `LOG_FORMAT` | `json` | Log format: `json` or `text` |
| `TRACING_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `GUARDRAILS_ENABLED` | `false` | Enable guardrails |
| `AUTH_ENABLED` | `false` | Enable HTTP Basic Auth |

See [PRD.md](PRD.md) for the full configuration reference.

## Evaluation

Built-in benchmark framework for measuring RAG quality:

```bash
# Run benchmark (Pro Git Q&A)
python evals/run_benchmark.py

# With LLM-as-judge scoring
python evals/run_benchmark.py --llm-judge

# Multi-model comparison
python evals/run_benchmark.py --models qwen2.5:1.5b,qwen2.5:3b

# Faithfulness (hallucination detection)
python evals/faithfulness.py

# Negative questions (refusal accuracy)
python evals/negative_eval.py
```

**Scoring dimensions:** keyword matching, semantic similarity, LLM-as-judge (1-5 factual correctness), conciseness, faithfulness (claim-level grounding).

| Model | Pass Rate | Keyword | Semantic Sim | Composite | Avg Latency |
|-------|-----------|---------|-------------|-----------|-------------|
| qwen2.5:1.5b | 100% | 82.2% | 0.906 | 0.873 | 4,468ms |
| qwen2.5:7b | 97% | 80.0% | 0.893 | 0.858 | ~7,500ms |

## Security

- All processing is local — no external API calls
- Opt-in HTTP Basic Auth (`AUTH_ENABLED=true`)
- Prompt injection detection (14 patterns)
- PII redaction in outputs (email, phone, SSN, API keys)
- XSS prevention in web UI
- Non-root Docker container

## License

MIT
