# RTFM For Me — Product Requirements Document

## 1. Product Overview

**RTFM For Me** is a self-hosted, privacy-first AI documentation assistant that ingests documents (PDF, Markdown, text, and web pages), stores them as vector embeddings, and answers natural language questions grounded exclusively in the ingested content. It uses Retrieval-Augmented Generation (RAG) with local LLMs via Ollama, ensuring no data leaves the user's infrastructure.

### 1.1 Vision

Enable teams and individuals to build a private knowledge base from their documentation and get accurate, cited answers in seconds — without sending sensitive content to third-party APIs.

### 1.2 Target Users

- **Development teams** seeking a private documentation Q&A tool
- **Small/medium businesses** (e.g., ~80 employees) wanting chat-based access to HR policies, internal wikis, or compliance docs
- **Individual developers** who want to query technical books and reference material locally
- **Security-conscious organizations** that cannot use cloud-based AI services

### 1.3 Key Value Propositions

- **Fully local**: All processing (embedding, search, LLM inference) runs on-premise
- **Grounded answers**: Responses cite specific sources; the system refuses when context is insufficient
- **Multi-format ingestion**: PDF, Markdown, plain text, and full website crawling
- **Production-ready**: Docker deployment, health monitoring, structured logging, Prometheus metrics
- **Extensible evaluation**: Built-in benchmark framework with multiple scoring dimensions

---

## 2. Architecture

### 2.1 System Architecture

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
     | (Vector Index, Cache, Sessions,     |
     |  Metrics, Full-Text Search)         |
     +-------------------------------------+
```

### 2.2 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | Python 3.11+ / FastAPI | REST API, SSE streaming, middleware |
| **LLM** | Ollama (qwen2.5:7b default) | Answer generation via OpenAI-compatible API |
| **Embeddings** | nomic-ai/nomic-embed-text-v1.5 (768d) | Document and query embedding (local) |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder relevance scoring |
| **Database** | Redis Stack | Vector search, full-text search, caching, sessions |
| **Memory** | Redis Agent Memory Server | Long-term memory with topic/entity extraction |
| **CLI** | Typer | Command-line interface |
| **Frontend** | Vanilla HTML/JS/CSS | Single-file web chat UI |
| **Deployment** | Docker Compose | Multi-service orchestration |
| **Observability** | Custom + OpenTelemetry (optional) | Logging, tracing, Prometheus metrics |

### 2.3 Data Flow

```
Document → Load → Chunk (500 tokens, 150 overlap) → Embed (768d) → Store in Redis
                                                                         |
Question → Guardrails → Cache Check → Hybrid Search → Rerank → LLM → Answer
                              |                                    |
                         (cache hit)                          (cache store)
```

---

## 3. Functional Requirements

### 3.1 Document Ingestion

| ID | Requirement | Status |
|----|-------------|--------|
| ING-01 | Ingest PDF files with markdown-preserving extraction | Done |
| ING-02 | Ingest Markdown (.md) and plain text (.txt) files | Done |
| ING-03 | Ingest from URLs (single page, HTML-to-markdown conversion) | Done |
| ING-04 | Recursive website crawling with rate limiting (configurable delay) | Done |
| ING-05 | Drag-and-drop file upload in web UI | Done |
| ING-06 | Idempotent re-ingestion (deterministic chunk IDs via SHA256) | Done |
| ING-07 | Auto-flush semantic cache on re-ingestion | Done |
| ING-08 | PDF artifact cleanup (page numbers, headers/footers, broken identifiers) | Done |
| ING-09 | HTML noise removal (nav, footer, sidebar, cookie banners, ads) | Done |
| ING-10 | Preserve code blocks, tables, headings, lists during HTML-to-markdown | Done |

**Chunking Strategy:**
- Paragraph-aware splitting at ~500 tokens (~2000 chars)
- 150-token overlap between chunks
- Section heading tracking (markdown `#` headings)
- Section headings prepended to chunk body for embedding context

**Storage Schema (Redis Hash per chunk):**
- `text`: Full chunk content (full-text indexed)
- `source_file`: Filename tag
- `source_url`: URL tag (empty for file sources)
- `source_type`: "file" or "web" tag
- `section`: Section heading tag
- `chunk_index`: Sequential number
- `embedding`: 768-dim HNSW cosine vector

### 3.2 Retrieval & RAG Pipeline

| ID | Requirement | Status |
|----|-------------|--------|
| RAG-01 | Hybrid search: vector similarity + BM25 full-text, combined via Reciprocal Rank Fusion | Done |
| RAG-02 | Cross-encoder reranking of top candidates | Done |
| RAG-03 | Source citation in answers using [Source N] labels | Done |
| RAG-04 | Refusal when context is insufficient ("I don't have enough information...") | Done |
| RAG-05 | Streaming responses via Server-Sent Events (SSE) | Done |
| RAG-06 | Source/section/URL/type filtering on search | Done |
| RAG-07 | Temperature 0 for deterministic outputs | Done |
| RAG-08 | 1024 max token output limit | Done |

**Search Pipeline (4 stages):**
1. **Vector Search**: Top-20 candidates via HNSW cosine similarity
2. **BM25 Full-Text**: Top-20 candidates via Redis FT.SEARCH
3. **Reciprocal Rank Fusion**: Combine vector + BM25 scores (50/50 weight, k=60)
4. **Cross-Encoder Rerank**: Score (query, chunk) pairs, return top-5

**System Prompt Rules:**
1. Answer ONLY from provided context
2. Cite sources with [Source N] labels
3. Use exact terminology from documentation
4. Respect qualifiers (e.g., "without", "only")
5. Answer the specific topic asked, not related ones
6. Keep answers under 3-4 sentences unless necessary
7. Code examples only for "how-to" questions
8. Refuse if context insufficient; no hedging on valid answers

### 3.3 Caching

| ID | Requirement | Status |
|----|-------------|--------|
| CACHE-01 | Semantic cache via RedisVL (cosine distance threshold 0.10) | Done |
| CACHE-02 | Cache stores answer + source attribution | Done |
| CACHE-03 | Configurable TTL (default: 1 hour) | Done |
| CACHE-04 | Manual cache flush via API and CLI | Done |
| CACHE-05 | Auto-flush on document re-ingestion | Done |
| CACHE-06 | Cache metrics (hit rate, latency savings) | Done |

### 3.4 Session & Memory

| ID | Requirement | Status |
|----|-------------|--------|
| MEM-01 | Session conversation history in Redis lists (24h TTL) | Done |
| MEM-02 | Token-budget-aware history truncation (4000 tokens max) | Done |
| MEM-03 | Automatic conversation summarization when history exceeds threshold | Done |
| MEM-04 | Long-term memory via Redis Agent Memory Server | Done |
| MEM-05 | Cross-session context awareness (topics, entities) | Done |
| MEM-06 | Session ID auto-generation, persistence in localStorage | Done |

### 3.5 Web UI

| ID | Requirement | Status |
|----|-------------|--------|
| UI-01 | 3-panel layout: sources sidebar, chat area, source viewer drawer | Done |
| UI-02 | Real-time streaming chat with markdown rendering | Done |
| UI-03 | Source sidebar with chunk counts, sections, type badges | Done |
| UI-04 | Source viewer: browse chunks/sections of any ingested document | Done |
| UI-05 | Per-source delete with confirmation dialog | Done |
| UI-06 | URL ingestion with recursive crawl option | Done |
| UI-07 | File upload (PDF, TXT, MD) with multi-file support | Done |
| UI-08 | Drag-and-drop file ingestion onto sidebar | Done |
| UI-09 | Dark/light theme toggle with localStorage persistence | Done |
| UI-10 | Health status indicator (green/yellow/red dot) | Done |
| UI-11 | Mobile responsive layout with hamburger menu | Done |
| UI-12 | Source-filtered queries (click source to scope chat) | Done |
| UI-13 | Clickable source citations in chat messages | Done |
| UI-14 | XSS protection (HTML escaping, safe attribute encoding) | Done |

### 3.6 CLI

| ID | Requirement | Status |
|----|-------------|--------|
| CLI-01 | `rtfm ingest <path_or_url>` with --recursive and --delay flags | Done |
| CLI-02 | `rtfm ask "<question>"` with --source and --section filters | Done |
| CLI-03 | `rtfm chat` interactive REPL with /quit, /clear, /sources commands | Done |
| CLI-04 | `rtfm search "<query>"` with --top-k and --no-rerank options | Done |
| CLI-05 | `rtfm list-sources` grouped by file/web | Done |
| CLI-06 | `rtfm clear-cache` flush semantic cache | Done |

### 3.7 Guardrails (Opt-in)

| ID | Requirement | Status |
|----|-------------|--------|
| GRD-01 | Prompt injection detection (14 regex patterns) | Done |
| GRD-02 | Query length enforcement (configurable max) | Done |
| GRD-03 | Off-topic detection via vector similarity threshold | Done |
| GRD-04 | PII redaction in outputs (email, phone, SSN, API keys) | Done |
| GRD-05 | Response length truncation | Done |
| GRD-06 | Graceful degradation (guardrail failures don't break RAG) | Done |
| GRD-07 | Opt-in via GUARDRAILS_ENABLED environment variable | Done |

---

## 4. API Specification

### 4.1 Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve web chat UI |
| POST | `/ingest` | Upload and ingest file (multipart) |
| POST | `/ingest/path` | Ingest from local filesystem path |
| POST | `/ingest/url` | Ingest from URL (with recursive option) |
| POST | `/ask` | Single question (no session) |
| POST | `/chat` | Chat with session history + optional SSE streaming |
| GET | `/health` | System health check (Redis, Ollama, Memory Server) |
| GET | `/metrics` | Cache and usage metrics (JSON) |
| GET | `/metrics/prometheus` | Prometheus-compatible metrics export |
| POST | `/cache/flush` | Flush semantic cache |
| POST | `/session/{id}/clear` | Clear session conversation history |
| GET | `/sources` | List all ingested sources with metadata |
| GET | `/sources/{source}/chunks` | Get all chunks for a source |
| DELETE | `/sources/{source}` | Delete a specific source |
| POST | `/documents/clear` | Delete all documents and drop index |

### 4.2 Health Check Response

```json
{
  "status": "healthy | degraded | unhealthy",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "redis": true,
  "ollama": true,
  "memory_server": false,
  "index_exists": true,
  "doc_count": 1619
}
```

**Status Logic:**
- `healthy`: Redis + Ollama both up
- `degraded`: Redis up, Ollama or Memory Server down
- `unhealthy`: Redis down

---

## 5. Non-Functional Requirements

### 5.1 Observability

| ID | Requirement | Status |
|----|-------------|--------|
| OBS-01 | Structured JSON logging (configurable json/text format) | Done |
| OBS-02 | Per-request logging: question, latency, cache hit, chunks, tokens, model | Done |
| OBS-03 | HTTP request middleware logging (method, path, status, latency) | Done |
| OBS-04 | OpenTelemetry tracing with span wrapping (optional, no-op fallback) | Done |
| OBS-05 | Trace spans: cache.check, search, llm.call, guardrails, cache.store | Done |
| OBS-06 | Prometheus counters: queries, cache hits/misses, guardrail blocks, errors, ingestion | Done |
| OBS-07 | Prometheus histograms: query/search/LLM latency, tokens per query, chunks retrieved | Done |
| OBS-08 | Prometheus gauges: active sessions, index size | Done |
| OBS-09 | `/metrics/prometheus` endpoint in exposition format | Done |
| OBS-10 | Optional Jaeger integration for trace visualization | Done |

### 5.2 Deployment

| ID | Requirement | Status |
|----|-------------|--------|
| DEP-01 | Multi-stage Docker build (builder, models, runtime) | Done |
| DEP-02 | Pre-cached embedding/reranker models in Docker image | Done |
| DEP-03 | Non-root container user | Done |
| DEP-04 | Docker Compose with 5 services (app, redis, ollama, memory, jaeger) | Done |
| DEP-05 | Persistent Redis volume | Done |
| DEP-06 | Health check-based service dependency ordering | Done |
| DEP-07 | Environment variable configuration via .env | Done |
| DEP-08 | 4GB memory limit for app container | Done |

### 5.3 Performance Targets

| Metric | Target | Measured |
|--------|--------|----------|
| Query latency (uncached) | < 10s | ~4.5-7.5s (model-dependent) |
| Query latency (cached) | < 100ms | ~5-20ms |
| Search latency | < 500ms | ~50-100ms |
| Ingestion throughput | > 10 pages/min | ~15-20 pages/min |
| Concurrent users | 10+ | Supported (single worker) |

### 5.4 Security

- All processing is local (no external API calls)
- Optional PII redaction in outputs
- Prompt injection detection (14 patterns)
- HTML escaping in web UI (XSS prevention)
- Non-root Docker container
- No credential storage in code

---

## 6. Evaluation Framework

### 6.1 Benchmark Suite

**Core Benchmark** (`evals/run_benchmark.py`):
- 30 Pro Git Q&A pairs across 8 categories
- Keyword matching (exact + stemmed + sliding window)
- Semantic similarity (embedding cosine)
- Conciseness ratio
- Refusal detection (5 regex patterns with hedging logic)
- Per-category breakdown
- Worst-N question reporting
- Regression tracking via history.jsonl

**Composite Score Formula:**
- Without judge: `0.40 * keyword + 0.60 * semantic`
- With judge: `0.40 * judge/5 + 0.20 * keyword + 0.20 * semantic + 0.20 * conciseness`

### 6.2 LLM-as-Judge (`evals/llm_judge.py`)

- Factual correctness rating (1-5 scale)
- File-based deterministic caching
- Catches wrong answers that semantic similarity misses
- Configurable judge model

### 6.3 Faithfulness Eval (`evals/faithfulness.py`)

- Hallucination detection via claim-level grounding
- Decomposes answers into individual claims
- Checks each claim against retrieved context chunks
- Reports: faithfulness score, hallucinated claims list, worst offenders

### 6.4 Negative Questions Eval (`evals/negative_eval.py`)

- 15 questions: 11 unanswerable + 4 answerable controls
- Categories: completely off-topic, plausible but absent, misleading premise, answerable control
- Metrics: accuracy, false answer rate (hallucination), false refusal rate (over-caution)

### 6.5 Multi-Model Comparison

- Side-by-side benchmark across multiple Ollama models
- N-way comparison matrix
- Per-question winner tracking
- Category-level model strengths

### 6.6 Current Benchmark Results

| Model | Pass Rate | Keyword | Semantic Sim | Composite | Avg Latency |
|-------|-----------|---------|-------------|-----------|-------------|
| qwen2.5:3b | 100% | 82.2% | 0.906 | 0.873 | 4,468ms |
| qwen2.5:7b | 97% | 80.0% | 0.893 | 0.858 | ~7,500ms |

---

## 7. Test Coverage

### 7.1 End-to-End Tests (Playwright)

27 browser tests across 6 test files:
- **Chat flow** (8): welcome, header, sidebar, streaming response, empty input, keyboard shortcuts, button states
- **Ingestion** (3): UI elements, empty URL handling, recursive checkbox
- **Sources** (5): sidebar listing, viewer open/close, clear button, action buttons
- **Source actions** (1): delete button visibility
- **Markdown rendering** (4): inline code, code blocks, bold, italic
- **Theme & state** (4): toggle, persistence, health indicator, session ID

### 7.2 Unit & Integration Tests

~25 test files covering:
- RAG pipeline logic and error handling
- Vector/BM25 search
- Document chunking and loading
- Web scraping and ingestion
- Session management and long-term memory
- Guardrails (17 injection patterns, 5 PII types)
- Cache operations
- API endpoints and SSE streaming
- CLI interface

---

## 8. Configuration Reference

All settings configurable via environment variables or `.env` file:

| Setting | Default | Description |
|---------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `MEMORY_SERVER_URL` | `http://localhost:9200` | Agent Memory Server URL |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API endpoint |
| `LLM_MODEL` | `qwen2.5:7b` | LLM model for answer generation |
| `EMBEDDING_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | Embedding model |
| `EMBEDDING_DIMS` | `768` | Embedding vector dimensions |
| `CHUNK_SIZE` | `500` | Chunk size in tokens |
| `CHUNK_OVERLAP` | `150` | Chunk overlap in tokens |
| `TOP_K` | `5` | Number of search results |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `BM25_WEIGHT` | `0.5` | BM25 weight in hybrid search (0-1) |
| `CACHE_DISTANCE_THRESHOLD` | `0.10` | Semantic cache similarity threshold |
| `CACHE_TTL` | `3600` | Cache TTL in seconds |
| `SESSION_TTL` | `86400` | Session TTL in seconds (24h) |
| `LOG_FORMAT` | `json` | Log format: "json" or "text" |
| `TRACING_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `GUARDRAILS_ENABLED` | `false` | Enable input/output guardrails |
| `MAX_QUERY_LENGTH` | `1000` | Max query length (guardrails) |
| `PII_REDACTION` | `true` | Enable PII redaction (guardrails) |
| `OFFTOPIC_THRESHOLD` | `0.15` | Off-topic detection threshold |

---

## 9. Deployment

### 9.1 Quick Start

```bash
docker compose up redis -d
ollama pull qwen2.5:7b
pip install -e .
python -m rtfm ingest ./docs/
python -m rtfm ask "How do I get started?"
```

### 9.2 Full Docker Deployment

```bash
docker compose up -d              # All services
docker compose --profile tracing up -d  # With Jaeger tracing
```

### 9.3 Services

| Service | Port | Purpose |
|---------|------|---------|
| app | 8000 | RTFM API + Web UI |
| redis | 6379, 8001 | Vector DB + RedisInsight |
| ollama | 11434 | Local LLM inference |
| redis-agent-memory | 9200 | Long-term memory |
| jaeger (optional) | 16686, 4317 | Trace visualization |

---

## 10. Known Limitations & Future Work

### 10.1 Current Limitations

- **Single-worker deployment**: Not horizontally scaled (single uvicorn worker)
- **Source-level summarization removed**: Chunk sampling for large documents (900+ chunks) produces low-quality summaries; removed pending better approach (e.g., map-reduce summarization)
- **FAQ generation removed**: Same sampling issue; produces shallow questions from TOC/preface rather than substantive content
- **No authentication**: API is open; intended for private network deployment
- **No rate limiting**: Relies on network-level access control
- **Memory server optional**: Long-term memory degrades gracefully but loses cross-session context

### 10.2 Potential Future Enhancements

- **Map-reduce summarization**: Summarize chunks in batches, then summarize summaries for full-document coverage
- **Authentication & RBAC**: API key or SSO integration for multi-tenant deployment
- **Horizontal scaling**: Multiple workers with shared Redis state
- **GPU acceleration**: Ollama GPU passthrough for faster inference
- **Retrieval quality eval**: Score search results independently from LLM answer quality
- **Multi-hop reasoning eval**: Questions requiring synthesis across multiple sources
- **Paraphrase invariance eval**: Same question asked multiple ways should produce consistent answers
- **Citation accuracy eval**: Verify [Source N] references match relevant chunks
- **Notebook/workspace support**: Group sources into projects with separate chat histories
- **ONNX embedding backend**: Reduce Docker image size (~2GB to ~200MB)
