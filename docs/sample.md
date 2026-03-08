# Getting Started with RTFM

RTFM (Read The Fine Manual) is an AI-powered documentation assistant that helps you find answers in your documentation quickly and accurately.

## Installation

To install RTFM, you need Python 3.11+ and Docker:

1. Clone the repository
2. Copy `.env.example` to `.env` and add your Anthropic API key
3. Run `docker compose up redis` to start Redis Stack
4. Install the package: `pip install -e .`

## Ingesting Documents

To ingest documentation, use the CLI:

```bash
rtfm ingest ./docs/
```

This will:
- Discover all `.md`, `.txt`, and `.pdf` files
- Split them into semantic chunks
- Generate embeddings using a local model
- Store everything in Redis for fast vector search

## Asking Questions

Once your docs are ingested, you can ask questions:

```bash
rtfm ask "How do I install RTFM?"
```

The system uses RAG (Retrieval Augmented Generation) to:
1. Find the most relevant document chunks via vector similarity
2. Pass them as context to Claude
3. Generate a grounded answer with source citations

## Interactive Chat

For multi-turn conversations:

```bash
rtfm chat
```

This starts an interactive session where you can ask follow-up questions. The system remembers conversation context within a session.

### Chat Commands

- `/quit` — Exit the chat
- `/clear` — Clear conversation history
- `/sources` — Show sources from the last answer

## Architecture

RTFM uses Redis Stack as its backbone:
- **Vector Search** — Find relevant document chunks by semantic similarity
- **Semantic Cache** — Avoid redundant LLM calls for similar questions
- **Session Storage** — Maintain conversation context with TTL
- **Long-term Memory** — Remember user preferences across sessions

The embedding model (`all-MiniLM-L6-v2`) runs locally, so there are no API costs for embeddings.

## Configuration

Key settings in `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 500 | Target tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Token overlap between chunks |
| `TOP_K` | 5 | Number of chunks to retrieve |
| `CACHE_TTL` | 3600 | Cache expiry in seconds |
| `SESSION_TTL` | 86400 | Session expiry (24h) |

## Troubleshooting

### Redis Connection Issues

If you see "Connection refused", make sure Redis Stack is running:

```bash
docker compose up redis -d
```

Check the status:

```bash
docker compose ps
```

### No Results Found

If queries return no results:
1. Verify documents were ingested: `redis-cli FT.INFO rtfm-docs`
2. Check the document count in the index
3. Try re-ingesting: `rtfm ingest ./docs/`

### Slow First Query

The first query may be slow as the embedding model loads into memory. Subsequent queries will be much faster.
