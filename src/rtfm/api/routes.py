"""FastAPI endpoints for RTFM."""

import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated

_START_TIME = time.time()

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from sse_starlette.sse import EventSourceResponse

from rtfm.cache.semantic_cache import flush_cache, get_cache_metrics
from rtfm.config import settings
from rtfm.ingest.pipeline import ingest_path, ingest_url
from rtfm.memory.longterm import format_memories, search_memory, store_memory
from rtfm.memory.session import (
    add_message,
    clear_session,
    get_history_truncated,
    summarize_if_needed,
)
from rtfm.observability.logging import get_logger, setup_logging
from rtfm.observability.metrics import metrics
from rtfm.retrieval.rag import ask, ask_stream

# Initialize structured logging on import
setup_logging(log_level=settings.log_level, log_format=settings.log_format)
logger = get_logger("api")

app = FastAPI(title="RTFM", version="0.1.0")

_STATIC_DIR = Path(__file__).parent / "static"


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every HTTP request with method, path, status, and latency."""
    start = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start) * 1000

    # Skip noisy health/metrics polling
    if request.url.path not in ("/health", "/metrics", "/metrics/prometheus"):
        logger.info("HTTP request",
                     extra={"method": request.method, "path": request.url.path,
                            "status_code": response.status_code,
                            "latency_ms": round(latency_ms, 1)})
    return response


@app.get("/")
async def index():
    """Serve the chat UI."""
    return FileResponse(_STATIC_DIR / "index.html")


@app.post("/ingest")
async def ingest_endpoint(
    file: Annotated[UploadFile, File()],
):
    """Ingest an uploaded file."""
    tmp_dir = Path(tempfile.gettempdir()) / "rtfm_uploads"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / file.filename

    content = await file.read()
    tmp_path.write_bytes(content)

    try:
        stats = ingest_path(tmp_path)
        metrics.ingestion_files.inc(stats.get("files", 0))
        metrics.ingestion_chunks.inc(stats.get("chunks", 0))
        logger.info("File ingested",
                     extra={"files_processed": stats.get("files", 0),
                            "chunks_created": stats.get("chunks", 0),
                            "path": str(tmp_path)})
        return {"status": "ok", **stats}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/ingest/path")
async def ingest_path_endpoint(path: str = Form(...)):
    """Ingest files from a local path."""
    p = Path(path)
    if not p.exists():
        return JSONResponse(status_code=404, content={"error": f"Path not found: {path}"})
    stats = ingest_path(p)
    metrics.ingestion_files.inc(stats.get("files", 0))
    metrics.ingestion_chunks.inc(stats.get("chunks", 0))
    logger.info("Path ingested",
                 extra={"files_processed": stats.get("files", 0),
                        "chunks_created": stats.get("chunks", 0),
                        "path": path})
    return {"status": "ok", **stats}


@app.post("/ingest/url")
async def ingest_url_endpoint(
    url: str = Form(...),
    recursive: bool = Form(False),
    delay: float = Form(1.0),
):
    """Ingest content from a URL."""
    stats = ingest_url(url, recursive=recursive, delay=delay)
    metrics.ingestion_chunks.inc(stats.get("chunks", 0))
    logger.info("URL ingested",
                 extra={"url": url, "chunks_created": stats.get("chunks", 0),
                        "files_processed": stats.get("pages", 0)})
    return {"status": "ok", **stats}


@app.post("/ask")
async def ask_endpoint(
    question: str = Form(...),
    source_filter: str | None = Form(None),
    section_filter: str | None = Form(None),
):
    """Ask a question (non-streaming, no session)."""
    result = ask(
        question,
        source_filter=source_filter,
        section_filter=section_filter,
    )
    return result


@app.post("/chat")
async def chat_endpoint(
    question: str = Form(...),
    session_id: str | None = Form(None),
    source_filter: str | None = Form(None),
    section_filter: str | None = Form(None),
    stream: bool = Form(False),
):
    """Chat with session history and optional streaming."""
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Get session history
    session_history = get_history_truncated(session_id)

    # Search long-term memory
    memories = search_memory(question)
    memory_context = format_memories(memories)

    if stream:
        async def event_generator():
            full_answer = ""
            for chunk in ask_stream(
                question,
                session_history=session_history,
                long_term_memories=memory_context,
                source_filter=source_filter,
                section_filter=section_filter,
            ):
                full_answer += chunk
                yield {"data": json.dumps({"token": chunk})}

            # Save to session after streaming completes
            add_message(session_id, "user", question)
            add_message(session_id, "assistant", full_answer)
            summarize_if_needed(session_id)

            # Store in long-term memory
            store_memory(session_id, [
                {"role": "user", "content": question},
                {"role": "assistant", "content": full_answer},
            ])

        return EventSourceResponse(event_generator())

    # Non-streaming
    result = ask(
        question,
        session_history=session_history,
        long_term_memories=memory_context,
        source_filter=source_filter,
        section_filter=section_filter,
    )

    # Save to session
    add_message(session_id, "user", question)
    add_message(session_id, "assistant", result["answer"])
    summarize_if_needed(session_id)

    # Store in long-term memory
    store_memory(session_id, [
        {"role": "user", "content": question},
        {"role": "assistant", "content": result["answer"]},
    ])

    result["session_id"] = session_id
    return result


@app.get("/metrics")
async def metrics_endpoint():
    """Get cache and usage metrics."""
    return get_cache_metrics()


@app.get("/metrics/prometheus")
async def prometheus_metrics_endpoint():
    """Prometheus-compatible metrics endpoint."""
    # Update index size gauge
    try:
        from rtfm.redis_client import get_redis

        r = get_redis()
        info = r.ft("rtfm-docs").info()
        metrics.index_size.set(int(info.get("num_docs", 0)))
    except Exception:
        pass

    return PlainTextResponse(metrics.to_prometheus(), media_type="text/plain; version=0.0.4")


@app.post("/cache/flush")
async def flush_cache_endpoint():
    """Flush the semantic cache."""
    flush_cache()
    return {"status": "ok", "message": "Cache flushed"}


@app.post("/session/{session_id}/clear")
async def clear_session_endpoint(session_id: str):
    """Clear a session's conversation history."""
    clear_session(session_id)
    return {"status": "ok", "message": f"Session {session_id} cleared"}


@app.get("/health")
async def health_endpoint():
    """Check health of all subsystems."""
    import httpx

    from rtfm.redis_client import get_redis

    # Check Redis
    redis_ok = False
    try:
        r = get_redis()
        redis_ok = r.ping()
    except Exception:
        pass

    # Check Ollama
    ollama_ok = False
    try:
        ollama_url = settings.ollama_base_url.rstrip("/v1").rstrip("/")
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    # Check Memory Server
    memory_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.memory_server_url}/health")
            memory_ok = resp.status_code == 200
    except Exception:
        pass

    # Check index existence
    index_ok = False
    doc_count = 0
    if redis_ok:
        try:
            info = r.ft("rtfm-docs").info()
            index_ok = True
            doc_count = int(info.get("num_docs", 0))
        except Exception:
            pass

    # Determine overall status
    if redis_ok and ollama_ok:
        status = "healthy"
    elif redis_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    uptime_s = int(time.time() - _START_TIME)

    return {
        "status": status,
        "version": "0.1.0",
        "uptime_seconds": uptime_s,
        "redis": redis_ok,
        "ollama": ollama_ok,
        "memory_server": memory_ok,
        "index_exists": index_ok,
        "doc_count": doc_count,
    }


@app.post("/documents/clear")
async def clear_documents_endpoint():
    """Delete all ingested documents and flush cache."""
    from rtfm.redis_client import get_redis

    r = get_redis()
    cursor = 0
    deleted = 0
    while True:
        cursor, keys = r.scan(cursor, match="doc:*", count=200)
        if keys:
            deleted += len(keys)
            r.delete(*keys)
        if cursor == 0:
            break

    # Also drop the search index so it's recreated on next ingest
    try:
        r.ft("rtfm-docs").dropindex()
    except Exception:
        pass

    # Flush cache since it references deleted docs
    try:
        flush_cache()
    except Exception:
        pass

    logger.info("Documents cleared", extra={"chunks_created": deleted})
    return {"status": "ok", "deleted_chunks": deleted}


@app.get("/sources")
async def sources_endpoint():
    """List all ingested document sources with chunk counts."""
    from rtfm.redis_client import get_redis

    r = get_redis()

    sources: dict[str, dict] = {}
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match="doc:*", count=200)
        for key in keys:
            fields = r.hmget(key, "source_file", "section")
            source_file = fields[0] or "unknown"
            section = fields[1] or ""

            if source_file not in sources:
                sources[source_file] = {"chunks": 0, "sections": set()}
            sources[source_file]["chunks"] += 1
            if section:
                sources[source_file]["sections"].add(section)
        if cursor == 0:
            break

    # Convert sets to sorted lists for JSON serialization
    result = []
    for src, info in sorted(sources.items()):
        source_type = "web" if src.startswith(("http://", "https://")) else "file"
        result.append({
            "source": src,
            "type": source_type,
            "chunks": info["chunks"],
            "sections": sorted(info["sections"]),
        })

    return result
