"""FastAPI endpoints for RTFM."""

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from rtfm.cache.semantic_cache import flush_cache, get_cache_metrics
from rtfm.ingest.pipeline import ingest_path, ingest_url
from rtfm.memory.longterm import format_memories, search_memory, store_memory
from rtfm.memory.session import (
    add_message,
    clear_session,
    get_history_truncated,
    summarize_if_needed,
)
from rtfm.retrieval.rag import ask, ask_stream

app = FastAPI(title="RTFM", version="0.1.0")


@app.post("/ingest")
async def ingest_endpoint(
    file: Annotated[UploadFile, File()],
):
    """Ingest an uploaded file."""
    tmp_dir = Path("/tmp/rtfm_uploads")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / file.filename

    content = await file.read()
    tmp_path.write_bytes(content)

    try:
        stats = ingest_path(tmp_path)
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
    return {"status": "ok", **stats}


@app.post("/ingest/url")
async def ingest_url_endpoint(
    url: str = Form(...),
    recursive: bool = Form(False),
    delay: float = Form(1.0),
):
    """Ingest content from a URL."""
    stats = ingest_url(url, recursive=recursive, delay=delay)
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
                yield {"data": chunk}

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
