"""RAG orchestrator — ties together search, cache, session, memory, and LLM."""

import time

from openai import OpenAI

from rtfm.config import settings
from rtfm.retrieval.search import SearchResult, search_documents

SYSTEM_PROMPT = """You are RTFM, a documentation assistant. Answer the user's question using ONLY the provided context from documentation.

Rules:
1. Only use information from the provided document context to answer.
2. Cite your sources by mentioning the source file name (e.g., "According to sample.md...").
3. If the context doesn't contain enough information to answer, say "I don't have enough information in the documentation to answer that question."
4. Be concise and direct.
5. If code examples are present in the context, include them in your answer when relevant.

{memory_context}"""


def _get_client() -> OpenAI:
    """Return an OpenAI client pointed at Ollama."""
    return OpenAI(base_url=settings.ollama_base_url, api_key="ollama")


def _format_context(results: list[SearchResult]) -> str:
    """Format search results into context for the LLM."""
    if not results:
        return "No relevant documentation found."

    parts = []
    for i, r in enumerate(results, 1):
        section_info = f" (section: {r.section})" if r.section else ""
        parts.append(
            f"[Source {i}: {r.source_file}{section_info}]\n{r.text}"
        )
    return "\n\n---\n\n".join(parts)


def _build_messages(
    question: str,
    context: str,
    system: str,
    session_history: list[dict] | None = None,
) -> list[dict]:
    """Build the messages list for the OpenAI-compatible API."""
    messages = [{"role": "system", "content": system}]

    # Add session history if present
    if session_history:
        messages.extend(session_history)

    # Add current question with context
    user_content = f"""Documentation context:
{context}

Question: {question}"""

    messages.append({"role": "user", "content": user_content})
    return messages


def ask(
    question: str,
    session_history: list[dict] | None = None,
    long_term_memories: str = "",
    source_filter: str | None = None,
    section_filter: str | None = None,
) -> dict:
    """Full RAG pipeline: cache check → search → LLM → cache store.

    Returns dict with: answer, sources, cached, latency_ms, tokens_used
    """
    start = time.time()

    # Check semantic cache
    cached_answer = None
    try:
        from rtfm.cache.semantic_cache import check_cache

        cached_answer = check_cache(question)
    except Exception:
        pass  # Graceful degradation

    if cached_answer is not None:
        latency = (time.time() - start) * 1000
        _record_metrics(latency, cached=True, tokens=0)
        return {
            "answer": cached_answer,
            "sources": [],
            "cached": True,
            "latency_ms": round(latency, 1),
            "tokens_used": 0,
        }

    # Vector search
    results = search_documents(
        question,
        source_filter=source_filter,
        section_filter=section_filter,
    )
    context = _format_context(results)

    # Build system prompt with optional long-term memory
    memory_context = ""
    if long_term_memories:
        memory_context = f"\nUser context from previous sessions:\n{long_term_memories}"
    system = SYSTEM_PROMPT.format(memory_context=memory_context)

    # Build messages
    messages = _build_messages(question, context, system, session_history)

    # Call LLM via Ollama (OpenAI-compatible API)
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        max_tokens=2048,
    )

    answer = response.choices[0].message.content
    tokens = (response.usage.prompt_tokens + response.usage.completion_tokens) if response.usage else 0

    # Store in semantic cache
    try:
        from rtfm.cache.semantic_cache import store_cache

        store_cache(question, answer)
    except Exception:
        pass  # Graceful degradation

    latency = (time.time() - start) * 1000
    _record_metrics(latency, cached=False, tokens=tokens)

    sources = [
        {"file": r.source_file, "section": r.section, "score": r.score}
        for r in results
    ]

    return {
        "answer": answer,
        "sources": sources,
        "cached": False,
        "latency_ms": round(latency, 1),
        "tokens_used": tokens,
    }


def ask_stream(
    question: str,
    session_history: list[dict] | None = None,
    long_term_memories: str = "",
    source_filter: str | None = None,
    section_filter: str | None = None,
):
    """Streaming RAG pipeline. Yields text chunks."""
    # Check cache first
    try:
        from rtfm.cache.semantic_cache import check_cache

        cached_answer = check_cache(question)
        if cached_answer is not None:
            yield cached_answer
            return
    except Exception:
        pass

    results = search_documents(
        question,
        source_filter=source_filter,
        section_filter=section_filter,
    )
    context = _format_context(results)

    memory_context = ""
    if long_term_memories:
        memory_context = f"\nUser context from previous sessions:\n{long_term_memories}"
    system = SYSTEM_PROMPT.format(memory_context=memory_context)

    messages = _build_messages(question, context, system, session_history)

    client = _get_client()
    full_answer = ""

    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        max_tokens=2048,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_answer += text
            yield text

    # Cache the full answer
    try:
        from rtfm.cache.semantic_cache import store_cache

        store_cache(question, full_answer)
    except Exception:
        pass


def _record_metrics(latency_ms: float, cached: bool, tokens: int) -> None:
    """Record query metrics in Redis."""
    try:
        from rtfm.redis_client import get_redis

        r = get_redis()
        if cached:
            r.incr("rtfm:metrics:cache_hits")
        else:
            r.incr("rtfm:metrics:cache_misses")
        r.incrbyfloat("rtfm:metrics:total_latency_ms", latency_ms)
        r.incr("rtfm:metrics:total_queries")
        r.incrby("rtfm:metrics:total_tokens", tokens)
    except Exception:
        pass
