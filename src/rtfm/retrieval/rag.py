"""RAG orchestrator — ties together search, cache, session, memory, and LLM."""

import time

from openai import OpenAI

from rtfm.config import settings
from rtfm.retrieval.search import SearchResult, search_documents

SYSTEM_PROMPT = """You are RTFM, a documentation assistant. Answer the user's question using ONLY the provided context from documentation.

Rules:
1. Only use information from the provided document context to answer.
2. Cite your sources using [Source N] labels that match the context headers.
3. Use exact terminology, command names, and technical terms from the documentation context.
4. Pay close attention to qualifiers and constraints in the question (e.g., "without", "only", "before").
5. When context contains multiple related but distinct topics, answer only about the one specifically asked.
6. Keep answers under 3-4 sentences unless a longer explanation is necessary.
7. Only include code examples when the question is specifically a "how-to" question.
8. If the context doesn't contain enough information to answer, say "I don't have enough information in the documentation to answer that question." — but if you CAN answer, do NOT add disclaimers or hedging.

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
            f"[Source {i}: {r.source_file}{section_info} (relevance: {r.score:.2f})]\n{r.text}"
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
    source_url_filter: str | None = None,
    source_type_filter: str | None = None,
) -> dict:
    """Full RAG pipeline: cache check → search → LLM → cache store.

    Returns dict with: answer, sources, cached, latency_ms, tokens_used
    """
    start = time.time()

    # Check semantic cache
    cached_answer = None
    cached_sources: list[dict] = []
    try:
        from rtfm.cache.semantic_cache import check_cache

        cached_answer, cached_sources = check_cache(question)
    except Exception:
        pass  # Graceful degradation

    if cached_answer is not None:
        latency = (time.time() - start) * 1000
        _record_metrics(latency, cached=True, tokens=0)
        return {
            "answer": cached_answer,
            "sources": cached_sources,
            "cached": True,
            "latency_ms": round(latency, 1),
            "tokens_used": 0,
        }

    # Vector search
    results = search_documents(
        question,
        source_filter=source_filter,
        section_filter=section_filter,
        source_url_filter=source_url_filter,
        source_type_filter=source_type_filter,
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
        temperature=0,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    tokens = (response.usage.prompt_tokens + response.usage.completion_tokens) if response.usage else 0

    sources = [
        {"file": r.source_file, "section": r.section, "score": r.score, "url": r.source_url}
        for r in results
    ]

    # Store in semantic cache (with sources)
    try:
        from rtfm.cache.semantic_cache import store_cache

        store_cache(question, answer, sources=sources)
    except Exception:
        pass  # Graceful degradation

    latency = (time.time() - start) * 1000
    _record_metrics(latency, cached=False, tokens=tokens)

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
    source_url_filter: str | None = None,
    source_type_filter: str | None = None,
):
    """Streaming RAG pipeline. Yields text chunks."""
    # Check cache first
    try:
        from rtfm.cache.semantic_cache import check_cache

        cached_answer, _cached_sources = check_cache(question)
        if cached_answer is not None:
            yield cached_answer
            return
    except Exception:
        pass

    results = search_documents(
        question,
        source_filter=source_filter,
        section_filter=section_filter,
        source_url_filter=source_url_filter,
        source_type_filter=source_type_filter,
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
        temperature=0,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_answer += text
            yield text

    # Cache the full answer (with sources)
    sources = [
        {"file": r.source_file, "section": r.section, "score": r.score, "url": r.source_url}
        for r in results
    ]
    try:
        from rtfm.cache.semantic_cache import store_cache

        store_cache(question, full_answer, sources=sources)
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
