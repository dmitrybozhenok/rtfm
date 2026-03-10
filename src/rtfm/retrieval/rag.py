"""RAG orchestrator — ties together search, cache, session, memory, and LLM."""

import time

from openai import OpenAI

from rtfm.config import settings
from rtfm.observability.logging import get_logger
from rtfm.observability.metrics import metrics
from rtfm.observability.tracing import trace_span
from rtfm.retrieval.search import SearchResult, search_documents

logger = get_logger("rag")

SYSTEM_PROMPT = """You are RTFM, a documentation assistant. Answer the user's question using ONLY the provided context from documentation.

Rules:
1. Only use information from the provided document context to answer.
2. Cite your sources using [Source N] labels that match the context headers.
3. Use exact terminology, command names, and technical terms from the documentation context.
4. Pay close attention to qualifiers and constraints in the question (e.g., "without", "only", "before").
5. When context contains multiple related but distinct topics, answer only about the one specifically asked.
6. Keep answers under 3-4 sentences for simple factual questions. If the user asks for details, explanation, or "how to", provide a thorough answer using all relevant context.
7. Only include code examples when the question is specifically a "how-to" question.
8. If the context doesn't contain enough information to answer, say "I don't have enough information in the documentation to answer that question." — but if you CAN answer, do NOT add disclaimers or hedging.
9. Always respond in the same language the user asked their question in. Keep technical terms, command names, and code in English.

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
    model_override: str | None = None,
) -> dict:
    """Full RAG pipeline: cache check → search → LLM → cache store.

    Args:
        model_override: Use this model instead of settings.llm_model for the LLM call.

    Returns dict with: answer, sources, cached, latency_ms, tokens_used
    """
    start = time.time()
    llm_model = model_override or settings.llm_model

    with trace_span("rag.ask", {"question": question, "model": llm_model}) as root_span:
        # Input guardrails (opt-in)
        if settings.guardrails_enabled:
            try:
                from rtfm.guardrails.input import validate_query

                with trace_span("guardrails.input") as g_span:
                    is_valid, reason = validate_query(question, settings)
                if not is_valid:
                    latency = (time.time() - start) * 1000
                    metrics.guardrail_blocks.inc()
                    logger.info("Query blocked by guardrails",
                                extra={"question": question, "guardrail_action": "blocked",
                                       "guardrail_reason": reason, "latency_ms": round(latency, 1)})
                    return {
                        "answer": reason,
                        "sources": [],
                        "cached": False,
                        "latency_ms": round(latency, 1),
                        "tokens_used": 0,
                    }
            except Exception:
                pass  # Graceful degradation

        # Check semantic cache
        cached_answer = None
        cached_sources: list[dict] = []
        with trace_span("cache.check") as cache_span:
            try:
                from rtfm.cache.semantic_cache import check_cache

                cached_answer, cached_sources = check_cache(question)
                cache_span.set_attribute("cache_hit", cached_answer is not None)
            except Exception:
                pass  # Graceful degradation

        if cached_answer is not None:
            latency = (time.time() - start) * 1000
            _record_metrics(latency, cached=True, tokens=0)
            metrics.queries_total.inc()
            metrics.cache_hits.inc()
            metrics.query_latency.observe(latency)
            logger.info("Query answered from cache",
                        extra={"question": question, "latency_ms": round(latency, 1),
                               "cached": True, "model": llm_model})
            return {
                "answer": cached_answer,
                "sources": cached_sources,
                "cached": True,
                "latency_ms": round(latency, 1),
                "tokens_used": 0,
            }

        # Vector search
        with trace_span("search", {"query": question}) as search_span:
            results = search_documents(
                question,
                source_filter=source_filter,
                section_filter=section_filter,
            )
            search_span.set_attribute("results_count", len(results))
            search_ms = search_span.duration_ms
        metrics.search_latency.observe(search_ms)
        metrics.chunks_retrieved.observe(len(results))

        context = _format_context(results)

        # Build system prompt with optional long-term memory
        memory_context = ""
        if long_term_memories:
            memory_context = f"\nUser context from previous sessions:\n{long_term_memories}"
        system = SYSTEM_PROMPT.format(memory_context=memory_context)

        # Build messages
        messages = _build_messages(question, context, system, session_history)

        # Call LLM via Ollama (OpenAI-compatible API)
        with trace_span("llm.call", {"model": llm_model}) as llm_span:
            client = _get_client()
            response = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=0,
                max_tokens=1024,
                extra_body={"repeat_penalty": 1.3},
            )
            llm_ms = llm_span.duration_ms
        metrics.llm_latency.observe(llm_ms)

        answer = response.choices[0].message.content
        tokens = (response.usage.prompt_tokens + response.usage.completion_tokens) if response.usage else 0
        metrics.tokens_per_query.observe(tokens)

        # Output guardrails (opt-in)
        if settings.guardrails_enabled:
            try:
                from rtfm.guardrails.output import filter_output

                with trace_span("guardrails.output"):
                    answer = filter_output(answer, settings)
            except Exception:
                pass  # Graceful degradation

        sources = [
            {"file": r.source_file, "section": r.section, "score": r.score,
             "snippet": r.text[:200].strip()}
            for r in results
        ]

        # Store in semantic cache (with sources)
        with trace_span("cache.store"):
            try:
                from rtfm.cache.semantic_cache import store_cache

                store_cache(question, answer, sources=sources)
            except Exception:
                pass  # Graceful degradation

        latency = (time.time() - start) * 1000
        _record_metrics(latency, cached=False, tokens=tokens)

        # Record Prometheus metrics
        metrics.queries_total.inc()
        metrics.cache_misses.inc()
        metrics.query_latency.observe(latency)

        source_files = [r.source_file for r in results]
        logger.info("Query processed",
                    extra={"question": question, "latency_ms": round(latency, 1),
                           "cached": False, "chunks_retrieved": len(results),
                           "tokens_used": tokens, "source_files": source_files,
                           "model": llm_model})

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
    start = time.time()

    # Check cache first
    try:
        from rtfm.cache.semantic_cache import check_cache

        cached_answer, cached_sources = check_cache(question)
        if cached_answer is not None:
            metrics.queries_total.inc()
            metrics.cache_hits.inc()
            latency = (time.time() - start) * 1000
            metrics.query_latency.observe(latency)
            logger.info("Streaming query answered from cache",
                        extra={"question": question, "latency_ms": round(latency, 1), "cached": True})
            yield {"token": cached_answer}
            yield {"sources": cached_sources or []}
            return
    except Exception:
        pass

    with trace_span("search", {"query": question}) as search_span:
        results = search_documents(
            question,
            source_filter=source_filter,
            section_filter=section_filter,
        )
        search_span.set_attribute("results_count", len(results))
        search_ms = search_span.duration_ms
    metrics.search_latency.observe(search_ms)
    metrics.chunks_retrieved.observe(len(results))

    context = _format_context(results)

    memory_context = ""
    if long_term_memories:
        memory_context = f"\nUser context from previous sessions:\n{long_term_memories}"
    system = SYSTEM_PROMPT.format(memory_context=memory_context)

    messages = _build_messages(question, context, system, session_history)

    client = _get_client()
    full_answer = ""
    llm_start = time.time()

    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        stream=True,
        extra_body={"repeat_penalty": 1.3},
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_answer += text
            yield {"token": text}

    llm_ms = (time.time() - llm_start) * 1000
    metrics.llm_latency.observe(llm_ms)

    sources = [
        {"file": r.source_file, "section": r.section, "score": r.score,
         "snippet": r.text[:200].strip()}
        for r in results
    ]
    try:
        from rtfm.cache.semantic_cache import store_cache

        store_cache(question, full_answer, sources=sources)
    except Exception:
        pass

    latency = (time.time() - start) * 1000
    metrics.queries_total.inc()
    metrics.cache_misses.inc()
    metrics.query_latency.observe(latency)

    logger.info("Streaming query processed",
                extra={"question": question, "latency_ms": round(latency, 1),
                       "cached": False, "chunks_retrieved": len(results),
                       "source_files": [r.source_file for r in results]})

    yield {"sources": sources}


def _get_source_chunks(source: str) -> list[dict]:
    """Retrieve all chunks for a given source from Redis."""
    from rtfm.redis_client import get_redis

    r = get_redis()
    chunks = []
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match="doc:*", count=200)
        for key in keys:
            fields = r.hmget(key, "source_file", "section", "text", "chunk_index")
            if fields[0] == source:
                chunks.append({
                    "source_file": fields[0] or "",
                    "section": fields[1] or "",
                    "text": fields[2] or "",
                    "chunk_index": int(fields[3]) if fields[3] else 0,
                })
        if cursor == 0:
            break

    chunks.sort(key=lambda c: c["chunk_index"])
    return chunks




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
