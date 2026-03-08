"""Typer CLI for RTFM: ingest, ask, chat commands."""

import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

app = typer.Typer(name="rtfm", help="RTFM For Me — AI documentation assistant")
console = Console()


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to file or directory to ingest"),
):
    """Ingest documents from a file path into the vector store."""
    from rtfm.ingest.pipeline import ingest_path

    p = Path(path)
    if not p.exists():
        console.print(f"[red]Error:[/red] Path not found: {p}")
        raise typer.Exit(1)

    console.print(f"Ingesting documents from [bold]{p}[/bold]...")
    stats = ingest_path(p)
    console.print(
        f"[green]Done![/green] Ingested {stats['files']} files, {stats['chunks']} chunks."
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    source: str | None = typer.Option(None, "--source", "-s", help="Filter by source file"),
    section: str | None = typer.Option(None, "--section", help="Filter by section"),
):
    """Ask a question about the ingested documentation."""
    from rtfm.retrieval.rag import ask as rag_ask

    with console.status("Thinking..."):
        result = rag_ask(
            question,
            source_filter=source,
            section_filter=section,
        )

    # Display answer
    console.print()
    console.print(Markdown(result["answer"]))
    console.print()

    # Display metadata
    if result["cached"]:
        console.print("[dim]Served from cache[/dim]")
    console.print(f"[dim]Latency: {result['latency_ms']}ms | Tokens: {result['tokens_used']}[/dim]")

    if result["sources"]:
        console.print("\n[bold]Sources:[/bold]")
        for src in result["sources"]:
            section_str = f" ({src['section']})" if src["section"] else ""
            console.print(f"  - {src['file']}{section_str} [dim](score: {src['score']:.3f})[/dim]")


@app.command()
def chat(
    source: str | None = typer.Option(None, "--source", "-s", help="Filter by source file"),
    section: str | None = typer.Option(None, "--section", help="Filter by section"),
):
    """Start an interactive chat session."""
    from rtfm.memory.longterm import format_memories, search_memory
    from rtfm.memory.session import (
        add_message,
        clear_session,
        get_history_truncated,
        summarize_if_needed,
    )
    from rtfm.retrieval.rag import ask as rag_ask

    session_id = str(uuid.uuid4())
    console.print(
        Panel(
            "RTFM Interactive Chat\n"
            "Type your questions. Special commands:\n"
            "  /quit  - Exit\n"
            "  /clear - Clear conversation history\n"
            "  /sources - Show sources from last answer",
            title="RTFM Chat",
        )
    )

    last_sources: list[dict] = []

    while True:
        try:
            question = console.input("\n[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not question:
            continue

        if question == "/quit":
            # Store conversation in long-term memory before exiting
            from rtfm.memory.session import get_history

            history = get_history(session_id)
            if history:
                from rtfm.memory.longterm import store_memory

                store_memory(session_id, history)
            console.print("Goodbye!")
            break

        if question == "/clear":
            clear_session(session_id)
            last_sources = []
            console.print("[dim]Conversation cleared.[/dim]")
            continue

        if question == "/sources":
            if last_sources:
                console.print("\n[bold]Sources from last answer:[/bold]")
                for src in last_sources:
                    section_str = f" ({src['section']})" if src["section"] else ""
                    console.print(f"  - {src['file']}{section_str} [dim](score: {src['score']:.3f})[/dim]")
            else:
                console.print("[dim]No sources available.[/dim]")
            continue

        # Get session history and long-term memory
        session_history = get_history_truncated(session_id)
        memories = search_memory(question)
        memory_context = format_memories(memories)

        with console.status("Thinking..."):
            result = rag_ask(
                question,
                session_history=session_history,
                long_term_memories=memory_context,
                source_filter=source,
                section_filter=section,
            )

        # Save to session
        add_message(session_id, "user", question)
        add_message(session_id, "assistant", result["answer"])
        summarize_if_needed(session_id)

        last_sources = result["sources"]

        # Display
        console.print()
        console.print(Markdown(result["answer"]))

        if result["cached"]:
            console.print("[dim]Served from cache[/dim]")
        console.print(
            f"[dim]Latency: {result['latency_ms']}ms | Tokens: {result['tokens_used']}[/dim]"
        )


@app.command(name="list-sources")
def list_sources():
    """List all ingested document sources with chunk counts."""
    from rtfm.redis_client import get_redis

    r = get_redis()

    # Scan all doc: keys and collect source info
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

    if not sources:
        console.print("[yellow]No documents ingested yet.[/yellow]")
        raise typer.Exit()

    total_chunks = sum(v["chunks"] for v in sources.values())
    console.print(f"\n[bold]Ingested Sources:[/bold] {len(sources)} sources, {total_chunks} total chunks\n")

    for src, info in sorted(sources.items()):
        sections_str = ", ".join(sorted(info["sections"])) if info["sections"] else "—"
        console.print(f"  {src}  [dim]({info['chunks']} chunks)[/dim]")
        console.print(f"    Sections: [dim]{sections_str}[/dim]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    source: str | None = typer.Option(None, "--source", "-s", help="Filter by source file"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable reranking"),
):
    """Search documents without invoking the LLM."""
    from rtfm.retrieval.search import search_documents

    with console.status("Searching..."):
        results = search_documents(
            query,
            top_k=top_k,
            source_filter=source,
            hybrid=not no_rerank,
        )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit()

    console.print(f"\n[bold]Search Results[/bold] ({len(results)} hits)\n")
    for i, r in enumerate(results, 1):
        section_str = f" ({r.section})" if r.section else ""
        preview = r.text[:200].replace("\n", " ")
        if len(r.text) > 200:
            preview += "..."
        console.print(f"[bold]{i}.[/bold] [cyan]{r.source_file}[/cyan]{section_str} [dim](score: {r.score:.4f})[/dim]")
        console.print(f"   {preview}\n")


@app.command(name="clear-cache")
def clear_cache():
    """Flush the semantic cache."""
    from rtfm.cache.semantic_cache import flush_cache

    flush_cache()
    console.print("[green]Cache flushed.[/green]")


def _get_session_history(session_id):
    """Import helper to avoid circular imports."""
    from rtfm.memory.session import get_history

    return get_history(session_id)


if __name__ == "__main__":
    app()
