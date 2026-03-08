"""Faithfulness eval — detect hallucinated claims not grounded in retrieved chunks.

Uses the LLM as judge to check if each claim in the generated answer is
supported by the provided context chunks. Catches hallucinations that
keyword/semantic similarity evals miss.

Usage:
    python evals/faithfulness.py                    # run on all benchmark questions
    python evals/faithfulness.py --limit 5          # first 5 only
    python evals/faithfulness.py --model qwen2.5:7b # specify model
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI

from rtfm.config import settings
from rtfm.retrieval.rag import ask

BENCHMARK_FILE = Path(__file__).parent / "progit_benchmark.json"

FAITHFULNESS_PROMPT = """You are evaluating whether an AI assistant's answer is faithful to the provided source context.

For each claim in the generated answer, determine if it is:
- SUPPORTED: directly backed by information in the context
- NOT SUPPORTED: not found in the context (hallucinated or from external knowledge)

Be strict. If the context doesn't explicitly state something, it is NOT SUPPORTED — even if it's true in general.

Respond with ONLY a JSON object:
{
  "claims": [
    {"claim": "brief claim text", "supported": true/false}
  ],
  "faithfulness_score": <float 0.0-1.0 = fraction of supported claims>,
  "hallucinated_claims": ["list of unsupported claims, if any"]
}"""


def check_faithfulness(
    question: str,
    answer: str,
    context_chunks: list[dict],
    model: str = "qwen2.5:7b",
    ollama_base_url: str = "http://localhost:11434/v1",
) -> dict:
    """Check if an answer is faithful to the retrieved context chunks.

    Returns dict with faithfulness_score (0-1), claims breakdown, hallucinated_claims.
    """
    # Build context from source chunks
    context_text = "\n\n---\n\n".join(
        f"[Source: {c.get('file', 'unknown')}]\n{c.get('text', '')}"
        for c in context_chunks
        if c.get("text")
    )

    if not context_text:
        context_text = "(no context chunks retrieved)"

    client = OpenAI(base_url=ollama_base_url, api_key="ollama")

    user_content = f"""Context (retrieved chunks):
{context_text}

Question: {question}

Generated answer: {answer}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": FAITHFULNESS_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()
    return _parse_faithfulness(raw)


def _parse_faithfulness(raw: str) -> dict:
    """Parse LLM faithfulness response."""
    import re

    # Try JSON parse
    try:
        parsed = json.loads(raw)
        return _validate_faithfulness(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting JSON from response
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return _validate_faithfulness(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback
    return {
        "claims": [],
        "faithfulness_score": 0.0,
        "hallucinated_claims": [],
        "parse_error": raw[:200],
    }


def _validate_faithfulness(parsed: dict) -> dict:
    """Validate and normalize parsed faithfulness result."""
    claims = parsed.get("claims", [])
    hallucinated = parsed.get("hallucinated_claims", [])

    if claims:
        supported = sum(1 for c in claims if c.get("supported", False))
        score = supported / len(claims)
    else:
        score = float(parsed.get("faithfulness_score", 0.0))

    return {
        "claims": claims,
        "faithfulness_score": round(score, 3),
        "hallucinated_claims": hallucinated,
    }


def _get_chunk_texts(sources: list[dict]) -> list[dict]:
    """Fetch actual chunk text for the sources returned by RAG."""
    from rtfm.retrieval.search import search_documents

    return [{"file": s.get("file", ""), "text": s.get("text", "")} for s in sources]


def run_faithfulness_eval(
    qa_pairs: list[dict],
    model: str | None = None,
    limit: int | None = None,
) -> dict:
    """Run faithfulness eval on benchmark questions.

    Asks each question through the full RAG pipeline, then checks if the
    answer is faithful to the retrieved chunks (not hallucinated).
    """
    llm_model = model or settings.llm_model

    if limit:
        qa_pairs = qa_pairs[:limit]

    print(f"\nFaithfulness Eval: {len(qa_pairs)} questions (model: {llm_model})")
    print("=" * 60)

    results = []
    total_score = 0.0
    total_hallucinated = 0

    for i, qa in enumerate(qa_pairs, 1):
        question = qa["question"]
        print(f"\n[{i}/{len(qa_pairs)}] {question}")

        # Get RAG answer with sources
        start = time.time()
        rag_result = ask(question, model_override=llm_model)
        elapsed_ms = (time.time() - start) * 1000

        answer = rag_result["answer"]
        sources = rag_result["sources"]

        # We need the actual chunk text — re-search to get it
        from rtfm.retrieval.search import search_documents
        search_results = search_documents(question)
        chunk_contexts = [
            {"file": r.source_file, "text": r.text}
            for r in search_results
        ]

        # Check faithfulness
        faith = check_faithfulness(
            question=question,
            answer=answer,
            context_chunks=chunk_contexts,
            model=llm_model,
            ollama_base_url=settings.ollama_base_url,
        )

        score = faith["faithfulness_score"]
        hallucinated = faith["hallucinated_claims"]
        total_score += score
        total_hallucinated += len(hallucinated)

        status = "FAITHFUL" if score >= 0.8 else ("PARTIAL" if score >= 0.5 else "HALLUCINATED")
        print(f"  Faithfulness: {score:.0%} | Claims: {len(faith['claims'])} | "
              f"Hallucinated: {len(hallucinated)} | {status}")
        if hallucinated:
            for h in hallucinated[:3]:
                print(f"    ! {h[:100]}")

        results.append({
            "question": question,
            "answer": answer,
            "faithfulness_score": score,
            "claims": faith["claims"],
            "hallucinated_claims": hallucinated,
            "latency_ms": round(elapsed_ms, 1),
            "category": qa.get("category", ""),
        })

    n = len(qa_pairs)
    summary = {
        "total_questions": n,
        "avg_faithfulness": round(total_score / n, 3) if n else 0,
        "total_hallucinated_claims": total_hallucinated,
        "fully_faithful_rate": round(
            sum(1 for r in results if r["faithfulness_score"] >= 1.0) / n, 3
        ) if n else 0,
        "hallucination_rate": round(
            sum(1 for r in results if r["faithfulness_score"] < 0.8) / n, 3
        ) if n else 0,
        "model": llm_model,
    }

    print(f"\n{'=' * 60}")
    print("FAITHFULNESS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Questions:            {summary['total_questions']}")
    print(f"  Avg faithfulness:     {summary['avg_faithfulness']:.0%}")
    print(f"  Fully faithful:       {summary['fully_faithful_rate']:.0%}")
    print(f"  Hallucination rate:   {summary['hallucination_rate']:.0%}")
    print(f"  Total halluc. claims: {summary['total_hallucinated_claims']}")
    print(f"  Model:                {summary['model']}")

    # Worst offenders
    worst = sorted(results, key=lambda r: r["faithfulness_score"])
    if worst and worst[0]["faithfulness_score"] < 1.0:
        print(f"\n{'-' * 60}")
        print("WORST FAITHFULNESS")
        print(f"{'-' * 60}")
        for r in worst[:5]:
            if r["faithfulness_score"] >= 1.0:
                break
            print(f"\n  [{r['faithfulness_score']:.0%}] {r['question']}")
            for h in r["hallucinated_claims"][:3]:
                print(f"    ! {h[:100]}")

    print(f"{'=' * 60}")

    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Run faithfulness eval")
    parser.add_argument("--file", default=str(BENCHMARK_FILE))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    qa_pairs = json.loads(Path(args.file).read_text(encoding="utf-8"))
    result = run_faithfulness_eval(qa_pairs, model=args.model, limit=args.limit)

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
