"""Benchmark RTFM against NotebookLM-generated Q&A pairs.

Usage:
    python evals/run_benchmark.py                        # run all
    python evals/run_benchmark.py --limit 5              # run first 5
    python evals/run_benchmark.py --source progit.pdf    # filter by source
    python evals/run_benchmark.py --output results.json  # save full results

The benchmark file (evals/progit_benchmark.json) should contain:
[
  {
    "question": "How do you create a branch?",
    "expected_answer": "git branch <name>",
    "expected_keywords": ["git branch", "branch"],
    "category": "branching"           // optional
  }
]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rtfm.embeddings import embed_query
from rtfm.retrieval.rag import ask


BENCHMARK_FILE = Path(__file__).parent / "progit_benchmark.json"


def keyword_score(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using the embedding model."""
    vec_a = embed_query(text_a)
    vec_b = embed_query(text_b)
    return float(1 - cosine(vec_a, vec_b))


def run_benchmark(
    qa_pairs: list[dict],
    source_filter: str | None = None,
    limit: int | None = None,
) -> dict:
    """Run all Q&A pairs through RTFM and score results."""
    if limit:
        qa_pairs = qa_pairs[:limit]

    results = []
    total_keyword = 0.0
    total_semantic = 0.0
    total_latency = 0.0
    refusals = 0
    cache_hits = 0

    print(f"\nRunning benchmark: {len(qa_pairs)} questions\n{'=' * 60}")

    for i, qa in enumerate(qa_pairs, 1):
        question = qa["question"]
        expected = qa["expected_answer"]
        keywords = qa.get("expected_keywords", [])
        category = qa.get("category", "")

        print(f"\n[{i}/{len(qa_pairs)}] {question}")

        start = time.time()
        result = ask(question, source_filter=source_filter)
        elapsed = time.time() - start

        answer = result["answer"]
        is_refusal = "don't have enough information" in answer.lower()

        kw_score = keyword_score(answer, keywords)
        sem_score = semantic_similarity(answer, expected)

        total_keyword += kw_score
        total_semantic += sem_score
        total_latency += result["latency_ms"]
        if is_refusal:
            refusals += 1
        if result["cached"]:
            cache_hits += 1

        # Composite score: 40% keyword match + 60% semantic similarity
        composite = 0.4 * kw_score + 0.6 * sem_score

        status = "REFUSE" if is_refusal else ("GOOD" if composite >= 0.5 else "WEAK")
        print(f"  Keywords: {kw_score:.0%} | Semantic: {sem_score:.2f} | "
              f"Composite: {composite:.2f} | {status}")

        results.append({
            "question": question,
            "expected_answer": expected,
            "rtfm_answer": answer,
            "keyword_score": round(kw_score, 3),
            "semantic_similarity": round(sem_score, 3),
            "composite_score": round(composite, 3),
            "is_refusal": is_refusal,
            "cached": result["cached"],
            "latency_ms": result["latency_ms"],
            "tokens_used": result["tokens_used"],
            "sources": result["sources"],
            "category": category,
        })

    n = len(qa_pairs)
    summary = {
        "total_questions": n,
        "avg_keyword_score": round(total_keyword / n, 3) if n else 0,
        "avg_semantic_similarity": round(total_semantic / n, 3) if n else 0,
        "avg_composite_score": round(
            (0.4 * total_keyword / n + 0.6 * total_semantic / n), 3
        ) if n else 0,
        "refusal_rate": round(refusals / n, 3) if n else 0,
        "cache_hit_rate": round(cache_hits / n, 3) if n else 0,
        "avg_latency_ms": round(total_latency / n, 1) if n else 0,
        "pass_rate": round(
            sum(1 for r in results if r["composite_score"] >= 0.5) / n, 3
        ) if n else 0,
    }

    return {"summary": summary, "results": results}


def print_summary(summary: dict) -> None:
    """Print a formatted summary table."""
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Questions:          {summary['total_questions']}")
    print(f"  Pass rate (>=0.5):  {summary['pass_rate']:.0%}")
    print(f"  Avg keyword score:  {summary['avg_keyword_score']:.1%}")
    print(f"  Avg semantic sim:   {summary['avg_semantic_similarity']:.3f}")
    print(f"  Avg composite:      {summary['avg_composite_score']:.3f}")
    print(f"  Refusal rate:       {summary['refusal_rate']:.0%}")
    print(f"  Cache hit rate:     {summary['cache_hit_rate']:.0%}")
    print(f"  Avg latency:        {summary['avg_latency_ms']:.0f}ms")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Run RTFM benchmark")
    parser.add_argument("--file", default=str(BENCHMARK_FILE), help="Q&A JSON file")
    parser.add_argument("--limit", type=int, help="Max questions to run")
    parser.add_argument("--source", help="Filter by source file")
    parser.add_argument("--output", help="Save full results to JSON file")
    args = parser.parse_args()

    qa_path = Path(args.file)
    if not qa_path.exists():
        print(f"Error: {qa_path} not found")
        sys.exit(1)

    qa_pairs = json.loads(qa_path.read_text(encoding="utf-8"))
    if not qa_pairs:
        print("Error: No Q&A pairs found in benchmark file.")
        print(f"Add questions to {qa_path}")
        sys.exit(1)

    benchmark = run_benchmark(qa_pairs, source_filter=args.source, limit=args.limit)
    print_summary(benchmark["summary"])

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(benchmark, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
