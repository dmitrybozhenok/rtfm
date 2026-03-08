"""Benchmark RTFM against NotebookLM-generated Q&A pairs.

Usage:
    python evals/run_benchmark.py                        # run all
    python evals/run_benchmark.py --limit 5              # run first 5
    python evals/run_benchmark.py --source progit.pdf    # filter by source
    python evals/run_benchmark.py --output results.json  # save full results
    python evals/run_benchmark.py --no-cache             # clear cache before run
    python evals/run_benchmark.py --worst 5              # show worst N questions

The benchmark file (evals/progit_benchmark.json) should contain:
[
  {
    "question": "How do you create a branch?",
    "expected_answer": "git branch <name>",
    "expected_keywords": ["git branch", "branch"],
    "category": "branching"
  }
]
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rtfm.embeddings import embed_query
from rtfm.retrieval.rag import ask

BENCHMARK_FILE = Path(__file__).parent / "progit_benchmark.json"
HISTORY_FILE = Path(__file__).parent / "history.jsonl"

# Refusal patterns — multiple ways the LLM might refuse
_REFUSAL_PATTERNS = [
    re.compile(r"don'?t have enough information", re.IGNORECASE),
    re.compile(r"cannot (find|answer|determine)", re.IGNORECASE),
    re.compile(r"no relevant (documentation|information|context)", re.IGNORECASE),
    re.compile(r"not (mentioned|covered|discussed) in the (documentation|context)", re.IGNORECASE),
    re.compile(r"the (documentation|context) does not (contain|mention|cover)", re.IGNORECASE),
]


def _is_refusal(answer: str) -> bool:
    """Detect refusal using multiple patterns.

    A hedged answer (refusal after substantive content) is NOT a refusal.
    """
    for pattern in _REFUSAL_PATTERNS:
        match = pattern.search(answer)
        if match:
            # If the refusal appears in the first 30% of the answer, it's a real refusal
            # If it appears late, it's hedging on an otherwise substantive answer
            pos_ratio = match.start() / max(len(answer), 1)
            if pos_ratio < 0.3:
                return True
    return False


def _stem_word(word: str) -> str:
    """Simple Porter-style suffix stripping for fuzzy keyword matching."""
    word = word.lower()
    for suffix in ("ing", "tion", "sion", "ment", "ness", "ed", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def keyword_score(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer.

    Uses exact match first, then falls back to stemmed/sliding-window matching.
    """
    if not keywords:
        return 1.0

    answer_lower = answer.lower()
    answer_words = answer_lower.split()
    answer_stems = [_stem_word(w) for w in answer_words]

    hits = 0
    for kw in keywords:
        kw_lower = kw.lower()
        # Exact substring match
        if kw_lower in answer_lower:
            hits += 1
            continue
        # Stemmed match for single words
        kw_parts = kw_lower.split()
        if len(kw_parts) == 1:
            if _stem_word(kw_lower) in answer_stems:
                hits += 1
                continue
        else:
            # Sliding window for multi-word keywords
            kw_stems = [_stem_word(w) for w in kw_parts]
            window_size = len(kw_stems)
            for i in range(len(answer_stems) - window_size + 1):
                window = answer_stems[i : i + window_size]
                if window == kw_stems:
                    hits += 1
                    break

    return hits / len(keywords)


def semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using the embedding model."""
    vec_a = embed_query(text_a)
    vec_b = embed_query(text_b)
    return float(1 - cosine(vec_a, vec_b))


def conciseness_ratio(expected: str, generated: str) -> float:
    """Ratio of expected to generated answer length. Closer to 1.0 is better."""
    if not generated:
        return 0.0
    return min(len(expected) / max(len(generated), 1), 2.0)


def run_benchmark(
    qa_pairs: list[dict],
    source_filter: str | None = None,
    limit: int | None = None,
    no_cache: bool = False,
) -> dict:
    """Run all Q&A pairs through RTFM and score results."""
    if no_cache:
        try:
            from rtfm.cache.semantic_cache import flush_cache
            flush_cache()
            print("Semantic cache flushed.")
        except Exception as e:
            print(f"Warning: Could not flush cache: {e}")

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
        is_refusal = _is_refusal(answer)

        kw_score = keyword_score(answer, keywords)
        sem_score = semantic_similarity(answer, expected)
        conc_ratio = conciseness_ratio(expected, answer)

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
        print(
            f"  Keywords: {kw_score:.0%} | Semantic: {sem_score:.2f} | "
            f"Composite: {composite:.2f} | Conciseness: {conc_ratio:.2f} | {status}"
        )

        results.append({
            "question": question,
            "expected_answer": expected,
            "rtfm_answer": answer,
            "keyword_score": round(kw_score, 3),
            "semantic_similarity": round(sem_score, 3),
            "composite_score": round(composite, 3),
            "conciseness_ratio": round(conc_ratio, 3),
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

    # Per-category breakdown
    categories: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        cat = r["category"] or "Uncategorized"
        categories[cat].append(r)

    category_breakdown = {}
    for cat, cat_results in sorted(categories.items()):
        cn = len(cat_results)
        category_breakdown[cat] = {
            "count": cn,
            "avg_composite": round(sum(r["composite_score"] for r in cat_results) / cn, 3),
            "avg_keyword": round(sum(r["keyword_score"] for r in cat_results) / cn, 3),
            "avg_semantic": round(sum(r["semantic_similarity"] for r in cat_results) / cn, 3),
            "pass_rate": round(sum(1 for r in cat_results if r["composite_score"] >= 0.5) / cn, 3),
            "refusals": sum(1 for r in cat_results if r["is_refusal"]),
        }

    summary["category_breakdown"] = category_breakdown

    return {"summary": summary, "results": results}


def print_summary(summary: dict, worst_n: int = 0, results: list[dict] | None = None) -> None:
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

    # Per-category breakdown
    cat_breakdown = summary.get("category_breakdown", {})
    if cat_breakdown:
        print(f"\n{'-' * 60}")
        print("PER-CATEGORY BREAKDOWN")
        print(f"{'-' * 60}")
        print(f"  {'Category':<25} {'Pass%':>6} {'Comp':>6} {'KW':>6} {'Sem':>6} {'n':>3}")
        print(f"  {'-' * 25} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 3}")
        for cat, stats in sorted(cat_breakdown.items(), key=lambda x: x[1]["avg_composite"]):
            print(
                f"  {cat:<25} {stats['pass_rate']:>5.0%} "
                f"{stats['avg_composite']:>6.3f} {stats['avg_keyword']:>5.1%} "
                f"{stats['avg_semantic']:>6.3f} {stats['count']:>3}"
            )

    # Worst-N report
    if worst_n and results:
        print(f"\n{'-' * 60}")
        print(f"WORST {worst_n} QUESTIONS")
        print(f"{'-' * 60}")
        sorted_results = sorted(results, key=lambda r: r["composite_score"])
        for r in sorted_results[:worst_n]:
            status = "REFUSE" if r["is_refusal"] else "WEAK"
            print(f"\n  [{status}] {r['question']}")
            print(f"    Composite: {r['composite_score']:.3f} | "
                  f"KW: {r['keyword_score']:.0%} | Sem: {r['semantic_similarity']:.3f}")
            print(f"    Category: {r['category']}")
            # Truncate answer for display
            short_answer = r["rtfm_answer"][:120] + "..." if len(r["rtfm_answer"]) > 120 else r["rtfm_answer"]
            print(f"    Answer: {short_answer}")

    print(f"{'=' * 60}")


def save_history(summary: dict) -> None:
    """Append timestamped summary to history.jsonl for regression tracking."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in summary.items() if k != "category_breakdown"},
    }
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nResults appended to {HISTORY_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Run RTFM benchmark")
    parser.add_argument("--file", default=str(BENCHMARK_FILE), help="Q&A JSON file")
    parser.add_argument("--limit", type=int, help="Max questions to run")
    parser.add_argument("--source", help="Filter by source file")
    parser.add_argument("--output", help="Save full results to JSON file")
    parser.add_argument("--no-cache", action="store_true", help="Flush semantic cache before run")
    parser.add_argument("--worst", type=int, default=5, help="Show N worst questions (default: 5)")
    parser.add_argument("--no-history", action="store_true", help="Don't append to history.jsonl")
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

    benchmark = run_benchmark(
        qa_pairs,
        source_filter=args.source,
        limit=args.limit,
        no_cache=args.no_cache,
    )
    print_summary(benchmark["summary"], worst_n=args.worst, results=benchmark["results"])

    if not args.no_history:
        save_history(benchmark["summary"])

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(benchmark, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
