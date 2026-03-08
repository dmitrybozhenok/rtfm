"""Analyze saved multi-model benchmark results.

Usage:
    python evals/compare_models.py evals/model_comparison.json
    python evals/compare_models.py evals/model_comparison.json --markdown
    python evals/compare_models.py evals/model_comparison.json --heatmap
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_results(path: str) -> dict[str, dict]:
    """Load multi-model results JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def print_summary_table(all_runs: dict[str, dict], markdown: bool = False) -> None:
    """Print per-model summary table."""
    models = list(all_runs.keys())

    metrics = [
        ("Pass rate", "pass_rate", ".0%"),
        ("Avg keyword", "avg_keyword_score", ".1%"),
        ("Avg semantic", "avg_semantic_similarity", ".3f"),
        ("Avg composite", "avg_composite_score", ".3f"),
        ("Refusal rate", "refusal_rate", ".0%"),
        ("Avg latency (ms)", "avg_latency_ms", ".0f"),
    ]

    has_judge = any("avg_judge_score" in all_runs[m]["summary"] for m in models)
    if has_judge:
        metrics.append(("Avg judge", "avg_judge_score", ".2f"))

    if markdown:
        # Markdown table
        header = "| Metric |"
        sep = "| --- |"
        for m in models:
            header += f" {m} |"
            sep += " --- |"
        print(header)
        print(sep)
        for name, key, fmt in metrics:
            row = f"| {name} |"
            for m in models:
                v = all_runs[m]["summary"].get(key, 0)
                row += f" {v:{fmt}} |"
            print(row)
    else:
        col_w = max(16, max(len(m) for m in models) + 2)
        print(f"\n{'MODEL SUMMARY':^{20 + col_w * len(models)}}")
        print("=" * (20 + col_w * len(models)))
        header = f"  {'Metric':<20}"
        for m in models:
            header += f" {m:>{col_w}}"
        print(header)
        print(f"  {'-' * 20}" + f" {'-' * col_w}" * len(models))
        for name, key, fmt in metrics:
            row = f"  {name:<20}"
            for m in models:
                v = all_runs[m]["summary"].get(key, 0)
                row += f" {v:{fmt}:>{col_w}}"
            print(row)


def print_heatmap(all_runs: dict[str, dict]) -> None:
    """Print per-question pass/fail heatmap across models."""
    models = list(all_runs.keys())
    n = len(all_runs[models[0]]["results"])

    print(f"\nPER-QUESTION HEATMAP (P=pass, F=fail, R=refusal)")
    print("=" * 80)

    col_w = max(8, max(len(m) for m in models) + 1)
    header = f"  {'#':<4} {'Question':<40}"
    for m in models:
        header += f" {m:>{col_w}}"
    print(header)
    print(f"  {'-' * 4} {'-' * 40}" + f" {'-' * col_w}" * len(models))

    for i in range(n):
        q = all_runs[models[0]]["results"][i]["question"][:38]
        row = f"  {i+1:<4} {q:<40}"
        for m in models:
            r = all_runs[m]["results"][i]
            if r["is_refusal"]:
                marker = "R"
            elif r["composite_score"] >= 0.5:
                marker = "P"
            else:
                marker = "F"
            row += f" {marker:>{col_w}}"
        print(row)


def print_category_breakdown(all_runs: dict[str, dict]) -> None:
    """Print category breakdown per model."""
    models = list(all_runs.keys())

    print(f"\nCATEGORY BREAKDOWN (pass rate per model)")
    print("=" * 80)

    # Collect all categories
    all_cats = set()
    for m in models:
        cats = all_runs[m]["summary"].get("category_breakdown", {})
        all_cats.update(cats.keys())

    col_w = max(10, max(len(m) for m in models) + 1)
    header = f"  {'Category':<25}"
    for m in models:
        header += f" {m:>{col_w}}"
    header += f" {'Best':>{col_w}}"
    print(header)
    print(f"  {'-' * 25}" + f" {'-' * col_w}" * (len(models) + 1))

    for cat in sorted(all_cats):
        row = f"  {cat:<25}"
        rates = {}
        for m in models:
            cats = all_runs[m]["summary"].get("category_breakdown", {})
            rate = cats.get(cat, {}).get("pass_rate", 0)
            rates[m] = rate
            row += f" {rate:>{col_w}.0%}"
        best = max(rates, key=rates.get)
        row += f" {best:>{col_w}}"
        print(row)


def print_winner_report(all_runs: dict[str, dict]) -> None:
    """Show which model scored highest on each question."""
    models = list(all_runs.keys())
    n = len(all_runs[models[0]]["results"])

    wins = defaultdict(int)
    ties = 0

    print(f"\nPER-QUESTION WINNERS")
    print("=" * 80)

    for i in range(n):
        scores = {m: all_runs[m]["results"][i]["composite_score"] for m in models}
        max_score = max(scores.values())
        winners = [m for m, s in scores.items() if s == max_score]
        q = all_runs[models[0]]["results"][i]["question"][:60]

        if len(winners) == 1:
            wins[winners[0]] += 1
            print(f"  Q{i+1:>2}: {q:<60} -> {winners[0]} ({max_score:.3f})")
        else:
            ties += 1
            print(f"  Q{i+1:>2}: {q:<60} -> TIE ({max_score:.3f})")

    print(f"\n  TOTALS:")
    for m in models:
        print(f"    {m}: {wins[m]} wins")
    print(f"    Ties: {ties}")


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-model benchmark results")
    parser.add_argument("results_file", help="Path to multi-model results JSON")
    parser.add_argument("--markdown", action="store_true", help="Output summary as markdown table")
    parser.add_argument("--heatmap", action="store_true", help="Show per-question pass/fail heatmap")
    parser.add_argument("--categories", action="store_true", help="Show category breakdown per model")
    parser.add_argument("--winners", action="store_true", help="Show per-question winner report")
    parser.add_argument("--all", action="store_true", help="Show all reports")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)

    all_runs = load_results(args.results_file)
    models = list(all_runs.keys())
    print(f"Loaded results for {len(models)} models: {', '.join(models)}")

    show_all = args.all or not (args.heatmap or args.categories or args.winners)

    if show_all or not (args.heatmap or args.categories or args.winners):
        print_summary_table(all_runs, markdown=args.markdown)

    if args.heatmap or args.all:
        print_heatmap(all_runs)

    if args.categories or args.all:
        print_category_breakdown(all_runs)

    if args.winners or args.all:
        print_winner_report(all_runs)


if __name__ == "__main__":
    main()
