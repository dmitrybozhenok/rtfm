"""Negative questions eval — test that the system refuses unanswerable questions.

Checks two things:
1. Questions NOT in the docs → system should refuse (say "I don't have enough info")
2. Questions IN the docs → system should answer (not falsely refuse)

Catches false confidence (hallucinating answers) and false refusals.

Usage:
    python evals/negative_eval.py                    # run all
    python evals/negative_eval.py --limit 5          # first 5 only
    python evals/negative_eval.py --model qwen2.5:7b # specify model
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rtfm.config import settings
from rtfm.retrieval.rag import ask

NEGATIVE_FILE = Path(__file__).parent / "negative_questions.json"

# Refusal patterns — same as run_benchmark.py
_REFUSAL_PATTERNS = [
    re.compile(r"don'?t have enough information", re.IGNORECASE),
    re.compile(r"cannot (find|answer|determine)", re.IGNORECASE),
    re.compile(r"no relevant (documentation|information|context)", re.IGNORECASE),
    re.compile(r"not (mentioned|covered|discussed) in the (documentation|context)", re.IGNORECASE),
    re.compile(r"the (documentation|context) does not (contain|mention|cover)", re.IGNORECASE),
    re.compile(r"(isn't|is not) (covered|mentioned|discussed) in", re.IGNORECASE),
    re.compile(r"(no|without) (sufficient|enough) (information|context|data)", re.IGNORECASE),
]


def _is_refusal(answer: str) -> bool:
    """Detect if the answer is a refusal."""
    for pattern in _REFUSAL_PATTERNS:
        match = pattern.search(answer)
        if match:
            pos_ratio = match.start() / max(len(answer), 1)
            if pos_ratio < 0.4:
                return True
    return False


def run_negative_eval(
    questions: list[dict],
    model: str | None = None,
    limit: int | None = None,
) -> dict:
    """Run negative question eval.

    For each question, checks if the system's refusal/answer behavior
    matches the expected behavior (should_refuse).
    """
    llm_model = model or settings.llm_model

    if limit:
        questions = questions[:limit]

    print(f"\nNegative Questions Eval: {len(questions)} questions (model: {llm_model})")
    print("=" * 60)

    results = []
    correct = 0
    false_answers = 0  # Should have refused but answered
    false_refusals = 0  # Should have answered but refused

    for i, q in enumerate(questions, 1):
        question = q["question"]
        should_refuse = q["should_refuse"]
        category = q.get("category", "")

        print(f"\n[{i}/{len(questions)}] {question}")
        print(f"  Expected: {'REFUSE' if should_refuse else 'ANSWER'}")

        start = time.time()
        rag_result = ask(question, model_override=llm_model)
        elapsed_ms = (time.time() - start) * 1000

        answer = rag_result["answer"]
        did_refuse = _is_refusal(answer)

        if should_refuse and did_refuse:
            verdict = "CORRECT_REFUSAL"
            correct += 1
        elif not should_refuse and not did_refuse:
            verdict = "CORRECT_ANSWER"
            correct += 1
        elif should_refuse and not did_refuse:
            verdict = "FALSE_ANSWER"
            false_answers += 1
        else:
            verdict = "FALSE_REFUSAL"
            false_refusals += 1

        short_answer = answer[:120] + "..." if len(answer) > 120 else answer
        print(f"  Got: {'REFUSE' if did_refuse else 'ANSWER'} -> {verdict}")
        print(f"  Answer: {short_answer}")

        results.append({
            "question": question,
            "should_refuse": should_refuse,
            "did_refuse": did_refuse,
            "verdict": verdict,
            "answer": answer,
            "category": category,
            "reason": q.get("reason", ""),
            "latency_ms": round(elapsed_ms, 1),
        })

    n = len(questions)
    n_should_refuse = sum(1 for q in questions if q["should_refuse"])
    n_should_answer = n - n_should_refuse

    summary = {
        "total_questions": n,
        "accuracy": round(correct / n, 3) if n else 0,
        "false_answer_rate": round(false_answers / n_should_refuse, 3) if n_should_refuse else 0,
        "false_refusal_rate": round(false_refusals / n_should_answer, 3) if n_should_answer else 0,
        "correct": correct,
        "false_answers": false_answers,
        "false_refusals": false_refusals,
        "questions_should_refuse": n_should_refuse,
        "questions_should_answer": n_should_answer,
        "model": llm_model,
    }

    print(f"\n{'=' * 60}")
    print("NEGATIVE EVAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Questions:           {n} ({n_should_refuse} should refuse, {n_should_answer} should answer)")
    print(f"  Accuracy:            {summary['accuracy']:.0%}")
    print(f"  False answer rate:   {summary['false_answer_rate']:.0%}  (hallucinated answer when should refuse)")
    print(f"  False refusal rate:  {summary['false_refusal_rate']:.0%}  (refused when should answer)")
    print(f"  Model:               {summary['model']}")

    # Show failures
    failures = [r for r in results if r["verdict"] in ("FALSE_ANSWER", "FALSE_REFUSAL")]
    if failures:
        print(f"\n{'-' * 60}")
        print(f"FAILURES ({len(failures)})")
        print(f"{'-' * 60}")
        for r in failures:
            print(f"\n  [{r['verdict']}] {r['question']}")
            print(f"    Category: {r['category']}")
            print(f"    Reason: {r['reason']}")
            short = r["answer"][:150] + "..." if len(r["answer"]) > 150 else r["answer"]
            print(f"    Answer: {short}")

    print(f"{'=' * 60}")

    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Run negative questions eval")
    parser.add_argument("--file", default=str(NEGATIVE_FILE))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    questions = json.loads(Path(args.file).read_text(encoding="utf-8"))
    result = run_negative_eval(questions, model=args.model, limit=args.limit)

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
