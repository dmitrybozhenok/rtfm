"""LLM-as-judge scoring for benchmark answers.

Sends (question, expected_answer, generated_answer) to a local LLM and gets
a factual correctness rating on a 1-5 scale. Catches factually wrong answers
that semantic similarity misses.

Usage:
    from evals.llm_judge import judge_answer, judge_batch

    result = judge_answer(question, expected, generated)
    # {"score": 4, "justification": "..."}

    results = judge_batch(qa_results, model="qwen2.5:7b")
"""

import hashlib
import json
from pathlib import Path

from openai import OpenAI

JUDGE_PROMPT = """You are an impartial judge evaluating a documentation Q&A system.

Given a question, the expected (reference) answer, and the system's generated answer, rate the FACTUAL CORRECTNESS of the generated answer on a scale of 1-5:

5 = Perfectly correct — all key facts match the expected answer
4 = Mostly correct — minor omissions but no wrong facts
3 = Partially correct — some correct facts but also missing important ones
2 = Mostly wrong — contains significant factual errors or answers a different question
1 = Completely wrong — contradicts the expected answer or is a false refusal

Important guidelines:
- Focus on FACTUAL CORRECTNESS, not style or verbosity
- A shorter answer that is factually correct should score high
- An answer that is correct but adds extra (correct) detail should score high
- An answer that refuses when it should have answered scores 1-2
- An answer that gets the key technical details wrong scores 1-2
- Paraphrasing is fine as long as the facts are correct

Respond with ONLY a JSON object in this format:
{"score": <1-5>, "justification": "<brief explanation>"}"""

CACHE_DIR = Path(__file__).parent / ".judge_cache"


def _cache_key(question: str, expected: str, generated: str, model: str) -> str:
    """Deterministic cache key from inputs."""
    content = f"{model}|{question}|{expected}|{generated}"
    return hashlib.sha256(content.encode()).hexdigest()


def _load_cached(key: str) -> dict | None:
    """Load cached judgment if it exists."""
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_cached(key: str, result: dict) -> None:
    """Save judgment to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{key}.json"
    cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def judge_answer(
    question: str,
    expected: str,
    generated: str,
    model: str = "qwen2.5:7b",
    ollama_base_url: str = "http://localhost:11434/v1",
) -> dict:
    """Score a single answer using LLM-as-judge.

    Returns:
        {"score": int (1-5), "justification": str}
    """
    cache_key = _cache_key(question, expected, generated, model)
    cached = _load_cached(cache_key)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    client = OpenAI(base_url=ollama_base_url, api_key="ollama")

    user_content = f"""Question: {question}

Expected answer: {expected}

Generated answer: {generated}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=256,
    )

    raw = response.choices[0].message.content.strip()
    result = _parse_judgment(raw)

    _save_cached(cache_key, result)
    result["from_cache"] = False
    return result


def _parse_judgment(raw: str) -> dict:
    """Parse LLM judge response into structured result."""
    # Try direct JSON parse
    try:
        parsed = json.loads(raw)
        score = int(parsed.get("score", 0))
        if 1 <= score <= 5:
            return {
                "score": score,
                "justification": parsed.get("justification", ""),
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try extracting JSON from markdown code block
    import re

    json_match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            score = int(parsed.get("score", 0))
            if 1 <= score <= 5:
                return {
                    "score": score,
                    "justification": parsed.get("justification", ""),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: look for a bare number 1-5
    number_match = re.search(r"\b([1-5])\b", raw)
    if number_match:
        return {
            "score": int(number_match.group(1)),
            "justification": raw[:200],
        }

    # Could not parse — return score 0 to flag the issue
    return {
        "score": 0,
        "justification": f"Failed to parse judge response: {raw[:200]}",
    }


def judge_batch(
    results: list[dict],
    model: str = "qwen2.5:7b",
    ollama_base_url: str = "http://localhost:11434/v1",
) -> list[dict]:
    """Score a batch of benchmark results.

    Args:
        results: List of benchmark result dicts with question, expected_answer, rtfm_answer
        model: Ollama model to use as judge
        ollama_base_url: Ollama API base URL

    Returns:
        List of judgment dicts with score, justification, from_cache
    """
    judgments = []
    cache_hits = 0

    for i, r in enumerate(results, 1):
        judgment = judge_answer(
            question=r["question"],
            expected=r["expected_answer"],
            generated=r["rtfm_answer"],
            model=model,
            ollama_base_url=ollama_base_url,
        )
        if judgment.get("from_cache"):
            cache_hits += 1
        judgments.append(judgment)

        print(
            f"  Judge [{i}/{len(results)}] score={judgment['score']}/5"
            f"{' (cached)' if judgment.get('from_cache') else ''}"
        )

    print(f"  Judge cache hits: {cache_hits}/{len(results)}")
    return judgments
