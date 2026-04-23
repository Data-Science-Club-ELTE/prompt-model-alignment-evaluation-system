from __future__ import annotations

import re
import json
import requests
import numpy as np
from typing import Optional, List
from rapidfuzz import fuzz

OLLAMA_URL = "http://localhost:11434"

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"  Warning: CrossEncoder unavailable ({e})")
    return _cross_encoder


def _call_ollama(prompt: str, model: str = "llama3.1:8b", timeout: int = 120) -> str:
    """Send a prompt to Ollama and return the response text."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"  Warning: Ollama call failed ({e})")
        return ""


def _is_valid_json(text: str) -> bool:
    try:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        json.loads(match.group(1) if match else text)
        return True
    except Exception:
        return False


def _count_sentences(text: str) -> int:
    return len(re.findall(r"[.!?]+", text))


def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)



def score_instruction_following(prompt: str, output: str, model: str = "llama3.1:8b") -> dict:
    """LLM-as-judge: check whether the output followed all prompt instructions."""
    judge_prompt = f"""You are a strict evaluation assistant. Your job is to check whether a model output correctly followed all instructions in a given prompt.

Original prompt:
\"\"\"
{prompt}
\"\"\"

Model output:
\"\"\"
{output}
\"\"\"

Carefully read the prompt and identify every explicit instruction (format, length, tone, content rules, etc.).
Then check whether the output followed each one.

Respond ONLY with a valid JSON object in exactly this format, no extra text:
{{
  "score": <float between 0.0 and 1.0>,
  "passed": ["instruction 1", "instruction 2"],
  "failed": ["instruction 3"],
  "reasoning": "one sentence explanation"
}}"""

    raw = _call_ollama(judge_prompt, model)

    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r"\{[\s\S]*\}", clean)
        if not match:
            raise ValueError("No JSON object found in response")
        data = json.loads(match.group())
        score = float(data.get("score", 0.75))
        score = max(0.0, min(1.0, score))
        passed = data.get("passed", [])
        failed = data.get("failed", [])
        reasoning = data.get("reasoning", "")
        note = f"Passed: {passed} | Failed: {failed} | {reasoning}"
        return {"score": round(score, 2), "note": note, "rules_found": passed + failed}
    except Exception as e:
        print(f"  Warning: LLM judge parse failed ({e}), falling back to heuristics")
        return _score_instruction_following_heuristic(prompt, output)


def _score_instruction_following_heuristic(prompt: str, output: str) -> dict:
    """Fallback regex heuristics if LLM judge fails."""
    prompt_lower = prompt.lower()
    rules_found, passed, failed = [], [], []

    checks = [
        (r"\bjson\b",         lambda o: _is_valid_json(o),                    "respond in JSON"),
        (r"\bbullet.?point",  lambda o: bool(re.search(r"^[-•*]", o, re.M)), "use bullet points"),
        (r"\bnumbered.?list", lambda o: bool(re.search(r"^\d+\.", o, re.M)), "use numbered list"),
        (r"\bmarkdown\b",     lambda o: bool(re.search(r"[#*_`]", o)),       "use markdown"),
        (r"\bsummar",         lambda o: len(o.split()) < 200,                 "summarise"),
        (r"\btranslat",       lambda o: len(o) > 0,                           "translate"),
    ]

    for pattern, check_fn, label in checks:
        if re.search(pattern, prompt_lower):
            rules_found.append(label)
            try:
                result = check_fn(output)
                (passed if result else failed).append(label)
            except Exception:
                failed.append(label)

    word_match = re.search(r"under (\d+) word", prompt_lower)
    if word_match:
        rules_found.append("under N words")
        limit = int(word_match.group(1))
        (passed if len(output.split()) < limit else failed).append("under N words")

    sentence_match = re.search(r"(\d+) sentence", prompt_lower)
    if sentence_match:
        rules_found.append("N sentences")
        target = int(sentence_match.group(1))
        (passed if abs(_count_sentences(output) - target) <= 1 else failed).append("N sentences")

    if not rules_found:
        return {"score": 0.75, "note": "No explicit instructions detected", "rules_found": []}

    score = len(passed) / len(rules_found)
    note = f"Passed: {passed} | Failed: {failed}"
    return {"score": round(score, 2), "note": note, "rules_found": rules_found}


def score_format_match(expected: Optional[str], output: str) -> dict:
    if not expected:
        return {"score": 0.75, "note": "No expected output provided — skipped"}

    checks = {
        "is_json":            _is_valid_json(expected) and _is_valid_json(output),
        "length_similar":     abs(len(output.split()) - len(expected.split())) < len(expected.split()) * 0.5,
        "has_bullets":        bool(re.search(r"^[-•*]", expected, re.M)) == bool(re.search(r"^[-•*]", output, re.M)),
        "has_numbers":        bool(re.search(r"^\d+\.", expected, re.M)) == bool(re.search(r"^\d+\.", output, re.M)),
        "has_headers":        bool(re.search(r"^#+", expected, re.M)) == bool(re.search(r"^#+", output, re.M)),
        "line_count_similar": abs(len(output.splitlines()) - len(expected.splitlines())) < max(3, len(expected.splitlines()) * 0.4),
    }

    passed = [k for k, v in checks.items() if v]
    score = len(passed) / len(checks)
    note = f"Passed: {passed} | Failed: {[k for k, v in checks.items() if not v]}"
    return {"score": round(score, 2), "note": note}


def score_semantic_accuracy(expected_text: Optional[str], output_text: str) -> dict:
    """Cross-encoder semantic similarity between expected and actual output."""
    if not expected_text:
        return {"score": 0.75, "note": "No expected output provided — skipped"}
    if not output_text:
        return {"score": 0.0, "note": "Model produced no output"}

    encoder = _get_cross_encoder()
    if encoder is None:
        return {"score": 0.75, "note": "Cross-encoder unavailable — skipped"}

    try:
        raw_score = encoder.predict([(expected_text, output_text)])
        score = float(1 / (1 + np.exp(-raw_score[0])))
        score = round(min(max(score, 0.0), 1.0), 2)
        return {"score": score, "note": f"Cross-encoder similarity: {score:.3f}"}
    except Exception as e:
        print(f"  Warning: Cross-encoder failed ({e})")
        return {"score": 0.75, "note": f"Cross-encoder failed: {e}"}


def _get_embedding(text: str, model: str = "nomic-embed-text") -> Optional[list]:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"  Warning: Embedding call failed ({e})")
        return None


def score_consistency(outputs: List[str]) -> dict:
    """Embedding cosine similarity across multiple runs."""
    if len(outputs) < 2:
        return {"score": 1.0, "note": "Only one run — consistency not measured"}

    embeddings = [_get_embedding(o) for o in outputs]

    if all(e is not None for e in embeddings):
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        score = sum(similarities) / len(similarities) if similarities else 1.0
        note = f"Embedding cosine similarity across {len(outputs)} runs: {score:.2f}"
        return {"score": round(score, 2), "note": note}

    print("  Warning: Embeddings unavailable, falling back to fuzzy string matching")
    similarities = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            sim = fuzz.ratio(outputs[i], outputs[j]) / 100
            similarities.append(sim)
    score = sum(similarities) / len(similarities) if similarities else 1.0
    note = f"Fuzzy string similarity across {len(outputs)} runs: {score:.2f} (fallback)"
    return {"score": round(score, 2), "note": note}



def score_edge_cases(test_results: list) -> dict:
    """test_results: list of {"input": str, "output": str, "passed": bool, "rule": str}"""
    if not test_results:
        return {"score": 0.75, "note": "No test cases provided — skipped"}

    passed = [t for t in test_results if t["passed"]]
    score = len(passed) / len(test_results)
    failed_rules = [t["rule"] for t in test_results if not t["passed"]]
    note = f"{len(passed)}/{len(test_results)} test cases passed. Failed: {failed_rules or 'none'}"
    return {"score": round(score, 2), "note": note}



WEIGHTS = {
    "instruction_following": 0.30,
    "format_match":          0.25,
    "semantic_accuracy":     0.25,
    "consistency":           0.10,
    "edge_cases":            0.10,
}


def compute_final_score(scores: dict) -> float:
    total = 0.0
    for k, w in WEIGHTS.items():
        dim = scores.get(k)
        if dim and isinstance(dim, dict):
            total += dim.get("score", 0.0) * w
    return round(total * 100, 1)


def score_label(score: float) -> str:
    if score >= 85: return "Well Aligned ✅"
    if score >= 65: return "Mostly Aligned 🟡"
    if score >= 40: return "Partially Aligned 🟠"
    return "Misaligned ❌"