from __future__ import annotations

import json
import re
import requests
from typing import Dict, List, Any, Optional

from rapidfuzz import fuzz

OLLAMA_URL = "http://localhost:11434"



def _call_ollama(prompt: str, model: str, timeout: int = 120) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"  Warning: Ollama call failed ({e})")
        return ""


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



def _cosine_similarity(a: list, b: list) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _similarity_0_to_1(
    a: str, b: str, use_embeddings: bool = False, model: str = "llama3.1:8b"
) -> float:
    if not a and not b:
        return 1.0

    if use_embeddings:
        emb_a = _get_embedding(a)
        emb_b = _get_embedding(b)
        if emb_a and emb_b:
            return _cosine_similarity(emb_a, emb_b)

    return fuzz.ratio(a, b) / 100.0



def _generate_variations_with_llm(prompt: str, model: str, n: int = 5) -> List[str]:
    instruction = f"""You are a prompt rewriting assistant. Your job is to generate {n} different rephrasings of the prompt below.

Rules:
- Every rephrasing must have EXACTLY the same meaning and intent as the original
- Vary the wording, structure, formality, and phrasing — but never change what is being asked
- Do not add new instructions or remove existing ones

Respond ONLY with a valid JSON array of {n} strings. No extra text, no markdown, no explanation.

Original prompt:
\"\"\"{prompt}\"\"\"

JSON array:"""

    raw = _call_ollama(instruction, model)

    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r"\[[\s\S]*\]", clean)
        if not match:
            raise ValueError("No JSON array found in response")
        variations = json.loads(match.group())
        if not isinstance(variations, list):
            raise ValueError("Response is not a list")
        variations = [
            v for v in variations
            if isinstance(v, str) and v.strip() and v.strip() != prompt.strip()
        ]
        if len(variations) >= n:
            return variations[:n]
        print(f"  Warning: LLM returned {len(variations)} variations. Topping up.")
        fallbacks = _generate_variations_fallback(prompt, n - len(variations))
        return (variations + fallbacks)[:n]
    except Exception as e:
        print(f"  Warning: LLM variation failed ({e}). Using fallback.")
        return _generate_variations_fallback(prompt, n)


def _generate_variations_fallback(prompt: str, n: int = 5) -> List[str]:
    variations: List[str] = []

    def replace_ci(text, old, new):
        return re.sub(re.escape(old), new, text, flags=re.IGNORECASE)

    if prompt:
        variations.append(f"Please {prompt[0].lower() + prompt[1:]}")
    if prompt:
        variations.append(f"Can you {prompt[0].lower() + prompt[1:]}")

    v3 = prompt
    for old, new in [
        ("provide", "give"), ("assist", "help"),
        ("explain", "break down"), ("summarize", "sum up"),
        ("summarise", "sum up"), ("evaluate", "check"),
    ]:
        v3 = replace_ci(v3, old, new)
    variations.append(v3)

    v4 = replace_ci(prompt, "summarise", "summarize")
    v4 = replace_ci(v4, "summarize this", "give me a summary of this")
    variations.append(v4)

    parts = re.split(r"(?<=[.!?])\s+", prompt.strip())
    parts = [p for p in parts if p]
    if len(parts) > 1:
        variations.append(" ".join(parts[1:] + [parts[0]]))
    else:
        variations.append(f"{prompt} Please be concise.")

    seen = set()
    unique = []
    for v in variations:
        key = v.strip()
        if key and key not in seen and key != prompt.strip():
            seen.add(key)
            unique.append(v)

    i = 1
    while len(unique) < n:
        unique.append(f"{prompt.strip()} (variation {i})")
        i += 1

    return unique[:n]




def test_sensitivity(
    prompt_text: str,
    model_name: str,
    num_variations: int = 5,
    use_embeddings: bool = True,
) -> Dict[str, Any]:
    """
    Test how robust a prompt is by generating semantically equivalent
    rephrasings and measuring how much the model's output changes.
    """
    if not prompt_text or not prompt_text.strip():
        raise ValueError("prompt_text must be a non-empty string.")
    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    print("  Running original prompt...")
    original_output = _call_ollama(prompt_text, model_name)

    print(f"  Generating {num_variations} variations with LLM...")
    variations = _generate_variations_with_llm(prompt_text, model_name, n=num_variations)


    if not variations:
        return {
            "sensitivity_score": 0.0,
            "average_similarity": 1.0,
            "variations": [],
            "most_stable_variation": None,
            "least_stable_variation": None,
        }

    variation_results: List[Dict[str, Any]] = []
    for i, variation_text in enumerate(variations):
        print(f"  Running variation {i + 1}/{len(variations)}...")
        variation_output = _call_ollama(variation_text, model_name)
        similarity_score = _similarity_0_to_1(
            original_output,
            variation_output,
            use_embeddings=use_embeddings,
            model=model_name,
        )
        variation_results.append({
            "variation_text": variation_text,
            "output": variation_output,
            "similarity_score": similarity_score,
        })

    similarities = [v["similarity_score"] for v in variation_results]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
    sensitivity_score = 1.0 - avg_similarity

    most_stable  = max(variation_results, key=lambda x: x["similarity_score"])
    least_stable = min(variation_results, key=lambda x: x["similarity_score"])

    return {
        "sensitivity_score":      round(sensitivity_score, 4),
        "average_similarity":     round(avg_similarity, 4),
        "variations":             variation_results,
        "most_stable_variation":  most_stable,
        "least_stable_variation": least_stable,
    }