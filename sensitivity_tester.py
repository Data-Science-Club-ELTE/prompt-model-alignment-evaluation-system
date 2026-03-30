import json
import random
import re
from typing import Dict, List, Any

import requests
from rapidfuzz import fuzz


def _call_ollama(prompt: str, model: str, timeout: int = 120) -> str:
    """
    Send a prompt to a local Ollama instance and return generated text.
    Uses the non-streaming generate endpoint for simplicity.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama: {e}")
        return ""


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter keeping punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _replace_phrase_case_insensitive(text: str, old: str, new: str) -> str:
    return re.sub(re.escape(old), new, text, flags=re.IGNORECASE)


def _generate_variations(prompt: str, num_variations: int = 5) -> List[str]:
    """
    Generate prompt variations with simple string manipulation only.
    Ensures at least 5 variations.
    """
    target_count = max(5, num_variations)
    variations: List[str] = []

    # 1) Prefix politeness
    v1 = f"Please {prompt[0].lower() + prompt[1:]}" if prompt else prompt
    variations.append(v1)

    # 2) Prefix task framing
    v2 = f"Can you {prompt[0].lower() + prompt[1:]}" if prompt else prompt
    variations.append(v2)

    # 3) Summarise/summarize/summary swaps
    v3 = prompt
    v3 = _replace_phrase_case_insensitive(v3, "summarise this", "give me a summary of this")
    v3 = _replace_phrase_case_insensitive(v3, "summarize this", "give me a summary of this")
    v3 = _replace_phrase_case_insensitive(v3, "summarise", "summarize")
    variations.append(v3)

    # 4) Casual synonym swaps
    v4 = prompt
    replacements = [
        ("provide", "give"),
        ("assist", "help"),
        ("explain", "break down"),
        ("summarize", "sum up"),
        ("summarise", "sum up"),
        ("evaluate", "check"),
    ]
    for old, new in replacements:
        v4 = _replace_phrase_case_insensitive(v4, old, new)
    variations.append(v4)

    # 5) Reorder sentences (if multiple sentences)
    sentences = _split_sentences(prompt)
    if len(sentences) > 1:
        reordered = " ".join(sentences[1:] + [sentences[0]])
    else:
        reordered = f"{prompt} Make sure to keep it clear."
    variations.append(reordered)

    # Additional variations if requested > 5
    while len(variations) < target_count:
        base = random.choice(variations) if variations else prompt
        mutated = base
        # Toggle ending politeness
        if not mutated.lower().endswith("please."):
            mutated = mutated.rstrip() + " Please."
        else:
            mutated = mutated[:-7].rstrip() + "."
        variations.append(mutated)

    seen = set()
    unique_variations = []
    for v in variations:
        key = v.strip()
        if key and key not in seen and key != prompt.strip():
            seen.add(key)
            unique_variations.append(key)

    i = 1
    while len(unique_variations) < target_count:
        unique_variations.append(f"{prompt.strip()} (variation {i})")
        i += 1

    return unique_variations[:target_count]


def _similarity_0_to_1(a: str, b: str) -> float:
    """rapidfuzz.fuzz.ratio returns 0..100, we need 0.0 to 1.0"""
    if not a and not b:
        return 1.0
    return fuzz.ratio(a, b) / 100.0


def test_sensitivity(prompt_text: str, model_name: str, num_variations: int = 5) -> Dict[str, Any]:
    """
    Test how fragile or robust a prompt is by comparing wording variations.
    """
    if not prompt_text or not prompt_text.strip():
        raise ValueError("prompt_text must be a non-empty string.")
    if not model_name or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    # Get baseline output
    original_output = _call_ollama(prompt_text, model_name)
    
    # Generate variations
    variations = _generate_variations(prompt_text, num_variations=num_variations)

    variation_results: List[Dict[str, Any]] = []
    
    # Run each variation and calculate similarity
    for variation_text in variations:
        variation_output = _call_ollama(variation_text, model_name)
        similarity_score = _similarity_0_to_1(original_output, variation_output)

        variation_results.append({
            "variation_text": variation_text,
            "output": variation_output,
            "similarity_score": similarity_score,
        })

    # Calculate sensitivity score
    similarities = [v["similarity_score"] for v in variation_results]
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    sensitivity_score = 1.0 - avg_similarity

    most_stable = max(variation_results, key=lambda x: x["similarity_score"]) if variation_results else None
    least_stable = min(variation_results, key=lambda x: x["similarity_score"]) if variation_results else None

    return {
        "sensitivity_score": sensitivity_score,
        "variations": variation_results,
        "most_stable_variation": most_stable,
        "least_stable_variation": least_stable,
    }


if __name__ == "__main__":
    # Test block 
    sample_prompt = """Summarise this research text about prompt interpretation into 3 practical takeaways:

# Research: How Models Interpret Prompts
This work investigates how large language models (LLMs) parse prompts internally and which factors influence interpretation:
- Tokenization effects on meaning  
- Instruction vs. context interaction  
- Prompt formatting patterns (roles, lists, delimiters)  
- Ambiguity and uncertainty signals  

## Why This Matters
In real-world systems, unstable prompts can lead to:
- Inconsistent answers  
- Format violations  
- Hallucinated content  
- Brittle performance under small wording changes  

A measurable evaluation pipeline improves reliability, testing, and debugging in production AI workflows."""

    # Using a common default model name for Ollama
    sample_model = "llama3.2" 

    print("Running sensitivity test (this will call the local Ollama API a few times)...")
    result = test_sensitivity(sample_prompt, sample_model, num_variations=5)
    
    # Print the result as formatted JSON
    print("\n" + "="*50 + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))