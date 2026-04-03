import json
import math
import re
from typing import Dict, List, Tuple

import numpy as np
import spacy
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Setup:
# pip install transformers torch spacy numpy
# python -m spacy download en_core_web_sm

# Run with:
# python uncertainty_scorer.py

# Configuration constants
MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128
SIMILARITY_THRESHOLD = 0.5

# Load Hugging Face model & spaCy once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()
nlp = spacy.load("en_core_web_sm")

# Text cleaning helpers
def normalize_text(text: str) -> str:
    """
    Lowercase and simplify spacing/punctuation for loose matching.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def token_set(text: str) -> set:
    """
    Convert text into a set of normalized words.
    """
    return set(normalize_text(text).split())


def jaccard_similarity(a: str, b: str) -> float:
    """
    Word-overlap similarity between two strings.
    """
    set_a = token_set(a)
    set_b = token_set(b)

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)

# Phrase extraction & grounding
def extract_candidate_phrases(text: str) -> List[str]:
    """
    Extract noun phrases and named entities from text.
    Remove duplicates while preserving order.
    """
    doc = nlp(text)
    phrases = []

    for chunk in doc.noun_chunks:
        cleaned = chunk.text.strip()
        if cleaned:
            phrases.append(cleaned)

    for ent in doc.ents:
        cleaned = ent.text.strip()
        if cleaned:
            phrases.append(cleaned)

    seen = set()
    unique_phrases = []

    for phrase in phrases:
        norm = normalize_text(phrase)
        if norm and norm not in seen:
            seen.add(norm)
            unique_phrases.append(phrase)

    return unique_phrases


def build_prompt_reference_list(prompt_text: str) -> List[str]:
    """
    Build a list of prompt phrases to compare against output phrases.
    """
    doc = nlp(prompt_text)
    refs = []

    for chunk in doc.noun_chunks:
        cleaned = chunk.text.strip()
        if cleaned:
            refs.append(cleaned)

    for ent in doc.ents:
        cleaned = ent.text.strip()
        if cleaned:
            refs.append(cleaned)

    refs.append(prompt_text)

    seen = set()
    unique_refs = []

    for ref in refs:
        norm = normalize_text(ref)
        if norm and norm not in seen:
            seen.add(norm)
            unique_refs.append(ref)

    return unique_refs


def is_phrase_grounded(phrase: str, prompt_text: str, prompt_refs: List[str]) -> bool:
    """
    Decide whether an output phrase is grounded in the prompt.

    Uses:
    1. Exact normalized substring match
    2. Jaccard similarity against prompt phrases
    3. spaCy similarity as a loose semantic check
    """
    norm_phrase = normalize_text(phrase)
    norm_prompt = normalize_text(prompt_text)

    if not norm_phrase:
        return True

    # direct appearance in prompt
    if norm_phrase in norm_prompt:
        return True

    # similarity to extracted prompt phrases
    for ref in prompt_refs:
        if jaccard_similarity(phrase, ref) >= SIMILARITY_THRESHOLD:
            return True

    # optional: loose semantic similarity
    phrase_doc = nlp(phrase)
    for ref in prompt_refs:
        ref_doc = nlp(ref)
        try:
            if phrase_doc.similarity(ref_doc) >= 0.75:
                return True
        except Exception:
            pass

    return False


def compute_hallucination_ratio(
    prompt_text: str, output_text: str
) -> Tuple[float, List[str], List[str]]:
    """
    Compute hallucination ratio based on unsupported output phrases.

    Returns:
        hallucination_ratio
        all_output_phrases
        unsupported_phrases
    """
    output_phrases = extract_candidate_phrases(output_text)

    if not output_phrases:
        return 0.0, [], []

    prompt_refs = build_prompt_reference_list(prompt_text)

    unsupported_phrases = []
    for phrase in output_phrases:
        if not is_phrase_grounded(phrase, prompt_text, prompt_refs):
            unsupported_phrases.append(phrase)

    ratio = len(unsupported_phrases) / len(output_phrases)
    return ratio, output_phrases, unsupported_phrases

# Token uncertainty scoring
def generate_with_scores(prompt_text: str) -> Tuple[str, List[float], List[str]]:
    """
    Run FLAN-T5 on the prompt with output_scores=True.

    Returns:
        generated_text
        entropies_per_generated_token
        decoded_generated_tokens
    """
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_OUTPUT_LENGTH,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )

    generated_ids = generation_output.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # generation_output.scores is a list of logits tensors, one per generated step
    entropies = []
    for step_scores in generation_output.scores:
        probs = torch.softmax(step_scores[0], dim=-1).cpu().numpy()
        entropy = -np.sum(probs * np.log2(probs + 1e-12))  # 1e-12 to avoid problems w/ log2(0)
        entropies.append(float(entropy))

    # Extract only the newly generated tokens, not the input prompt tokens
    input_length = inputs["input_ids"].shape[1]
    new_token_ids = generated_ids[input_length:] if len(generated_ids) > input_length else []

    decoded_tokens = [
        tokenizer.decode([token_id], skip_special_tokens=False)
        for token_id in new_token_ids
    ]

    # Sometimes the number of scores and number of extracted tokens may differ slightly.
    # Trim to the shorter length so they stay aligned.
    min_len = min(len(entropies), len(decoded_tokens))
    entropies = entropies[:min_len]
    decoded_tokens = decoded_tokens[:min_len]

    return generated_text, entropies, decoded_tokens


def compute_perplexity(entropies: List[float]) -> float:
    """
    Perplexity = 2^(average entropy)
    """
    if not entropies:
        return 1.0

    avg_entropy = float(np.mean(entropies))
    return float(2 ** avg_entropy)


def normalize_uncertainty_score(perplexity: float) -> float:
    """
    Normalize perplexity into a 0..1 score.

    Uses:
        score = 1 - exp(-perplexity / 10)

    This keeps the score bounded between 0 and 1.
    """
    score = 1.0 - math.exp(-perplexity / 10.0)
    return float(max(0.0, min(1.0, score)))

# High uncertainty spans
def find_high_uncertainty_spans(
    entropies: List[float],
    decoded_tokens: List[str],
    top_k: int = 3,
    window_size: int = 3
) -> List[str]:
    """
    Return short spans around the highest-entropy generated tokens.
    """
    if not entropies or not decoded_tokens:
        return []

    candidate_spans = []

    for idx, entropy in enumerate(entropies):
        start = max(0, idx - window_size)
        end = min(len(decoded_tokens), idx + window_size + 1)

        span_tokens = decoded_tokens[start:end]
        span_text = "".join(span_tokens)
        span_text = span_text.replace("<pad>", "").replace("</s>", "").strip()

        if span_text:
            candidate_spans.append((entropy, idx, span_text))

    candidate_spans.sort(key=lambda x: x[0], reverse=True)

    selected_spans = []
    selected_indices = []

    for _, idx, span_text in candidate_spans:
        if all(abs(idx - prev_idx) > window_size for prev_idx in selected_indices):
            selected_spans.append(span_text)
            selected_indices.append(idx)

        if len(selected_spans) >= top_k:
            break

    return selected_spans

# Main function
def score_uncertainty(prompt_text: str, output_text: str) -> Dict:
    """
    Returns a dictionary with:
    - uncertainty_score
    - perplexity
    - hallucination_ratio
    - high_uncertainty_spans

    Notes:
    - uncertainty_score is based only on model entropy/perplexity
    - hallucination_ratio is computed separately from prompt grounding
    - high_uncertainty_spans come from the model's generated output
    """
    generated_text, entropies, decoded_tokens = generate_with_scores(prompt_text)

    perplexity = compute_perplexity(entropies)
    uncertainty_score = normalize_uncertainty_score(perplexity)

    hallucination_ratio, _, _ = compute_hallucination_ratio(prompt_text, output_text)

    high_uncertainty_spans = find_high_uncertainty_spans(
        entropies=entropies,
        decoded_tokens=decoded_tokens,
        top_k=3,
        window_size=3
    )

    return {
        "uncertainty_score": round(uncertainty_score, 4),
        "perplexity": round(perplexity, 4),
        "hallucination_ratio": round(hallucination_ratio, 4),
        "high_uncertainty_spans": high_uncertainty_spans
    }

# Test block
if __name__ == "__main__":
    test_cases = [
        {
            "name": "1. Clean grounded answer",
            "prompt": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "note": "Expected: low hallucination_ratio"
        },
        {
            "name": "2. Hallucinated extra detail",
            "prompt": "What is the capital of France?",
            "output": "The capital of France is Paris and it is ruled by King Napoleon III in 2026.",
            "note": "Expected: higher hallucination_ratio"
        },
        {
            "name": "3. Empty output",
            "prompt": "What is the capital of France?",
            "output": "",
            "note": "Expected: hallucination_ratio = 0.0 or no crash"
        }
    ]

    for case in test_cases:
        print("\n" + "=" * 70)
        print(case["name"])
        print(case["note"])
        print("PROMPT:", case["prompt"])
        print("OUTPUT:", case["output"])

        try:
            result = score_uncertainty(case["prompt"], case["output"])
            print(json.dumps(result, indent=2))
        except Exception as e:
            print("ERROR:", str(e))

