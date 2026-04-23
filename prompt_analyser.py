from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Set, Tuple

import tiktoken
import spacy

try:
    _NLP = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' not found.\n"
        "Run:  python -m spacy download en_core_web_sm"
    )



_TYPE_VOCAB: Dict[str, Set[str]] = {
    "structured": {
        "table", "list", "format", "schema", "json", "xml", "csv", "template",
        "structure", "outline", "columns", "rows", "fields", "headers",
        "numbered", "bullet", "enumerate", "categorise", "categorize",
        "organise", "organize", "sort", "rank", "compare",
    },
    "creative": {
        "write", "story", "poem", "imagine", "create", "invent", "fiction",
        "narrative", "character", "plot", "scene", "describe", "creative",
        "metaphor", "analogy", "dream", "fantasy", "song", "verse", "prose",
        "essay", "dialogue", "monologue", "vivid", "evocative",
    },
    "factual": {
        "what", "when", "where", "who", "why", "how", "define", "explain",
        "summarise", "summarize", "history", "science", "fact", "research",
        "evidence", "data", "statistic", "calculate", "convert", "translate",
        "identify", "describe", "information", "detail", "background",
    },
    "instructional": {
        "do", "make", "build", "implement", "code", "fix", "debug", "refactor",
        "generate", "produce", "step", "steps", "instructions", "guide",
        "tutorial", "procedure", "process", "task", "complete", "perform",
        "execute", "run", "install", "configure", "setup", "deploy",
    },
}

_VAGUE_WORDS: Set[str] = {
    "somehow", "something", "anything", "whatever", "whenever", "wherever",
    "whoever", "perhaps", "maybe", "possibly", "probably", "generally",
    "usually", "often", "sometimes", "occasionally", "rather", "quite",
    "fairly", "somewhat", "mostly", "largely", "roughly", "approximately",
    "around", "about", "few", "many", "several", "various", "some", "any",
    "good", "bad", "nice", "great", "better", "best", "worse", "worst",
    "important", "interesting", "relevant", "appropriate", "suitable",
    "reasonable", "significant", "considerable", "notable",
    "proper", "correct", "right", "wrong", "enough", "sufficient",
    "adequate", "decent", "acceptable", "stuff", "things", "thing",
    "aspect", "aspects", "issue", "issues", "matter", "matters",
    "sort", "kind", "type", "way", "ways", "etc", "etcetera",
}

_CONFLICT_PAIRS: List[Tuple[Set[str], Set[str], str]] = [
    (
        {"brief", "short", "concise", "succinct", "terse", "quick", "summary"},
        {"comprehensive", "detailed", "exhaustive", "thorough", "complete",
         "in-depth", "extensive", "elaborate", "fully", "all"},
        "brevity vs. comprehensiveness",
    ),
    (
        {"json", "xml", "csv", "yaml"},
        {"bullet", "bullets", "list", "numbered", "markdown"},
        "structured data format vs. list/markdown format",
    ),
    (
        {"formal", "professional", "academic"},
        {"casual", "informal", "conversational", "friendly", "simple"},
        "formal tone vs. informal tone",
    ),
    (
        {"simple", "beginner", "basic", "elementary", "layman", "plain"},
        {"advanced", "technical", "expert", "detailed", "in-depth", "deep"},
        "simple/beginner level vs. advanced/technical level",
    ),
    (
        {"objective", "neutral", "unbiased", "balanced", "impartial"},
        {"opinionated", "argue", "persuade", "convince", "advocate",
         "defend", "support", "against"},
        "neutral/objective vs. opinionated/persuasive",
    ),
    (
        {"short", "one-line", "one line", "single sentence", "brief"},
        {"step-by-step", "steps", "step by step", "walkthrough",
         "tutorial", "explain each", "explain every"},
        "brief output vs. step-by-step walkthrough",
    ),
]

_LENGTH_SIGNALS: Dict[str, Set[str]] = {
    "short": {
        "briefly", "short", "quick", "one-line", "one line", "tldr", "tl;dr",
        "summary", "summarise", "summarize", "in a sentence", "in one sentence",
        "in two sentences", "simple answer", "quick answer",
    },
    "long": {
        "comprehensive", "detailed", "thorough", "exhaustive", "in-depth",
        "step-by-step", "step by step", "fully", "complete", "elaborate",
        "expand", "all", "every", "each", "tutorial", "guide",
        "explain everything", "cover all",
    },
}

_TASK_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"\b(write|create|build|make|generate|produce|implement|"
        r"code|design|explain|summarise|summarize|describe|list|"
        r"find|identify|calculate|convert|translate|compare|"
        r"analyse|analyze|evaluate|review|fix|debug|refactor|"
        r"tell me|show me|give me|help me)\b",
        re.IGNORECASE,
    ),
]

_CONTEXT_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"\b(context|background|given that|assuming|as a|in the context of|"
        r"for a|for the|the goal is|our goal|we are|i am|i'm|we're|"
        r"the purpose|this is for|this will be|here is|here's|"
        r"the following)\b",
        re.IGNORECASE,
    ),
]

_FORMAT_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"\b(format|json|xml|csv|yaml|markdown|bullet|numbered|list|"
        r"table|outline|structure|in the form|as a|respond with|"
        r"output should|return a|return the|provide a)\b",
        re.IGNORECASE,
    ),
]


def analyse_prompt(prompt: str, model_name: str) -> Dict[str, Any]:
    """
    Analyse a prompt before it is sent to any model.
    Returns token_count, prompt_type, ambiguity_score, missing_components,
    conflict_flags, complexity_score, estimated_output_length.
    """
    doc = _NLP(prompt)
    tokens_spacy = [t for t in doc if not t.is_space]
    word_tokens  = [t for t in tokens_spacy if t.is_alpha]

    token_count            = _count_tokens(prompt, model_name)
    prompt_type            = _classify_prompt_type(word_tokens)
    ambiguity_score        = _score_ambiguity(word_tokens)
    missing_components     = _find_missing_components(prompt)
    conflict_flags         = _detect_conflicts(prompt)
    sentences              = list(doc.sents)
    complexity_score       = _compute_complexity(token_count, sentences, prompt)
    estimated_output_length = _estimate_output_length(prompt)

    return {
        "token_count":            token_count,
        "prompt_type":            prompt_type,
        "ambiguity_score":        ambiguity_score,
        "missing_components":     missing_components,
        "conflict_flags":         conflict_flags,
        "complexity_score":       complexity_score,
        "estimated_output_length": estimated_output_length,
    }



def _count_tokens(prompt: str, model_name: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(prompt))


def _classify_prompt_type(word_tokens: list) -> str:
    prompt_lower = {t.lower_ for t in word_tokens}
    scores = {
        category: len(prompt_lower & vocab)
        for category, vocab in _TYPE_VOCAB.items()
    }
    best_category = max(scores, key=lambda c: scores[c])
    return best_category if scores[best_category] > 0 else "factual"


def _score_ambiguity(word_tokens: list) -> float:
    if not word_tokens:
        return 0.0
    vague_count = sum(1 for t in word_tokens if t.lower_ in _VAGUE_WORDS)
    return round(min(vague_count / len(word_tokens), 1.0), 4)


def _find_missing_components(prompt: str) -> List[str]:
    missing = []
    if not any(p.search(prompt) for p in _TASK_PATTERNS):
        missing.append("task")
    if not any(p.search(prompt) for p in _CONTEXT_PATTERNS):
        missing.append("context")
    if not any(p.search(prompt) for p in _FORMAT_PATTERNS):
        missing.append("format")
    return missing


def _detect_conflicts(prompt: str) -> List[str]:
    prompt_lower = prompt.lower()
    words_in_prompt = set(re.findall(r"[a-z\-]+", prompt_lower))
    conflicts = []
    for signal_a, signal_b, label in _CONFLICT_PAIRS:
        if bool(signal_a & words_in_prompt) and bool(signal_b & words_in_prompt):
            conflicts.append(label)
    return conflicts


def _count_distinct_instructions(prompt: str) -> int:
    separators = re.split(
        r"[.!?;]|\band\b|\bthen\b|\balso\b|\badditionally\b|\bfinally\b",
        prompt, flags=re.IGNORECASE
    )
    instruction_re = re.compile(
        r"\b(write|create|build|make|generate|produce|implement|code|"
        r"explain|summarise|summarize|describe|list|find|identify|"
        r"calculate|convert|compare|analyse|analyze|evaluate|review|"
        r"fix|debug|refactor|tell|show|give|help|do|use|include|"
        r"ensure|avoid|follow|provide|return|output)\b",
        re.IGNORECASE,
    )
    return sum(1 for seg in separators if instruction_re.search(seg))


def _compute_complexity(token_count: int, sentences: list, prompt: str) -> float:
    norm_tokens       = min(token_count / 512, 1.0)
    norm_sents        = min(len(sentences) / 20, 1.0)
    instruction_count = _count_distinct_instructions(prompt)
    norm_instructions = min(instruction_count / 10, 1.0)
    score = 0.30 * norm_tokens + 0.25 * norm_sents + 0.45 * norm_instructions
    return round(score, 4)


def _estimate_output_length(prompt: str) -> str:
    prompt_lower = prompt.lower()
    words = set(re.findall(r"[a-z\-]+", prompt_lower))
    short_hits = len(words & _LENGTH_SIGNALS["short"])
    long_hits  = len(words & _LENGTH_SIGNALS["long"])

    for phrase in ("step by step", "step-by-step", "in depth", "in-depth",
                   "explain everything", "cover all", "in a sentence",
                   "one line", "one sentence"):
        if phrase in prompt_lower:
            if phrase in ("in a sentence", "one line", "one sentence"):
                short_hits += 1
            else:
                long_hits += 1

    if long_hits > short_hits:
        return "long"
    if short_hits > long_hits:
        return "short"
    return "medium"



if __name__ == "__main__":
    TEST_PROMPT = (
        "You are a senior software engineer. Given the following Python codebase, "
        "write a comprehensive and detailed technical review. Be concise but also "
        "exhaustive in your analysis. Format the output as both JSON and a numbered "
        "bullet list. The review should be formal and professional, but also "
        "conversational and easy to understand for beginners. Include step-by-step "
        "explanations for each issue found, and also provide a brief one-line summary "
        "at the top. Identify any security vulnerabilities, performance issues, and "
        "code style violations. Make sure the response is thorough and covers all "
        "aspects of the code."
    )
    result = analyse_prompt(TEST_PROMPT, model_name="gpt-4")
    print(json.dumps(result, indent=2))