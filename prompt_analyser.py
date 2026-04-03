import json
import re
from typing import Any

import tiktoken
import spacy

# Requires:  python -m spacy download en_core_web_sm
try:
    _NLP = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' not found.\n"
        "Run:  python -m spacy download en_core_web_sm"
    )


# Vocabulary bank 
# A prompt is scored against all four and the highest-scoring bucket wins.
_TYPE_VOCAB: dict[str, set[str]] = {
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

# Vague language to raise ambiguity 
_VAGUE_WORDS: set[str] = {
    "somehow", "something", "anything", "whatever", "whenever", "wherever",
    "whoever", "perhaps", "maybe", "possibly", "probably", "generally",
    "usually", "often", "sometimes", "occasionally", "rather", "quite",
    "fairly", "somewhat", "mostly", "largely", "roughly", "approximately",
    "around", "about", "few", "many", "several", "various", "some", "any",
    "good", "bad", "nice", "great", "better", "best", "worse", "worst",
    "important", "interesting", "relevant", "appropriate", "suitable",
    "reasonable", "significant", "considerable", "notable", "notable",
    "proper", "correct", "right", "wrong", "enough", "sufficient",
    "adequate", "decent", "acceptable", "stuff", "things", "thing",
    "aspect", "aspects", "issue", "issues", "matter", "matters",
    "sort", "kind", "type", "way", "ways", "etc", "etcetera",
}

# Pairs of conflicting instruction signals.
# Each tuple is (signal_a_words, signal_b_words, human-readable conflict label).
_CONFLICT_PAIRS: list[tuple[set[str], set[str], str]] = [
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

# Heuristic keyword sets for output length estimation.
_LENGTH_SIGNALS: dict[str, set[str]] = {
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

# Phrases / patterns that signal a clear task is present.
_TASK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(write|create|build|make|generate|produce|implement|"
               r"code|design|explain|summarise|summarize|describe|list|"
               r"find|identify|calculate|convert|translate|compare|"
               r"analyse|analyze|evaluate|review|fix|debug|refactor|"
               r"tell me|show me|give me|help me)\b", re.IGNORECASE),
]

# Patterns that signal context / background is present.
_CONTEXT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(context|background|given that|assuming|as a|in the context of|"
               r"for a|for the|the goal is|our goal|we are|i am|i'm|we're|"
               r"the purpose|this is for|this will be|here is|here's|"
               r"the following)\b", re.IGNORECASE),
]

# Patterns that signal a format instruction is present.
_FORMAT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(format|json|xml|csv|yaml|markdown|bullet|numbered|list|"
               r"table|outline|structure|in the form|as a|respond with|"
               r"output should|return a|return the|provide a)\b", re.IGNORECASE),
]


# Public API

def analyse_prompt(prompt: str, model_name: str) -> dict[str, Any]:
    """
    Analyse a prompt before it is sent to any model.

    Parameters
    ----------
    prompt     : The raw prompt string to analyse.
    model_name : Target model identifier (used for tiktoken encoding selection).

    Returns
    -------
    dict with keys:
        token_count          int
        prompt_type          str  — "structured" | "creative" | "factual" | "instructional"
        ambiguity_score      float  0.0–1.0
        missing_components   list[str]
        conflict_flags       list[str]
        complexity_score     float  0.0–1.0
        estimated_output_length str  — "short" | "medium" | "long"
    """
    # Shared initialisation 
    doc = _NLP(prompt)
    tokens_spacy = [t for t in doc if not t.is_space]
    word_tokens = [t for t in tokens_spacy if t.is_alpha]

    #1. token_count  (tiktoken BPE tokenization)
    token_count = _count_tokens(prompt, model_name)

    #2. prompt_type  — vocabulary scoring
    prompt_type = _classify_prompt_type(word_tokens)

    #3. ambiguity_score  — vague-word ratio
    ambiguity_score = _score_ambiguity(word_tokens)

    # 4. missing_components  — task / context / format presence checks
    missing_components = _find_missing_components(prompt)

    # 5. conflict_flags  — contradicting instruction detection
    conflict_flags = _detect_conflicts(prompt)

    # 6. complexity_score  — normalised composite
    sentences = list(doc.sents)
    complexity_score = _compute_complexity(token_count, sentences, prompt)

    # 7. estimated_output_length  — wording heuristics
    estimated_output_length = _estimate_output_length(prompt)

    return {
        "token_count": token_count,
        "prompt_type": prompt_type,
        "ambiguity_score": ambiguity_score,
        "missing_components": missing_components,
        "conflict_flags": conflict_flags,
        "complexity_score": complexity_score,
        "estimated_output_length": estimated_output_length,
    }


#Private helpers

def _count_tokens(prompt: str, model_name: str) -> int:
    """tiktoken BPE token count with cl100k_base fallback."""
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(prompt))


def _classify_prompt_type(word_tokens: list) -> str:
    """
    Score each category by counting matching vocabulary words and return
    the highest-scoring label. Ties broken by category insertion order.
    """
    prompt_lower = {t.lower_ for t in word_tokens}
    scores = {
        category: len(prompt_lower & vocab)
        for category, vocab in _TYPE_VOCAB.items()
    }
    # Fall back to "factual" when no signal at all (the most generic bucket).
    best = max(scores, key=lambda c: (scores[c], c == "factual"))
    return best if scores[best] > 0 else "factual"


def _score_ambiguity(word_tokens: list) -> float:
    """Ratio of vague words to total alpha-token count, clamped to [0, 1]."""
    if not word_tokens:
        return 0.0
    vague_count = sum(1 for t in word_tokens if t.lower_ in _VAGUE_WORDS)
    return round(min(vague_count / len(word_tokens), 1.0), 4)


def _find_missing_components(prompt: str) -> list[str]:
    """
    Check for three structural components and return labels for absent ones.
    Components:  "task", "context", "format"
    """
    missing = []

    has_task = any(p.search(prompt) for p in _TASK_PATTERNS)
    if not has_task:
        missing.append("task")

    has_context = any(p.search(prompt) for p in _CONTEXT_PATTERNS)
    if not has_context:
        missing.append("context")

    has_format = any(p.search(prompt) for p in _FORMAT_PATTERNS)
    if not has_format:
        missing.append("format")

    return missing


def _detect_conflicts(prompt: str) -> list[str]:
    """
    Scan for pairs of opposing instruction signals and return a list of
    human-readable conflict descriptions for every pair that fires.
    """
    prompt_lower = prompt.lower()
    # Tokenise at word-boundary level for whole-word matching.
    words_in_prompt = set(re.findall(r"[a-z\-]+", prompt_lower))

    conflicts = []
    for signal_a, signal_b, label in _CONFLICT_PAIRS:
        hit_a = bool(signal_a & words_in_prompt)
        hit_b = bool(signal_b & words_in_prompt)
        if hit_a and hit_b:
            conflicts.append(label)

    return conflicts


def _count_distinct_instructions(prompt: str) -> int:
    """
    Heuristic count of distinct imperative instructions.
    Looks for imperative-style verb phrases and instruction separators.
    """
    separators = re.split(
        r"[.!?;]|\band\b|\bthen\b|\balso\b|\badditionally\b|\bfinally\b",
        prompt, flags=re.IGNORECASE
    )
    # An instruction segment contains a task-like verb.
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
    """
    Normalised complexity score combining:
      - token count        (cap at 512 for normalisation)
      - sentence count     (cap at 20)
      - distinct instructions (cap at 10)

    Returns a float in [0.0, 1.0].
    """
    norm_tokens = min(token_count / 512, 1.0)
    norm_sents = min(len(sentences) / 20, 1.0)

    instruction_count = _count_distinct_instructions(prompt)
    norm_instructions = min(instruction_count / 10, 1.0)

    # Weighted average   
    # instruction density is the strongest complexity indicator.
    score = 0.30 * norm_tokens + 0.25 * norm_sents + 0.45 * norm_instructions
    return round(score, 4)


def _estimate_output_length(prompt: str) -> str:
    """
    Three-class heuristic: "short" | "medium" | "long"
    based on wording signals in the prompt.
    """
    prompt_lower = prompt.lower()
    words = set(re.findall(r"[a-z\-]+", prompt_lower))

    short_hits = len(words & _LENGTH_SIGNALS["short"])
    long_hits = len(words & _LENGTH_SIGNALS["long"])

    #check multi-word phrases that a bag-of-words misses.
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


# Test block — runs when: python prompt_analyser.py

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