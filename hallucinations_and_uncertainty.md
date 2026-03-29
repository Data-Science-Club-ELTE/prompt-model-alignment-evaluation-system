# 8: Analyze hallucination and uncertainty in LLMs

## What causes hallucinations?
- Definition: Hallucination
    - When language models “produce overconfident, plausible falsehoods, which diminish their utility and trustworthiness” (Kalai et al., 2025)
    - i.e., when AI confidently produces a believable response that is actually false
- Causes:
    - Evaluation methods (Kalai et al., 2025)
        - LLMs evaluated based solely on accuracy - Test-taking Mode
            - Encourages guessing (higher chance of getting it correct)
            - Discourages admitting uncertainty (I don't know)
    - Pretraining
        - Pretraining = “process of predicting the next word in huge amounts of text” (OpenAI, 2025)
        - Uses IIV (Is-It-Valid) binary classification to determine successfulness of responses (Kalai et al., 2025)
            - Easier to find patterns for grammar, spelling, counting
            - Difficult to find patterns for data such as birthdays
            - Leads to hallucinations when asked for someone's birthday
    - Bias in data sources (MIT Sloan, n.d.)
        - Models trained on data that can contain inaccuracies, bias (cultural, societal)
        - Models can mimic these biases (model itself cannot differentiate biased/inaccurate data)
    - Inherent Challenges in Design (MIT Sloan, n.d.)
        - Generative AI designed to predict patterns and predict the next word
        - Not designed to distinguish true/false information
        - Can produce plausible responses that are actually inaccurate
- Solutions & Existing Challenges:
    - OpenAI (Kalai et al., 2025)
        - Penalize confident errors more over uncertainty (explicit confidence targets)
    - Possible problems with OpenAI's solution (Xing, 2025)
        - If a model frequently responds with "I don't know," users may be dissatisfied
        - Shift to using a different model

Sources:
1. [Why Language Models Hallucinate (Kalai et al., 2025)] (https://arxiv.org/abs/2509.04664)
2. [Why Language Models Hallucinate (OpenAI, 2025)] (https://openai.com/index/why-language-models-hallucinate/)
3. [Why OpenAI’s solution to AI hallucinations would kill ChatGPT tomorrow (Xing, 2025)] (https://theconversation.com/why-openais-solution-to-ai-hallucinations-would-kill-chatgpt-tomorrow-265107)

## How does entropy relate?
- Definition: Entropy (Correa, 2025)
    - Entropy = "mathematical measure of disorder or unpredictability in a system"
    - Use with LLMs = "measures how spread out the probability distribution is over possible next words or tokens"
    - Entropy & Uncertainty: entropy is one of the ways to measure uncertainty (but it does not cover all aspects of uncertainty)
    - High entropy: uncertain; considering many options as equally likely
    - Low entropy: confident; considering fewer options
- Relation with Hallucinations (Correa, 2025)
    - We may expect high entropy -> more hallucinations, low entropy -> less hallucinations
        - But not always true
        - High entropy -> sometimes correct uncertainty
        - Low entropy -> often confident nonsense
    - Other factors to consider to analyze hallucinations
        - Epistemic uncertainty
            - The model doesn't know what it doesn't know
            - Cannot reliably distinguish when operating outside of training data
        - Calibration Problems
            - The model's certainty/confidence does not correlate with correctness/accuracy
            - i.e., the model can be confident but wrong
        - Semantic vs. syntactic uncertainty
            - The model can be certain about semantics (meaning of words) but uncertain about syntax (wording, grammar) and vice versa
            - Difficult to differentiate these two using traditional entropy
- Solutions:
    - Standard Methods
        - Retrieval-Augmented Generation (RAG)
            - One of most effective
            - RAG allows model to search external databases or the web
            - Can use this information in responses
        - Chain-of-Verification (CoVe)
            - Makes models check their own response
            - Generate response -> create questions to verify -> revise response
        - Constitutional AI
            - Has a set of principles to critique and revise outputs
            - Generate response -> checks it against the principles (rules) -> make revisions
    - Innovative Approaches
        - Semantic Entropy
            - Measures uncertainty over meanings (semantics)
            - e.g., If a model can say "Paris," "The capital of France," or "The City of Light" with equal probability
                - Traditional entropy: high uncertainty
                - Semantic entropy: all mean the same, so low uncertainty
        - Semantic Entropy Probes (SEPs)
            - Application of semantic entropy
            - "Extracts semantic entropy directly from the model's internal states without generating multiple outputs"
            - "Makes hallucination detection 5-10x more efficient"
        - Conformal Prediction
            - Provides statistical guarantees for model's outputs
            - Get a set of possible answers
            - And a guarantee that there is a high probability the correct answer is in that set
            - If the set is too large or empty, can admit uncertainty
- Existing Challenges:
    - Chain of Thought & Reasoning
        - Chain of Thought (CoT) = prompts models to show thinking process
        - Chain of Reasoning (CoR) = on top of CoT, make models generate multi-step logical arguments
        - Would making a model think step by step (CoT, CoR) reduce hallucinations?
        - Actually, it can worsen hallucinations
        - e.g., OpenAI's models with extended reasoning chains
            - Sometimes makes up the reasoning process
            - If model does not show all processes, cannot tell if it was real or hallucinated

Sources:
1. [Hallucinations in Large Language Models: The Entropy Problem and Current Solutions (Correa, 2025)] (https://monostate.com/blog/hallucinations-entropy-llms)

## Can uncertainty signals predict bad outputs?
- Uncertainty Quantification (PrajnaAI, 2025)
    - Uncertainty Quantification (UQ) = field in ML that estimates confidence in model predictions
    - Recent UQ framework for LLMs
        - Produce a confidence score between 0 and 1
        - Higher the score, model is more confident
        - Give hallucinations a lower score
        - Examples of approaches
            - Black box: only see end result
                - e.g., checking consistency of responses
                - Higher consistency -> higher confidence
            - White box: can see internal process
                - e.g., see how the model chose words for response
                - If one of words are 0 score, it lowers overall score
            - LLM-as-a-Judge
                - Have another model (multiple judge models) to evaluate response
                - If more judges agree correct -> higher confidence
                - If more judges disagree -> lower confidence
            - Ensemble scores
                - Can combine multiple scoring methods to get a better overall view
        - UQLM Toolkit
            - UQLM = Uncertainty Quantification for Language Models
            - Python library on [GitHub] (https://github.com/cvs-health/uqlm)
            - Built for uncertainty-based hallucination detection
            - Combines 4 UQ methods
                - Consistency-Based (Black-Box) Scorers
                - Token-Probability (White-Box) Scorers
                - LLM-as-a-Judge Scorers
                - Ensemble Scorers


Sources:
1. [Hallucinations Are Not Bugs, They’re Warnings: Why Uncertainty Quantification Matters in LLMs (PrajnaAI, 2025)] (https://prajnaaiwisdom.medium.com/hallucinations-are-not-bugs-theyre-warnings-why-uncertainty-quantification-matters-in-llms-289454d79a59)
2. [UQLM on GitHub] (https://github.com/cvs-health/uqlm)
