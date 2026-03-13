# Tokenization and Next-Token Prediction fundamentals

It may look like large language models are processing human readable text, however under the hood our prompts are represented as a string of smaller chuncks called tokens. Each of these tokens is an embedding *(high dimension real number vector)* that is stored in the models vocabulary *(typically between 32.000 - 128.000 entries)*. These tokens are also used for generating output, as LLM-s are simply put a very large neural network that is trained to predict the next token based on the all the previous tokens it has processed *(context)*.

Tokenization is the process of turning gramatical structures *(ie. sentences, words)* into these smaller chunks that are able to be processed by an LLM model. It is important to mention that the tokenizer is a completely seperate module from the actual language model, however it can greatly effect the output of the LLM in ways that have nothing to do with the underlying neural network architecture.

## How it works conceptually

Different model providers use different tokenization techniques to suit their own LLM-s, however the pipeline is generally as follows:

1. **Normalization:** The raw text is standardized *(lowercased, excess spaces/punctuation removed, cleaned of inconsistencies)*.
2. **Splitting (Segmentation):** The text is divided into individual characters, subwords, or words based on a predetermined algorithm.
3. **Assigning Token IDs:** Each unique token is mapped to a specific integer ID based on the model's "vocabulary".
4. **Sequence Formatting:** The resulting sequence of IDs is grouped into chunks *(sequences)* to be processed.

![Tokenization pipeline](https://miro.medium.com/v2/resize:fit:1200/0*F4xBvaO68VzNrUsn.png)

These are the most common tokenization algorithms that are used in the industry today:

- **Byte-Pair Encoding (BPE):** Popular in models like GPT-2, GPT-3, and GPT-4. It starts with individual characters and iteratively merges the most frequently occurring adjacent pairs of characters or subwords until a target vocabulary size is reached.
- **WordPiece:** Used in BERT *(Bidirectional Encoder Representations from Transformers)*. Similar to BPE, but it chooses merges that maximize the training data's likelihood rather than just frequency.
- **SentencePiece:** Treats input as a raw stream and handles whitespace as a character, making it effective for multilingual applications and languages without spaces.

## Subword Tokenization Effects

Most modern models use algorithms like Byte-Pair Encoding (BPE) or WordPiece to break rare words into smaller, meaningful chunks *(subwords)*.

![Subword Tokenization of the word "unfriendly"](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT3uHVwmKAFncnahZ8rSAyL840SAvAhcDw5Cg&s)

Some unfavory effects of subword tokenization includes:

- **Morphological Misalignment:** Ideally, subwords should align with linguistic units *(e.g., un- + friend + -ly)*. However, tokenizers often split words arbitrarily *(e.g., unf + ri + endly)*. This "broken word" effect forces the model to work harder to reconstruct the original meaning, leading to lower accuracy in reasoning tasks.
- **Tokenization Bias:** Different tokenization strategies can lead to different predictive distributions even for statistically equivalent data. Research shows that "tokenization bias" can cause models to fail when a prompt ends mid-token *(e.g., the model expects "Apple" but the prompt ends at "App")*, a phenomenon mitigated by "token healing."
- **Multilingual Unfairness:** Tokenizers are often skewed toward English. In other languages, a single word might be broken into many more tokens than its English equivalent, increasing computational cost and decreasing the "context window" available for that language.

## Next-Token Probability Distribution

The heart of an LLM is the Softmax layer, which produces a probability distribution over the entire vocabulary for the next token. The softmax function converts a vector of real-numbered raw scores (logits) into a probability distribution of probabilities that sum to 1.

![Softmax Function Definition](https://media.geeksforgeeks.org/wp-content/uploads/20220802180348/pastedimage0.png)

![Softmax Function Graph](https://www.researchgate.net/profile/Shen-Leixian/publication/325856086/figure/fig1/AS:723221292789765@1549440801787/Softmax-function-image.png)

- **The Generative Horizon:** Research using the Branching Factor *(BF)* shows that as a model generates text, its probability distribution usually "sharpens" *(becomes more concentrated)*. This means the model becomes more certain about its path the further it goes.
- **Alignment and Sharpness:** Alignment tuning *(like RLHF)* significantly sharpens this distribution. While this makes the model more "helpful" and "safe" by steering it toward predictable, high-confidence responses *(e.g., starting a response with "Sure!")*, it also drastically reduces output diversity.
- **The Confidence Metric:** Model confidence is typically measured as the maximum probability in the softmax distribution. However, this can be misleading; a model can be "confidently wrong" if its training data was biased or if it is hallucinating a fact it doesn't actually "know".

## Why Prediction Uncertainty Matters

Uncertainty is the model's way of saying, "I'm not sure what comes next Understanding this is critical for safety and reliability.

- **Hallucination Detection:** High entropy *(uncertainty)* in the next-token distribution is a strong signal for potential hallucinations. Strategies like Cautious Next Token Prediction *(CNTP)* adaptively sample more paths when uncertainty is high to find a more reliable output.
- **Abstention for Safety:** Research suggests that models should "abstain" *(refuse to answer)* when statistical uncertainty exceeds a certain threshold. This can improve safety by up to 99% in high-stakes domains like medicine or law.
- **The Reliability Gap:** Performance and reliability are distinct. A model might be 95% accurate but "unreliable" if it presents its 5% of errors with the same high confidence as its correct answers. Quantifying epistemic uncertainty *(uncertainty due to lack of knowledge)* helps developers build guardrails against these confident failures.

Prediction stability can be viewed geometrically. A "stable" prediction is one where the internal state of the model can withstand minor perturbations without the top-predicted token changing.

## Token Probability to confidence

In the context of LLMs, confidence is a measure of how "peaky" the probability distribution is at a specific step *t*.The Softmax MetricThe model generates a vector of raw scores *(logits)* for every word in its vocabulary. These are converted into probabilities using the Softmax function.

- **High Confidence (Low Entropy):** One token has a probability near 1.0, while all others are near 0. This happens in structured tasks *(e.g., "The capital of France is Paris")*.

- **Low Confidence (High Entropy):** Many tokens have similar probabilities *(e.g., "Once upon a time/day/forest")*.

### The Calibration Gap

A major research hurdle is calibration: does a model's 80% probability actually mean it is right 80% of the time? Often, models are "overconfident"—they assign high probability to a hallucination because that specific sequence of tokens appeared frequently in their training data, even if it's factually incorrect.

## From Confidence to Alignment Scoring

Alignment is the process of ensuring a model's outputs match human intent, safety, and helpfulness. We use confidence metrics to "score" how well a model is behaving.

### RLHF and Reward Modeling

In Reinforcement Learning from Human Feedback *(RLHF)*, we train a Reward Model *(RM)*. The RM looks at a generated sentence and assigns it a scalar score.

- The Log-Prob Connection: We often calculate the Kullback-Leibler *(KL)* Divergence between the "aligned" model and the original "base" model.

- If the aligned model starts assigning high probability to "unsafe" tokens, the alignment score drops. We essentially "punish" the model for being confident in the wrong direction.

### The "Sycophancy" Problem

A fascinating link between confidence and alignment is sycophancy. If a user says, "I think 2+2=5," an over-aligned model might assign high probability to "You are right!" because its alignment training prioritized "being helpful and agreeable" over "being factually confident."

|    **Metric**    |                    **Definition**                    |                                      **Impact On Alignment**                                     |
|:----------------:|:----------------------------------------------------:|:------------------------------------------------------------------------------------------------:|
| Perplexity       | How "surprised" the model is by a sequence.          | Lower perplexity usually means better alignment with the training distribution.                  |
| Top-1 Prob       | The probability of the most likely token.            | Used to trigger ""refusal"" mechanisms if the model is uncertain about safety.                   |
| Semantic Entropy | Uncertainty across meanings rather than just tokens. | High semantic entropy is a red flag for a "hallucinated" alignment *(lying to please the user)*. |

## The "Alignment Tax" on Probability

There is a direct trade-off between a model's raw predictive power and its alignment.

When we force a model to be "aligned" *(e.g., preventing it from generating toxic language)*, we are essentially warping the probability distribution. We tell the model: "Even if 'toxic_word' is the most statistically likely next token based on the internet data you read, you must artificially depress its probability to zero."

This "Alignment Tax" can sometimes lower the model's confidence in complex reasoning tasks because the "safe" path is not always the most "logical" path based on its raw pre-training.
