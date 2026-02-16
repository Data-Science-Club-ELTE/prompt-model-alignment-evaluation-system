# Data Science Club Project Template


## Brief project description

> This project builds a Prompt-Model Alignment Evaluation Pipeline to move beyond "trial-and-error" prompting. We aim to mathematically measure how well a Large Language Model (LLM) understands and follows a specific instruction by analysing its internal "brain" states, such as logics and embeddings.
## Team


> Please fill out the following table with the name and expected responsibilities of each team member.

|               | Expected responsibilities |
| :------------ | :------------------------ |
| Team member 1 | Team Leader, etc.         |
| Team member 2 | Data Modeling, etc.       |
| $\vdots$      | $\vdots$                  |
| Team member N | Data Visualization, etc.  |

## The *Problem* behind the project

Most users interact with AI as a "black box," relying on subjective feelings to decide if a prompt works. In professional environments, we need a way to quantify "Alignment." Currently, there is a lack of accessible tools that tell a developer: "This prompt has high uncertainty (entropy) and is likely to hallucinate," before the model even finishes typing.

## Challenges

1. High dimensionality: comparing vectors in a 4,096 space (like Llama-3) is computationally expensive and hard to visualise.
2. Token Artifacts: Rare words or complexformatting can break into "nonsense" tokens, artificially lowering alignment scores.
3. Internal Access: Unlike using an API (e.g., ChatGPT), we must run models locally to "intercept" the probability distributions, which requires significant GPU/RAM management.

## Expectations

The final output will be a functional Python pipeline where a user inputs a prompt and receives a Technical Alignment Report. This report will include a percentage score based on mathematical certainty (Entropy), logical flow (Perplexity), and thematic stay-on-topic (Cosine Similarity).


## Tools & Technologies

> Language: Python

> Deep Learning: PyTorch, Hugging Face Transformers

> Math & Vectors: NumPy, SciPy, Scikit-learn

> Data Handling: Pandas

> Visualization: Matplotlib, Seaborn (for plotting the "distance" between prompt and response)

> Model Interpretability: Captum (optional)

## How to run the Project

> [!WARNING]
> This section is reserved for guidance on how to run the project components. This becomes relevant after code has been pushed to the repository, and is expected to be maintained according to the evolving state of the project.

## Results

> [!CAUTION]
> This section is reserved for discussing the project results at the **end of the semester**.

