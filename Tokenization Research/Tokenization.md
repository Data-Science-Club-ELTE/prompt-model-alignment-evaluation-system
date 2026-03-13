# Tokenization and Next-Token Prediction fundamentals

It may look like large language models are processing human readable text, however under the hood our prompts are represented as a string of smaller chuncks called tokens. Each of these tokens is an embedding (high dimension real number vector) that is stored in the models vocabulary (typically between 32.000 - 128.000 entries). These tokens are also used for generating output, as LLM-s are simply put a very large neural network that is trained to predict the next token based on the all the previous tokens it has processed (context).

Tokenization is the process of turning gramatical structures (ie. sentences, words) into these smaller chunks that are able to be processed by an LLM model. It is important to mention that the tokenizer is a completely seperate module from the actual language model, however it can greatly effect the output of the LLM in ways that have nothing to do with the underlying neural network architecture.

## Tokens

Different model providers use different tokenization techniques to suit their own LLM-s. Each of these tokens are then represented as an embedding (a high dimension vector).