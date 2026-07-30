
# Homework 1: Tokenization, Embeddings, and Sequence Modeling with PyTorch

## Overview

This homework integrates the main concepts from Week 2: tokenization, token IDs, word embeddings, and sequence modeling. You will first examine how GPT-2 and BERT tokenize the same text differently. Then, using PyTorch, you will build and compare a SimpleRNN and an LSTM for next-word prediction.

## Learning objectives

By completing this homework, you should be able to:

1. Use Hugging Face tokenizers to convert text into tokens and token IDs.
2. Compare GPT-2 and BERT tokenization behavior.
3. Create a vocabulary and numerical dataset for a PyTorch language-modeling task.
4. Explain and visualize word embeddings.
5. Build, train, and evaluate SimpleRNN and LSTM models in PyTorch.
6. Explain why LSTMs can better preserve information in sequential text.

## Files and submission

Download the starter notebook, `Homework1_Tokenization_Embeddings_RNN_LSTM.ipynb`, and complete every required code and written-response cell.

Rename your completed notebook:

```text
LastName_FirstName_Homework1.ipynb
```

Submit it through the course submission system by **[insert due date and time]**. Your notebook must run from top to bottom without errors.

## Part 1: Tokenization with GPT-2 and BERT

Install and import the required libraries:

```python
!pip -q install transformers tokenizers datasets scikit-learn

from transformers import AutoTokenizer
```

Use the following text:

```python
text1 = "Large Language Models are changing AI."
text2 = "人工智能很可怕吗"
```

Load the GPT-2 and BERT tokenizers:

```python
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

Complete the following tasks:

1. Tokenize the text using GPT-2 and print the tokens.
2. Encode the text using GPT-2 and print the token IDs.
3. Tokenize the same text using BERT and print the tokens.
4. Encode the text using BERT and print the token IDs.
5. Create a Markdown table comparing the two tokenizers for each text.

Your table should include:

| Tokenizer | Tokens | Token IDs | Number of tokens |
| --------- | ------ | --------- | ---------------- |
| GPT-2     |        |           |                  |
| BERT      |        |           |                  |

Answer the following questions:

1. Do GPT-2 and BERT divide the text into exactly the same tokens? Give one example.
2. Why do language models need token IDs instead of raw text?
3. Which tokenizer would you choose for a GPT-style text-generation application? Why?
4. Why might different tokenization strategies affect model efficiency or output quality?

## Part 2: Create a Dataset for Next-Word Prediction

Use the provided corpus or create your own corpus with at least 10 meaningful sentences on a topic of your choice.

```python
sentences = [
    "healthy plants need sunlight and water",
    "disease symptoms can appear on leaves",
    "early detection helps protect crops",
    "machine learning can identify plant diseases",
    "farmers use images to monitor plant health",
    "deep learning models learn patterns from data",
    "sensors can support precision agriculture",
    "timely treatment can reduce crop damage",
    "leaf spots may indicate fungal infection",
    "data quality affects model performance"
]
```

For the PyTorch models, create a simple word-level vocabulary from this corpus. Unlike the pre-trained GPT-2 and BERT tokenizers in Part 1, this vocabulary should be built from your own training text.

Complete the following tasks:

1. Convert the corpus to lowercase and split it into words.
2. Create `word_to_id` and `id_to_word` dictionaries.
3. Convert each sentence to a sequence of word IDs.
4. Generate inputâ€“target pairs for next-word prediction.
5. Pad input sequences to the same length.
6. Split the data into training and validation sets.
7. Use `TensorDataset` and `DataLoader` to prepare the data for PyTorch training.

Print:

- vocabulary size;
- one original sentence;
- its word-level token IDs;
- one input sequence and its target word;
- the shapes of `X_train`, `y_train`, `X_val`, and `y_val`.

Answer:

1. Why must text be converted to numbers before a neural network can process it?
2. What does a word ID represent in your custom vocabulary?
3. Why is padding needed for batches of input sequences?
4. What is the prediction target in this task?

## Part 3: Train a SimpleRNN Model

Build a next-word prediction model using PyTorch. Your model must include:

1. `nn.Embedding`
2. `nn.RNN`
3. `nn.Linear`

Use:

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Train the model for at least 50 epochs. Your notebook must include:

- the model architecture;
- a complete training loop;
- training and validation loss for every epoch;
- training and validation accuracy for every epoch;
- a loss plot;
- an accuracy plot;
- final training and validation accuracy.

Answer:

1. What does the `nn.Embedding` layer do?
2. What information can the RNN hidden state carry from earlier words in the sequence?
3. Based on the training curves, does the model underfit, overfit, or perform reasonably well? Explain briefly.

## Part 4: Train an LSTM Model

Build an LSTM model using the same dataset and, as closely as possible, the same hyperparameters used for your SimpleRNN.

Your model must include:

1. `nn.Embedding`
2. `nn.LSTM`
3. `nn.Linear`

Train and evaluate the model using the same loss function, optimizer type, epoch count, and train/validation procedure.

Include:

- the model architecture;
- training and validation loss and accuracy;
- loss and accuracy plots;
- final training and validation accuracy.

## Part 5: Visualize and Analyze Embeddings

Use the trained embedding layer from your better-performing model.

Complete the following tasks:

1. Extract the embedding matrix from the model.
2. Select at least 10 meaningful words from your vocabulary.
3. Use PCA to reduce their embeddings to two dimensions.
4. Create a labeled scatter plot of the selected word embeddings.
5. Use cosine similarity to identify the three nearest neighbors of one selected word.

Answer:

1. Which words appear close together in your PCA visualization?
2. Do the nearest-neighbor results make semantic sense? Explain with one example.
3. Why are word embeddings useful compared with treating every word ID as an unrelated number?
4. What is one limitation of interpreting a two-dimensional PCA plot?

## Part 6: Compare the Models

Create the following table in a Markdown cell:

| Criterion                 | SimpleRNN | LSTM |
| ------------------------- | --------- | ---- |
| Final training accuracy   |           |      |
| Final validation accuracy |           |      |
| Training behavior         |           |      |
| Generated-text quality    |           |      |

Then answer:

1. Which model achieved better validation performance?
2. Which model produced more coherent next-word predictions?
3. Why can an LSTM handle longer-term information more effectively than a SimpleRNN?
4. Did either model show signs of overfitting? Support your answer with evidence from the plots.
5. How do the token IDs from Part 1 connect to the embedding layer used in Parts 3 and 4?

## Part 7: Generate Text

Write a function that accepts a seed phrase and generates at least five additional words, one word at a time.

Test both models with at least two seed phrases, such as:

```text
machine learning
plant disease
```

Include the generated outputs from both models and briefly state which continuation is more meaningful.

## Part 8: Reflection

In a final Markdown cell, write a reflection of approximately 150â€“250 words addressing:

- what you learned about tokenization and token IDs;
- how embeddings represent words for a neural network;
- the difference between a SimpleRNN and an LSTM;
- one limitation of this small-corpus experiment;
- one improvement you would make with a larger dataset and more computing resources.

## Grading rubric

| Category                                              | Points  |
| ----------------------------------------------------- | ------- |
| GPT-2/BERT tokenization comparison and analysis       | 15      |
| Dataset preparation and custom vocabulary             | 15      |
| SimpleRNN implementation and evaluation               | 20      |
| LSTM implementation and evaluation                    | 20      |
| Embedding visualization and nearest-neighbor analysis | 15      |
| Model comparison, generation, and written responses   | 10      |
| Code quality and reproducibility                      | 5       |
| **Total**                                             | **100** |

## Academic integrity and AI use

You may use course materials, official documentation, and the provided starter code. If you use an AI coding assistant or another external resource, include a brief Markdown note describing what you used and how it helped.

You are responsible for understanding, running, and explaining all submitted code. Do not submit code that you cannot explain.

## Submission checklist

- [ ] Notebook runs from top to bottom with results show up but without errors.
- [ ] No API keys, passwords, or private information are included.
- [ ] GPT-2 and BERT tokenization are compared.
- [ ] Custom vocabulary and PyTorch data loaders are created.
- [ ] Both SimpleRNN and LSTM models are trained and evaluated.
- [ ] Embeddings are visualized with PCA and analyzed with nearest neighbors.
- [ ] Generated-text examples are included for both models.
- [ ] All written questions and the reflection are answered.
- [ ] Filename follows the required format.
