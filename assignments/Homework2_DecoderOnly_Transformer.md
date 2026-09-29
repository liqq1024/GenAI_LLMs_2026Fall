# Homework 2 Description: Implement a Simplified Decoder-Only Transformer

## Overview

This homework builds directly on the Week 3 self-attention activity and the Week 4 lecture on GPT-style language models. You will implement and train a small decoder-only Transformer in PyTorch for next-token prediction.

Your model will use token embeddings, positional embeddings, masked self-attention, feed-forward layers, residual connections, and a final vocabulary prediction layer. You will apply a causal attention mask so the model can use previous tokens but cannot see future tokens when predicting the next token.

Because the provided corpus is small, this homework emphasizes understanding the architecture and training process rather than producing high-quality language generation.

## Learning Objectives

By completing this homework, you should be able to:

1. Construct a simplified GPT-style decoder-only Transformer using PyTorch.
2. Explain token embeddings and positional embeddings.
3. Create and apply a causal attention mask.
4. Train a model for next-token prediction using integer token IDs and cross-entropy loss.
5. Generate text autoregressively from a prompt.
6. Analyze how model architecture choices and dataset size affect results.
7. Explain how this simplified model differs from production-scale GPT models.

## Files and Submission

You will receive:

- Homework2_DecoderOnly_Transformer.ipynb
- toy_transformer_corpus.txt
- decoder_skeleton.py

Complete every required code and written-response cell.

Rename and submit:

~~~text
LastName_FirstName_Homework2.ipynb
LastName_FirstName_Homework2_Report.pdf
~~~

Submit both files through the course submission system by **11:59PM October 12th, 2026**. Your notebook must run from top to bottom without errors, and all required outputs, plots, and generated-text examples must be visible.

## Task 1: Prepare the Dataset for Next-Token Prediction

Use the provided toy corpus and tokenizer/vocabulary code in the starter notebook. The corpus will be converted into token IDs and divided into fixed-length input sequences. The input is a sequence of token IDs and the target is the same sequence shifted one token to the left.

| Input tokens | Target tokens |
|---|---|
| the model learns from | model learns from data |

Complete the following tasks:

1. Inspect the provided vocabulary and token-ID mapping.
2. Create fixed-length input and target sequences.
3. Split examples into training and validation sets.
4. Use TensorDataset and DataLoader to prepare the data.
5. Print the vocabulary size and shapes of one input and target batch.

Answer:

1. Why is the target sequence shifted relative to the input sequence?
2. What does each target token ID represent?
3. Why should all sequences in a batch have the same length?

## Task 2: Implement a Decoder Block

Complete the provided PyTorch decoder-block skeleton. Your block must include:

1. Token embeddings using nn.Embedding
2. Positional embeddings using nn.Embedding
3. Masked multi-head self-attention using nn.MultiheadAttention
4. A residual connection and layer normalization after attention
5. A feed-forward network
6. A second residual connection and layer normalization

The decoder block must preserve the tensor shape:

~~~text
(batch_size, sequence_length, embedding_dimension)
~~~

Print the model architecture and answer:

1. What information is added by positional embeddings?
2. Why are residual connections useful in deep neural networks?
3. What is the purpose of the feed-forward network after self-attention?
4. Why does the decoder use masked self-attention instead of unrestricted self-attention?

## Task 3: Create and Apply a Causal Attention Mask

Create a causal mask for a sequence of length T. The mask must prevent a token at position i from attending to positions greater than i.

| Query position | Allowed key positions |
|---|---|
| 0 | 0 |
| 1 | 0, 1 |
| 2 | 0, 1, 2 |
| 3 | 0, 1, 2, 3 |

Complete the following tasks:

1. Create a causal mask using PyTorch.
2. Apply the mask in the self-attention layer.
3. Print or visualize the mask.
4. Verify that future-token positions are masked.
5. Explain why the mask is essential for next-token prediction.

Answer:

1. What problem would occur if a language model could attend to future tokens during training?
2. How does causal masking make training consistent with text generation?
3. Does a token attend to itself? Explain.

## Task 4: Train the Simplified Decoder-Only Transformer

Train your model for at least 50 epochs, or longer if needed to observe learning behavior.

Use:

~~~python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
~~~

For next-token prediction, reshape logits and targets before calculating loss:

~~~python
# logits:  (batch_size, sequence_length, vocab_size)
# targets: (batch_size, sequence_length)
loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
~~~

Your notebook must include a complete training loop, training and validation loss and token accuracy for each epoch, loss and accuracy plots, and final training and validation metrics.

Answer:

1. What does each row of the reshaped logits tensor represent?
2. Why are logits passed directly to nn.CrossEntropyLoss without applying softmax first?
3. Based on your curves, does the model appear to underfit, overfit, or perform reasonably well? Explain using evidence from the plots.

## Task 5: Generate Text

Write a function that generates text one token at a time from a seed prompt. At each step, use the model output at the final input position to select the next token, append it to the prompt, and repeat.

Use greedy decoding or the sampling method provided in the starter notebook. Test at least three prompts. For each prompt, generate at least 10 additional tokens, display the complete generated text, and comment on whether it is meaningful, repetitive, or inconsistent.

## Task 6: Analyze Model Architecture Choices

In a Markdown cell, discuss:

1. How causal masking changes the model’s behavior during training and generation.
2. One limitation of training this model on a small toy corpus.
3. How increasing embedding dimension, number of heads, number of decoder blocks, or context length might affect the model.
4. One architecture change you would try first and why.
5. At least three differences between this simplified decoder-only Transformer and a production-scale GPT model.

Possible differences include model size, training data, number of layers and heads, context-window length, tokenizer design, training resources, alignment methods, and deployment infrastructure.

## PDF Report Requirements

Prepare a 3–4 page PDF report containing:

1. **Introduction:** The goal of decoder-only language modeling.
2. **Model Design:** Embeddings, positional embeddings, masked attention, and the feed-forward network.
3. **Causal Masking:** Your mask visualization and explanation.
4. **Training Results:** Loss and accuracy plots and final metrics.
5. **Text Generation:** Examples from all three prompts and a discussion of quality.
6. **Analysis:** Dataset limitations, model behavior, and differences from production-scale GPT models.
7. **Conclusion:** One key concept learned from the implementation.

Use clear figure labels and refer to each figure in your discussion.

## Grading Rubric

| Category | Points |
|---|---:|
| Decoder implementation | 30 |
| Causal masking | 20 |
| Training and inference | 20 |
| Analysis and discussion | 20 |
| Code quality and documentation | 10 |
| **Total** | **100** |

## Academic Integrity and AI Use

You may use course materials, official PyTorch documentation, and the provided starter code. If you use an AI coding assistant or another external resource, include a brief Markdown note describing what you used and how it helped.

You are responsible for understanding, running, and explaining all submitted code. Do not submit code that you cannot explain.

## Optional Background Module

Students who would like additional preparation may review:

- Cross-entropy loss
- Gradient descent and backpropagation
- PyTorch nn.Module
- PyTorch training loops
- Adam optimization

These concepts will become increasingly important when we study fine-tuning in Week 8.

## Reading Assignment

### Required

- Brown et al., *Language Models are Few-Shot Learners*: Introduction and Model sections
- Sebastian Raschka, *Build a Large Language Model (From Scratch)*: chapters on decoder-only architectures and autoregressive generation

### Recommended

- OpenAI’s original GPT blog post for historical context
- Hugging Face Course sections on causal language modeling

## Submission Checklist

- [ ] Notebook runs from top to bottom with outputs shown and without errors.
- [ ] No API keys, passwords, or private information are included.
- [ ] Input and target sequences for next-token prediction are prepared correctly.
- [ ] A decoder-only Transformer block is implemented.
- [ ] Positional embeddings are included.
- [ ] A causal attention mask is created and applied.
- [ ] Training and validation loss and accuracy plots are included.
- [ ] Text is generated from at least three prompts.
- [ ] The PDF report is 3–4 pages and includes required figures and discussion.
- [ ] All written questions are answered.
- [ ] Both filenames follow the required format.
