# PaliGemma Vision-Language Model Fine-Tuning for Charts and Graphs

This repository contains code and resources for fine-tuning the **PaliGemma vision-language model** to generate descriptive captions for charts and graphs. The goal is to train the model to understand the relationships between visual entities and extract meaningful captions based on these relationships.

## Project Overview

The notebook demonstrates how to fine-tune **PaliGemma**, a powerful vision-language model, on a custom dataset of charts and graphs. The key objectives of the project are:
- Fine-tune the model's language component while freezing the vision parameters.
- Train the model to describe visual content with complex relationships between entities.
- Evaluate the model using metrics such as **Perplexity** and **BLEU Score**.

### Key Features:
- Data preprocessing for both images and text (tokenization, resizing, and normalization).
- Training loop using **Stochastic Gradient Descent (SGD)**.
- Inference loop to generate predictions (captions) for unseen images.
- Model evaluation using **Perplexity**, **BLEU Score**, and **Human Evaluation**.

## Setup and Installation

### Requirements
To run this project, you'll need the following:
- Python 3.x
- JAX
- TensorFlow
- Hugging Face `transformers` and `datasets` libraries
- SentencePiece tokenizer
- Google Colab or a GPU-enabled environment for faster execution

### Results
The fine-tuned model was able to generate descriptive captions for charts and graphs with moderate accuracy. Further fine-tuning with larger datasets or more training iterations could improve performance.
