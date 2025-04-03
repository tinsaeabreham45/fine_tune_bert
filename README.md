# Test it : https://huggingface.co/spaces/TinsaeA/sentiment-analysis-demo

# Sentiment Analysis with BERT

This repository contains a BERT-based model for binary sentiment classification, trained on a large dataset of 1.5 million sentences. The model achieves an accuracy of 85.29% and an F1 score of 0.8528 on the test set.

## Project Overview

The goal of this project was to build a sentiment analysis model to classify text as **positive/neutral** (label 1) or **negative** (label 0). The model was fine-tuned using a pretrained BERT (`bert-base-uncased`) model from the `transformers` library with TensorFlow.

### Dataset
- **Size**: 1.5 million sentences.
- **Source**: Not specified (e.g., social media, reviews—update this if applicable).
- **Labels**:
  - Original labels: 1 (positive/neutral) and 0 (negative).
  - Remapped to: 1 (positive/neutral) and 0 (negative) for binary classification.
- **Class Distribution**:
  - Negative: 156,155 samples (49.9%)
  - Positive: 156,942 samples (50.1%)
  - Nearly balanced classes.

### Preprocessing
- **Text Cleaning**:
  - Removed special characters using regex: `[^a-zA-Z0-9\s]`.
  - Example: `"@switchfoot http://twitpic.com/2y1zl - Awww, t..."` → `"switchfoot httptwitpiccom2y1zl Awww t"`.
- **Tokenization**:
  - Used `BertTokenizer` from `transformers`.
  - Parameters: `max_length=128`, `padding='max_length'`, `truncation=True`.

### Data Splitting
- **Train/Validation/Test Split**:
  - Train: 60% (~900,000 samples)
  - Validation: 20% (~300,000 samples)
  - Test: 20% (~313,097 samples)
- Used `train_test_split` from `sklearn` with `random_state=42`.

## Model Details

- **Model**: `TFBertForSequenceClassification` (from `transformers` library).
- **Pretrained Base**: `bert-base-uncased`.
- **Number of Labels**: 2 (binary classification).
- **Training**:
  - Epochs: 2
  - Batch Size: 16
  - Learning Rate: 2e-5
  - Optimizer: Adam
  - Loss: `SparseCategoricalCrossentropy` (from logits)
  - Metrics: Accuracy
- **Hardware**: Trained on Kaggle with GPU (P100).

### Achievements
- **Training Accuracy**: 87.47%
- **Validation Accuracy**: 85.26%
- **Test Accuracy**: 85.29%
- **F1 Score (Weighted)**: 0.8528
- **Classification Report** (Test Set, 313,097 samples):
