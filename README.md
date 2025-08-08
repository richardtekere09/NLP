# Text Classification with Neural Networks: A Comparative Study

A comprehensive implementation comparing three different text representation approaches for neural network-based text classification using the 20 Newsgroups dataset.

## Overview

This project implements and compares three distinct approaches to text classification:

1. **Bag-of-Words (BoW)** with various preprocessing techniques
2. **Character-level CNN** for sequence modeling
3. **Word-level embedding** with neural networks

Each approach is evaluated on the 20 Newsgroups dataset to demonstrate the impact of different text representation methods on classification performance.

## Features

- Multiple text preprocessing strategies (basic, NLTK-based, stemming, lemmatization)
- Character-level convolutional neural networks
- Word-level embedding models with configurable parameters
- Comprehensive evaluation metrics and visualizations
- Confusion matrix analysis
- Example text prediction functionality

## Requirements
- torch>=1.9.0
- numpy>=1.20.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- nltk>=3.6.0
  
## Model Architectures

 **Bag-of-Words Classifier**
- Input: Vocabulary-sized sparse vectors
- Architecture: Linear → ReLU → Dropout → Linear
- Hidden size: 256 units

**Character-level CNN**
- Input: Character sequences (max length: 250-1000)
- Architecture: Embedding → Conv1D → MaxPool → Conv1D → MaxPool → FC
- Filters: 128 per convolutional layer
- Kernel sizes: 5, 50 (configurable)

**Word-level Embedding Model**
- Input: Word sequences with padding
- Architecture: Embedding → Average Pooling → FC → ReLU → Dropout → FC
- Embedding dimensions: 50, 150, 300 (configurable)

## Experiments and Results

**Preprocessing Comparison (BoW)**
Basic preprocessing (lowercasing, punctuation removal)
NLTK with stopword removal
NLTK with stemming
NLTK with lemmatization

**Character-level Analysis**
- Sequence lengths: 250, 1000 characters
- Kernel sizes: 5, 50
- Classes tested: alt.atheism vs comp.graphics

**Embedding Parameter Study**
- Vocabulary sizes: 2,500, 5,000, 10,000
- Embedding dimensions: 50, 150, 300
- Maximum sequence lengths: 50, 100, 250

## Evaluation Metrics
**Classification accuracy**
- Detailed classification reports (precision, recall, F1-score)
- Confusion matrices with normalization
- Training loss curves
- Comparative visualizations

## Key Results
The experiments demonstrate:
- Preprocessing Impact: NLTK-based preprocessing generally improves BoW performance
- Character vs Word Level: Word-level embeddings typically outperform character-level approaches for this dataset
- Parameter Sensitivity: Embedding dimension and vocabulary size significantly affect performance
- Trade-offs: Longer sequences improve accuracy but increase computational cost
