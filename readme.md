# Natural Language Processing - Text Reconstruction Project

## Overview

This project focuses on text reconstruction using various Natural Language Processing (NLP) techniques and transformer-based models. The goal is to transform ambiguous or grammatically incorrect texts into semantically accurate, grammatically correct, and syntactically well-formed versions.

## Team Members

| Name | Student ID |
|------|------------|
| **Dimitris Lazanas** | P22082 |
| **Ioanna Andrianou** | P22010 |
| **Danai Charzaka** | P22194 |

## Project Structure

```
├── documentation.md          # Complete project documentation
├── question-1.ipynb         # Question 1A - Custom NLP Pipeline with Stanza
├── question-1A.ipynb        # Additional Question 1A implementation
├── question-1-b.ipynb       # Question 1B - Transformer-based paraphrasing
├── part2.ipynb              # Part 2 - Word Embeddings Analysis
└── readme.md                # This file
```

## Requirements

### Python Dependencies

The project requires the following Python libraries:

```bash
# Core NLP libraries
transformers
torch
stanza
spacy
nltk

# Text augmentation
textattack
nlpaug

# Machine learning and analysis
scikit-learn
sentence-transformers
gensim
textstat

# Data visualization and manipulation
matplotlib
pandas
numpy
plotly

# Jupyter notebook environment
ipywidgets
```

### Model Downloads

Some models and datasets need to be downloaded:

```python
# spaCy English model
python -m spacy download en_core_web_sm

# NLTK data
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Running the Notebooks

### Important Note: Kaggle Environment

**These notebooks were primarily developed and tested on Kaggle**, which provides:
- Pre-installed GPU support (CUDA)
- Most required libraries pre-installed
- Sufficient computational resources for transformer models

### Local Setup

If running locally, ensure you have:

1. **GPU Support** (recommended): Install CUDA-compatible PyTorch
2. **Sufficient RAM**: Transformer models require significant memory
3. **All dependencies**: Install using pip or conda

### Notebook Descriptions

#### 1. question-1.ipynb & question-1A.ipynb
**Custom NLP Pipeline with Stanza**
- Implements custom phrase and grammar-based substitutions
- Uses Stanza for POS tagging, dependency parsing, and lemmatization
- Reconstructs individual sentences from provided texts

#### 2. question-1-b.ipynb
**Transformer-based Paraphrasing**
- Implements three transformer models:
  - Pegasus T5: `tuner007/pegasus_paraphrase`
  - BART: `facebook/bart-base`
  - T5: `t5-base`
- Processes entire texts using iterative sentence-level paraphrasing
- Compares model performance and limitations

#### 3. part2.ipynb
**Word Embeddings Analysis**
- Analyzes semantic similarity using BERT and GloVe embeddings
- Calculates cosine similarity between original and reconstructed texts
- Visualizes embeddings using PCA dimensionality reduction
- Includes readability analysis with TextStat

## Usage Instructions

### Running on Kaggle (Recommended)

1. Upload the notebook files to Kaggle
2. Enable GPU acceleration in Kaggle settings
3. Install any missing dependencies using `!pip install` in the first cell
4. Run cells sequentially


## Key Features

### Question 1A - Custom Pipeline
- **Stanza Integration**: Advanced NLP pipeline for grammatical analysis
- **Context-aware Substitutions**: Grammar-based modifications using dependency parsing
- **Phrase-level Corrections**: Direct translation fixes for Chinese-to-English patterns

### Question 1B - Transformer Models
- **Multiple Model Comparison**: Pegasus, BART, and T5 implementations
- **Iterative Refinement**: Multiple passes for improved results
- **Hardware Optimization**: Automatic CUDA/CPU selection

### Part 2 - Embeddings Analysis
- **Multi-modal Analysis**: BERT contextual and GloVe static embeddings
- **Similarity Metrics**: Cosine similarity calculations
- **Visualization**: PCA plots showing semantic space shifts
- **Readability Assessment**: TextStat metrics for quality evaluation

## Results and Observations

- **Custom Pipeline**: Effective for targeted grammatical corrections
- **Transformer Models**: Limited success with complex grammatically-challenged texts
- **Sentence-level Processing**: More effective than full-text processing
- **Model Ranking**: Pegasus showed best overall performance among transformers

## Documentation

For detailed methodology, results, and analysis, refer to `documentation.md`.

## Hardware Requirements

- **Optimal**: Kaggle environment with GPU acceleration enabled

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch sizes or use CPU
2. **Model Download Failures**: Check internet connection, retry downloads
3. **Import Errors**: Verify all dependencies are installed
4. **Slow Execution**: Enable GPU if available, reduce iterations

### Performance Tips

- Use Kaggle or Google Colab for GPU access
- Process texts sentence-by-sentence for better memory management
- Adjust model parameters (temperature, beam search) based on available resources
