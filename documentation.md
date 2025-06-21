# Natural Language Processing
## Assignment 2025 

---

## Team Members

| Name | Student ID |
|------|------------|
| **Dimitris Lazanas** | P22082 |
| **Ioanna Andrianou** | P22010 |
| **Danai Charzaka** | P22194 |

---

## Introduction

Text reconstruction is a process within the field of **Natural Language Processing (NLP)** that aims to transform ambiguous or grammatically incorrect texts into semantically accurate, grammatically correct, and syntactically well-formed versions. The goal is to enhance the clarity, coherence, and overall structure of the original text, making it more understandable and accessible.

In this project, we focus on implementing text reconstruction through various methodologies and subsequently comparing the results. The reconstruction process involves transforming two entire texts, as well as individual sentences within them, to ensure correctness in terms of syntax, grammar, and meaning.

Moreover, the comparative evaluation of the reconstruction results is conducted using a range of tools designed to quantify and visualize the improvements. This process is supported by advanced NLP techniques, including morphosyntactic analysis, the identification of grammatical patterns, and the resolution of semantic ambiguities.

### Methodology

#### Main concepts
**What is ambiguity?**
Ambiguity refers to situations where a word, phrase, or sentence has more than one possible interpretation or lacks clarity. In our project, ambiguity is not measured directly, but inferred using tools like cosine similarity, embeddings, and PCA visualizations.

**What is similarity?**
Similarity in Natural Language Processing refers to how semantically close two pieces of text are. It’s not just about using the same words—it's about expressing the same meaning, even when different wording is used.
In this project, similarity is measured using cosine similarity and embeddings.

**What is an NLP pipeline?**
An NLP pipeline is a structured sequence of steps or modules that process raw text and transform it into something that can be analyzed, interpreted, or used in downstream tasks (e.g., classification, summarization, semantic analysis). Each step addresses a specific layer of linguistic understanding, from basic structure to deep semantics.

---

## Part 1

### **Question 1A - Custom NLP Pipeline** 
> *"Reconstruction of 2 sentences from our choice, 1 from each text, with the use of a custom pipeline"*

For this question, **Stanza** was selected to create a custom NLP pipeline that performs custom phrase substitutions and grammar modifications to sentences. The goal was to reconstruct 2 sentences of our choosing from the provided texts and thus to gain a better grasp of Stanza as a tool and familiarize ourselves with it.

#### Tools used:

- **Stanza**
- **Pandas**

#### What features we used from Stanza:

- **Tokenization** - Splits text into individual words or tokens
- **POS Tagging** - Assigns part-of-speech tags to tokens
- **Lemmatization** - Converts words to their base or dictionary form
- **Dependency Parsing** - Analyzes grammatical relationships between words in a sentence

#### The following play a crucial role for the grammar-based modifications which are mentioned later on:

1. **Head** - Represents the index of the head word in the dependency tree
   
   Dependency parsing identifies relationships between words in a sentence, each word (except the root) is "governed" by another word, called its "head". The head is essentially an index pointing to the head word of a sentence. If `word.head == 0`, it means the word is the root of the dependency tree (it has no governing word). For example, in the sentence *"I love programming"*, "love" is the root because it is the main action and governs the other words.

2. **Lemma** - Base or dictionary form of a word
   
   Lemmatization reduces a word to its canonical form, for example for the word "running", the lemma is "run".

3. **Deprel** - Describes the dependency relation between a word and its head
   
   For example, "obj" means object, "nsubj" means nominal subject and "root" means root of the sentence.

4. **UPOS** - Universal Part-of-Speech tags
   
   These are the standardized tags used across different languages to represent the grammatical category of words like noun, verb, adjective.

5. **Head_lemma** - The lemma of the head word
   
   For this question, this is used to ensure that the word we are replacing is governed by a specific head word in the sentence, since want to substitute it specifically, in a specific context.

6. **Replacement** - Used to specify what the replacement of a token will be.

#### Challenges

The main challenge posed, was to figure out how the grammar-based substitutions would be applied for certain words in certain contexts. For example, a word may appear multiple times in a sentence but only once may it need to be modified (removed or replaced). We needed a way to identify when the "target" word (word to be modified) was in a certain context. That's where Stanza's POS tagging feature comes in.

#### Methodology

##### Using Stanza for POS Tagging 
We utilized the POS Tagging feature of Stanza to analyze the grammatical structure of the sentences. This helped greatly in understanding the part-of-speech of each word in the context of its sentence, which was crucial to make the grammar-based substitutions we implemented.

##### Using Pandas for POS Tags Visualization
After the POS tagging, we use Pandas to create a DataFrame that visualizes the POS tags of each word in the sentences.

#### Overview of the Pipeline
As mentioned in the beginning, this custom NLP pipeline demonstrates how we used Stanza to perform:

1. **Custom Phrase-Based Substitutions**: Replacing, among others, mostly phrases that were directly translated from Chinese to English, hence did not adhere to the syntax rules of the English language.

2. **Custom Grammar-Based Substitutions**: Modifying words based on grammatical rules and the context they're placed in. To identify the context for which we want a modification to happen, we utilize: Head, lemma, deprel, UPOS, head_lemma and replacement.

3. **Sentence Reconstruction**: Combining both substitution methods to reconstruct sentences.

---

### **Question 1B**
> *"Reconstruction of the entire 2 texts using 3 different automatic Python library pipelines."*

#### Tools used:

> *[To be filled in]*

#### Challenges:

> *[To be filled in]*

#### Methodology:
A complete reconstruction of two texts was implemented using 3 different pipelines, utilizing...

---

### **Question 1C**
> *"Compare the results of each approach using appropriate techniques."*

#### Tools used:

> *[To be filled in]*

#### Challenges:

> *[To be filled in]*

#### Methodology:
Quality and quantity comparison of the above techniques was conducted...

---

## Part 2 - Word Embeddings Analysis

> *"Use your own word embeddings (Word2Vec, GloVe, FastText, BERT embeddings, etc.) and your own custom automatic NLP pipelines (preprocessing, vocabulary, word embeddings, semantic trees etc.) to analyze the word similarity before and after the reconstruction. Calculate the cosine similarity between the original and reconstructed versions. Compare the methods with respect to A, B of deliverable 1. Visualize the word embeddings for A,B using PCA/t-SNE to demonstrate the shifts in the semantic space."*

The goal of this part of the project is to evaluate the semantic similarity between original and reconstructed texts (produced by various NLP models such as Pegasus, BART, T5, and ChatGPT) using different word embedding techniques and a custom NLP pipeline. The task involves comparing the semantic shifts before and after text reconstruction using cosine similarity and visualizing these embeddings in a reduced dimensional space (PCA).

### Word Embeddings Analysis

**Word embedding** is a technique in Natural Language Processing (NLP) where words or phrases from a vocabulary are mapped to vectors of real numbers in a continuous vector space. The idea is to capture the semantic meaning of words so that similar words have similar vector representations.

In our case we're using BERT, GloVe as well as a custom NLP pipeline to find the similarity between our original text and the reconstructed versions.

### Experiments & Results

For this part of the project we get the previously modified text from the models (**Pegasus**, **BART**, **T5**) as well as a **ChatGPT** modified version of the original text (that we assume is the best version of the original text) and we do word by word comparison of them using word embedding techniques.

#### Libraries used:
- **textstat** - for quick assessment of text readability
- **sentence-transformers** - for BERT-based model embeddings
- **gensim** - for GloVe embeddings
- **nltk** - for natural language processing tasks
- **numpy** - for numerical computations
- **cosine-similarity** - for similarity calculations
- **matplotlib** - for data visualization
- **PCA** - for dimensionality reduction
- **pandas** - for result tables

### Methodology

#### TextStat Analysis
We first evaluate the textual quality and complexity using a set of readability metrics provided by the textstat library. This includes:
- Flesch Reading Ease
- Flesch-Kincaid Grade Level
- Gunning Fog Index
- SMOG Index
- Automated Readability Index
- Dale-Chall Score
- Difficult Word Count
- Lexicon Count
- Sentence Count
This provides a baseline for understanding how the reconstructions differ from the original text in terms of readability.

#### Contextual Sentence Embeddings (BERT) (SentenceTransformer)
We use SentenceTransformer's 'all-MiniLM-L6-v2' model to obtain sentence-level embeddings. These embeddings account for the full context of the sentence, offering a high-fidelity semantic representation.

#### Static Word Embeddings (GloVe) (Gensim)
We use the GloVe-Twitter-25 pretrained model from Gensim to generate word-level embeddings. Sentence embeddings are computed by averaging the vectors of tokenized and lowercased words.
This method captures general semantic content but lacks contextual sensitivity.

#### Custom NLP Pipeline
We also designed a custom pipeline to build sentence embeddings using GloVe, emphasizing control over preprocessing and interpretability.

**Steps:**
1. Lowercasing and punctuation removal
2. Tokenization (NLTK)
3. Stopword removal
4. Optional: POS tagging, WordNet definitions, and dependency parsing (spaCy)
5. Mean pooling over valid word embeddings

This approach is modular and suitable for interpretability and integration with lexical tools like WordNet.

#### Cosine Similarity

Cosine similarity measures the similarity between two non-zero vectors by calculating the cosine of the angle between them. It is widely used in machine learning and data analysis, especially in text analysis, document comparison, search queries, and recommendation systems.

The formula to find the cosine similarity between two vectors is:

```
S_C(x, y) = (x · y) / (||x|| * ||y||)
```

Where:
- `x · y` = dot product of vectors 'x' and 'y'
- `||x||` and `||y||` = magnitude of vectors 'x' and 'y'
- `||x|| * ||y||` = cross product of the magnitudes

After generating word embeddings for each method, we apply cosine similarity and PCA for analysis.

#### Dimensionality Reduction and Visualization (PCA)
To visualize semantic shifts in embedding space, we reduce high-dimensional vectors to 2D using PCA (Principal Component Analysis).
Each method (BERT, GloVe, Custom) is visualized separately, highlighting the spatial proximity between original and reconstructed texts.

### Results

#### Cosine Similarity Analysis
At the end of the analysis, we use pandas to generate a table with all cosine similarity scores for each method across every text.

Our end results looked like this:

| Model   | Cosine BERT | Cosine GloVe | Cosine Custom |
|---------|-------------|--------------|---------------|
| Pegasus | 0.937536    | 0.997454     | 0.993679      |
| BART    | 0.977968    | 0.999684     | 0.997300      |
| T5      | 0.706616    | 0.997601     | 0.990438      |
| ChatGPT | 0.965841    | 0.998354     | 0.995629      |

**Observations:**

- **BART** achieved the highest similarity across all models, particularly with BERT (0.9780), indicating strong semantic preservation.
- **ChatGPT** also maintained high fidelity in all embeddings, especially within the GloVe (0.9984) and custom pipelines (0.9956).
- **Pegasus** performed very well overall, with high similarity in GloVe (0.9975) and custom embeddings (0.9937), suggesting that its reconstructions retain core semantic content. Its slightly lower BERT score (0.9375), however, may indicate subtle contextual differences not reflected in static embeddings.
- **T5**, despite scoring well in GloVe and custom embeddings, showed a significantly lower BERT similarity (0.7066), suggesting a semantic shift captured only by contextual models.
- Static embeddings (GloVe and custom) tend to produce uniformly high similarity scores, while BERT introduces more variability due to context sensitivity.

#### PCA Visualization Analysis
The PCA projections of sentence embeddings across the BERT, GloVe, and custom pipelines reveal important distinctions in how models represent semantic similarity:

**BERT PCA**
BERT captures deep contextual semantics. Pegasus appears far from the original, likely due to paraphrasing or restructuring that changes context, even if meaning is preserved. BART and ChatGPT stay closer, indicating they retain both structure and meaning more faithfully.

**GloVe PCA**
In contrast, GloVe averages static word vectors, so models like Pegasus appear closer to the original if they reuse similar vocabulary—even if phrasing changes. Pegasus clusters tightly despite being distant in BERT’s space because of that.

**Custom Pipeline PCA**
This model blends token filtering (e.g. stopword removal) with GloVe vectors, making it sensitive to lexical and structural shifts. Pegasus ends up further from the original, suggesting its paraphrasing altered important token-level content.

---

## General Discussion

**Key Questions:**
- How well did the word embeddings capture the meaning of the original text?
- Which were the biggest difficulties in reconstruction?
- How can this process be automated using NLP models?
- Were there differences in reconstruction quality between techniques, methods, libraries etc?
- Discuss your findings.

---

## Conclusions

*[Findings and difficulties to be filled in]*

---

## References

### **Part 1**

### **Question 1A**
- [GitHub Repository - NLP Lab UniPi](https://github.com/dimitris1pana/nlp_lab_unipi/tree/main/lab1)
- [Stanza - Tokenization and Sentence Segmentation](https://stanfordnlp.github.io/stanza/tokenize.html#tokenization-and-sentence-segmentation)
- [Stanza - POS Tagging](https://stanfordnlp.github.io/stanza/pos.html)
- [Stanza - Lemmatization](https://stanfordnlp.github.io/stanza/lemma.html)
- [Stanza - Dependency Parsing](https://stanfordnlp.github.io/stanza/depparse.html)
- [Stanza - Data Objects](https://stanfordnlp.github.io/stanza/data_objects.html)

### **Question 1B**
> *[To be filled in]*

### **Question 1C**
> *[To be filled in]*

### **Part 2**
- [Word Embeddings](https://www.ibm.com/think/topics/word-embeddings)
- [Sentence Transformers](https://sbert.net)
- [Gensim](https://radimrehurek.com/gensim/intro.html)
- [NLTK](https://www.nltk.org)
- [NumPy](https://numpy.org)
- [spaCy](https://spacy.io)
- [Cosine Similarity](https://www.geeksforgeeks.org/dbms/cosine-similarity/)
- [TextStat](https://pypi.org/project/textstat/)

---

## Υποδείξεις *(5/19/2025)*

> **Σημαντικές Παρατηρήσεις:**
> 
> - Όλο το documentation μπορεί να είναι στα αγγλικά
> 
> ### **1A.** Κυριολεκτικά ότι θέλουμε από το lab1
> 
> ### **1B.** Ένα ακόμα μοντέλο + περισσότερο experimentation με παραμέτρους
> 
> ### **1C.** Απλός σχολιασμός - κάτι πιο υποκειμενικό
> 
> ### **Παραδοτέο 2**
> - **ChatGPT** - παραγωγή ενός σωστού κειμένου "ground truth" για σύγκριση με τα παραγώμενα κείμενα από άλλα μοντέλα
> - **Visualization** - Πιο αντικειμενικός σχολιασμός

---

*✨ Made with love by Team NLP 2025 ✨*


