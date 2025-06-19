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

### Μεθοδολογία

**Τι είναι ασάφια;**
*(εξήγηση και άλλων τέτοιων εννοιών, θα εξηγηθούν σύμφωνα με τις τεχνικές που χρησιμοποιούνται σε κάθε ερώτημα π.χ. cosine similarity)*

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
> *"Ανακατασκευή του συνόλου των 2 κειμένων με χρήση 3 διαφορετικών αυτόματων βιβλιοθηκών python pipelines."*

#### Εργαλεία που χρησιμοποιήθηκαν:

> *[To be filled in]*

#### Προκλήσεις:

> *[To be filled in]*

#### Περιγραφή Μεθοδολογίας:
Υλοποιήθηκε πλήρης ανακατασκευή δύο κειμένων με χρήση 3 διαφορετικών pipelines, αξιοποιώντας...

---

### **Question 1C**
> *"Συγκρίνετε τα αποτελέσματα της κάθε προσέγγισης με τις κατάλληλες τεχνικές."*

#### Εργαλεία που χρησιμοποιήθηκαν:

> *[To be filled in]*

#### Προκλήσεις:

> *[To be filled in]*

#### Περιγραφή Μεθοδολογίας:
Πραγματοποιήθηκε ποιοτική και ποσοτική σύγκριση των παραπάνω τεχνικών...

---

## Part 2

### Word Embeddings Analysis

> *"Χρησιμοποιήστε ενσωματώσεις λέξεων (Word2Vec, GloVe, FastText, BERT(embeddings), κ.λπ.) και
δικές σας -custom- αυτόματες ροές εργασίας NLP (προεπεξεργασία, λεξιλόγιο, ενσωμάτωση
λέξεων, εννοιολογικά δέντρα κλπ) για να αναλύσετε την ομοιότητα των λέξεων πριν και μετά την
ανακατασκευή. Υπολογίστε βαθμολογίες συνημιτόνου (cosine similarity) μεταξύ των αρχικών και
των ανακατασκευασμένων εκδοχών. Συγκρίνετε τις μεθόδους ως προς τα A, B του παραδοτέου 1.
Οπτικοποιήστε τις ενσωματώσεις λέξεων για τα Α,B χρησιμοποιώντας PCA/t-SNE για να
αποδείξετε τις μετατοπίσεις στον σημασιολογικό χώρο."*

### Πειράματα & Αποτελέσματα - Παραδοτέο 2

For this part of the project we get the previously modified text from the models (**Pegasus**, **Bart**, **T5**) as well as a **ChatGPT** modified version of the original text (that we assume is the best version of the original text) and we do word by word comparison of them.

#### Libraries used:
- **textstat** *(for a quick-look of the text readability)*
- **sentence-transformers** *(for BERT based model)*
- **gensim** *(for Glove)*
- **nltk**
- **numpy**
- **cosine-similarity**
- **matplotlib**
- **PCA**
- **pandas** *(for result table)*

In order to find the similarity and the quality of the modified text we use textstat to show us the readability and the quality of the text as well as use BERT, Glove and a custom nlp pipeline to make different embeddings (based on each model) and use consine similarity to check how similar they are to the original text.

#### What is cosine similarity;

> *[To be filled in]*

#### Πριν την πλήρη ανακατασκευή

> *[To be filled in]*

#### Μετά την πλήρη ανακατασκευή

> *[To be filled in]*

---

## Συζήτηση

> **Κρίσιμες Ερωτήσεις:**
> - Πόσο καλά αποτύπωσαν οι ενσωματώσεις λέξεων το νόημα;
> - Ποιες ήταν οι μεγαλύτερες προκλήσεις στην ανακατασκευή;
> - Πώς μπορεί να αυτοματοποιηθεί αυτή η διαδικασία χρησιμοποιώντας μοντέλα NLP;
> - Υπήρξαν διαφορές στην ποιότητα ανακατασκευής μεταξύ τεχνικών, μεθόδων, βιβλιοθηκών κλπ;
> - Συζητήστε τα ευρήματά σας.

---

## Συμπέρασμα

Αναστοχασμός επί των ευρημάτων και των προκλήσεων της μελέτης.

---

## Βιβλιογραφία

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


