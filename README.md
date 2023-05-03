# readnext

![Pre-Commit](https://github.com/joel-beck/readnext/actions/workflows/pre-commit.yaml/badge.svg)
![Tests](https://github.com/joel-beck/readnext/actions/workflows/tests.yaml/badge.svg)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)

The `readnext` package provides a hybrid recommender system for computer science papers.

It is part of my master's thesis at the University of Göttingen supervised by [Corinna Breitinger](https://gipplab.org/team/corinna-breitinger/) and [Terry Ruas](https://gipplab.org/team/dr-terry-lima-ruas/).

The project is under active development.
Below you find a quick guide with the project background and installation instructions.

Comprehensive documentation with detailed explanations of how the package is structured and can be used to reproduce the results of my thesis will follow soon 🥳


## Quick Guide

### Background

Given a query document, the goal of the hybrid recommender is to propose a list of papers that are related to the query document and, thus, might be interesting follow-up reads.



The flow of the hybrid recommender system to generate recommendations is visualized in the following diagram:

![](./static/hybrid-architecture.png)

In the first step, the metadata, the bibliography and the abstract of the query document are extracted.

The metadata is used to identify the paper and retrieve a list of documents that cite the query document (here called the *citing* papers).
In contrast, the bibliography contains all papers that are cited by the query document (here called the *cited* papers).

Based on the list of citing and cited papers of the query document, the overlap of citing and cited papers of all documents in the training corpus is calculated.
This procedure is known as *co-citation analysis* and *bibliographic coupling*, respectively.

#### Component 1: The Citation Recommender

Next, separate rankings of all documents in the training corpus are created based on the following five features:

- **Publication Date**: More recent papers are rewarded by this feature and ranked higher.
- **Paper Citation Count**: Papers with a higher citation count are ranked higher.
- **Author Citation Count**: For each paper, the author with the highest lifetime citation count across all publications is identified. Papers with more popular authors in terms of citation count are ranked higher.
- **Co-Citation Analysis**: Papers with a higher number of common citing papers are ranked higher.
- **Bibliographic Coupling**: Papers with a higher number of common cited papers are ranked higher.

These individual rankings are then combined by the Citation Recommender into a single ranking using a linear combination of the individual ranks.
The top-n ranked papers are recommended to the user.
The weights of the Citation Recommender are chosen from a discrete set of weight vectors to maximize the Mean Average Precision of the generated recommendations.

#### Component 2: The Language Recommender

The raw abstract of the query document is first tokenized and preprocessed.
The tokens are then passed to each of the following 8 language models to generate a document embedding vector:

- **TF-IDF**
- **BM25**
- **Word2Vec**
- **GloVe**
- **FastText**
- **BERT**
- **SciBERT**
- **Longformer**

From the query document embedding vector, the cosine similarity to the embedding vectors of all documents in the training corpus is calculated.
The top-n papers with the highest cosine similarity are recommended to the user.

#### Hybrid Recommender

The Citation Recommender and the Language Recommender are combined into a Hybrid Recommender with the *cascade* strategy.
First, one of the two recommenders is used to generate a list of candidate papers.
Then, the other recommender re-ranks this candidate list and generates the final recommendations.

Both orderings of the recommenders are tested and evaluated.
The ordering with the highest Mean Average Precision is chosen as the final Hybrid Recommender.
The hybrid recommender scores are also compared to the scores of the candidate list to determine if the re-ranking step improved the recommendations.


### Installation