# Overview

This section elucidates the hybrid recommender's recommendation process during training and inference phases.

## Training: Precomputing Features

The following diagram presents a high-level overview of the hybrid recommender system for papers in the training corpus:

![Hybrid recommender system schematic](./assets/hybrid-architecture.png)

The primary concept involves a **Citation Recommender** that combines global document features and citation-based features, and a **Language Recommender** that generates embeddings from paper abstracts.
The hybrid recommender integrates these components in a *cascade* fashion, with one recommender initially producing a candidate list, which is then re-ranked by the second recommender to yield the final recommendations.

### Citation Recommender

The **Citation Recommender** extracts five features from each training document:

#### Global Document Features

These features are derived from the document metadata in the D3 dataset.

- **Publication Date**:
    A *novelty* metric. Recent publications score higher, as they build upon earlier papers and compare their findings with existing results.

- **Paper Citation Count**:
    A *document popularity* metric. Papers with more citations are considered more valuable and relevant.

- **Author Citation Count**:
    An *author popularity* metric. Authors with higher total citations across their publications are deemed more important in the research community.

Note that global document features are identical for each query document.

#### Citation-Based Features

These features are obtained from the citation data retrieved from the Semantic Scholar API and are *pairwise features* computed for each pair of documents in the training corpus.

- **Co-Citation Analysis**:
    Counts shared *citing* papers. Candidate documents with higher co-citation analysis scores are considered more relevant to the query document.

- **Bibliographic Coupling**:
    Counts shared *cited* papers. Candidate documents with higher bibliographic coupling scores are considered more relevant to the query document.

#### Feature Weighting

To combine features linearly, documents are first *ranked* by each feature. Then, a linear combination of these ranks is calculated to produce a weighted ranking, where papers with the lowest weighted rank are recommended. The weight vector yielding the best performance (Mean Average Precision) is selected.

### Language Recommender

The **Language Recommender** encodes paper abstracts into embedding vectors to capture semantic meaning. Papers with embeddings most similar to the query document (measured by cosine similarity) are recommended.

Abstracts are preprocessed and tokenized using the `spaCy` library. **Eight language models across three categories** are considered:

1. **Keyword-based models**:
    TF-IDF (implemented using `scikit-learn` library) and BM25.

2. **Static embedding models**:
    Word2Vec, GloVe, and fastText (all implemented using their `gensim` interface).

3. **Contextual embedding models**:
    BERT, SciBERT, and Longformer (all provided by the `transformers` library).

All static and contextual embedding models are pre-trained on extensive text corpora.

### Hybrid Recommender

The hybrid recommender combines the citation recommender and the language recommender in a *cascade* fashion. Both component orders are considered, and evaluation scores are computed to determine the best component order and if the cascade approach improves performance.

### Evaluation

**Mean Average Precision (MAP)** is used as the evaluation metric, as it considers the order of recommendations, includes all items on the list, and works with binary labels. The MAP averages Average Precision (AP) scores across the entire corpus, enabling comparison between different recommender systems.


## Inference: Retrieving Recommendations

When it comes to inference, our goal is to obtain a list of recommendations for a single query document.
For example, a user might have just read or is currently reading a paper, and now they want to find other papers that may interest them.

We need to consider two scenarios, depending on whether the query document is part of the training corpus or not.

### Query Document in the Training Corpus

If the query document is part of the training corpus, getting recommendations is fast – it's just a simple lookup of precomputed values!
The user needs to supply a unique identifier for the query document, which could be the document ID within the D3 dataset, the Semantic Scholar URL of the paper, or in most cases, even the paper's title.

From the database, you can retrieve any piece of information, including individual features, abstract embeddings, pairwise co-citation analysis, bibliographic coupling, cosine similarity scores, and the final recommendation list.

### Query Document Not in the Training Corpus

!!! info

    This part of the project is still a work in progress.
    We'll soon add detailed documentation on how users can get recommendations for unseen papers 🚀

When the query document isn't part of the training corpus, the inference process is a bit more complex.
In principle, users have to provide all model inputs, such as the query document's abstract, global document characteristics, arxiv labels, and complete lists of citing and cited papers.

Since citation information might not be readily available, our goal is to automatically fetch this data from the Semantic Scholar API using only the query document's Semantic Scholar URL.

Keep in mind that inference will be considerably slower for unseen papers compared to those in the training corpus.
This is because abstract embeddings and pairwise co-citation analysis, bibliographic coupling, and cosine similarity scores concerning all other papers in the training corpus need to be computed from scratch.
