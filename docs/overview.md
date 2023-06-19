# Overview

The following diagram presents a high-level overview of the hybrid recommender system for papers in the training corpus:

![Hybrid recommender system schematic](./assets/hybrid-architecture.png)

The hybrid structure involves a **Citation Recommender** that combines global document features and citation-based features, and a **Language Recommender** that generates embeddings from paper abstracts.

The hybrid recommender combines the Citation Recommender and the Language Recommender in a *cascade* fashion, i.e. one is used to generate a candidate list which is then re-ranked by the second recommender.

Both component orders as well as the two candidate lists are evaluated.The objectives of the evaluation study are

1. To determine the best component order for the cascade strategy, i.e. Citation -> Language or Language -> Citation.
1. To investigate if the hybrid approach improves performance over single component recommenders in the first place.


## Citation Recommender

The **Citation Recommender** extracts five features from each training document out of two categories: global document features and citation-based features.

### Global Document Features

These features are derived from the document metadata in the D3 dataset.

- **Publication Date**:
    A *novelty* metric. Recent publications score higher, as they build upon earlier papers and compare their findings with existing results.

- **Paper Citation Count**:
    A *document popularity* metric. Papers with more citations are considered more valuable and relevant.

- **Author Citation Count**:
    An *author popularity* metric. Authors with higher total citations across their publications are deemed more important in the research community.

Note that global document features are identical for each query document.


### Citation-Based Features

These features are obtained from the citation data retrieved from the Semantic Scholar API and are *pairwise features* computed for each pair of documents in the training corpus.

- **Co-Citation Analysis**:
    Counts the number of shared *citing* papers, i.e. papers that themselves cite both the query and the candidate paper. Candidate documents with higher co-citation analysis scores are considered more relevant to the query document.

- **Bibliographic Coupling**:
    Counts shared *cited* papers, i.e. papers that are cited by both the query and the candidate paper. Candidate documents with higher bibliographic coupling scores are considered more relevant to the query document.


### Feature Weighting

The five features are weighted in the following manner:

- To reduce memory load, only the top-100 of all precomputed scores are stored for each feature and query document. This number is configurable before running the setup scripts through the `readnext/config.py` file.

- All training documents are then ranked by each feature individually in ascending order. The candidate paper with the best score for a given feature is assigned rank 1 for this feature, the candidate paper with the 100th best score is assigned rank 100, all worse-scoring papers are assigned rank 101.

- Thus, the weighting scheme grows linearly with a threshold at rank 100. The absolute magnitude of the original scores is not considered, but only their ranks to diminish the impact of outliers.

- Instead of using the ranks directly for feature weighting, points are computed for better interpretability. They behave like inverse ranks, i.e. the best-scoring paper for a given feature receives 100 points, the 100th best-scoring paper receives 1 point, and all worse-scoring papers receive 0 points.

- The points for each feature are combined linearly with the user-specified feature weights. Papers with the highest weighted points score are recommended.


## Language Recommender

Note: The following section assumes basic familiarity with embeddings and language models in general.
For a more thorough introduction, check out the [documentation](https://joel-beck.github.io/readnext/background/#language-models).

The **Language Recommender** encodes paper abstracts into embedding vectors to capture semantic meaning. Papers with embeddings most similar to the query document (measured by cosine similarity) are recommended.

8 language models across 3 categories are considered: keyword-based models, static embedding models, and contextual embedding models.


### Keyword-based models

They produce sparse vector embeddings where the embedding dimension equals the vocabulary size of all document abstracts in the training corpus.
For these models, text preprocessing and tokenization is performed by the `spaCy` library using the `en_core_web_sm` model.

The following keyword-based models are considered:

- TF-IDF: Implemented according to the formula:

    $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$$

    with:

    $$\text{TF}(t, d) = \frac{\text{count}(t, d)}{\text{len}(d)}$$

    and:

    $$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{DF}(t)} + 1\right)$$

    where:

    - $t$ is a token,
    - $d$ is a document,
    - $\text{TF}(t, d)$ is the term frequency of token $t$ in document $d$ (interpreted as the relative frequency of a term in a document),
    - $\text{IDF}(t)$ is the inverse document frequency of token $t$ across all documents in the training corpus,
    - $\text{count}(t, d)$ is the count of token $t$ in document $d$,
    - $\text{len}(d)$ is the total number of tokens in document $d$,
    - $\text{DF}(t)$ is the document frequency of token $t$ (the number of documents in the corpus that contain the term $t$),
    - $N$ is the total number of documents in the corpus.

- BM25: Implemented in the BM25+ variant as proposed by (Lv & Zhai, 2011) and described in (Trotman et al., 2014).

    The formula is:

    $$\text{BM25}(t, d) = \text{BM25-TF}(t, d) \cdot \text{BM25-IDF}(t)$$

    with:

    $$\text{BM25-TF}(t, d) = \frac{(k + 1) \cdot \text{TF}(t, d)}{k \cdot (1 - b + b \cdot (\text{len}(d) / \text{avgdl})) + \text{TF}(t, d)} + \delta$$

    and:

    $$\text{BM25-IDF}(t) = \log\left(\frac{N+1}{\text{DF}(t)}\right)$$

    where:

    - $t$ is a token,
    - $d$ is a document,
    - $\text{BM25-TF}(t, d)$ is the BM25+ term frequency of token $t$ in document $d$,
    - $\text{BM25-IDF}(t)$ is the BM25+ inverse document frequency of token $t$ across all documents in the training corpus,
    - $\text{TF}(t, d)$ is the term frequency of token $t$ in document $d$ (interpreted as the relative frequency of a term in a document),
    - $\text{DF}(t)$ is the document frequency of token $t$ (the number of documents in the corpus that contain the term $t$),
    - $\text{len}(d)$ is the total number of tokens in document $d$,
    - $\text{avgdl}$ is the average document length across the corpus,
    - $N$ is the total number of documents in the corpus,
    - $k$, $b$, and $\delta$ are free parameters.

    Default values of $k = 1.5$, $b = 0.75$, and $\delta = 1.0$ are adapted from the [rank_bm25 package](https://github.com/dorianbrown/rank_bm25/blob/990470ebbe6b28c18216fd1a8b18fe7446237dd6/rank_bm25.py#L176).


### Static embedding models

They produce dense vector embeddings where the embedding dimension is fixed (here set to the default of 300) and independent of the vocabulary size.
Word embeddings are averaged dimension-wise to obtain a single embedding vector for each abstract.
Again, `spaCy` is used for text preprocessing and tokenization.
All three static embedding models are pretrained and implemented via their `gensim` interface:

- Word2Vec: Pretrained on the Google News corpus using the `word2vec-google-news-300` gensim model.
- GloVe: Pretrained on the Gigaword corpus and Wikipedia using the `glove.6B.300d` model from the NLP Stanford GloVe project.
- FastText: Pretrained on the Common Crawl corpus and Wikipedia using the `cc.en.300.bin` model from the FastText Website.


### Contextual embedding models

Similar to static embedding models, they produce dense vector embeddings where the embedding dimension is fixed (here set to the default of 768) and independent of the vocabulary size.
Instead of string tokens, contextual embedding models take integer token IDs as input which are mapped to words and subwords and learned during pretraining.
All three static embedding models are pretrained and implemented via the HuggingFace `transformers` library:

- BERT: Pretrained on the BooksCorpus and English Wikipedia using the `bert-base-uncased` model.
- SciBERT: Pretrained on the Semantic Scholar corpus (i.e. specific to scientific language) using the `allenai/scibert_scivocab_uncased` model.
- Longformer: Pretrained on the BooksCorpus and English Wikipedia using the `allenai/longformer-base-4096` model.

Instead of averaging word embeddings like static embedding models, these Transformer based models cut off the document abstracts at a maximum token length of 512 for BERT and SciBERT and 4096 for the Longformer model.
However, only 0.58% of all abstracts in the training corpus exceed the maximum token length of 512 such that the impact of this cutoff is negligible.



## Evaluation Metrics

The **Mean Average Precision (MAP)** is used as evaluation metric due to the following reasons:

1. It takes the order of recommendations into account, i.e. it is not only important to recommend relevant items but also to recommend them early in the list.
1. All items on the recommendation list are considered, i.e. it is not only important to recommend relevant items but also to avoid irrelevant items.
1. It works well with binary 0/1 encoded labels as in our case for irrelevant/relevant recommendations.

The **Average Precision (AP)** computes a scalar score for a single recommendation list according to the following definitions:

**Precision**

$$\text{Precision} = \frac{\# \text{ of relevant items}}{\# \text{ of items}}$$

**Average Precision (AP)**

$$\text{AP} = \frac{1}{r} \sum_{k=1}^{K} P(k) \cdot \text{rel}(k)$$

where:

- $K$ is the total number of items,
- $r$ is the total number of relevant items,
- $P(k)$ is the precision at $k$,
- $\text{rel}(k)$ is 1 if item $k$ is relevant and 0 otherwise.

If the labels are binary 0/1 encoded as in our case, the formula simplifies to:

$$\text{AP} = \frac{1}{r} \sum_{k=1}^{K} \frac{\sum_{i=1}^{k} \text{rel}(i)}{k}$$


The Mean Average Precision is then computed as the average over the Average Precision scores for the recommendations of all query documents in the training corpus.

**Mean Average Precision (MAP)**

$$\text{MAP} = \frac{1}{Q} \sum_{q=1}^{Q} \text{AP}(q)$$

where:

- $Q$ is the total number of query documents,
- $\text{AP}(q)$ is the average precision for query document $q$.

Within this project, the MAP computes a scalar score for a given combination of Language Model Choice and Feature Weights.
Thus, to determine which Recommender order works best within the Hybrid structure, we could e.g. aggregate the MAP scores for each order over all Language Model Choices and Feature Weights.
