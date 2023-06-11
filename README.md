# readnext

[![pre-commit](https://github.com/joel-beck/readnext/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/joel-beck/readnext/actions/workflows/pre-commit.yaml)
[![pytest](https://github.com/joel-beck/readnext/actions/workflows/tests.yaml/badge.svg)](https://github.com/joel-beck/readnext/actions/workflows/tests.yaml)
[![Codecov](https://codecov.io/github/joel-beck/readnext/branch/main/graph/badge.svg?token=JL9CGV7C73)](https://codecov.io/github/joel-beck/readnext)
[![pdm-managed](https://img.shields.io/static/v1?label=pdm&message=managed&color=blueviolet&style=flat)](https://pdm.fming.dev)
[![Docs](https://img.shields.io/static/v1?label=docs&message=mkdocs&color=blue&style=flat)](https://joel-beck.github.io/readnext/)
[![License](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat)](https://github.com/joel-beck/readnext)


The `readnext` package provides a hybrid recommender system for computer science papers.
Its main objective is to suggest relevant research papers based on a given query document you might be currently exploring, streamlining your journey to discover more intriguing academic literature.

It is part of my master's thesis at the University of Göttingen supervised by [Corinna Breitinger](https://gipplab.org/team/corinna-breitinger/) and [Terry Ruas](https://gipplab.org/team/dr-terry-lima-ruas/).

The project is under active development.
Below you find the installation instructions and a brief overview of the package.
Check out the [documentation](https://joel-beck.github.io/readnext/) for background information about the citation-based methods and language models that are used in this project as well as a comprehensive user guide.

## Quick Look

```python
from readnext import readnext, LanguageModelChoice, FeatureWeights

result = readnext(
    # `Attention is all you need` query paper
    arxiv_url="https://arxiv.org/abs/1706.03762",
    language_model_choice=LanguageModelChoice.FASTTEXT,
    feature_weights=FeatureWeights(
        publication_date=1,
        citationcount_document=2,
        citationcount_author=0.5,
        co_citation_analysis=2,
        bibliographic_coupling=2,
    ),
)

print(result.recommendations.language_to_citation)
```

| candidate_d3_document_id | weighted_points | publication_date_points | citationcount_document_points | citationcount_author_points | co_citation_analysis_points | bibliographic_coupling_points | title                                                                                                           | author                 | arxiv_labels                        | semanticscholar_url                                                            | arxiv_url                        | integer_label | publication_date | citationcount_document | citationcount_author | co_citation_analysis_score | bibliographic_coupling_score |
| -----------------------: | --------------: | ----------------------: | ----------------------------: | --------------------------: | --------------------------: | ----------------------------: | :-------------------------------------------------------------------------------------------------------------- | :--------------------- | :---------------------------------- | :----------------------------------------------------------------------------- | :------------------------------- | ------------: | :--------------- | ---------------------: | -------------------: | -------------------------: | ---------------------------: |
|                 11212020 |            76.9 |                       0 |                            88 |                          56 |                          93 |                          93.5 | Neural Machine Translation by Jointly Learning to Align and Translate                                           | Yoshua Bengio          | ['cs.CL' 'cs.LG' 'cs.NE' 'stat.ML'] | https://www.semanticscholar.org/paper/fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5 | https://arxiv.org/abs/1409.0473  |             1 | 2014-09-01       |                  19996 |               372099 |                         45 |                            4 |
|                  7961699 |            70.8 |                       0 |                            83 |                           0 |                          86 |                          96.5 | Sequence to Sequence Learning with Neural Networks                                                              | Ilya Sutskever         | ['cs.CL' 'cs.LG']                   | https://www.semanticscholar.org/paper/cea967b59209c6be22829699f05b8b1ac4dc092d | https://arxiv.org/abs/1409.3215  |             1 | 2014-09-10       |                  15342 |               234717 |                         25 |                            5 |
|                  6287870 |            54.8 |                       0 |                            77 |                           0 |                          30 |                          98.5 | TensorFlow: A system for large-scale machine learning                                                           | J. Dean                | ['cs.DC' 'cs.AI']                   | https://www.semanticscholar.org/paper/46200b99c40e8586c8a0f588488ab6414119fb28 | https://arxiv.org/abs/1605.08695 |             0 | 2016-05-27       |                  13266 |               115104 |                          4 |                            7 |
|                 10716717 |            53.6 |                       0 |                            70 |                           0 |                          70 |                            61 | Feature Pyramid Networks for Object Detection                                                                   | Kaiming He             | ['cs.CV']                           | https://www.semanticscholar.org/paper/b9b4e05faa194e5022edd9eb9dd07e3d675c2b36 | https://arxiv.org/abs/1612.03144 |             0 | 2016-12-09       |                  10198 |               251467 |                         14 |                            1 |
|                  4555207 |            51.3 |                       0 |                            56 |                           0 |                          59 |                          77.5 | MobileNetV2: Inverted Residuals and Linear Bottlenecks                                                          | Liang-Chieh Chen       | ['cs.CV']                           | https://www.semanticscholar.org/paper/dd9cfe7124c734f5a6fc90227d541d3dbcd72ba4 | https://arxiv.org/abs/1801.04381 |             0 | 2018-01-13       |                   7925 |                39316 |                         10 |                            2 |
|                225039882 |            51.1 |                       0 |                            16 |                           0 |                          98 |                          77.5 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale                                      | Jakob Uszkoreit        | ['cs.CV' 'cs.AI' 'cs.LG']           | https://www.semanticscholar.org/paper/268d347e8a55b5eb82fb5e7d2f800e33c75ab18a | https://arxiv.org/abs/2010.11929 |             1 | 2020-10-22       |                   5519 |                51813 |                        185 |                            2 |
|                  1114678 |            49.6 |                       0 |                            10 |                           0 |                          89 |                            87 | Neural Machine Translation of Rare Words with Subword Units                                                     | Alexandra Birch        | ['cs.CL']                           | https://www.semanticscholar.org/paper/1af68821518f03568f913ab03fc02080247a27ff | https://arxiv.org/abs/1508.07909 |             1 | 2015-08-31       |                   4963 |                16343 |                         34 |                            3 |
|                  3429309 |            49.5 |                       0 |                            69 |                           0 |                        55.5 |                            61 | DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs | A. Yuille              | ['cs.CV']                           | https://www.semanticscholar.org/paper/cab372bc3824780cce20d9dd1c22d4df39ed081a | https://arxiv.org/abs/1606.00915 |             0 | 2016-06-02       |                   9963 |                64894 |                          9 |                            1 |
|                218971783 |            49.5 |                       0 |                            12 |                           0 |                          96 |                          77.5 | Language Models are Few-Shot Learners                                                                           | Ilya Sutskever         | ['cs.CL']                           | https://www.semanticscholar.org/paper/6b85b63579a916f705a8e10a49bd8d849d91b1fc | https://arxiv.org/abs/2005.14165 |             1 | 2020-05-28       |                   5278 |               234717 |                        149 |                            2 |
|                 13740328 |            48.4 |                       0 |                            76 |                           0 |                        44.5 |                            61 | Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification                     | Kaiming He             | ['cs.CV' 'cs.AI' 'cs.LG']           | https://www.semanticscholar.org/paper/d6f2f611da110b5b5061731be3fc4c7f45d8ee23 | https://arxiv.org/abs/1502.01852 |             1 | 2015-02-06       |                  12933 |               251467 |                          6 |                            1 |
|                  1998416 |            47.7 |                       0 |                            33 |                           0 |                          59 |                            87 | Effective Approaches to Attention-based Neural Machine Translation                                              | Christopher D. Manning | ['cs.CL']                           | https://www.semanticscholar.org/paper/93499a7c7f699b6630a86fad964536f9423bb6d0 | https://arxiv.org/abs/1508.04025 |             1 | 2015-08-17       |                   6071 |               142015 |                         10 |                            3 |
|                  3603249 |            46.3 |                       0 |                             7 |                           0 |                        66.5 |                           100 | Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation              | J. Dean                | ['cs.CL' 'cs.AI' 'cs.LG']           | https://www.semanticscholar.org/paper/dbde7dfa6cae81df8ac19ef500c42db96c3d1edd | https://arxiv.org/abs/1609.08144 |             1 | 2016-09-26       |                   4847 |               115104 |                         12 |                           10 |
|                  1629541 |            45.6 |                       0 |                            91 |                           0 |                          80 |                             0 | Fully convolutional networks for semantic segmentation                                                          | Trevor Darrell         | ['cs.CV']                           | https://www.semanticscholar.org/paper/317aee7fc081f2b137a85c4f20129007fd8e717e | https://arxiv.org/abs/1411.4038  |             0 | 2014-11-14       |                  25471 |               142117 |                         20 |                            0 |
|                206593880 |            44.7 |                       0 |                            85 |                           0 |                        82.5 |                             0 | Rethinking the Inception Architecture for Computer Vision                                                       | Christian Szegedy      | ['cs.CV']                           | https://www.semanticscholar.org/paper/23ffaa0fe06eae05817f527a47ac3291077f9e58 | https://arxiv.org/abs/1512.00567 |             0 | 2015-12-02       |                  16562 |               128072 |                         21 |                            0 |
|                 16326763 |            43.6 |                       0 |                            47 |                           0 |                        55.5 |                            61 | Continuous control with deep reinforcement learning                                                             | David Silver           | ['cs.LG' 'stat.ML']                 | https://www.semanticscholar.org/paper/024006d4c2a89f7acacc6e4438d156525b60a98f | https://arxiv.org/abs/1509.02971 |             1 | 2015-09-09       |                   7002 |                89450 |                          9 |                            1 |
|                  6719686 |            38.1 |                       0 |                            23 |                           0 |                          59 |                            61 | Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks                                               | P. Abbeel              | ['cs.LG' 'cs.AI' 'cs.CV' 'cs.NE']   | https://www.semanticscholar.org/paper/c889d6f98e6d79b89c3a6adf8a921f88fa6ba518 | https://arxiv.org/abs/1703.03400 |             1 | 2017-03-09       |                   5757 |                74628 |                         10 |                            1 |
|                  9672033 |            36.9 |                       0 |                            72 |                           0 |                        66.5 |                             0 | Convolutional Neural Networks for Sentence Classification                                                       | Yoon Kim               | ['cs.CL' 'cs.NE']                   | https://www.semanticscholar.org/paper/1f6ba0782862ec12a5ec6d7fb608523d55b0c6ba | https://arxiv.org/abs/1408.5882  |             1 | 2014-08-25       |                  10420 |                13934 |                         12 |                            0 |
|                  1687220 |            36.5 |                       0 |                            24 |                           0 |                          52 |                            61 | Improved Techniques for Training GANs                                                                           | Ian J. Goodfellow      | ['cs.LG' 'cs.CV' 'cs.NE']           | https://www.semanticscholar.org/paper/571b0750085ae3d939525e62af510ee2cee9d5ea | https://arxiv.org/abs/1606.03498 |             1 | 2016-06-10       |                   5788 |               133593 |                          8 |                            1 |
|                  6875312 |            34.9 |                       0 |                            17 |                           0 |                        36.5 |                          77.5 | Asynchronous Methods for Deep Reinforcement Learning                                                            | K. Kavukcuoglu         | ['cs.LG']                           | https://www.semanticscholar.org/paper/69e76e16740ed69f4dc55361a3d319ac2f1293dd | https://arxiv.org/abs/1602.01783 |             1 | 2016-02-04       |                   5556 |               104649 |                          5 |                            2 |
|                 14955348 |            33.5 |                       0 |                             2 |                           0 |                        36.5 |                            87 | Communication-Efficient Learning of Deep Networks from Decentralized Data                                       | D. Ramage              | ['cs.LG']                           | https://www.semanticscholar.org/paper/d1dbf643447405984eeef098b1b320dee0b3b8a7 | https://arxiv.org/abs/1602.05629 |             1 | 2016-02-17       |                   4789 |                45639 |                          5 |                            3 |


See the [Usage](#usage) section for more details and examples.


## Table of Contents <!-- omit from toc -->

- [Quick Look](#quick-look)
- [Installation](#installation)
- [Overview](#overview)
    - [Citation Recommender](#citation-recommender)
        - [Global Document Features](#global-document-features)
        - [Citation-Based Features](#citation-based-features)
        - [Feature Weighting](#feature-weighting)
    - [Language Recommender](#language-recommender)
    - [Hybrid Recommender](#hybrid-recommender)
    - [Evaluation](#evaluation)
- [Setup](#setup)
    - [Requirements](#requirements)
    - [Data and Models](#data-and-models)
        - [D3 Dataset](#d3-dataset)
        - [Citation Information](#citation-information)
        - [Arxiv Labels](#arxiv-labels)
    - [Environment Variables](#environment-variables)
    - [Setup Scripts](#setup-scripts)
        - [Dataset Construction](#dataset-construction)
        - [Citation Models](#citation-models)
        - [Language Models](#language-models)
- [Usage](#usage)
    - [Examples](#examples)
        - [Seen Query Paper](#seen-query-paper)
        - [Unseen Query Paper](#unseen-query-paper)
    - [Input Validation](#input-validation)


## Installation

Currently, the `readnext` package is not available on PyPI but can be installed directly from GitHub.

Note: The project has recently been migrated from Pandas to Polars for massive performance improvements.
Thus, it is currently recommended to install the package from the `polars` branch.
Once all unit tests have been updated and changes have been merged into the `main` branch, the installation instructions will be updated accordingly.

    ```bash
    # via HTTPS
    pip install git+https://github.com/joel-beck/readnext.git@polars#egg=readnext

    # via SSH
    pip install git+ssh://git@github.com/joel-beck/readnext.git@polars#egg=readnext
    ```

If you are interested in customizing the `readnext` package to your own needs, learn about some tips for an efficient development workflow in the [documentation](https://joel-beck.github.io/readnext/setup/#development-workflow).


## Overview

The following diagram presents a high-level overview of the hybrid recommender system for papers in the training corpus.
Check out the [documentation](https://joel-beck.github.io/readnext/overview/#inference-retrieving-recommendations) for more information how the hybrid recommender works during inference for unseen papers.

![Hybrid recommender system schematic](./docs/assets/hybrid-architecture.png)

The primary concept involves a **Citation Recommender** that combines global document features and citation-based features, and a **Language Recommender** that generates embeddings from paper abstracts.
The hybrid recommender integrates these components in a *cascade* fashion, with one recommender initially producing a candidate list, which is then re-ranked by the second recommender to yield the final recommendations.

### Citation Recommender

The **Citation Recommender** extracts five features from each training document.

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
    Counts the number of shared *citing* papers, i.e. papers that themselves cite both the query and the candidate paper. Candidate documents with higher co-citation analysis scores are considered more relevant to the query document.

- **Bibliographic Coupling**:
    Counts shared *cited* papers, i.e. papers that are cited by both the query and the candidate paper. Candidate documents with higher bibliographic coupling scores are considered more relevant to the query document.

#### Feature Weighting

The five features are weighted in the following manner:

- To reduce memory load, only the top-100 of all precomputed scores are stored for each feature and query document. This number is configurable before running the setup scripts through the `readnext/config.py` file.

- All training documents are then ranked by each feature individually in ascending order. The candidate paper with the best score for a given feature is assigned rank 1 for this feature, the candidate paper with the 100th best score is assigned rank 100, all worse-scoring papers are assigned rank 101.

- Thus, the weighting scheme grows linearly with a threshold at rank 100. The absolute magnitude of the original scores is not considered, but only their ranks to diminish the impact of outliers.

- Instead of using the ranks directly for feature weighting, points are computed for better interpretability. They behave like inverse ranks, i.e. the best-scoring paper for a given feature receives 100 points, the 100th best-scoring paper receives 1 point, and all worse-scoring papers receive 0 points.

- The points for each feature are combined linearly with the user-specified feature weights. Papers with the highest weighted points score are recommended.


### Language Recommender

Note: The following section assumes basic familiarity with the concept of word embeddings and how language models work.
For a more detailed introduction, check out the [documentation](https://joel-beck.github.io/readnext/background/#language-models).

The **Language Recommender** encodes paper abstracts into embedding vectors to capture semantic meaning. Papers with embeddings most similar to the query document (measured by cosine similarity) are recommended.

Abstracts are preprocessed and tokenized using the `spaCy` library. **Eight language models across three categories** are considered:

1. **Keyword-based models**

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

        $$\text{BM25+}(t, d) = \text{BM25-TF}(t, d) \cdot \text{BM25-IDF}(t)$$

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


2. **Static embedding models**

    They produce dense vector embeddings where the embedding dimension is fixed (here set to the default of 300) and independent of the vocabulary size.
    Again, `spaCy` is used for text preprocessing and tokenization.
    All three static embedding models are pretrained and implemented via their `gensim` interface:

    - Word2Vec: Pretrained on the Google News corpus using the `word2vec-google-news-300` gensim model.
    - GloVe: Pretrained on the Gigaword corpus and Wikipedia using the `glove.6B.300d` model from the NLP Stanford GloVe project.
    - FastText: Pretrained on the Common Crawl corpus and Wikipedia using the `cc.en.300.bin` model from the FastText Website.


3. **Contextual embedding models**

    TODO
    BERT, SciBERT, and Longformer (all provided by the `transformers` library).



### Hybrid Recommender

The hybrid recommender combines the Citation Recommender and the Language Recommender in a *cascade* fashion, i.e. one is used to generate a candidate list which is then re-ranked by the second recommender.
Both component orders as well as the two candidate lists are evaluated to determine the best component order as well as to investigate if the hybrid approach improves performance over single component recommenders in the first place.

### Evaluation
TODO: add more details
**Mean Average Precision (MAP)** is used as the evaluation metric, as it considers the order of recommendations, includes all items on the list, and works with binary labels. The MAP averages Average Precision (AP) scores over all training documents, enabling comparison between different recommender systems.





## Setup

### Requirements

This project requires Python 3.10.
Earlier versions of Python are not supported.
Future support for higher versions will be available once the `torch` and `transformers` libraries are fully compatible with Python 3.11 and beyond.


### Data and Models

To execute all scripts and reproduce project results, the following **local downloads** are necessary:

- [D3 papers and authors dataset](https://zenodo.org/record/7071698#.ZFZnCi9ByLc)
- [Arxiv dataset from Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- Pretrained [word2vec-google-news-300 Word2Vec model](https://github.com/RaRe-Technologies/gensim-data) from Gensim
- Pretrained [glove.6B GloVe model](https://nlp.stanford.edu/projects/glove/) from the Stanford NLP website
- Pretrained [English FastText model](https://fasttext.cc/docs/en/crawl-vectors.html#models) from the FastText website


#### D3 Dataset

The hybrid recommender system's training data originates from multiple sources.
The [D3 DBLP Discovery Dataset](https://github.com/jpwahle/lrec22-d3-dataset/tree/main) serves as the foundation, offering information about computer science papers and their authors.
This dataset provides global document features for the text-independent recommender as well as paper abstracts for the content-based recommender.

#### Citation Information

The D3 dataset only includes total citation and reference counts for each paper.
To obtain individual citations and references, the [Semantic Scholar API](https://api.semanticscholar.org/api-docs/graph) is employed.
A [private API key](https://www.semanticscholar.org/product/api#api-key) is recommended for a higher request rate.

#### Arxiv Labels

Arxiv categories act as labels for the recommender system.
If two papers share at least one arxiv label, the recommendation is considered relevant, and irrelevant otherwise.
Arxiv labels are extracted from the [arxiv-metadata-oai-snapshot.json](https://www.kaggle.com/datasets/Cornell-University/arxiv) dataset on Kaggle.


### Environment Variables

`readnext` needs to know the locations of local data and model files in your file system, which can be stored in any directory.
User-specific information is provided through environment variables.
The `.env_template` file in the project root directory contains a template for the expected environment variables with default values (except for the Semantic Scholar API key):

```bash
# .env_template
DOCUMENTS_METADATA_FILENAME="2022-11-30_papers.jsonl"
AUTHORS_METADATA_FILENAME="2022-11-30_authors.jsonl"
ARXIV_METADATA_FILENAME="arxiv_metadata.json"

SEMANTICSCHOLAR_API_KEY="ABC123"

DATA_DIRPATH="data"
MODELS_DIRPATH="models"
RESULTS_DIRPATH="results"
```

Explanation of the environment variables:

-  `DOCUMENTS_METADATA_FILENAME` and `AUTHORS_METADATA_FILENAME` correspond to the downloaded D3 dataset files, `ARXIV_METADATA_FILENAME` to the downloaded arxiv dataset file.
-  `SEMANTICSCHOLAR_API_KEY` represents the API key for the Semantic Scholar API.
-  `DATA_DIRPATH` is the directory path for all local data files, including downloaded and generated data files.
-  `MODELS_DIRPATH` is the directory path for all pretrained model files.
-  `RESULTS_DIRPATH` is the directory path for all stored result files, such as tokenized abstracts, numeric embeddings of abstracts, and precomputed co-citation analysis, bibliographic coupling, and cosine similarity scores.


### Setup Scripts

The inference step of the `readnext` package leverages preprocessed and precomputed data such that all recommender features and abstract embeddings are readily available.
To generate these files locally, run the following setup scripts in the specified order.
All scripts are located in the `readnext/scripts` directory.


#### Dataset Construction

These scripts are located in the `readnext/scripts/data` directory.

1. `s1_read_raw_data.py`: Reads documents, authors and arxiv metadata from raw JSON files and write it out into Parquet format.
1. `s2_merge_arxiv_labels.py`: Merges the arxiv metadata with the D3 dataset via the arxiv id. Adds arxiv labels as new feature to the dataset which are later used as ground-truth labels for the recommender system.
1. `s3_merge_authors.py`: Adds the author citationcount to the dataset and selects the most popular author for each document.
1. `s4_add_citations.py`: Sends requests to the semanticscholar API to obtain citation and reference urls for all documents in the dataset and add them as features to the dataframe.
1. `s5_add_ranks.py`: Adds rank features for global document characteristics (publication date, document citation count and author citation count) to the dataset and selects a subset of the most cited documents for the final dataset.


All further script paths are relative to the `readnext/scripts/modeling` directory.

#### Citation Models

1. `run_co_citation_analysis.py`: Precomputes co-citation analysis scores for all document pairs in the dataset.
1. `bibliographic_coupling.py`: Precomputes bibliographic coupling scores for all document pairs in the dataset.


#### Language Models

1. `tokenizer/run_tokenizer.py`: Tokenizes the abstracts of all documents in the dataset by four different tokenizers into the appropriate format for all eight language models.
1. `embedder/run_embedder_*.py`: These scripts generate sparse or dense numeric embeddings of all document abstracts for each language model. The process is split into separate scripts for each model to allow for easy parallelization.
1. `cosine_similarities/run_cosine_similarities_*.py`: Precomputes cosine similarity scores for all document pairs in the dataset for each language model. Again, multiple scripts are used for parallelization purposes.








## Usage

The user interface for generating recommendations is designed to be simple and easy to use.
It relies on the top-level `readnext()` function, which takes two required arguments and one optional keyword argument:

- An identifier for the query paper.
This can be the Semanticscholar ID, Semanticscholar URL, Arxiv ID, or Arxiv URL of the paper.
This argument is required and should be provided as a string.

    **Term Definition**:
    - The Semanticscholar *ID* is a 40-digit hexadecimal string at the end of the Semanticscholar URL after the last forward slash.
    For example, the Semanticscholar ID for the URL `https://www.semanticscholar.org/paper/67c4ffa7f9c25e9e0f0b0eac5619070f6a5d143d` is `67c4ffa7f9c25e9e0f0b0eac5619070f6a5d143d`.
    - The Arxiv *ID* is a 4-digit number followed by a dot followed by a 5-digit number at the end of the Arxiv URL after the last forward slash.
    For example, the Arxiv ID for the URL `https://arxiv.org/abs/1234.56789` is `1234.56789`.

- The language model choice for the Language Recommender, which is used to tokenize and embed the query paper's abstract.
This argument is required and should be passed using the `LanguageModelChoice` Enum, which provides autocompletion for all eight available language models.

- The feature weighting for the Citation Recommender.
This argument is optional and is submitted using an instance of the `FeatureWeights` class.
If not specified, the five features (`publication_date`, `citationcount_document`, `citationcount_author`, `co_citation_analysis`, and `bibliographic_coupling`) are given equal weights of one.
Note that the weights are normalized to sum up to one, so the absolute values are irrelevant; only the relative ratios matter.


### Examples

Inference works for both 'seen' and 'unseen' query documents, depending on whether the query document is part of the training corpus or not.


#### Seen Query Paper

If the query paper is part of the training corpus, all feature values are precomputed and inference is fast.

In the [Quick Look](#quick-look) example at the beginning, we have already obtained recommendations for the query paper "Attention is all you need" by Vaswani et al. (2017) using the `FastText` language model and custom feature weights.

Assuming we have read the top recommendation "Neural Machine Translation by Jointly Learning to Align and Translate" by Yoshua Bengio, we can continue our reading flow by extracting the identifier for the previously recommended paper from the `semantischolar_url` or `arxiv_url` columns of the previous output.
In this case, we use the `SciBERT` language model and the default feature weights of 1 for each feature:

```python
from readnext import readnext, LanguageModelChoice, FeatureWeights

result = readnext(
    # `Attention is all you need` query paper
    arxiv_url="https://arxiv.org/abs/1706.03762",
    language_model_choice=LanguageModelChoice.FASTTEXT,
    feature_weights=FeatureWeights(
        publication_date=1,
        citationcount_document=2,
        citationcount_author=0.5,
        co_citation_analysis=2,
        bibliographic_coupling=2,
    ),
)

# extract one of the paper identifiers from the previous top recommendation
semanticscholar_url = result.recommendations.citation_to_language[0, "semanticscholar_url"]

result_seen_query = readnext(
    semanticscholar_url=semanticscholar_url,
    language_model_choice=LanguageModelChoice.SCIBERT,
)
```

A message is printed to the console indicating that the query paper is part of the training corpus:

```console
> ╭──────────────────────────────────────────────────╮
> │                                                  │
> │ Query document is contained in the training data │
> │                                                  │
> ╰──────────────────────────────────────────────────╯
```

The return value of the `readnext()` function contains the following attributes:

- `document_identifier`: Contains the identifiers of the query paper.

- `document_info`: Provides information about the query paper.

- `features`: Individual dataframes that include values for `publication_date`, `citationcount_document`, `citationcount_author`, `co_citation_analysis`, `bibliographic_coupling`, `cosine_similarity`, and `feature_weights`.

- `ranks`: Individual dataframes that list the ranks of individual features.

- `points`: Individual dataframes that specify the points of individual features.

- `labels`: Individual dataframes that present the arxiv labels for all candidate papers and binary 0/1 labels related to the query paper.
These binary labels are useful for 'seen' query papers where the arxiv labels of the query paper is known.
For 'unseen' papers this information is not availabels and all binary labels are set to 0.

- `recommendations`: Individual dataframes that offer the top paper recommendations.
Recommendations are calculated for both Hybrid-Recommender orders (Citation -> Language and Language -> Citation), and this includes both the intermediate candidate lists and the final hybrid recommendations.

Let's first take a look at our new query paper:

```python
print(result_seen_query.document_info)
```

```console
> Document 11212020
> ---------------------
> Title: Neural Machine Translation by Jointly Learning to Align and Translate
> Author: Yoshua Bengio
> Publication Date: 2014-09-01
> Arxiv Labels: ['cs.CL', 'cs.LG', 'cs.NE', 'stat.ML']
> Semanticscholar URL: https://www.semanticscholar.org/paper/fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5
> Arxiv URL: https://arxiv.org/abs/1409.0473
```

Now we want to get recommendations for which papers we should read next.
Here, we choose the recommendations for the Citation -> Language Hybrid-Recommender order.

The output is a dataframe where each row represents a recommendation.
The rows are sorted in descending order by the cosine similarity between the query paper and the candidate paper since the re-ranking step is performed by the Language Recommender.
For compactness we limit the output to the top three recommendations:


```python
print(result_seen_query.recommendations.citation_to_language.head(3))
```

| candidate_d3_document_id | cosine_similarity | title                                                                                         | author                 | arxiv_labels                        | semanticscholar_url                                                            | arxiv_url                        | integer_label |
| -----------------------: | ----------------: | :-------------------------------------------------------------------------------------------- | :--------------------- | :---------------------------------- | :----------------------------------------------------------------------------- | :------------------------------- | ------------: |
|                  7961699 |          0.959016 | Sequence to Sequence Learning with Neural Networks                                            | Ilya Sutskever         | ['cs.CL' 'cs.LG']                   | https://www.semanticscholar.org/paper/cea967b59209c6be22829699f05b8b1ac4dc092d | https://arxiv.org/abs/1409.3215  |             1 |
|                  5590763 |          0.953746 | Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation | Yoshua Bengio          | ['cs.CL' 'cs.LG' 'cs.NE' 'stat.ML'] | https://www.semanticscholar.org/paper/0b544dfe355a5070b60986319a3f51fb45d1348e | https://arxiv.org/abs/1406.1078  |             1 |
|                  1998416 |          0.946671 | Effective Approaches to Attention-based Neural Machine Translation                            | Christopher D. Manning | ['cs.CL']                           | https://www.semanticscholar.org/paper/93499a7c7f699b6630a86fad964536f9423bb6d0 | https://arxiv.org/abs/1508.04025 |             1 |


Hence, we read the paper "Sequence to Sequence Learning with Neural Networks" by Ilya Sutskever et al. next.



#### Unseen Query Paper

If the query paper is not part of the training corpus, the inference step takes longer since tokenization, embedding and the computation of co-citation analysis, bibliographic coupling and cosine similarity scores has to be performed from scratch.

However, apart from a longer waiting time, **the user does not have to care about** if the query paper is part of the training corpus or not since the user interface remains the same!

As an example, we fetch recommendations for the "GPT-4 Technical Report" paper by OpenAI. This paper is too recent to be part of the training corpus.

Due to its recency, it might not have been cited that often, so we lower the weight of the `co_citation_analysis` feature. Further, we increase the `publication_date` weight and decrease the `citationcount_author` weight.
For the Language Recommender we use the `GloVe` model to embed the paper abstract.

Note that we only need to specify the weights for the features we want to change from the default value of 1

```python
from readnext import readnext, LanguageModelChoice, FeatureWeights

result_unseen_query = readnext(
    arxiv_url="https://arxiv.org/abs/2303.08774",
    language_model_choice=LanguageModelChoice.GLOVE,
    feature_weights=FeatureWeights(
        publication_date=4,
        citationcount_author=0.2,
        co_citation_analysis=0.2,
    ),
)
```

The console output informs us that the query paper is not part of the training corpus and provides some progress updates for the ongoing computations:

```console
> ╭──────────────────────────────────────────────────────╮
> │                                                      │
> │ Query document is not contained in the training data │
> │                                                      │
> ╰──────────────────────────────────────────────────────╯

> Loading training corpus................. ✅ (0.07 seconds)
> Tokenizing query abstract............... ✅ (0.41 seconds)
> Loading pretrained Glove model.......... ✅ (26.38 seconds)
> Embedding query abstract................ ✅ (0.00 seconds)
> Loading pretrained embeddings........... ✅ (0.19 seconds)
> Computing cosine similarities........... ✅ (0.05 seconds)
```

The time distribution differs between the language models. For `GloVe`, loading the large pretrained model into memory allocates by far the most time.

Now, we generate the recommendations candidate list with the Language Recommender and re-rank the candidates with the Citation Recommender.
Since the second recommender of the hybrid structure is the Citation Recommender, the output is sorted by the weighted points score of the individual features:

```python
print(result_unseen_query.recommendations.language_to_citation.head(3))
```

| candidate_d3_document_id | weighted_points | publication_date_points | citationcount_document_points | citationcount_author_points | co_citation_analysis_points | bibliographic_coupling_points | title                                                            | author          | arxiv_labels      | semanticscholar_url                                                            | arxiv_url                        | integer_label | publication_date | citationcount_document | citationcount_author | co_citation_analysis_score | bibliographic_coupling_score |
| -----------------------: | --------------: | ----------------------: | ----------------------------: | --------------------------: | --------------------------: | ----------------------------: | :--------------------------------------------------------------- | :-------------- | :---------------- | :----------------------------------------------------------------------------- | :------------------------------- | ------------: | :--------------- | ---------------------: | -------------------: | -------------------------: | ---------------------------: |
|                247951931 |              80 |                      99 |                             0 |                           0 |                          99 |                            96 | PaLM: Scaling Language Modeling with Pathways                    | Noam M. Shazeer | ['cs.CL']         | https://www.semanticscholar.org/paper/094ff971d6a8b8ff870946c9b3ce5aa173617bfb | https://arxiv.org/abs/2204.02311 |             0 | 2022-04-05       |                    145 |                51316 |                         74 |                           77 |
|                230435736 |            13.9 |                      18 |                             0 |                           0 |                        83.5 |                             0 | The Pile: An 800GB Dataset of Diverse Text for Language Modeling | Jason Phang     | ['cs.CL']         | https://www.semanticscholar.org/paper/db1afe3b3cd4cd90e41fbba65d3075dd5aebb61e | https://arxiv.org/abs/2101.00027 |             0 | 2020-12-31       |                    154 |                 1303 |                         17 |                           48 |
|                227239228 |            10.9 |                     4.5 |                             0 |                           0 |                           0 |                          51.5 | Pre-Trained Image Processing Transformer                         | W. Gao          | ['cs.CV' 'cs.LG'] | https://www.semanticscholar.org/paper/43cb4886a8056d5005702edbc51be327542b2124 | https://arxiv.org/abs/2012.00364 |             0 | 2020-12-01       |                    379 |                13361 |                          1 |                           52 |

Note that the `integer_label` column is not informative for unseen query papers and only kept for consistency.
Since no arxiv labels are available for unseen query papers they can not intersect with the arxiv labels of the candidates such that all values of the `integer_label` column are set to 0.



### Input Validation

The `pydantic` library is used for basic input validation.
For invalid user inputs the command fails early before any computations are performed with an informative error message.

The following checks are performed:

- The Semanticscholar ID must be a 40-character hexadecimal string.

- The Semanticscholar URL must be a valid URL starting with `https://www.
semanticscholar.org/paper/`.

- The Arxiv ID must start with 4 digits followed by a dot followed by 5 more
digits (e.g. `1234.56789`).

- The Arxiv URL must be a valid URL starting with `https://arxiv.org/abs/`.

- At least one of the four query paper identifiers must be provided.

- The feature weights must be non-negative numeric values.


For example, the following command fails because we assigned a negative weight to the `publication_date` feature:

```python
from readnext import readnext, LanguageModelChoice, FeatureWeights

result = readnext(
    arxiv_id="2101.03041",
    language_model_choice=LanguageModelChoice.BM25,
    feature_weights=FeatureWeights(publication_date=-1),
)
```

```console
pydantic.error_wrappers.ValidationError: 1 validation error for FeatureWeights
publication_date
  ensure this value is greater than or equal to 0 (type=value_error.number.not_ge; limit_value=0)
```
