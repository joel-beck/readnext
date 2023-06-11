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


## Table of Contents <!-- omit from toc -->

- [Setup](#setup)
    - [Requirements](#requirements)
    - [Installation](#installation)
    - [Data and Models](#data-and-models)
        - [D3 Dataset](#d3-dataset)
        - [Citation Information](#citation-information)
        - [Arxiv Labels](#arxiv-labels)
    - [Environment Variables](#environment-variables)
    - [Setup Scripts](#setup-scripts)
- [Overview](#overview)
    - [Citation Recommender](#citation-recommender)
        - [Global Document Features](#global-document-features)
        - [Citation-Based Features](#citation-based-features)
        - [Feature Weighting](#feature-weighting)
    - [Language Recommender](#language-recommender)
    - [Hybrid Recommender](#hybrid-recommender)
    - [Evaluation](#evaluation)
- [Usage](#usage)
    - [Examples](#examples)
        - [Seen Query Paper](#seen-query-paper)
        - [Continue the Flow](#continue-the-flow)
        - [Unseen Query Paper](#unseen-query-paper)
    - [Input Validation](#input-validation)



## Setup

### Requirements

This project requires Python 3.10.
Earlier versions of Python are not supported.
Future support for higher versions will be available once the `torch` and `transformers` libraries are fully compatible with Python 3.11 and beyond.



### Installation

Currently, the `readnext` package is not available on PyPI but can be installed directly from GitHub:

    ```bash
    # via HTTPS
    pip install git+https://github.com/joel-beck/readnext.git

    # via SSH
    pip install git+ssh://git@github.com/joel-beck/readnext.git
    ```

If you are interested in customizing the `readnext` package to your own needs, learn about some tips for an efficient development workflow in the [documentation](https://joel-beck.github.io/readnext/setup/#development-workflow).



### Data and Models

To execute all scripts and reproduce project results, the following **local downloads** are necessary:

- [D3 papers and authors dataset](https://zenodo.org/record/7071698#.ZFZnCi9ByLc)
- [Arxiv dataset from Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- Pretrained [word2vec-google-news-300 Word2Vec model](https://github.com/RaRe-Technologies/gensim-data) from Gensim
- Pretrained [glove.6B GloVe model](https://nlp.stanford.edu/projects/glove/) from Stanford NLP website
- Pretrained [English FastText model](https://fasttext.cc/docs/en/crawl-vectors.html#models) from FastText website


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
All scripts are located in the `readnext/scripts/data` directory.

1. `s1_read_raw_data.py`: Reads documents, authors and arxiv metadata from raw JSON files and write it out into Parquet format.
1. `s2_merge_arxiv_labels.py`: Merges the arxiv metadata with the D3 dataset via the arxiv id. Adds arxiv labels as new feature to the dataset which are later used as ground-truth labels for the recommender system.
1. `s3_merge_authors.py`: Adds the author citationcount to the dataset and selects the most popular author for each document.
1. `s4_add_citations.py`: Sends requests to the semanticscholar API to obtain citation and reference urls for all documents in the dataset and add them as features to the dataframe.
1. `s5_add_ranks.py`: Adds rank features for global document characteristics (publication date, document citation count and author citation count) to the dataset and selects a subset of the most cited documents for the final dataset.



## Overview

The following diagram presents a high-level overview of the hybrid recommender system for papers in the training corpus.
Check out the [documentation](https://joel-beck.github.io/readnext/overview/#inference-retrieving-recommendations) for more information how the hybrid recommender works during inference for unseen papers.

![Hybrid recommender system schematic](./docs/assets/hybrid-architecture.png)

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
    TF-IDF and BM25.

2. **Static embedding models**:
    Word2Vec, GloVe, and fastText (all implemented using their `gensim` interface).

3. **Contextual embedding models**:
    BERT, SciBERT, and Longformer (all provided by the `transformers` library).

All static and contextual embedding models are pre-trained on extensive text corpora.

### Hybrid Recommender

The hybrid recommender combines the citation recommender and the language recommender in a *cascade* fashion. Both component orders are considered, and evaluation scores are computed to determine the best component order and if the cascade approach improves performance.

### Evaluation

**Mean Average Precision (MAP)** is used as the evaluation metric, as it considers the order of recommendations, includes all items on the list, and works with binary labels. The MAP averages Average Precision (AP) scores across the entire corpus, enabling comparison between different recommender systems.


## Usage

The user interface for inference, i.e. to retrieve recommendations, is kept simple and intuitive:
The top-level `readnext()` function has two required and one optional keyword arguments:

- An identifier for the query paper which the recommendations are based on. This can be either the Semanticscholar ID, the Semantischolar URL, the Arxiv ID, or the Arxiv URL of the query paper.
This input is required and passed as a string.

- The language model choice for the Language Recommender, i.e. for tokenizing and embedding the query paper's abstract. This input is required and passed via the `LangaugeModelChoice` Enum providing autocompletion for all eight available language models.

- The feature weighting for the Citation Recommender. This input is passed via an `FeatureWeights` instance. The argument is optional: By default, all five features (`publication_date`, `citationcount_document`, `citationcount_authot`, `co_citation_analysis` and `bibliographic_coupling`) contribute equally with weights of one. Note that the absolute magnitude of the weights is irrelevant and only their relative proportions matter, as the weights are normalized to sum to one.

### Examples

Inference works for both 'seen' and 'unseen' query documents, depending on whether the query document is part of the training corpus or not.

#### Seen Query Paper

If the query paper is part of the training corpus, all feature values are precomputed and inference is fast.
As an example we choose the popular "Attention is all you need" paper by Vaswani et al. (2017) using the "FastText" language model and the default feature weights:

```python
from readnext import readnext, LanguageModelChoice, FeatureWeights

result = readnext(
    arxiv_url="https://arxiv.org/abs/1706.03762", language_model_choice=LanguageModelChoice.FASTTEXT
)
```

A message is printed to the console to indicate that the query paper is part of the training corpus:

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
Recommendations are calculated for both Hybrid-Recommender orders (Citation -> Language and Language -> Citation), and this includes both the    intermediate candidate lists and the final hybrid recommendations.

Let's first take a look at our query paper:

```python
print(result.document_info)
```

```console
> Document 13756489
> ---------------------
> Title: Attention is All you Need
> Author: Lukasz Kaiser
> Arxiv Labels: ['cs.CL', 'cs.LG']
> Semanticscholar URL: https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776
> Arxiv URL: https://arxiv.org/abs/1706.03762
```

Now we want to get recommendations for which papers we should read next.
Here, we choose the recommendations for the Citation -> Language Hybrid-Recommender order.

The output is a dataframe where each row represents a recommendation.
The rows are sorted in descending order by the cosine similarity between the query paper and the candidate paper since the re-ranking step is performed by the Language Recommender:

```python
print(result.recommendations.citation_to_language)
```

```console
| candidate_d3_document_id | cosine_similarity | title                                                                                         | author               | arxiv_labels                        | semanticscholar_url                                                            | arxiv_url                        | integer_label |
| -----------------------: | ----------------: | :-------------------------------------------------------------------------------------------- | :------------------- | :---------------------------------- | :----------------------------------------------------------------------------- | :------------------------------- | ------------: |
|                 11212020 |          0.892914 | Neural Machine Translation by Jointly Learning to Align and Translate                         | Yoshua Bengio        | ['cs.CL' 'cs.LG' 'cs.NE' 'stat.ML'] | https://www.semanticscholar.org/paper/fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5 | https://arxiv.org/abs/1409.0473  |             1 |
|                  7961699 |          0.891895 | Sequence to Sequence Learning with Neural Networks                                            | Ilya Sutskever       | ['cs.CL' 'cs.LG']                   | https://www.semanticscholar.org/paper/cea967b59209c6be22829699f05b8b1ac4dc092d | https://arxiv.org/abs/1409.3215  |             1 |
|                 10716717 |          0.877233 | Feature Pyramid Networks for Object Detection                                                 | Kaiming He           | ['cs.CV']                           | https://www.semanticscholar.org/paper/b9b4e05faa194e5022edd9eb9dd07e3d675c2b36 | https://arxiv.org/abs/1612.03144 |             0 |
|                225039882 |          0.867628 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale                    | Jakob Uszkoreit      | ['cs.CV' 'cs.AI' 'cs.LG']           | https://www.semanticscholar.org/paper/268d347e8a55b5eb82fb5e7d2f800e33c75ab18a | https://arxiv.org/abs/2010.11929 |             1 |
|                  4555207 |          0.860428 | MobileNetV2: Inverted Residuals and Linear Bottlenecks                                        | Liang-Chieh Chen     | ['cs.CV']                           | https://www.semanticscholar.org/paper/dd9cfe7124c734f5a6fc90227d541d3dbcd72ba4 | https://arxiv.org/abs/1801.04381 |             0 |
|                  6287870 |          0.855834 | TensorFlow: A system for large-scale machine learning                                         | J. Dean              | ['cs.DC' 'cs.AI']                   | https://www.semanticscholar.org/paper/46200b99c40e8586c8a0f588488ab6414119fb28 | https://arxiv.org/abs/1605.08695 |             0 |
|                  1055111 |          0.854223 | Show, Attend and Tell: Neural Image Caption Generation with Visual Attention                  | Yoshua Bengio        | ['cs.LG' 'cs.CV']                   | https://www.semanticscholar.org/paper/4d8f2d14af5991d4f0d050d22216825cac3157bd | https://arxiv.org/abs/1502.03044 |             1 |
|                  6628106 |          0.846944 | Adam: A Method for Stochastic Optimization                                                    | Diederik P. Kingma   | ['cs.LG']                           | https://www.semanticscholar.org/paper/a6cb366736791bcccc5c8639de5a8f9636bf87e8 | https://arxiv.org/abs/1412.6980  |             1 |
|                  3626819 |          0.844104 | Deep Contextualized Word Representations                                                      | Kenton Lee           | ['cs.CL']                           | https://www.semanticscholar.org/paper/3febb2bed8865945e7fddc99efd791887bb7e14f | https://arxiv.org/abs/1802.05365 |             1 |
|                  5808102 |          0.843033 | Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift  | Christian Szegedy    | ['cs.LG']                           | https://www.semanticscholar.org/paper/4d376d6978dad0374edfa6709c9556b42d3594d3 | https://arxiv.org/abs/1502.03167 |             1 |
|                 12670695 |          0.836785 | MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications            | Hartwig Adam         | ['cs.CV']                           | https://www.semanticscholar.org/paper/3647d6d0f151dc05626449ee09cc7bce55be497e | https://arxiv.org/abs/1704.04861 |             0 |
|                  9433631 |          0.820209 | Densely Connected Convolutional Networks                                                      | Kilian Q. Weinberger | ['cs.CV' 'cs.LG']                   | https://www.semanticscholar.org/paper/5694e46284460a648fe29117cbc55f6c9be3fa3c | https://arxiv.org/abs/1608.06993 |             1 |
|                  5590763 |          0.813149 | Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation | Yoshua Bengio        | ['cs.CL' 'cs.LG' 'cs.NE' 'stat.ML'] | https://www.semanticscholar.org/paper/0b544dfe355a5070b60986319a3f51fb45d1348e | https://arxiv.org/abs/1406.1078  |             1 |
|                198953378 |          0.802215 | RoBERTa: A Robustly Optimized BERT Pretraining Approach                                       | Luke Zettlemoyer     | ['cs.CL']                           | https://www.semanticscholar.org/paper/077f8329a7b6fa3b7c877a57b81eb6c18b5f87de | https://arxiv.org/abs/1907.11692 |             1 |
|                 10328909 |          0.799677 | Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks                | Kaiming He           | ['cs.CV']                           | https://www.semanticscholar.org/paper/424561d8585ff8ebce7d5d07de8dbf7aae5e7270 | https://arxiv.org/abs/1506.01497 |             0 |
|                  6200260 |          0.799408 | Image-to-Image Translation with Conditional Adversarial Networks                              | Alexei A. Efros      | ['cs.CV']                           | https://www.semanticscholar.org/paper/8acbe90d5b852dadea7810345451a99608ee54c7 | https://arxiv.org/abs/1611.07004 |             0 |
|                206594692 |           0.78717 | Deep Residual Learning for Image Recognition                                                  | Kaiming He           | ['cs.CV']                           | https://www.semanticscholar.org/paper/2c03df8b48bf3fa39054345bafabfeff15bfd11d | https://arxiv.org/abs/1512.03385 |             0 |
|                  3144218 |          0.785984 | Semi-Supervised Classification with Graph Convolutional Networks                              | M. Welling           | ['cs.LG' 'stat.ML']                 | https://www.semanticscholar.org/paper/36eff562f65125511b5dfab68ce7f7a943c27478 | https://arxiv.org/abs/1609.02907 |             1 |
|                  5201925 |          0.783595 | Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling                  | Yoshua Bengio        | ['cs.NE' 'cs.LG']                   | https://www.semanticscholar.org/paper/adfcf065e15fd3bc9badf6145034c84dfb08f204 | https://arxiv.org/abs/1412.3555  |             1 |
|                 52967399 |          0.779854 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding              | Kenton Lee           | ['cs.CL']                           | https://www.semanticscholar.org/paper/df2b0e26d0599ce3e70df8a9da02e51594e0e992 | https://arxiv.org/abs/1810.04805 |             1 |
```

Hence, we read the "Neural Machine Translation by Jointly Learning to Align and Translate" paper by Yoshua Bengio next.


#### Continue the Flow

Since the recommendations dataframe contains identifiers for the candidate papers, we can continue our reading flow by generating recommendations for the next paper.

In this case, we want to use the `SciBERT` language model, assign a higher weight to the `co_citation_analysis` and `bibliographic coupling` features and disregard the author popularity by setting the `citationcount_author` weight to 0.

Note that we only have to specify the weights for the features we want to change from the default value of 1:

```python
# extract one of the paper identifiers from the previous top recommendation
semanticscholar_url = result.recommendations.citation_to_language[0, "semanticscholar_url"]

next_result = readnext(
    semanticscholar_url=semanticscholar_url,
    language_model_choice=LanguageModelChoice.SCIBERT,
    feature_weights=FeatureWeights(
        citationcount_author=0, co_citation_analysis=3, bibliographic_coupling=3
    ),
)
```

Now, we generate the recommendations candidate list with the Language Recommender and re-rank the candidates with the Citation Recommender:

```python
print(next_result.recommendations.language_to_citation)
```

Since the second recommender is the Citation Recommender, the output is sorted by the weighted points score of the individual features:

```console
| candidate_d3_document_id | weighted_points | publication_date_points | citationcount_document_points | citationcount_author_points | co_citation_analysis_points | bibliographic_coupling_points | title                                                                                                           | author                 | arxiv_labels                        | semanticscholar_url                                                            | arxiv_url                        | integer_label | publication_date | citationcount_document | citationcount_author | co_citation_analysis_score | bibliographic_coupling_score |
| -----------------------: | --------------: | ----------------------: | ----------------------------: | --------------------------: | --------------------------: | ----------------------------: | :-------------------------------------------------------------------------------------------------------------- | :--------------------- | :---------------------------------- | :----------------------------------------------------------------------------- | :------------------------------- | ------------: | :--------------- | ---------------------: | -------------------: | -------------------------: | ---------------------------: |
|                  7961699 |            85.4 |                       0 |                            83 |                           0 |                         100 |                           100 | Sequence to Sequence Learning with Neural Networks                                                              | Ilya Sutskever         | ['cs.CL' 'cs.LG']                   | https://www.semanticscholar.org/paper/cea967b59209c6be22829699f05b8b1ac4dc092d | https://arxiv.org/abs/1409.3215  |             1 | 2014-09-10       |                  15342 |               234717 |                        191 |                           12 |
|                  5590763 |            84.8 |                       0 |                            84 |                        56.5 |                          99 |                            99 | Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation                   | Yoshua Bengio          | ['cs.CL' 'cs.LG' 'cs.NE' 'stat.ML'] | https://www.semanticscholar.org/paper/0b544dfe355a5070b60986319a3f51fb45d1348e | https://arxiv.org/abs/1406.1078  |             1 | 2014-06-03       |                  15720 |               372099 |                        149 |                           11 |
|                 13756489 |            80.8 |                       0 |                            96 |                           0 |                          91 |                          92.5 | Attention is All you Need                                                                                       | Lukasz Kaiser          | ['cs.CL' 'cs.LG']                   | https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776 | https://arxiv.org/abs/1706.03762 |             1 | 2017-06-12       |                  38868 |                61163 |                         45 |                            4 |
|                  1998416 |            75.6 |                       0 |                            33 |                           0 |                          98 |                          92.5 | Effective Approaches to Attention-based Neural Machine Translation                                              | Christopher D. Manning | ['cs.CL']                           | https://www.semanticscholar.org/paper/93499a7c7f699b6630a86fad964536f9423bb6d0 | https://arxiv.org/abs/1508.04025 |             1 | 2015-08-17       |                   6071 |               142015 |                        116 |                            4 |
|                  3626819 |            75.3 |                       0 |                            58 |                           0 |                          92 |                          89.5 | Deep Contextualized Word Representations                                                                        | Kenton Lee             | ['cs.CL']                           | https://www.semanticscholar.org/paper/3febb2bed8865945e7fddc99efd791887bb7e14f | https://arxiv.org/abs/1802.05365 |             1 | 2018-02-15       |                   8314 |                50225 |                         53 |                            3 |
|                  5959482 |            74.5 |                       0 |                            89 |                           0 |                          94 |                            75 | Efficient Estimation of Word Representations in Vector Space                                                    | J. Dean                | ['cs.CL']                           | https://www.semanticscholar.org/paper/330da625c15427c6e42ccfa3b747fb29e5835bf0 | https://arxiv.org/abs/1301.3781  |             1 | 2013-01-16       |                  22770 |               115104 |                         65 |                            1 |
|                  9672033 |            74.4 |                       0 |                            72 |                           0 |                          89 |                          85.5 | Convolutional Neural Networks for Sentence Classification                                                       | Yoon Kim               | ['cs.CL' 'cs.NE']                   | https://www.semanticscholar.org/paper/1f6ba0782862ec12a5ec6d7fb608523d55b0c6ba | https://arxiv.org/abs/1408.5882  |             1 | 2014-08-25       |                  10420 |                13934 |                         36 |                            2 |
|                  1114678 |            71.8 |                       0 |                            10 |                           0 |                          93 |                            95 | Neural Machine Translation of Rare Words with Subword Units                                                     | Alexandra Birch        | ['cs.CL']                           | https://www.semanticscholar.org/paper/1af68821518f03568f913ab03fc02080247a27ff | https://arxiv.org/abs/1508.07909 |             1 | 2015-08-31       |                   4963 |                16343 |                         56 |                            5 |
|                  3603249 |            70.8 |                       0 |                             7 |                           0 |                          90 |                          96.5 | Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation              | J. Dean                | ['cs.CL' 'cs.AI' 'cs.LG']           | https://www.semanticscholar.org/paper/dbde7dfa6cae81df8ac19ef500c42db96c3d1edd | https://arxiv.org/abs/1609.08144 |             1 | 2016-09-26       |                   4847 |               115104 |                         44 |                            7 |
|                206741496 |            67.7 |                       0 |                            48 |                         3.5 |                          75 |                          89.5 | Speech recognition with deep recurrent neural networks                                                          | Geoffrey E. Hinton     | ['cs.NE' 'cs.CL']                   | https://www.semanticscholar.org/paper/4177ec52d1b80ed57f2e72b0f9a42365f1a8598d | https://arxiv.org/abs/1303.5778  |             1 | 2013-03-22       |                   7072 |               360601 |                         21 |                            3 |
|                207556454 |            65.2 |                       0 |                            46 |                           0 |                        83.5 |                            75 | Enriching Word Vectors with Subword Information                                                                 | Tomas Mikolov          | ['cs.CL' 'cs.LG']                   | https://www.semanticscholar.org/paper/e2dba792360873aef125572812f3673b1a85d850 | https://arxiv.org/abs/1607.04606 |             1 | 2016-07-15       |                   6876 |                92866 |                         25 |                            1 |
|                  5808102 |            65.1 |                       0 |                            92 |                           0 |                          68 |                            75 | Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift                    | Christian Szegedy      | ['cs.LG']                           | https://www.semanticscholar.org/paper/4d376d6978dad0374edfa6709c9556b42d3594d3 | https://arxiv.org/abs/1502.03167 |             1 | 2015-02-10       |                  30576 |               128072 |                         17 |                            1 |
|                  6254678 |            61.8 |                       0 |                             5 |                           0 |                          88 |                            75 | WaveNet: A Generative Model for Raw Audio                                                                       | K. Simonyan            | ['cs.SD' 'cs.LG']                   | https://www.semanticscholar.org/paper/df0402517a7338ae28bc54acaac400de6b456a46 | https://arxiv.org/abs/1609.03499 |             1 | 2016-09-12       |                   4834 |               117421 |                         29 |                            1 |
|                  2407601 |            61.3 |                       0 |                            51 |                           0 |                        71.5 |                            75 | Distributed Representations of Sentences and Documents                                                          | Quoc V. Le             | ['cs.CL' 'cs.AI' 'cs.LG']           | https://www.semanticscholar.org/paper/f527bcfb09f32e6a4a8afc0b37504941c1ba2cee | https://arxiv.org/abs/1405.4053  |             1 | 2014-05-16       |                   7273 |               107587 |                         18 |                            1 |
|                   351666 |            57.9 |                       0 |                            45 |                           0 |                        64.5 |                            75 | Natural Language Processing (Almost) from Scratch                                                               | K. Kavukcuoglu         | ['cs.LG' 'cs.CL']                   | https://www.semanticscholar.org/paper/bc1022b031dc6c7019696492e8116598097a8c12 | https://arxiv.org/abs/1103.0398  |             1 | 2011-02-01       |                   6779 |               104649 |                         16 |                            1 |
|                215827080 |            32.3 |                       0 |                            87 |                           0 |                          57 |                             0 | Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation                                | Ross B. Girshick       | ['cs.CV']                           | https://www.semanticscholar.org/paper/2f4df08d9072fc2ac181b7fced6a245315ce05c8 | https://arxiv.org/abs/1311.2524  |             0 | 2013-11-11       |                  18047 |               190081 |                         14 |                            0 |
|                  3429309 |            27.2 |                       0 |                            69 |                           0 |                        49.5 |                             0 | DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs | A. Yuille              | ['cs.CV']                           | https://www.semanticscholar.org/paper/cab372bc3824780cce20d9dd1c22d4df39ed081a | https://arxiv.org/abs/1606.00915 |             0 | 2016-06-02       |                   9963 |                64894 |                         12 |                            0 |
|                 17127188 |            22.8 |                       0 |                            20 |                           0 |                          54 |                             0 | Multi-Scale Context Aggregation by Dilated Convolutions                                                         | V. Koltun              | ['cs.CV']                           | https://www.semanticscholar.org/paper/7f5fc84819c0cf94b771fe15141f65b123f7b8ec | https://arxiv.org/abs/1511.07122 |             0 | 2015-11-23       |                   5655 |                37311 |                         13 |                            0 |
|                 11797475 |            16.8 |                       0 |                            19 |                           0 |                        38.5 |                             0 | Two-Stream Convolutional Networks for Action Recognition in Videos                                              | Andrew Zisserman       | ['cs.CV']                           | https://www.semanticscholar.org/paper/67dccc9a856b60bdc4d058d83657a089b8ad4486 | https://arxiv.org/abs/1406.2199  |             0 | 2014-06-09       |                   5636 |               226816 |                          9 |                            0 |
|                211096730 |            13.1 |                       0 |                            13 |                         3.5 |                        30.5 |                             0 | A Simple Framework for Contrastive Learning of Visual Representations                                           | Geoffrey E. Hinton     | ['cs.LG' 'cs.CV' 'stat.ML']         | https://www.semanticscholar.org/paper/34733eaf66007516347a40ad5d9bbe1cc9dacb6b | https://arxiv.org/abs/2002.05709 |             1 | 2020-02-13       |                   5312 |               360601 |                          7 |                            0 |
```

Thus, we continue our reading session with the "Sequence to Sequence Learning with Neural Networks" paper by Ilya Sutskever et al.


#### Unseen Query Paper

TODO


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
