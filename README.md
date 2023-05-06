# readnext

[![pre-commit](https://github.com/joel-beck/readnext/actions/workflows/pre-commit.yaml/badge.svg)](https://github.com/joel-beck/readnext/actions/workflows/pre-commit.yaml)
[![pytest](https://github.com/joel-beck/readnext/actions/workflows/tests.yaml/badge.svg)](https://github.com/joel-beck/readnext/actions/workflows/tests.yaml)
[![Codecov](https://codecov.io/github/joel-beck/readnext/branch/main/graph/badge.svg?token=JL9CGV7C73)](https://codecov.io/github/joel-beck/readnext)
[![pdm-managed](https://img.shields.io/static/v1?label=pdm&message=managed&color=blueviolet&style=flat)](https://pdm.fming.dev)
[![Docs](https://img.shields.io/static/v1?label=docs&message=mkdocs&color=blue&style=flat)](https://joel-beck.github.io/readnext/)
[![License](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat)](https://github.com/joel-beck/readnext)


The `readnext` package provides a hybrid recommender system for computer science papers.

It is part of my master's thesis at the University of Göttingen supervised by [Corinna Breitinger](https://gipplab.org/team/corinna-breitinger/) and [Terry Ruas](https://gipplab.org/team/dr-terry-lima-ruas/).

The project is under active development.
Below you find the installation instructions and a brief overview of the package.
Check out the [documentation](https://joel-beck.github.io/readnext/) for background information about the citation-based methods and language models that are used in this project as well as a comprehensive user guide.


## Setup

### Requirements

-   This project utilizes [pdm](https://pdm.fming.dev/) for package and dependency management.
    To install `pdm`, follow the [installation instructions](https://pdm.fming.dev/latest/#installation) on the pdm website.
-   This project requires Python 3.10.
    Earlier versions of Python are not supported.
    Future support for higher versions will be available once the `torch` and `transformers` libraries are fully compatible with Python 3.11 and beyond.



### Installation

1. Clone the repository from GitHub:

    ```bash
    # via HTTPS
    git clone https://github.com/joel-beck/readnext.git

    # via SSH
    git@github.com:joel-beck/readnext.git

    # via GitHub CLI
    gh repo clone joel-beck/readnext
    ```

2. Navigate into the project directory, build the package locally and install all dependencies:

    ```bash
    cd readnext
    pdm install
    ```

That's it! 🎉

If you are interested in customizing the `readnext` package to your own needs, learn about some tips for an efficient development workflow in the [documentation](https://joel-beck.github.io/readnext/setup/#development-workflow).



### Data and Models

To execute all scripts and reproduce project results, the following **local downloads** are necessary:

- [D3 papers and authors dataset](https://zenodo.org/record/7071698#.ZFZnCi9ByLc)
- [Arxiv dataset from Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- Pretrained [word2vec-google-news-300 Word2Vec model](https://github.com/RaRe-Technologies/gensim-data) from Gensim
- Pretrained [glove-wiki-gigaword-300 GloVe model](https://github.com/RaRe-Technologies/gensim-data) from Gensim
- Pretrained [English FastText model](https://fasttext.cc/docs/en/crawl-vectors.html#models) from FastText website


#### D3 Dataset

The hybrid recommender system's training data originates from multiple sources.
The [D3 DBLP Discovery Dataset](https://github.com/jpwahle/lrec22-d3-dataset/tree/main) serves as the foundation, offering information about computer science papers and their authors.
This dataset provides global document features for the text-independent recommender as well as paper abstracts for the content-based recommender.

#### Citation Information

The D3 dataset only includes total citation and reference counts for each paper.
To obtain individual citations and references, the [Semantic Scholar API](https://api.semanticscholar.org/api-docs/graph) is employed. A private API key is recommended for a higher request rate.

#### Arxiv Labels

Arxiv categories act as labels for the recommender system. If two papers share at least one arxiv label, the recommendation is considered relevant, and irrelevant otherwise.
Arxiv labels are extracted from the [arxiv-metadata-oai-snapshot.json](https://www.kaggle.com/datasets/Cornell-University/arxiv) dataset on Kaggle.


### Environment Variables

`readnext` needs to know the locations of local data and model files in your file system, which can be stored in any directory.
User-specific information is provided through environment variables.
The .env-template file in the project root directory contains a template for the expected environment variables with default values (except for the Semantic Scholar API key):

```bash
# .env-template
DOCUMENTS_METADATA_FILENAME="2022-11-30-papers.jsonl"
AUTHORS_METADATA_FILENAME="2022-11-30-authors.jsonl"

SEMANTICSCHOLAR_API_KEY="ABC123"

DATA_DIRPATH="data"
MODELS_DIRPATH="models"
RESULTS_DIRPATH = "results"
```

Explanation of the environment variables:

-  `DOCUMENTS_METADATA_FILENAME` and `AUTHORS_METADATA_FILENAME` correspond to the downloaded D3 dataset files.
-  `SEMANTICSCHOLAR_API_KEY` represents the API key for the Semantic Scholar API.
-  `DATA_DIRPATH` is the directory path for all local data files, including downloaded and generated data files.
-  `MODELS_DIRPATH` is the directory path for all pretrained model files.
-  `RESULTS_DIRPATH` is the directory path for all stored result files, such as tokenized abstracts, numeric embeddings of abstracts, and precomputed co-citation analysis, bibliographic coupling, and cosine similarity scores.



## Overview