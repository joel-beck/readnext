# Setup

!!! warning "Requirements"
    -   This project utilizes [pdm](https://pdm.fming.dev/) for package and dependency management.
        To install `pdm`, follow the [installation instructions](https://pdm.fming.dev/latest/#installation) on the pdm website.
    -   This project requires Python 3.10.
        Earlier versions of Python are not supported.
        Future support for higher versions will be available once the `torch` and `transformers` libraries are fully compatible with Python 3.11 and beyond.



## Installation

1. Clone the repository from GitHub:

    === "HTTPS"

        ```bash
        git clone https://github.com/joel-beck/readnext.git
        ```

    === "SSH"

        ```bash
        git clone ssh://git@github.com:joel-beck/readnext.git
        ```

    === "GitHub CLI"

        ```bash
        gh repo clone joel-beck/readnext
        ```

2. Navigate into the project directory, build the package locally and install all dependencies:

    ```bash
    cd readnext
    pdm install
    ```

That's it! 🎉



## Data and Models

!!! note

    To execute all scripts and reproduce project results, the following **local downloads** are necessary:

    - [D3 papers and authors dataset](https://zenodo.org/record/7071698#.ZFZnCi9ByLc)
    - [Arxiv dataset from Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
    - Pretrained [word2vec-google-news-300 Word2Vec model](https://github.com/RaRe-Technologies/gensim-data) from Gensim
    - Pretrained [glove-wiki-gigaword-300 GloVe model](https://github.com/RaRe-Technologies/gensim-data) from Gensim
    - Pretrained [English FastText model](https://fasttext.cc/docs/en/crawl-vectors.html#models) from FastText website



### D3 Dataset

The hybrid recommender system's training data originates from multiple sources.
The [D3 DBLP Discovery Dataset](https://github.com/jpwahle/lrec22-d3-dataset/tree/main) serves as the foundation, offering information about computer science papers and their authors.
This dataset provides global document features for the text-independent recommender as well as paper abstracts for the content-based recommender.

### Citation Information

The D3 dataset only includes total citation and reference counts for each paper.
To obtain individual citations and references, the [Semantic Scholar API](https://api.semanticscholar.org/api-docs/graph) is employed. A private API key is recommended for a higher request rate.


### Arxiv Labels

Arxiv categories act as labels for the recommender system. If two papers share at least one arxiv label, the recommendation is considered relevant, and irrelevant otherwise.
Arxiv labels are extracted from the [arxiv-metadata-oai-snapshot.json](https://www.kaggle.com/datasets/Cornell-University/arxiv) dataset on Kaggle.

## Environment Variables

`readnext` needs to know the locations of local data and model files in your file system, which can be stored in any directory.
User-specific information is provided through environment variables.
The `.env_template` file in the project root directory contains a template for the expected environment variables with default values (except for the Semantic Scholar API key):

```bash title=".env_template"
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



## Development Workflow

Pdm provides the option to define [user scripts](https://pdm.fming.dev/latest/usage/scripts/) that can be run from the command line.
These scripts are specified in the `pyproject.toml` file in the `[tool.pdm.scripts]` section.

The following built-in and custom user scripts are useful for the development workflow:

-  `pdm add <package name>`: Add and install a new (production) dependency to the project.
    They are automatically added to the `[project]` -> `dependencies` section of the `pyproject.toml` file.
-  `pdm add -dG dev <package name>`: Add and install a new development dependency to the project.
    They are automatically added to the `[tool.pdm.dev-dependencies]` section of the `pyproject.toml` file.
-  `pdm remove <package name>`: Remove and uninstall a dependency from the project.
-  `pdm format`: Format the entire project with [black](https://github.com/psf/black).
    The black configuration is specified in the `[tool.black]` section of the `pyproject.toml` file.
-  `pdm lint`: Lint the entire project with the [ruff](https://github.com/charliermarsh/ruff) linter.
    The ruff configuration is specified in the `[tool.ruff.*]` sections of the `pyproject.toml` file.
-  `pdm check`: Static type checking with [mypy](https://github.com/python/mypy).
    The mypy configuration is specified in the `[tool.mypy]` section of the `pyproject.toml` file.
-  `pdm test`: Run all unit tests with [pytest](https://github.com/pytest-dev/pytest).
    The pytest configuration is specified in the `[tool.pytest.ini_options]` section of the `pyproject.toml` file.
-  `pdm pre`: Run [pre-commit](https://github.com/pre-commit/pre-commit) on all files.
    All pre-commit hooks are specified in the `.pre-commit-config.yaml` file in the project root directory.
