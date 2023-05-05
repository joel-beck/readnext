# Setup

!!! warning "Requirements"
    -   This project uses [pdm](https://pdm.fming.dev/) as a package and dependency manager.
        To install `pdm`, follow the [installation instructions](https://pdm.fming.dev/latest/#installation) on the pdm website.
    -   This project requires Python 3.10.
        Earlier versions of Python are not supported.
        Higher versions will be supported in the future once the `torch` and `transformers` libraries are fully compatible with Python 3.11 and higher.



## Installation

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


## Data & Environment Variables

TODO:
- Explain and give examples of a `.env` file
- Give instructions how to download the D3 dataset to run all scripts and reproduce the results

## Development Workflow

Pdm provides the option to define [user scripts](https://pdm.fming.dev/latest/usage/scripts/) that can be run from the command line.
They are specified in the `pyproject.toml` file in the `[tool.pdm.scripts]` section.

The following built-in and custom user scripts are useful for the development workflow:

-  `pdm add <package name>`: Add and install a new (production) dependency to the project.
    They are automatically added to the `[project]` -> `dependencies` section of the `pyproject.toml` file.
-  `pdm add -dG dev <package name>`: Add and install a new development dependency to the project.
    They are automatically added to the `[tool.pdm.dev-dependencies]` section of the `pyproject.toml` file.
-  `pdm remove <package name>`: Remove and uninstall a dependency from the project.
-  `pdm lint`: Lint the entire project with the [ruff](https://github.com/charliermarsh/ruff) linter.
    The ruff configuration is specified in the `[tool.ruff.*]` section of the `pyproject.toml` file.
-  `pdm check`: Static type checking with [mypy](https://github.com/python/mypy).
    The mypy configuration is specified in the `[tool.mypy]` section of the `pyproject.toml` file.
-  `pdm test`: Run all unit tests with [pytest](https://github.com/pytest-dev/pytest).
    The pytest configuration is specified in the `[tool.pytest.ini_options]` section of the `pyproject.toml` file.
-  `pdm pre`: Run [pre-commit](https://github.com/pre-commit/pre-commit) on all files.
    All pre-commit hooks are specified in the `.pre-commit-config.yaml` file in the project root directory.
