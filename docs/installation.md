# Installation

Currently, the `readnext` package is not available on PyPI but can be installed directly from GitHub.

This project requires Python 3.10.
Earlier versions of Python are not supported.
Future support for higher versions will be available once the `torch` and `transformers` libraries are fully compatible with Python 3.11 and beyond.

!!! note

    The project has recently been migrated from Pandas to Polars for massive performance improvements.
    Thus, it is currently recommended to install the package from the `polars` branch.
    Once all unit tests have been updated and changes have been merged into the `main` branch, the installation instructions will be updated accordingly.


=== "HTTPS"

    ```bash
    pip install git+https://github.com/joel-beck/readnext.git@polars#egg=readnext
    ```

=== "SSH"

    ```bash
    pip install git+ssh://git@github.com/joel-beck/readnext.git@polars#egg=readnext
    ```
