"""
Read documents, authors and arxiv metadata from raw JSON files and write it out into
Parquet format.

The raw documents and authors metadata stem from the D3 dataset which can be downloaded
from https://zenodo.org/record/7071698#.ZA7jbC8w2Lc. This project uses version 2.1 of
the D3 dataset, published on 2022-11-25.

The raw arxiv metadata can be downloaded from
https://www.kaggle.com/datasets/Cornell-University/arxiv.

The D3 data builds the foundation of this project and contributes many of the core data
features. The arxiv metadata is used to add arxiv tags which are used as labels for
evaluating the recommender system.
"""

import polars as pl

from readnext.config import DataPaths
from readnext.utils import write_df_to_parquet


def main() -> None:
    # takes roughly 15 minutes
    documents = pl.scan_ndjson(DataPaths.raw.documents_json).collect()
    authors = pl.scan_ndjson(DataPaths.raw.authors_json).collect()
    arxiv_labels = pl.scan_ndjson(DataPaths.raw.arxiv_labels_json).collect()

    write_df_to_parquet(documents, DataPaths.raw.documents_parquet)
    write_df_to_parquet(authors, DataPaths.raw.authors_parquet)
    write_df_to_parquet(arxiv_labels, DataPaths.raw.arxiv_labels_parquet)


if __name__ == "__main__":
    main()
