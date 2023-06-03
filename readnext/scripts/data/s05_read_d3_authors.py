"""
Read authors metadata from the D3 dataset. The data can be downloaded from
https://zenodo.org/record/7071698#.ZA7jbC8w2Lc. This project uses version 2.1, published
on 2022-11-25.
"""

import polars as pl

from readnext.config import DataPaths
from readnext.utils import write_df_to_parquet


def main() -> None:
    authors = pl.scan_ndjson(DataPaths.d3.authors.raw_json).collect()
    authors_most_cited = authors.head(100_000)

    write_df_to_parquet(authors_most_cited, DataPaths.d3.authors.most_cited_parquet)
    write_df_to_parquet(authors, DataPaths.d3.authors.full_parquet)


if __name__ == "__main__":
    main()
