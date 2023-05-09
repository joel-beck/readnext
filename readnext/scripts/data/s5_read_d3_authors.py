"""
Read authors metadata from the D3 dataset. The data can be downloaded from
https://zenodo.org/record/7071698#.ZA7jbC8w2Lc. This project uses version 2.1, published
on 2022-11-25.
"""

import pandas as pd

from readnext.config import DataPaths


def main() -> None:
    authors: pd.DataFrame = pd.read_json(DataPaths.d3.authors.raw_json, lines=True).convert_dtypes()

    authors_most_cited = authors.sort_values(by="citationcount", ascending=False).iloc[:100_000]

    authors_most_cited.to_pickle(DataPaths.d3.authors.most_cited_pkl)
    authors.to_pickle(DataPaths.d3.authors.full_pkl)


if __name__ == "__main__":
    main()
