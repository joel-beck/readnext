"""
Compute cosine similarities of abstract embeddings for all documents with BM25.
"""
import polars as pl

from readnext.config import ResultsPaths
from readnext.evaluation.scoring import precompute_cosine_similarities_polars
from readnext.utils.io import write_df_to_parquet


def main() -> None:
    bm25_embeddings = pl.scan_parquet(ResultsPaths.language_models.bm25_embeddings_frame_parquet)

    bm25_cosine_similarities = precompute_cosine_similarities_polars(bm25_embeddings)

    write_df_to_parquet(
        bm25_cosine_similarities,
        ResultsPaths.language_models.bm25_cosine_similarities_parquet,
    )


if __name__ == "__main__":
    main()
