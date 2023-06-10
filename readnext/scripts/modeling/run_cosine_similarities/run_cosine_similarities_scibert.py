"""
Compute cosine similarities of abstract embeddings for all documents with SciBERT.
"""
import polars as pl

from readnext.config import ResultsPaths
from readnext.evaluation.scoring import precompute_cosine_similarities_polars
from readnext.utils.io import write_df_to_parquet


def main() -> None:
    scibert_embeddings = pl.scan_parquet(ResultsPaths.language_models.scibert_embeddings_parquet)

    scibert_cosine_similarities = precompute_cosine_similarities_polars(scibert_embeddings)

    write_df_to_parquet(
        scibert_cosine_similarities,
        ResultsPaths.language_models.scibert_cosine_similarities_parquet,
    )


if __name__ == "__main__":
    main()
