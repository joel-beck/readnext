"""
Compute cosine similarities of abstract embeddings for all documents with FastText.
"""
import polars as pl

from readnext.config import ResultsPaths
from readnext.evaluation.scoring import precompute_cosine_similarities_polars
from readnext.utils.io import write_df_to_parquet


def main() -> None:
    fasttext_embeddings = pl.scan_parquet(ResultsPaths.language_models.fasttext_embeddings_parquet)

    fasttext_cosine_similarities = precompute_cosine_similarities_polars(fasttext_embeddings)

    write_df_to_parquet(
        fasttext_cosine_similarities,
        ResultsPaths.language_models.fasttext_cosine_similarities_parquet,
    )


if __name__ == "__main__":
    main()
