"""
Compute cosine similarities of abstract embeddings for all documents with Word2Vec.
"""
import polars as pl

from readnext.config import ResultsPaths
from readnext.evaluation.scoring import precompute_cosine_similarities_polars
from readnext.utils.io import write_df_to_parquet


def main() -> None:
    word2vec_embeddings = pl.scan_parquet(ResultsPaths.language_models.word2vec_embeddings_parquet)

    word2vec_cosine_similarities = precompute_cosine_similarities_polars(word2vec_embeddings)

    write_df_to_parquet(
        word2vec_cosine_similarities,
        ResultsPaths.language_models.word2vec_cosine_similarities_parquet,
    )


if __name__ == "__main__":
    main()
