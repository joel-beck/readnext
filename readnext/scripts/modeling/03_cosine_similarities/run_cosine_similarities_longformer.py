"""
Compute cosine similarities of abstract embeddings for all documents with Longformer.
"""
from readnext.config import ResultsPaths
from readnext.evaluation.scoring import precompute_cosine_similarities
from readnext.utils import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    longformer_embeddings = read_df_from_parquet(
        ResultsPaths.language_models.longformer_embeddings_parquet
    )
    longformer_cosine_similarities = precompute_cosine_similarities(longformer_embeddings)

    write_df_to_parquet(
        longformer_cosine_similarities,
        ResultsPaths.language_models.longformer_cosine_similarities_parquet,
    )


if __name__ == "__main__":
    main()
