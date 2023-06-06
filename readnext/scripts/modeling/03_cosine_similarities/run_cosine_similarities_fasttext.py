"""
Compute cosine similarities of abstract embeddings for all documents with FastText.
"""
from readnext.config import ResultsPaths
from readnext.evaluation.scoring import precompute_cosine_similarities
from readnext.utils import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    fasttext_embeddings = read_df_from_parquet(
        ResultsPaths.language_models.fasttext_embeddings_parquet
    )
    fasttext_cosine_similarities = precompute_cosine_similarities(fasttext_embeddings)

    write_df_to_parquet(
        fasttext_cosine_similarities,
        ResultsPaths.language_models.fasttext_cosine_similarities_parquet,
    )


if __name__ == "__main__":
    main()
