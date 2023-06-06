"""
Compute cosine similarities of abstract embeddings for all documents with TF-IDF.
"""
from readnext.config import ResultsPaths
from readnext.evaluation.scoring import precompute_cosine_similarities
from readnext.utils import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    tfidf_embeddings = read_df_from_parquet(ResultsPaths.language_models.tfidf_embeddings_parquet)
    tfidf_cosine_similarities = precompute_cosine_similarities(tfidf_embeddings)

    write_df_to_parquet(
        tfidf_cosine_similarities,
        ResultsPaths.language_models.tfidf_cosine_similarities_parquet,
    )


if __name__ == "__main__":
    main()
