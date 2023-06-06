"""
Compute cosine similarities of abstract embeddings for all documents with BERT.
"""
from readnext.config import ResultsPaths
from readnext.evaluation.scoring import precompute_cosine_similarities
from readnext.utils import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    bert_embeddings = read_df_from_parquet(ResultsPaths.language_models.bert_embeddings_parquet)
    bert_cosine_similarities = precompute_cosine_similarities(bert_embeddings)

    write_df_to_parquet(
        bert_cosine_similarities,
        ResultsPaths.language_models.bert_cosine_similarities_parquet,
    )


if __name__ == "__main__":
    main()
