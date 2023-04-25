from readnext.config import ResultsPaths
from readnext.evaluation.pairwise_scores import precompute_cosine_similarities
from readnext.utils import load_df_from_pickle, save_df_to_pickle


def main() -> None:
    tfidf_embeddings_mapping = load_df_from_pickle(
        ResultsPaths.language_models.tfidf_embeddings_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    tfidf_embeddings_mapping = tfidf_embeddings_mapping.head(1000)
    tfidf_cosine_similarities = precompute_cosine_similarities(tfidf_embeddings_mapping)
    save_df_to_pickle(
        tfidf_cosine_similarities,
        ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl,
    )

    word2vec_embeddings_mapping = load_df_from_pickle(
        ResultsPaths.language_models.word2vec_embeddings_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    word2vec_embeddings_mapping = word2vec_embeddings_mapping.head(1000)
    word2vec_cosine_similarities = precompute_cosine_similarities(word2vec_embeddings_mapping)
    save_df_to_pickle(
        word2vec_cosine_similarities,
        ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_pkl,
    )

    fasttext_embeddings_mapping = load_df_from_pickle(
        ResultsPaths.language_models.fasttext_embeddings_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    fasttext_embeddings_mapping = fasttext_embeddings_mapping.head(1000)
    fasttext_cosine_similarities = precompute_cosine_similarities(fasttext_embeddings_mapping)
    save_df_to_pickle(
        fasttext_cosine_similarities,
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl,
    )


if __name__ == "__main__":
    main()
