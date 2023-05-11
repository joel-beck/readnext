"""Compute cosine similarities of abstract embeddings for all language models."""

from readnext.config import ResultsPaths
from readnext.evaluation.scoring import precompute_cosine_similarities
from readnext.utils import load_df_from_pickle, save_df_to_pickle


def main() -> None:
    # SUBSECTION: TF-IDF
    tfidf_embeddings_mapping = load_df_from_pickle(
        ResultsPaths.language_models.tfidf_embeddings_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    tfidf_embeddings_mapping = tfidf_embeddings_mapping.head(100)
    tfidf_cosine_similarities = precompute_cosine_similarities(tfidf_embeddings_mapping)
    save_df_to_pickle(
        tfidf_cosine_similarities,
        ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl,
    )

    # SUBSECTION: BM25
    bm25_embeddings_mapping = load_df_from_pickle(
        ResultsPaths.language_models.bm25_embeddings_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    bm25_embeddings_mapping = bm25_embeddings_mapping.head(100)
    bm25_cosine_similarities = precompute_cosine_similarities(bm25_embeddings_mapping)
    save_df_to_pickle(
        bm25_cosine_similarities,
        ResultsPaths.language_models.bm25_cosine_similarities_most_cited_pkl,
    )

    # SUBSECTION: Word2Vec
    word2vec_embeddings_mapping = load_df_from_pickle(
        ResultsPaths.language_models.word2vec_embeddings_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    word2vec_embeddings_mapping = word2vec_embeddings_mapping.head(100)
    word2vec_cosine_similarities = precompute_cosine_similarities(word2vec_embeddings_mapping)
    save_df_to_pickle(
        word2vec_cosine_similarities,
        ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_pkl,
    )

    # SUBSECTION: GloVe
    glove_embeddings_mapping = load_df_from_pickle(
        ResultsPaths.language_models.glove_embeddings_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    glove_embeddings_mapping = glove_embeddings_mapping.head(100)
    glove_cosine_similarities = precompute_cosine_similarities(glove_embeddings_mapping)
    save_df_to_pickle(
        glove_cosine_similarities,
        ResultsPaths.language_models.glove_cosine_similarities_most_cited_pkl,
    )

    # SUBSECTION: FastText
    fasttext_embeddings_mapping = load_df_from_pickle(
        ResultsPaths.language_models.fasttext_embeddings_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    fasttext_embeddings_mapping = fasttext_embeddings_mapping.head(100)
    fasttext_cosine_similarities = precompute_cosine_similarities(fasttext_embeddings_mapping)
    save_df_to_pickle(
        fasttext_cosine_similarities,
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl,
    )

    # SUBSECTION: BERT
    bert_embeddings_mapping = load_df_from_pickle(
        ResultsPaths.language_models.bert_embeddings_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    bert_embeddings_mapping = bert_embeddings_mapping.head(100)
    bert_cosine_similarities = precompute_cosine_similarities(bert_embeddings_mapping)
    save_df_to_pickle(
        bert_cosine_similarities,
        ResultsPaths.language_models.bert_cosine_similarities_most_cited_pkl,
    )

    # SUBSECTION: SciBERT
    scibert_embeddings_mapping = load_df_from_pickle(
        ResultsPaths.language_models.scibert_embeddings_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    scibert_embeddings_mapping = scibert_embeddings_mapping.head(100)
    scibert_cosine_similarities = precompute_cosine_similarities(scibert_embeddings_mapping)
    save_df_to_pickle(
        scibert_cosine_similarities,
        ResultsPaths.language_models.scibert_cosine_similarities_most_cited_pkl,
    )


if __name__ == "__main__":
    main()
