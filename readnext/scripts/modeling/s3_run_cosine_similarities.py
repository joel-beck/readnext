"""Compute cosine similarities of abstract embeddings for all language models."""

from readnext.config import ResultsPaths
from readnext.evaluation.scoring import precompute_cosine_similarities
from readnext.utils import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    # SUBSECTION: TF-IDF
    tfidf_embeddings = read_df_from_parquet(
        ResultsPaths.language_models.tfidf_embeddings_most_cited_parquet
    )
    # NOTE: Remove to train on full data
    tfidf_embeddings = tfidf_embeddings.head(1000)
    tfidf_cosine_similarities = precompute_cosine_similarities(tfidf_embeddings)
    write_df_to_parquet(
        tfidf_cosine_similarities,
        ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_parquet,
    )

    # SUBSECTION: BM25
    bm25_embeddings = read_df_from_parquet(
        ResultsPaths.language_models.bm25_embeddings_most_cited_parquet
    )
    # NOTE: Remove to train on full data
    bm25_embeddings = bm25_embeddings.head(1000)
    bm25_cosine_similarities = precompute_cosine_similarities(bm25_embeddings)
    write_df_to_parquet(
        bm25_cosine_similarities,
        ResultsPaths.language_models.bm25_cosine_similarities_most_cited_parquet,
    )

    # SUBSECTION: Word2Vec
    word2vec_embeddings = read_df_from_parquet(
        ResultsPaths.language_models.word2vec_embeddings_most_cited_parquet
    )
    # NOTE: Remove to train on full data
    word2vec_embeddings = word2vec_embeddings.head(1000)
    word2vec_cosine_similarities = precompute_cosine_similarities(word2vec_embeddings)
    write_df_to_parquet(
        word2vec_cosine_similarities,
        ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_parquet,
    )

    # SUBSECTION: GloVe
    glove_embeddings = read_df_from_parquet(
        ResultsPaths.language_models.glove_embeddings_most_cited_parquet
    )
    # NOTE: Remove to train on full data
    glove_embeddings = glove_embeddings.head(1000)
    glove_cosine_similarities = precompute_cosine_similarities(glove_embeddings)
    write_df_to_parquet(
        glove_cosine_similarities,
        ResultsPaths.language_models.glove_cosine_similarities_most_cited_parquet,
    )

    # SUBSECTION: FastText
    fasttext_embeddings = read_df_from_parquet(
        ResultsPaths.language_models.fasttext_embeddings_most_cited_parquet
    )
    # NOTE: Remove to train on full data
    fasttext_embeddings = fasttext_embeddings.head(1000)
    fasttext_cosine_similarities = precompute_cosine_similarities(fasttext_embeddings)
    write_df_to_parquet(
        fasttext_cosine_similarities,
        ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_parquet,
    )

    # SUBSECTION: BERT
    bert_embeddings = read_df_from_parquet(
        ResultsPaths.language_models.bert_embeddings_most_cited_parquet
    )
    # NOTE: Remove to train on full data
    bert_embeddings = bert_embeddings.head(1000)
    bert_cosine_similarities = precompute_cosine_similarities(bert_embeddings)
    write_df_to_parquet(
        bert_cosine_similarities,
        ResultsPaths.language_models.bert_cosine_similarities_most_cited_parquet,
    )

    # SUBSECTION: SciBERT
    scibert_embeddings = read_df_from_parquet(
        ResultsPaths.language_models.scibert_embeddings_most_cited_parquet
    )
    # NOTE: Remove to train on full data
    scibert_embeddings = scibert_embeddings.head(1000)
    scibert_cosine_similarities = precompute_cosine_similarities(scibert_embeddings)
    write_df_to_parquet(
        scibert_cosine_similarities,
        ResultsPaths.language_models.scibert_cosine_similarities_most_cited_parquet,
    )

    # SUBSECTION: Longformer
    longformer_embeddings = read_df_from_parquet(
        ResultsPaths.language_models.longformer_embeddings_most_cited_parquet
    )
    # NOTE: Remove to train on full data
    longformer_embeddings = longformer_embeddings.head(1000)
    longformer_cosine_similarities = precompute_cosine_similarities(longformer_embeddings)
    write_df_to_parquet(
        longformer_cosine_similarities,
        ResultsPaths.language_models.longformer_cosine_similarities_most_cited_parquet,
    )


if __name__ == "__main__":
    main()
