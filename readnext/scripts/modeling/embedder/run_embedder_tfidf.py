"""
Generate embedding frames of document abstracts with TF-IDF.
"""

from readnext.config import ResultsPaths
from readnext.modeling.language_models import (
    LanguageModelChoice,
    TFIDFEmbedder,
    load_language_model,
)
from readnext.utils.io import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    spacy_tokens_frame = read_df_from_parquet(
        ResultsPaths.language_models.spacy_tokens_frame_parquet
    )

    tfidf_vectorizer = load_language_model(LanguageModelChoice.TFIDF)
    tfidf_embedder = TFIDFEmbedder(
        tokens_frame=spacy_tokens_frame, tfidf_vectorizer=tfidf_vectorizer
    )
    tfidf_embeddings_frame = tfidf_embedder.compute_embeddings_frame()

    write_df_to_parquet(
        tfidf_embeddings_frame,
        ResultsPaths.language_models.tfidf_embeddings_frame_parquet,
    )


if __name__ == "__main__":
    main()
