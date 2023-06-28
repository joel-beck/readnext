"""
Generate embedding frames of document abstracts with FastText.
"""

from gensim.models.fasttext import FastText, load_facebook_model

from readnext.config import ModelPaths, ResultsPaths
from readnext.modeling.language_models import GensimEmbedder
from readnext.utils.io import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    spacy_tokens_frame = read_df_from_parquet(
        ResultsPaths.language_models.spacy_tokens_frame_parquet
    )

    # requires pre-downloaded model from fasttext website:
    # https://fasttext.cc/docs/en/crawl-vectors.html#models
    fasttext_model: FastText = load_facebook_model(ModelPaths.fasttext)
    fasttext_embedder = GensimEmbedder(
        tokens_frame=spacy_tokens_frame, keyed_vectors=fasttext_model.wv
    )
    fasttext_embeddings_frame = fasttext_embedder.compute_embeddings_frame()

    write_df_to_parquet(
        fasttext_embeddings_frame,
        ResultsPaths.language_models.fasttext_embeddings_frame_parquet,
    )


if __name__ == "__main__":
    main()
