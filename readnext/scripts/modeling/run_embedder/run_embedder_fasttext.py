"""
Generate embedding frames of document abstracts with FastText.
"""

from gensim.models.fasttext import load_facebook_model

from readnext.config import ModelPaths, ResultsPaths
from readnext.modeling.language_models import FastTextEmbedder
from readnext.utils.io import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    spacy_tokens_frame = read_df_from_parquet(
        ResultsPaths.language_models.spacy_tokenized_abstracts_parquet
    )

    # requires pre-downloaded model from fasttext website:
    # https://fasttext.cc/docs/en/crawl-vectors.html#models
    fasttext_model = load_facebook_model(ModelPaths.fasttext)
    fasttext_embedder = FastTextEmbedder(
        tokens_frame=spacy_tokens_frame,
        embedding_model=fasttext_model,  # type: ignore
    )
    fasttext_embeddings_frame = fasttext_embedder.compute_embeddings_frame()

    write_df_to_parquet(
        fasttext_embeddings_frame,
        ResultsPaths.language_models.fasttext_embeddings_parquet,
    )


if __name__ == "__main__":
    main()
