"""
Generate embedding frames of document abstracts with Word2Vec.
"""
from gensim.models.keyedvectors import KeyedVectors

from readnext.config import ResultsPaths
from readnext.modeling.language_models import (
    GensimEmbedder,
    LanguageModelChoice,
    load_language_model,
)
from readnext.utils.io import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    spacy_tokens_frame = read_df_from_parquet(
        ResultsPaths.language_models.spacy_tokens_frame_parquet
    )

    # requires pre-downloaded model from gensim data repository:
    # https://github.com/RaRe-Technologies/gensim-data
    #
    # download and save model locally with the commands
    # `import gensim.downloader as api`
    # `api.load(ModelVersions.word2vec, return_path=True)`
    #
    # then unzip the model file and move it to the local `models` directory
    keyed_vectors: KeyedVectors = load_language_model(LanguageModelChoice.WORD2VEC)
    word2vec_embedder = GensimEmbedder(tokens_frame=spacy_tokens_frame, keyed_vectors=keyed_vectors)
    word2vec_embeddings_frame = word2vec_embedder.compute_embeddings_frame()

    write_df_to_parquet(
        word2vec_embeddings_frame,
        ResultsPaths.language_models.word2vec_embeddings_frame_parquet,
    )


if __name__ == "__main__":
    main()
