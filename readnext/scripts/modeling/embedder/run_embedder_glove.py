"""
Generate embedding frames of document abstracts with GloVe.
"""

from gensim.models.keyedvectors import KeyedVectors, load_word2vec_format

from readnext.config import ModelPaths, ResultsPaths
from readnext.modeling.language_models import GensimEmbedder
from readnext.utils.io import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    spacy_tokens_frame = read_df_from_parquet(
        ResultsPaths.language_models.spacy_tokens_frame_parquet
    )

    # requires pre-downloaded `glove.6B` model from Stanford NLP website:
    # https://nlp.stanford.edu/projects/glove/
    #
    # `load_word2vec_format` with `no_header=True` converts the GloVe model to the
    # word2vec format. After conversion the user interface of the two models is
    # identical.
    keyed_vectors: KeyedVectors = load_word2vec_format(
        ModelPaths.glove, binary=False, no_header=True
    )
    glove_embedder = GensimEmbedder(tokens_frame=spacy_tokens_frame, keyed_vectors=keyed_vectors)
    glove_embeddings_frame = glove_embedder.compute_embeddings_frame()

    write_df_to_parquet(
        glove_embeddings_frame,
        ResultsPaths.language_models.glove_embeddings_frame_parquet,
    )


if __name__ == "__main__":
    main()
