"""
Generate embedding frames of document abstracts with GloVe.
"""

from gensim.models.keyedvectors import KeyedVectors, load_word2vec_format

from readnext.config import ModelPaths, ResultsPaths
from readnext.modeling.language_models import Word2VecEmbedder
from readnext.utils.io import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    spacy_tokens_frame = read_df_from_parquet(
        ResultsPaths.language_models.spacy_tokenized_abstracts_parquet
    )

    # requires pre-downloaded `glove.6B` model from Stanford NLP website:
    # https://nlp.stanford.edu/projects/glove/
    #
    # `load_word2vec_format` with `no_header=True` converts the GloVe model to the
    # word2vec format.
    # After conversion the user interface of the two models is identical, thus the same
    # `Word2VecEmbedder` can be used!
    glove_model: KeyedVectors = load_word2vec_format(ModelPaths.glove, binary=False, no_header=True)
    glove_embedder = Word2VecEmbedder(
        tokens_frame=spacy_tokens_frame,
        embedding_model=glove_model,  # type: ignore
    )
    glove_embeddings_frame = glove_embedder.compute_embeddings_frame()

    write_df_to_parquet(
        glove_embeddings_frame,
        ResultsPaths.language_models.glove_embeddings_parquet,
    )


if __name__ == "__main__":
    main()
