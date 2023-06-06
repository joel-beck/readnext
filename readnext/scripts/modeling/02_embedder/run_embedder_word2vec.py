"""
Generate embedding frames of document abstracts with Word2Vec.
"""

from gensim.models.keyedvectors import KeyedVectors, load_word2vec_format

from readnext.config import ModelPaths, ResultsPaths
from readnext.modeling.language_models import Word2VecEmbedder
from readnext.utils import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    spacy_tokens_frame = read_df_from_parquet(
        ResultsPaths.language_models.spacy_tokenized_abstracts_parquet
    )

    # requires pre-downloaded model from gensim data repository:
    # https://github.com/RaRe-Technologies/gensim-data
    #
    # download and save model locally with the commands
    # `import gensim.downloader as api`
    # `api.load(ModelVersions.word2vec, return_path=True)`
    #
    # then unzip the model file and move it to the local `models` directory
    word2vec_model: KeyedVectors = load_word2vec_format(ModelPaths.word2vec, binary=True)
    word2vec_embedder = Word2VecEmbedder(
        tokens_frame=spacy_tokens_frame,
        embedding_model=word2vec_model,  # type: ignore
    )
    word2vec_embeddings_frame = word2vec_embedder.compute_embeddings_frame()

    write_df_to_parquet(
        word2vec_embeddings_frame,
        ResultsPaths.language_models.word2vec_embeddings_parquet,
    )


if __name__ == "__main__":
    main()
