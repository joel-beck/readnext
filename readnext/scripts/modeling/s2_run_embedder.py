"""Generate mappings of document ids to embeddings for all language models."""

from gensim.models.fasttext import load_facebook_model
from gensim.models.keyedvectors import KeyedVectors, load_word2vec_format
from transformers import BertModel, LongformerModel

from readnext.config import ModelPaths, ModelVersions, ResultsPaths
from readnext.modeling.language_models import (
    BERTEmbedder,
    FastTextEmbedder,
    LongformerEmbedder,
    TFIDFEmbedder,
    Word2VecEmbedder,
    bm25,
    tfidf,
)
from readnext.utils import (
    read_df_from_parquet,
    suppress_transformers_logging,
    write_df_to_parquet,
)


def main() -> None:
    suppress_transformers_logging()

    spacy_tokens_frame = read_df_from_parquet(
        ResultsPaths.language_models.spacy_tokenized_abstracts_parquet
    )
    # NOTE: Remove to train on full data
    spacy_tokens_frame = spacy_tokens_frame.head(1000)

    bert_token_ids_frame = read_df_from_parquet(
        ResultsPaths.language_models.bert_tokenized_abstracts_parquet
    )
    # NOTE: Remove to train on full data
    bert_token_ids_frame = bert_token_ids_frame.head(1000)

    scibert_token_ids_frame = read_df_from_parquet(
        ResultsPaths.language_models.scibert_tokenized_abstracts_parquet
    )
    # NOTE: Remove to train on full data
    scibert_token_ids_frame = scibert_token_ids_frame.head(1000)

    longformer_token_ids_frame = read_df_from_parquet(
        ResultsPaths.language_models.longformer_tokenized_abstracts_parquet
    )
    # NOTE: Remove to train on full data
    longformer_token_ids_frame = longformer_token_ids_frame.head(1000)

    tfidf_embedder = TFIDFEmbedder(tokens_frame=spacy_tokens_frame, keyword_algorithm=tfidf)
    tfidf_embeddings_frame = tfidf_embedder.compute_embeddings_frame()
    write_df_to_parquet(
        tfidf_embeddings_frame,
        ResultsPaths.language_models.tfidf_embeddings_parquet,
    )

    # interface of tfidf and bm25 is identical, thus the same embedder can be used
    bm25_embedder = TFIDFEmbedder(tokens_frame=spacy_tokens_frame, keyword_algorithm=bm25)
    bm25_embeddings_frame = bm25_embedder.compute_embeddings_frame()
    write_df_to_parquet(
        bm25_embeddings_frame,
        ResultsPaths.language_models.bm25_embeddings_parquet,
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

    bert_model = BertModel.from_pretrained(ModelVersions.bert)  # type: ignore
    bert_embedder = BERTEmbedder(
        token_ids_frame=bert_token_ids_frame,
        torch_model=bert_model,  # type: ignore
    )
    bert_embeddings_frame = bert_embedder.compute_embeddings_frame()
    write_df_to_parquet(
        bert_embeddings_frame,
        ResultsPaths.language_models.bert_embeddings_parquet,
    )

    scibert_model = BertModel.from_pretrained(ModelVersions.scibert)  # type: ignore
    scibert_embedder = BERTEmbedder(
        token_ids_frame=scibert_token_ids_frame,
        torch_model=scibert_model,  # type: ignore
    )
    scibert_embeddings_frame = scibert_embedder.compute_embeddings_frame()
    write_df_to_parquet(
        scibert_embeddings_frame,
        ResultsPaths.language_models.scibert_embeddings_parquet,
    )

    longformer_model = LongformerModel.from_pretrained(ModelVersions.longformer)  # type: ignore
    longformer_embedder = LongformerEmbedder(
        token_ids_frame=longformer_token_ids_frame,
        torch_model=longformer_model,  # type: ignore
    )
    longformer_embeddings_frame = longformer_embedder.compute_embeddings_frame()
    write_df_to_parquet(
        longformer_embeddings_frame,
        ResultsPaths.language_models.longformer_embeddings_parquet,
    )


if __name__ == "__main__":
    main()
