"""Generate mappings of document ids to embeddings for all language models."""

from gensim.models.fasttext import load_facebook_model
from gensim.models.keyedvectors import KeyedVectors, load_word2vec_format
from transformers import BertModel, LongformerModel

from readnext.config import ModelPaths, ModelVersions, ResultsPaths
from readnext.modeling.language_models import (
    BERTEmbedder,
    BERTTokenizer,
    FastTextEmbedder,
    LongformerEmbedder,
    LongformerTokenizer,
    SpacyTokenizer,
    TFIDFEmbedder,
    Word2VecEmbedder,
    bm25,
    embeddings_mapping_to_frame,
    tfidf,
)
from readnext.utils import slice_mapping, suppress_transformers_logging, write_df_to_pickle


def main() -> None:
    suppress_transformers_logging()

    spacy_tokens_mapping = SpacyTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.spacy_tokenized_abstracts_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    spacy_tokens_mapping = slice_mapping(spacy_tokens_mapping, size=1000)

    bert_token_ids_mapping = BERTTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.bert_tokenized_abstracts_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    bert_token_ids_mapping = slice_mapping(bert_token_ids_mapping, size=1000)

    scibert_token_ids_mapping = BERTTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.scibert_tokenized_abstracts_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    scibert_token_ids_mapping = slice_mapping(scibert_token_ids_mapping, size=1000)

    longformer_token_ids_mapping = LongformerTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.longformer_tokenized_abstracts_mapping_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    longformer_token_ids_mapping = slice_mapping(longformer_token_ids_mapping, size=1000)

    tfidf_embedder = TFIDFEmbedder(tokens_mapping=spacy_tokens_mapping, keyword_algorithm=tfidf)
    tfidf_embeddings_mapping = tfidf_embedder.compute_embeddings_mapping()
    write_df_to_pickle(
        embeddings_mapping_to_frame(tfidf_embeddings_mapping),
        ResultsPaths.language_models.tfidf_embeddings_most_cited_pkl,
    )

    # interface of tfidf and bm25 is identical, thus the same embedder can be used
    bm25_embedder = TFIDFEmbedder(tokens_mapping=spacy_tokens_mapping, keyword_algorithm=bm25)
    bm25_embeddings_mapping = bm25_embedder.compute_embeddings_mapping()
    write_df_to_pickle(
        embeddings_mapping_to_frame(bm25_embeddings_mapping),
        ResultsPaths.language_models.bm25_embeddings_most_cited_pkl,
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
        tokens_mapping=spacy_tokens_mapping,
        embedding_model=word2vec_model,  # type: ignore
    )
    word2vec_embeddings_mapping = word2vec_embedder.compute_embeddings_mapping()
    write_df_to_pickle(
        embeddings_mapping_to_frame(word2vec_embeddings_mapping),
        ResultsPaths.language_models.word2vec_embeddings_most_cited_pkl,
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
        tokens_mapping=spacy_tokens_mapping, embedding_model=glove_model  # type: ignore
    )
    glove_embeddings_mapping = glove_embedder.compute_embeddings_mapping()
    write_df_to_pickle(
        embeddings_mapping_to_frame(glove_embeddings_mapping),
        ResultsPaths.language_models.glove_embeddings_most_cited_pkl,
    )

    # requires pre-downloaded model from fasttext website:
    # https://fasttext.cc/docs/en/crawl-vectors.html#models
    fasttext_model = load_facebook_model(ModelPaths.fasttext)
    fasttext_embedder = FastTextEmbedder(
        tokens_mapping=spacy_tokens_mapping, embedding_model=fasttext_model  # type: ignore
    )
    fasttext_embeddings_mapping = fasttext_embedder.compute_embeddings_mapping()
    write_df_to_pickle(
        embeddings_mapping_to_frame(fasttext_embeddings_mapping),
        ResultsPaths.language_models.fasttext_embeddings_most_cited_pkl,
    )

    bert_model = BertModel.from_pretrained(ModelVersions.bert)  # type: ignore
    bert_embedder = BERTEmbedder(
        tokens_tensor_mapping=bert_token_ids_mapping,
        torch_model=bert_model,  # type: ignore
    )
    bert_embeddings_mapping = bert_embedder.compute_embeddings_mapping()
    write_df_to_pickle(
        embeddings_mapping_to_frame(bert_embeddings_mapping),
        ResultsPaths.language_models.bert_embeddings_most_cited_pkl,
    )

    scibert_model = BertModel.from_pretrained(ModelVersions.scibert)  # type: ignore
    scibert_embedder = BERTEmbedder(
        tokens_tensor_mapping=scibert_token_ids_mapping,
        torch_model=scibert_model,  # type: ignore
    )
    scibert_embeddings = scibert_embedder.compute_embeddings_mapping()
    write_df_to_pickle(
        embeddings_mapping_to_frame(scibert_embeddings),
        ResultsPaths.language_models.scibert_embeddings_most_cited_pkl,
    )

    longformer_model = LongformerModel.from_pretrained(ModelVersions.longformer)  # type: ignore
    longformer_embedder = LongformerEmbedder(
        tokens_tensor_mapping=longformer_token_ids_mapping,
        torch_model=longformer_model,  # type: ignore
    )
    longformer_embeddings = longformer_embedder.compute_embeddings_mapping()
    write_df_to_pickle(
        embeddings_mapping_to_frame(longformer_embeddings),
        ResultsPaths.language_models.longformer_embeddings_most_cited_pkl,
    )


if __name__ == "__main__":
    main()
