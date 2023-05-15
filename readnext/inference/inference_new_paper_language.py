from collections.abc import Callable
from typing import TypeAlias

import spacy
from gensim.models.fasttext import load_facebook_model
from gensim.models.keyedvectors import KeyedVectors, load_word2vec_format

from readnext.config import ModelPaths, ModelVersions, ResultsPaths
from readnext.modeling import DocumentInfo, DocumentsInfo
from readnext.modeling.language_models import (
    Embedding,
    FastTextEmbedder,
    LanguageModelChoice,
    SpacyTokenizer,
    TFIDFEmbedder,
    Tokens,
    TokensMapping,
    Word2VecEmbedder,
    bm25,
    tfidf,
)

QueryEmbeddingFunction: TypeAlias = Callable[[DocumentInfo], Embedding]


def spacy_tokenize_query(query_document_info: DocumentInfo) -> Tokens:
    spacy_model = spacy.load(ModelVersions.spacy)

    query_documents_info = DocumentsInfo([query_document_info])
    spacy_tokenizer = SpacyTokenizer(documents_info=query_documents_info, spacy_model=spacy_model)

    return spacy_tokenizer.tokenize_single_document(query_document_info.abstract)


def spacy_load_training_tokens_mapping() -> TokensMapping:
    return SpacyTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.spacy_tokenized_abstracts_mapping_most_cited_pkl
    )


def tfidf_embed_query(query_document_info: DocumentInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_document_info)

    tfidf_embedder = TFIDFEmbedder(
        keyword_algorithm=tfidf, tokens_mapping=learned_spacy_tokens_mapping
    )

    return tfidf_embedder.compute_embedding_single_document(query_abstract_tokenized)


def bm25_embed_query(query_document_info: DocumentInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_document_info)

    bm25_embedder = TFIDFEmbedder(
        keyword_algorithm=bm25, tokens_mapping=learned_spacy_tokens_mapping
    )

    return bm25_embedder.compute_embedding_single_document(query_abstract_tokenized)


def word2vec_embed_query(query_document_info: DocumentInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_document_info)

    print("Loading pretrained Word2Vec Model...")
    word2vec_model: KeyedVectors = load_word2vec_format(ModelPaths.word2vec, binary=True)
    print("Word2Vec Model loaded.")

    word2vec_embedder = Word2VecEmbedder(word2vec_model, learned_spacy_tokens_mapping)

    return word2vec_embedder.compute_embedding_single_document(query_abstract_tokenized)


def glove_embed_query(query_document_info: DocumentInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_document_info)

    print("Loading pretrained GloVe Model...")
    glove_model: KeyedVectors = load_word2vec_format(ModelPaths.glove, binary=False, no_header=True)
    print("GloVe Model loaded.")

    glove_embedder = Word2VecEmbedder(glove_model, learned_spacy_tokens_mapping)

    return glove_embedder.compute_embedding_single_document(query_abstract_tokenized)


def fasttest_embed_query(query_document_info: DocumentInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_document_info)

    print("Loading pretrained FastText Model...")
    fasttext_model = load_facebook_model(ModelPaths.fasttext)
    print("FastText Model loaded.")

    fasttext_embedder = FastTextEmbedder(fasttext_model, learned_spacy_tokens_mapping)

    return fasttext_embedder.compute_embedding_single_document(query_abstract_tokenized)


def bert_embed_query(query_document_info: DocumentInfo) -> Embedding:
    pass


def scibert_embed_query(query_document_info: DocumentInfo) -> Embedding:
    pass


def longformer_embed_query(query_document_info: DocumentInfo) -> Embedding:
    pass


def select_query_embedding_function(
    language_model_choice: LanguageModelChoice,
) -> QueryEmbeddingFunction:
    match language_model_choice:
        case LanguageModelChoice.tfidf:
            return tfidf_embed_query
        case LanguageModelChoice.bm25:
            return bm25_embed_query
        case LanguageModelChoice.word2vec:
            return word2vec_embed_query
        case LanguageModelChoice.glove:
            return glove_embed_query
        case LanguageModelChoice.fasttext:
            return fasttest_embed_query
        case LanguageModelChoice.bert:
            return bert_embed_query
        case LanguageModelChoice.scibert:
            return scibert_embed_query
        case LanguageModelChoice.longformer:
            return longformer_embed_query
        case _:
            raise ValueError(f"Language model choice {language_model_choice} not supported")
