import spacy
from gensim.models.fasttext import load_facebook_model
from gensim.models.keyedvectors import KeyedVectors, load_word2vec_format
from transformers import BertModel, BertTokenizerFast, LongformerModel, LongformerTokenizerFast

from readnext.config import ModelPaths, ModelVersions, ResultsPaths
from readnext.modeling import DocumentInfo, DocumentsInfo
from readnext.modeling.language_models import (
    BERTEmbedder,
    BERTTokenizer,
    FastTextEmbedder,
    LanguageModelChoice,
    LongformerEmbedder,
    LongformerTokenizer,
    SpacyTokenizer,
    TFIDFEmbedder,
    Word2VecEmbedder,
    bm25,
    tfidf,
)
from readnext.utils import (
    Embedding,
    QueryEmbeddingFunction,
    TokenIds,
    TokenIdsMapping,
    Tokens,
    TokensMapping,
)


def spacy_load_training_tokens_mapping() -> TokensMapping:
    return SpacyTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.spacy_tokenized_abstracts_parquet
    )


def spacy_tokenize_query(query_document_info: DocumentInfo) -> Tokens:
    spacy_model = spacy.load(ModelVersions.spacy)

    query_documents_info = DocumentsInfo([query_document_info])
    spacy_tokenizer = SpacyTokenizer(documents_info=query_documents_info, spacy_model=spacy_model)

    return spacy_tokenizer.tokenize_single_document(query_document_info.abstract)


def bert_load_training_tokens_mapping() -> TokenIdsMapping:
    return BERTTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.bert_tokenized_abstracts_parquet
    )


def bert_tokenize_query(query_document_info: DocumentInfo) -> TokenIds:
    bert_tokenizer_transformers = BertTokenizerFast.from_pretrained(
        ModelVersions.bert, do_lower_case=True, clean_text=True
    )
    query_documents_info = DocumentsInfo([query_document_info])

    bert_tokenizer = BERTTokenizer(query_documents_info, bert_tokenizer_transformers)

    return bert_tokenizer.tokenize_into_ids(query_document_info.abstract)


def scibert_load_training_tokens_mapping() -> TokenIdsMapping:
    return BERTTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.scibert_tokenized_abstracts_parquet
    )


def scibert_tokenize_query(query_document_info: DocumentInfo) -> TokenIds:
    scibert_tokenizer_transformers = BertTokenizerFast.from_pretrained(
        ModelVersions.scibert, do_lower_case=True, clean_text=True
    )
    query_documents_info = DocumentsInfo([query_document_info])

    scibert_tokenizer = BERTTokenizer(query_documents_info, scibert_tokenizer_transformers)

    return scibert_tokenizer.tokenize_into_ids(query_document_info.abstract)


def longformer_load_training_tokens_mapping() -> TokenIdsMapping:
    return LongformerTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.longformer_tokenized_abstracts_parquet
    )


def longformer_tokenize_query(query_document_info: DocumentInfo) -> TokenIds:
    longformer_tokenizer_transformers = LongformerTokenizerFast.from_pretrained(
        ModelVersions.longformer
    )
    query_documents_info = DocumentsInfo([query_document_info])

    longformer_tokenizer = LongformerTokenizer(
        query_documents_info, longformer_tokenizer_transformers
    )

    return longformer_tokenizer.tokenize_into_ids(query_document_info.abstract)


def tfidf_embed_query(query_document_info: DocumentInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_document_info)

    tfidf_embedder = TFIDFEmbedder(
        tokens_frame=learned_spacy_tokens_mapping,
        keyword_algorithm=tfidf,
    )

    return tfidf_embedder.compute_embedding_single_document(query_abstract_tokenized)


def bm25_embed_query(query_document_info: DocumentInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_document_info)

    bm25_embedder = TFIDFEmbedder(tokens_frame=learned_spacy_tokens_mapping, keyword_algorithm=bm25)

    return bm25_embedder.compute_embedding_single_document(query_abstract_tokenized)


def word2vec_embed_query(query_document_info: DocumentInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_document_info)

    print("Loading pretrained Word2Vec Model...")
    word2vec_model: KeyedVectors = load_word2vec_format(ModelPaths.word2vec, binary=True)
    print("Word2Vec Model loaded.")

    word2vec_embedder = Word2VecEmbedder(
        tokens_mapping=learned_spacy_tokens_mapping,
        embedding_model=word2vec_model,  # type: ignore
    )

    return word2vec_embedder.compute_embedding_single_document(query_abstract_tokenized)


def glove_embed_query(query_document_info: DocumentInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_document_info)

    print("Loading pretrained GloVe Model...")
    glove_model: KeyedVectors = load_word2vec_format(ModelPaths.glove, binary=False, no_header=True)
    print("GloVe Model loaded.")

    glove_embedder = Word2VecEmbedder(
        tokens_mapping=learned_spacy_tokens_mapping,
        embedding_model=glove_model,  # type: ignore
    )

    return glove_embedder.compute_embedding_single_document(query_abstract_tokenized)


def fasttest_embed_query(query_document_info: DocumentInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_document_info)

    print("Loading pretrained FastText Model...")
    fasttext_model = load_facebook_model(ModelPaths.fasttext)
    print("FastText Model loaded.")

    fasttext_embedder = FastTextEmbedder(
        tokens_mapping=learned_spacy_tokens_mapping,
        embedding_model=fasttext_model,  # type: ignore
    )

    return fasttext_embedder.compute_embedding_single_document(query_abstract_tokenized)


def bert_embed_query(query_document_info: DocumentInfo) -> Embedding:
    learned_bert_tokens_mapping = bert_load_training_tokens_mapping()
    query_abstract_tokenized = bert_tokenize_query(query_document_info)

    bert_model = BertModel.from_pretrained(ModelVersions.bert)
    bert_embedder = BERTEmbedder(
        token_ids_frame=learned_bert_tokens_mapping,
        torch_model=bert_model,  # type: ignore
    )

    return bert_embedder.compute_embedding_single_document(query_abstract_tokenized)


def scibert_embed_query(query_document_info: DocumentInfo) -> Embedding:
    learned_scibert_tokens_mapping = scibert_load_training_tokens_mapping()
    query_abstract_tokenized = scibert_tokenize_query(query_document_info)

    scibert_model = BertModel.from_pretrained(ModelVersions.scibert)
    scibert_embedder = BERTEmbedder(
        token_ids_frame=learned_scibert_tokens_mapping,
        torch_model=scibert_model,  # type: ignore
    )

    return scibert_embedder.compute_embedding_single_document(query_abstract_tokenized)


def longformer_embed_query(query_document_info: DocumentInfo) -> Embedding:
    learned_longformer_tokens_mapping = longformer_load_training_tokens_mapping()
    query_abstract_tokenized = longformer_tokenize_query(query_document_info)

    longformer_model = LongformerModel.from_pretrained(ModelVersions.longformer)
    longformer_embedder = LongformerEmbedder(
        token_ids_frame=learned_longformer_tokens_mapping,
        torch_model=longformer_model,  # type: ignore
    )

    return longformer_embedder.compute_embedding_single_document(query_abstract_tokenized)


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
