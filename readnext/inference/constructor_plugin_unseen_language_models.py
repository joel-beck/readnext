import polars as pl
import spacy
from gensim.models.fasttext import FastText, load_facebook_model
from gensim.models.keyedvectors import KeyedVectors, load_word2vec_format
from transformers import BertModel, BertTokenizerFast, LongformerModel, LongformerTokenizerFast

from readnext.config import ModelPaths, ModelVersions, ResultsPaths
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
    TokenIdsFrame,
    Tokens,
    TokensFrame,
    read_df_from_parquet,
    status_update,
)


@status_update("tokenizing query abstract with Spacy")
def spacy_tokenize_query(query_documents_data: pl.DataFrame) -> Tokens:
    spacy_model = spacy.load(ModelVersions.spacy)
    spacy_tokenizer = SpacyTokenizer(query_documents_data, spacy_model)

    return spacy_tokenizer.tokenize_single_document(query_documents_data["abstract"][0])


@status_update("tokenizing query abstract with BERT")
def bert_tokenize_query(query_documents_data: pl.DataFrame) -> TokenIds:
    bert_tokenizer_transformers = BertTokenizerFast.from_pretrained(
        ModelVersions.bert, do_lower_case=True, clean_text=True
    )
    bert_tokenizer = BERTTokenizer(query_documents_data, bert_tokenizer_transformers)

    return bert_tokenizer.tokenize_into_ids(query_documents_data["abstract"][0])


@status_update("tokenizing query abstract with SciBERT")
def scibert_tokenize_query(query_documents_data: pl.DataFrame) -> TokenIds:
    scibert_tokenizer_transformers = BertTokenizerFast.from_pretrained(
        ModelVersions.scibert, do_lower_case=True, clean_text=True
    )
    scibert_tokenizer = BERTTokenizer(query_documents_data, scibert_tokenizer_transformers)

    return scibert_tokenizer.tokenize_into_ids(query_documents_data["abstract"][0])


@status_update("tokenizing query abstract with Longformer")
def longformer_tokenize_query(query_documents_data: pl.DataFrame) -> TokenIds:
    longformer_tokenizer_transformers = LongformerTokenizerFast.from_pretrained(
        ModelVersions.longformer
    )
    longformer_tokenizer = LongformerTokenizer(
        query_documents_data, longformer_tokenizer_transformers
    )

    return longformer_tokenizer.tokenize_into_ids(query_documents_data["abstract"][0])


def spacy_load_training_tokens_frame() -> TokensFrame:
    return read_df_from_parquet(ResultsPaths.language_models.spacy_tokenized_abstracts_parquet)


@status_update("embedding query abstract with TF-IDF")
def tfidf_embed_query(query_documents_data: pl.DataFrame) -> Embedding:
    learned_spacy_tokens_frame = spacy_load_training_tokens_frame()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_data)

    tfidf_embedder = TFIDFEmbedder(
        tokens_frame=learned_spacy_tokens_frame,
        keyword_algorithm=tfidf,
    )

    return tfidf_embedder.compute_embedding_single_document(query_abstract_tokenized)


@status_update("embedding query abstract with BM25")
def bm25_embed_query(query_documents_data: pl.DataFrame) -> Embedding:
    learned_spacy_tokens_frame = spacy_load_training_tokens_frame()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_data)

    bm25_embedder = TFIDFEmbedder(tokens_frame=learned_spacy_tokens_frame, keyword_algorithm=bm25)

    return bm25_embedder.compute_embedding_single_document(query_abstract_tokenized)


@status_update("loading pretrained Word2Vec model")
def word2vec_load_model() -> KeyedVectors:
    return load_word2vec_format(ModelPaths.word2vec, binary=True)


@status_update("embedding query abstract with Word2Vec")
def word2vec_embed_query(query_documents_data: pl.DataFrame) -> Embedding:
    learned_spacy_tokens_frame = spacy_load_training_tokens_frame()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_data)

    word2vec_model = word2vec_load_model()
    word2vec_embedder = Word2VecEmbedder(
        tokens_frame=learned_spacy_tokens_frame,
        embedding_model=word2vec_model,  # type: ignore
    )

    return word2vec_embedder.compute_embedding_single_document(query_abstract_tokenized)


@status_update("loading pretrained Glove model")
def glove_load_model() -> KeyedVectors:
    return load_word2vec_format(ModelPaths.glove, binary=False, no_header=True)


@status_update("embedding query abstract with Word2Vec")
def glove_embed_query(query_documents_data: pl.DataFrame) -> Embedding:
    learned_spacy_tokens_frame = spacy_load_training_tokens_frame()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_data)

    glove_model = glove_load_model()
    glove_embedder = Word2VecEmbedder(
        tokens_frame=learned_spacy_tokens_frame,
        embedding_model=glove_model,  # type: ignore
    )

    return glove_embedder.compute_embedding_single_document(query_abstract_tokenized)


@status_update("loading pretrained FastText model")
def fasttext_load_model() -> FastText:
    return load_facebook_model(ModelPaths.fasttext)


@status_update("embedding query abstract with FastText")
def fasttest_embed_query(query_documents_data: pl.DataFrame) -> Embedding:
    learned_spacy_tokens_frame = spacy_load_training_tokens_frame()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_data)

    fasttext_model = fasttext_load_model()
    fasttext_embedder = FastTextEmbedder(
        tokens_frame=learned_spacy_tokens_frame,
        embedding_model=fasttext_model,  # type: ignore
    )

    return fasttext_embedder.compute_embedding_single_document(query_abstract_tokenized)


def bert_load_training_tokens_frame() -> TokenIdsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.bert_tokenized_abstracts_parquet)


@status_update("loading pretrained BERT model")
def bert_load_model() -> BertModel:
    return BertModel.from_pretrained(ModelVersions.bert)  # type: ignore


@status_update("embedding query abstract with BERT")
def bert_embed_query(query_documents_data: pl.DataFrame) -> Embedding:
    learned_bert_tokens_frame = bert_load_training_tokens_frame()
    query_abstract_tokenized = bert_tokenize_query(query_documents_data)

    bert_model = bert_load_model()
    bert_embedder = BERTEmbedder(
        token_ids_frame=learned_bert_tokens_frame,
        torch_model=bert_model,  # type: ignore
    )

    return bert_embedder.compute_embedding_single_document(query_abstract_tokenized)


def scibert_load_training_tokens_frame() -> TokenIdsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.scibert_tokenized_abstracts_parquet)


@status_update("loading pretrained SciBERT model")
def scibert_load_model() -> BertModel:
    return BertModel.from_pretrained(ModelVersions.scibert)  # type: ignore


@status_update("embedding query abstract with SciBERT")
def scibert_embed_query(query_documents_data: pl.DataFrame) -> Embedding:
    learned_scibert_tokens_frame = scibert_load_training_tokens_frame()
    query_abstract_tokenized = scibert_tokenize_query(query_documents_data)

    scibert_model = scibert_load_model()
    scibert_embedder = BERTEmbedder(
        token_ids_frame=learned_scibert_tokens_frame,
        torch_model=scibert_model,  # type: ignore
    )

    return scibert_embedder.compute_embedding_single_document(query_abstract_tokenized)


def longformer_load_training_tokens_frame() -> TokenIdsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.longformer_tokenized_abstracts_parquet)


@status_update("loading pretrained Longformer model")
def longformer_load_model() -> LongformerModel:
    return LongformerModel.from_pretrained(ModelVersions.longformer)  # type: ignore


@status_update("embedding query abstract with Longformer")
def longformer_embed_query(query_documents_data: pl.DataFrame) -> Embedding:
    learned_longformer_tokens_frame = longformer_load_training_tokens_frame()
    query_abstract_tokenized = longformer_tokenize_query(query_documents_data)

    longformer_model = longformer_load_model()
    longformer_embedder = LongformerEmbedder(
        token_ids_frame=learned_longformer_tokens_frame,
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
