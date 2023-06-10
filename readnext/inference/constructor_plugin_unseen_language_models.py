import spacy
from gensim.models.fasttext import FastText, load_facebook_model
from gensim.models.keyedvectors import KeyedVectors, load_word2vec_format
from transformers import BertModel, BertTokenizerFast, LongformerModel, LongformerTokenizerFast

from readnext import LanguageModelChoice
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
    tfidf,
)
from readnext.utils.io import read_df_from_parquet
from readnext.utils.decorators import status_update
from readnext.utils.aliases import (
    DocumentsFrame,
    Embedding,
    QueryEmbeddingFunction,
    TokenIds,
    TokenIdsFrame,
    Tokens,
    TokensFrame,
)


@status_update("Loading training corpus")
def spacy_load_training_tokens_frame() -> TokensFrame:
    return read_df_from_parquet(ResultsPaths.language_models.spacy_tokenized_abstracts_parquet)


@status_update("Tokenizing query abstract")
def spacy_tokenize_query(query_documents_frame: DocumentsFrame) -> Tokens:
    spacy_model = spacy.load(ModelVersions.spacy)
    spacy_tokenizer = SpacyTokenizer(query_documents_frame, spacy_model)

    return spacy_tokenizer.tokenize_single_document(query_documents_frame["abstract"][0])


@status_update("Embedding query abstract")
def tfidf_compute_embedding(
    tfidf_embedder: TFIDFEmbedder, query_abstract_tokenized: Tokens
) -> Embedding:
    return tfidf_embedder.compute_embedding_single_document(query_abstract_tokenized)


def tfidf_embed_query(query_documents_frame: DocumentsFrame) -> Embedding:
    learned_spacy_tokens_frame = spacy_load_training_tokens_frame()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_frame)

    tfidf_embedder = TFIDFEmbedder(
        tokens_frame=learned_spacy_tokens_frame,
        keyword_algorithm=tfidf,
    )

    return tfidf_compute_embedding(tfidf_embedder, query_abstract_tokenized)


@status_update("Embedding query abstract")
def bm25_compute_embedding(
    bm25_embedder: TFIDFEmbedder, query_abstract_tokenized: Tokens
) -> Embedding:
    return bm25_embedder.compute_embedding_single_document(query_abstract_tokenized)


def bm25_embed_query(query_documents_frame: DocumentsFrame) -> Embedding:
    learned_spacy_tokens_frame = spacy_load_training_tokens_frame()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_frame)

    bm25_embedder = TFIDFEmbedder(tokens_frame=learned_spacy_tokens_frame, keyword_algorithm=bm25)

    return bm25_compute_embedding(bm25_embedder, query_abstract_tokenized)


@status_update("Loading pretrained Word2Vec model")
def word2vec_load_model() -> KeyedVectors:
    return load_word2vec_format(ModelPaths.word2vec, binary=True)


@status_update("Embedding query abstract")
def word2vec_compute_embedding(
    word2vec_embedder: Word2VecEmbedder, query_abstract_tokenized: Tokens
) -> Embedding:
    return word2vec_embedder.compute_embedding_single_document(query_abstract_tokenized)


def word2vec_embed_query(query_documents_frame: DocumentsFrame) -> Embedding:
    learned_spacy_tokens_frame = spacy_load_training_tokens_frame()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_frame)

    word2vec_model = word2vec_load_model()
    word2vec_embedder = Word2VecEmbedder(
        tokens_frame=learned_spacy_tokens_frame,
        embedding_model=word2vec_model,  # type: ignore
    )

    return word2vec_compute_embedding(word2vec_embedder, query_abstract_tokenized)


@status_update("Loading pretrained Glove model")
def glove_load_model() -> KeyedVectors:
    return load_word2vec_format(ModelPaths.glove, binary=False, no_header=True)


@status_update("Embedding query abstract")
def glove_compute_embedding(
    glove_embedder: Word2VecEmbedder, query_abstract_tokenized: Tokens
) -> Embedding:
    return glove_embedder.compute_embedding_single_document(query_abstract_tokenized)


def glove_embed_query(query_documents_frame: DocumentsFrame) -> Embedding:
    learned_spacy_tokens_frame = spacy_load_training_tokens_frame()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_frame)

    glove_model = glove_load_model()
    glove_embedder = Word2VecEmbedder(
        tokens_frame=learned_spacy_tokens_frame,
        embedding_model=glove_model,  # type: ignore
    )

    return glove_compute_embedding(glove_embedder, query_abstract_tokenized)


@status_update("Loading pretrained FastText model")
def fasttext_load_model() -> FastText:
    return load_facebook_model(ModelPaths.fasttext)


@status_update("Embedding query abstract")
def fasttext_compute_embedding(
    fasttext_embedder: FastTextEmbedder, query_abstract_tokenized: Tokens
) -> Embedding:
    return fasttext_embedder.compute_embedding_single_document(query_abstract_tokenized)


def fasttest_embed_query(query_documents_frame: DocumentsFrame) -> Embedding:
    learned_spacy_tokens_frame = spacy_load_training_tokens_frame()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_frame)

    fasttext_model = fasttext_load_model()
    fasttext_embedder = FastTextEmbedder(
        tokens_frame=learned_spacy_tokens_frame,
        embedding_model=fasttext_model,  # type: ignore
    )

    return fasttext_compute_embedding(fasttext_embedder, query_abstract_tokenized)


status_update("Loading training corpus")


def bert_load_training_tokens_frame() -> TokenIdsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.bert_tokenized_abstracts_parquet)


@status_update("Tokenizing query abstract")
def bert_tokenize_query(query_documents_frame: DocumentsFrame) -> TokenIds:
    bert_tokenizer_transformers = BertTokenizerFast.from_pretrained(
        ModelVersions.bert, do_lower_case=True, clean_text=True
    )
    bert_tokenizer = BERTTokenizer(query_documents_frame, bert_tokenizer_transformers)

    return bert_tokenizer.tokenize_into_ids(query_documents_frame["abstract"][0])


@status_update("Loading pretrained BERT model")
def bert_load_model() -> BertModel:
    return BertModel.from_pretrained(ModelVersions.bert)  # type: ignore


@status_update("Embedding query abstract")
def bert_compute_embedding(
    bert_embedder: BERTEmbedder, query_abstract_tokenized: TokenIds
) -> Embedding:
    return bert_embedder.compute_embedding_single_document(query_abstract_tokenized)


def bert_embed_query(query_documents_frame: DocumentsFrame) -> Embedding:
    learned_bert_tokens_frame = bert_load_training_tokens_frame()
    query_abstract_tokenized = bert_tokenize_query(query_documents_frame)

    bert_model = bert_load_model()
    bert_embedder = BERTEmbedder(
        token_ids_frame=learned_bert_tokens_frame,
        torch_model=bert_model,  # type: ignore
    )

    return bert_compute_embedding(bert_embedder, query_abstract_tokenized)


@status_update("Loading training corpus")
def scibert_load_training_tokens_frame() -> TokenIdsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.scibert_tokenized_abstracts_parquet)


@status_update("Tokenizing query abstract")
def scibert_tokenize_query(query_documents_frame: DocumentsFrame) -> TokenIds:
    scibert_tokenizer_transformers = BertTokenizerFast.from_pretrained(
        ModelVersions.scibert, do_lower_case=True, clean_text=True
    )
    scibert_tokenizer = BERTTokenizer(query_documents_frame, scibert_tokenizer_transformers)

    return scibert_tokenizer.tokenize_into_ids(query_documents_frame["abstract"][0])


@status_update("Loading pretrained SciBERT model")
def scibert_load_model() -> BertModel:
    return BertModel.from_pretrained(ModelVersions.scibert)  # type: ignore


@status_update("Embedding query abstract")
def scibert_compute_embedding(
    scibert_embedder: BERTEmbedder, query_abstract_tokenized: TokenIds
) -> Embedding:
    return scibert_embedder.compute_embedding_single_document(query_abstract_tokenized)


def scibert_embed_query(query_documents_frame: DocumentsFrame) -> Embedding:
    learned_scibert_tokens_frame = scibert_load_training_tokens_frame()
    query_abstract_tokenized = scibert_tokenize_query(query_documents_frame)

    scibert_model = scibert_load_model()
    scibert_embedder = BERTEmbedder(
        token_ids_frame=learned_scibert_tokens_frame,
        torch_model=scibert_model,  # type: ignore
    )

    return scibert_compute_embedding(scibert_embedder, query_abstract_tokenized)


@status_update("Loading training corpus")
def longformer_load_training_tokens_frame() -> TokenIdsFrame:
    return read_df_from_parquet(ResultsPaths.language_models.longformer_tokenized_abstracts_parquet)


@status_update("Tokenizing query abstract")
def longformer_tokenize_query(query_documents_frame: DocumentsFrame) -> TokenIds:
    longformer_tokenizer_transformers = LongformerTokenizerFast.from_pretrained(
        ModelVersions.longformer
    )
    longformer_tokenizer = LongformerTokenizer(
        query_documents_frame, longformer_tokenizer_transformers
    )

    return longformer_tokenizer.tokenize_into_ids(query_documents_frame["abstract"][0])


@status_update("Loading pretrained Longformer model")
def longformer_load_model() -> LongformerModel:
    return LongformerModel.from_pretrained(ModelVersions.longformer)  # type: ignore


@status_update("Embedding query abstract")
def longformer_compute_embedding(
    longformer_embedder: LongformerEmbedder,
    query_abstract_tokenized: TokenIds,
) -> Embedding:
    return longformer_embedder.compute_embedding_single_document(query_abstract_tokenized)


def longformer_embed_query(query_documents_frame: DocumentsFrame) -> Embedding:
    learned_longformer_tokens_frame = longformer_load_training_tokens_frame()
    query_abstract_tokenized = longformer_tokenize_query(query_documents_frame)

    longformer_model = longformer_load_model()
    longformer_embedder = LongformerEmbedder(
        token_ids_frame=learned_longformer_tokens_frame,
        torch_model=longformer_model,  # type: ignore
    )

    return longformer_compute_embedding(longformer_embedder, query_abstract_tokenized)


def select_query_embedding_function(
    language_model_choice: LanguageModelChoice,
) -> QueryEmbeddingFunction:
    match language_model_choice:
        case LanguageModelChoice.TFIDF:
            return tfidf_embed_query
        case LanguageModelChoice.BM25:
            return bm25_embed_query
        case LanguageModelChoice.WORD2VEC:
            return word2vec_embed_query
        case LanguageModelChoice.GLOVE:
            return glove_embed_query
        case LanguageModelChoice.FASTTEXT:
            return fasttest_embed_query
        case LanguageModelChoice.BERT:
            return bert_embed_query
        case LanguageModelChoice.SCIBERT:
            return scibert_embed_query
        case LanguageModelChoice.LONGFORMER:
            return longformer_embed_query
        case _:
            raise ValueError(f"Language model choice {language_model_choice} not supported")
