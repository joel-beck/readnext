import pytest
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, LongformerModel

from readnext.modeling.language_models import (
    BERTEmbedder,
    BM25Embedder,
    GensimEmbedder,
    LongformerEmbedder,
    TFIDFEmbedder,
)
from readnext.utils.aliases import TokenIdsFrame, TokensFrame


@pytest.fixture(scope="session")
def tfidf_embedder(
    test_spacy_tokens_frame: TokensFrame, tfidf_model: TfidfVectorizer
) -> TFIDFEmbedder:
    return TFIDFEmbedder(tokens_frame=test_spacy_tokens_frame.head(3), tfidf_vectorizer=tfidf_model)


@pytest.fixture(scope="session")
def bm25_embedder(test_spacy_tokens_frame: TokensFrame) -> BM25Embedder:
    return BM25Embedder(tokens_frame=test_spacy_tokens_frame.head(3))


@pytest.fixture(scope="session")
def word2vec_embedder(
    test_spacy_tokens_frame: TokensFrame, word2vec_model: KeyedVectors
) -> GensimEmbedder:
    return GensimEmbedder(
        tokens_frame=test_spacy_tokens_frame.head(3), keyed_vectors=word2vec_model
    )


@pytest.fixture(scope="session")
def glove_embedder(
    test_spacy_tokens_frame: TokensFrame, glove_model: KeyedVectors
) -> GensimEmbedder:
    return GensimEmbedder(tokens_frame=test_spacy_tokens_frame.head(3), keyed_vectors=glove_model)


@pytest.fixture(scope="session")
def fasttext_embedder(
    test_spacy_tokens_frame: TokensFrame, fasttext_model: FastText
) -> GensimEmbedder:
    return GensimEmbedder(
        tokens_frame=test_spacy_tokens_frame.head(3),
        keyed_vectors=fasttext_model.wv,
    )


@pytest.fixture(scope="session")
def bert_embedder(test_bert_token_ids_frame: TokenIdsFrame, bert_model: BertModel) -> BERTEmbedder:
    return BERTEmbedder(token_ids_frame=test_bert_token_ids_frame.head(3), torch_model=bert_model)


@pytest.fixture(scope="session")
def scibert_embedder(
    test_bert_token_ids_frame: TokenIdsFrame, scibert_model: BertModel
) -> BERTEmbedder:
    return BERTEmbedder(
        token_ids_frame=test_bert_token_ids_frame.head(3), torch_model=scibert_model
    )


@pytest.fixture(scope="session")
def longformer_embedder(
    test_longformer_token_ids_frame: TokenIdsFrame, longformer_model: LongformerModel
) -> LongformerEmbedder:
    return LongformerEmbedder(
        token_ids_frame=test_longformer_token_ids_frame.head(3), torch_model=longformer_model
    )
