import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, LongformerModel

from readnext.modeling.language_models import (
    BERTEmbedder,
    BM25Embedder,
    LongformerEmbedder,
    TFIDFEmbedder,
    GensimEmbedder,
)
from readnext.utils.aliases import TokenIdsFrame, TokensFrame
from readnext.utils.protocols import FastTextModelProtocol, Word2VecModelProtocol


@pytest.fixture(scope="session")
def tfidf_embedder(
    test_spacy_tokens_frame: TokensFrame, tfidf_vectorizer: TfidfVectorizer
) -> TFIDFEmbedder:
    return TFIDFEmbedder(
        tokens_frame=test_spacy_tokens_frame.head(3), tfidf_vectorizer=tfidf_vectorizer
    )


@pytest.fixture(scope="session")
def bm25(test_spacy_tokens_frame: TokensFrame) -> BM25Embedder:
    return BM25Embedder(tokens_frame=test_spacy_tokens_frame.head(3))


@pytest.fixture(scope="session")
def word2vec_embedder(
    test_spacy_tokens_frame: TokensFrame, word2vec_model: Word2VecModelProtocol
) -> GensimEmbedder:
    return GensimEmbedder(
        tokens_frame=test_spacy_tokens_frame.head(3), keyed_vectors=word2vec_model  # type: ignore
    )


@pytest.fixture(scope="session")
def fasttext_embedder(
    test_spacy_tokens_frame: TokensFrame, fasttext_model: FastTextModelProtocol
) -> GensimEmbedder:
    return GensimEmbedder(
        tokens_frame=test_spacy_tokens_frame.head(3), keyed_vectors=fasttext_model.wv  # type: ignore
    )


@pytest.fixture(scope="session")
def bert_embedder(test_bert_token_ids_frame: TokenIdsFrame, bert_model: BertModel) -> BERTEmbedder:
    return BERTEmbedder(token_ids_frame=test_bert_token_ids_frame.head(3), torch_model=bert_model)


@pytest.fixture(scope="session")
def longformer_embedder(
    test_longformer_token_ids_frame: TokenIdsFrame, longformer_model: LongformerModel
) -> LongformerEmbedder:
    return LongformerEmbedder(
        token_ids_frame=test_longformer_token_ids_frame.head(3), torch_model=longformer_model
    )
