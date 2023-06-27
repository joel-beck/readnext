import pytest

from readnext.modeling.language_models import (
    BERTEmbedder,
    FastTextEmbedder,
    LongformerEmbedder,
    TFIDFEmbedder,
    Word2VecEmbedder,
    tfidf,
)
from readnext.utils.aliases import TokenIdsFrame, TokensFrame
from readnext.utils.protocols import (
    BertModelProtocol,
    FastTextModelProtocol,
    LongformerModelProtocol,
    Word2VecModelProtocol,
)


@pytest.fixture(scope="session")
def tfidf_embedder(test_spacy_tokens_frame: TokensFrame) -> TFIDFEmbedder:
    return TFIDFEmbedder(tokens_frame=test_spacy_tokens_frame.head(3), keyword_algorithm=tfidf)


@pytest.fixture(scope="session")
def word2vec_embedder(
    test_spacy_tokens_frame: TokensFrame, word2vec_model: Word2VecModelProtocol
) -> Word2VecEmbedder:
    return Word2VecEmbedder(
        tokens_frame=test_spacy_tokens_frame.head(3), embedding_model=word2vec_model
    )


@pytest.fixture(scope="session")
def fasttext_embedder(
    test_spacy_tokens_frame: TokensFrame, fasttext_model: FastTextModelProtocol
) -> FastTextEmbedder:
    return FastTextEmbedder(
        tokens_frame=test_spacy_tokens_frame.head(3), embedding_model=fasttext_model
    )


@pytest.fixture(scope="session")
def bert_embedder(
    test_bert_token_ids_frame: TokenIdsFrame, bert_model: BertModelProtocol
) -> BERTEmbedder:
    return BERTEmbedder(token_ids_frame=test_bert_token_ids_frame.head(3), torch_model=bert_model)


@pytest.fixture(scope="session")
def longformer_embedder(
    test_longformer_token_ids_frame: TokenIdsFrame, longformer_model: LongformerModelProtocol
) -> LongformerEmbedder:
    return LongformerEmbedder(
        token_ids_frame=test_longformer_token_ids_frame.head(3), torch_model=longformer_model
    )
