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
def tfidf_embedder(spacy_tokens_frame: TokensFrame) -> TFIDFEmbedder:
    return TFIDFEmbedder(tokens_frame=spacy_tokens_frame, keyword_algorithm=tfidf)


@pytest.fixture(scope="session")
def word2vec_embedder(
    spacy_tokens_frame: TokensFrame, word2vec_model: Word2VecModelProtocol
) -> Word2VecEmbedder:
    return Word2VecEmbedder(tokens_frame=spacy_tokens_frame, embedding_model=word2vec_model)


@pytest.fixture(scope="session")
def fasttext_embedder(
    spacy_tokens_frame: TokensFrame, fasttext_model: FastTextModelProtocol
) -> FastTextEmbedder:
    return FastTextEmbedder(tokens_frame=spacy_tokens_frame, embedding_model=fasttext_model)


@pytest.fixture(scope="session")
def bert_embedder(
    bert_tokens_id_frame: TokenIdsFrame, bert_model: BertModelProtocol
) -> BERTEmbedder:
    return BERTEmbedder(token_ids_frame=bert_tokens_id_frame, torch_model=bert_model)


@pytest.fixture(scope="session")
def longformer_embedder(
    longformer_tokens_id_frame: TokenIdsFrame, longformer_model: LongformerModelProtocol
) -> LongformerEmbedder:
    return LongformerEmbedder(
        token_ids_frame=longformer_tokens_id_frame, torch_model=longformer_model
    )
