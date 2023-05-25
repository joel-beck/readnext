import pytest

from readnext.modeling.language_models import (
    BERTEmbedder,
    FastTextEmbedder,
    LongformerEmbedder,
    TFIDFEmbedder,
    Word2VecEmbedder,
    tfidf,
)
from readnext.utils import (
    BertModelProtocol,
    FastTextModelProtocol,
    LongformerModelProtocol,
    TokensMapping,
    Word2VecModelProtocol,
)
from readnext.utils.aliases import TokensIdMapping


@pytest.fixture(scope="session")
def tfidf_embedder(spacy_tokens_mapping: TokensMapping) -> TFIDFEmbedder:
    return TFIDFEmbedder(tokens_mapping=spacy_tokens_mapping, keyword_algorithm=tfidf)


@pytest.fixture(scope="session")
def word2vec_embedder(
    spacy_tokens_mapping: TokensMapping, word2vec_model: Word2VecModelProtocol
) -> Word2VecEmbedder:
    return Word2VecEmbedder(tokens_mapping=spacy_tokens_mapping, embedding_model=word2vec_model)


@pytest.fixture(scope="session")
def fasttext_embedder(
    spacy_tokens_mapping: TokensMapping, fasttext_model: FastTextModelProtocol
) -> FastTextEmbedder:
    return FastTextEmbedder(tokens_mapping=spacy_tokens_mapping, embedding_model=fasttext_model)


@pytest.fixture(scope="session")
def bert_embedder(
    bert_tokens_id_mapping: TokensIdMapping, bert_model: BertModelProtocol
) -> BERTEmbedder:
    return BERTEmbedder(tokens_tensor_mapping=bert_tokens_id_mapping, torch_model=bert_model)


@pytest.fixture(scope="session")
def longformer_embedder(
    longformer_tokens_id_mapping: TokensIdMapping, longformer_model: LongformerModelProtocol
) -> LongformerEmbedder:
    return LongformerEmbedder(
        tokens_tensor_mapping=longformer_tokens_id_mapping,
        torch_model=longformer_model,
    )
