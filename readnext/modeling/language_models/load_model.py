from typing import Any, Literal, overload

from gensim.models.fasttext import FastText, load_facebook_model
from gensim.models.keyedvectors import KeyedVectors, load_word2vec_format
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, LongformerModel

from readnext.config import ModelPaths, ModelVersions
from readnext.modeling.language_models.model_choice import LanguageModelChoice


@overload
def load_language_model(
    language_model_choice: Literal[LanguageModelChoice.TFIDF],
) -> TfidfVectorizer:
    ...


@overload
def load_language_model(
    language_model_choice: Literal[LanguageModelChoice.WORD2VEC],
) -> KeyedVectors:
    ...


@overload
def load_language_model(language_model_choice: Literal[LanguageModelChoice.GLOVE]) -> KeyedVectors:
    ...


@overload
def load_language_model(language_model_choice: Literal[LanguageModelChoice.FASTTEXT]) -> FastText:
    ...


@overload
def load_language_model(language_model_choice: Literal[LanguageModelChoice.BERT]) -> BertModel:
    ...


@overload
def load_language_model(language_model_choice: Literal[LanguageModelChoice.SCIBERT]) -> BertModel:
    ...


@overload
def load_language_model(
    language_model_choice: Literal[LanguageModelChoice.LONGFORMER],
) -> LongformerModel:
    ...


def load_language_model(language_model_choice: LanguageModelChoice) -> Any:
    match language_model_choice:
        case LanguageModelChoice.TFIDF:
            return TfidfVectorizer()
        case LanguageModelChoice.WORD2VEC:
            return load_word2vec_format(ModelPaths.word2vec, binary=True)
        case LanguageModelChoice.GLOVE:
            return load_word2vec_format(ModelPaths.glove, binary=False, no_header=True)
        case LanguageModelChoice.FASTTEXT:
            return load_facebook_model(ModelPaths.fasttext)
        case LanguageModelChoice.BERT:
            return BertModel.from_pretrained(ModelVersions.bert)
        case LanguageModelChoice.SCIBERT:
            return BertModel.from_pretrained(ModelVersions.scibert)
        case LanguageModelChoice.LONGFORMER:
            return LongformerModel.from_pretrained(ModelVersions.longformer)
        case _:
            raise ValueError(f"Language model {language_model_choice} not implemented.")
