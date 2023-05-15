import os
from collections.abc import Callable
from typing import TypeAlias

import pandas as pd
import spacy
from dotenv import load_dotenv
from gensim.models.fasttext import load_facebook_model
from gensim.models.keyedvectors import KeyedVectors, load_word2vec_format

from readnext.config import DataPaths, ModelPaths, ModelVersions, ResultsPaths
from readnext.evaluation.metrics import CosineSimilarity
from readnext.inference.inference_new_paper_base import (
    QueryLanguageModelDataConstructor,
    send_semanticscholar_request,
)
from readnext.modeling import DocumentInfo, DocumentsInfo, LanguageModelData
from readnext.modeling.document_info import DocumentScore
from readnext.modeling.language_models import (
    BERTEmbedder,
    BERTTokenizer,
    Embedding,
    FastTextEmbedder,
    LanguageModelChoice,
    LongformerEmbedder,
    SpacyTokenizer,
    TFIDFEmbedder,
    Tokens,
    TokensMapping,
    Word2VecEmbedder,
    bm25,
    load_embeddings_from_choice,
    tfidf,
)
from readnext.utils import (
    get_arxiv_id_from_arxiv_url,
    get_semanticscholar_id_from_semanticscholar_url,
    get_semanticscholar_url_from_semanticscholar_id,
    load_df_from_pickle,
)

QueryEmbeddingFunction: TypeAlias = Callable[[DocumentsInfo], Embedding]


def spacy_tokenize_query(query_documents_info: DocumentsInfo) -> Tokens:
    spacy_model = spacy.load(ModelVersions.spacy)
    spacy_tokenizer = SpacyTokenizer(documents_info=query_documents_info, spacy_model=spacy_model)
    return spacy_tokenizer.tokenize_single_document(query_document_info.abstract)


def spacy_load_training_tokens_mapping() -> TokensMapping:
    return SpacyTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.spacy_tokenized_abstracts_mapping_most_cited_pkl
    )


def tfidf_embed_query(query_documents_info: DocumentsInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_info)

    tfidf_embedder = TFIDFEmbedder(
        keyword_algorithm=tfidf, tokens_mapping=learned_spacy_tokens_mapping
    )

    return tfidf_embedder.compute_embedding_single_document(query_abstract_tokenized)


def bm25_embed_query(query_documents_info: DocumentsInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_info)

    bm25_embedder = TFIDFEmbedder(
        keyword_algorithm=bm25, tokens_mapping=learned_spacy_tokens_mapping
    )

    return bm25_embedder.compute_embedding_single_document(query_abstract_tokenized)


def word2vec_embed_query(query_documents_info: DocumentsInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_info)

    print("Loading pretrained Word2Vec Model...")
    word2vec_model: KeyedVectors = load_word2vec_format(ModelPaths.word2vec, binary=True)
    print("Word2Vec Model loaded.")

    word2vec_embedder = Word2VecEmbedder(word2vec_model, learned_spacy_tokens_mapping)

    return word2vec_embedder.compute_embedding_single_document(query_abstract_tokenized)


def glove_embed_query(query_documents_info: DocumentsInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_info)

    print("Loading pretrained GloVe Model...")
    glove_model: KeyedVectors = load_word2vec_format(ModelPaths.glove, binary=False, no_header=True)
    print("GloVe Model loaded.")

    glove_embedder = Word2VecEmbedder(glove_model, learned_spacy_tokens_mapping)

    return glove_embedder.compute_embedding_single_document(query_abstract_tokenized)


def fasttest_embed_query(query_documents_info: DocumentsInfo) -> Embedding:
    learned_spacy_tokens_mapping = spacy_load_training_tokens_mapping()
    query_abstract_tokenized = spacy_tokenize_query(query_documents_info)

    print("Loading pretrained FastText Model...")
    fasttext_model = load_facebook_model(ModelPaths.fasttext)
    print("FastText Model loaded.")

    fasttext_embedder = FastTextEmbedder(fasttext_model, learned_spacy_tokens_mapping)

    return fasttext_embedder.compute_embedding_single_document(query_abstract_tokenized)


def bert_embed_query(query_documents_info: DocumentsInfo) -> Embedding:
    pass


def scibert_embed_query(query_documents_info: DocumentsInfo) -> Embedding:
    pass


def longformer_embed_query(query_documents_info: DocumentsInfo) -> Embedding:
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


# BOOKMARK: Inputs
semanticscholar_url = (
    "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
)
arxiv_url = "https://arxiv.org/abs/1706.03762"
language_model_choice = LanguageModelChoice.fasttext

load_dotenv()

documents_authors_labels_citations_most_cited: pd.DataFrame = load_df_from_pickle(
    DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
).set_index("document_id")


SEMANTICSCHOLAR_API_KEY = os.getenv("SEMANTICSCHOLAR_API_KEY", "")
request_headers = {"x-api-key": SEMANTICSCHOLAR_API_KEY}

response = send_semanticscholar_request(
    semanticscholar_id=get_semanticscholar_id_from_semanticscholar_url(semanticscholar_url),
    request_headers=request_headers,
)

assert response.abstract is not None

# set document id of unseen document to -1
query_document_info = DocumentInfo(document_id=-1, abstract=response.abstract)
query_documents_info = DocumentsInfo(documents_info=[query_document_info])

query_embedding_function = select_query_embedding_function(language_model_choice)

query_abstract_embedding = query_embedding_function(query_documents_info)

candidate_embeddings: pd.DataFrame = load_embeddings_from_choice(language_model_choice)

cosine_similarity_scores: list[DocumentScore] = []

for document_id, candidate_embedding in zip(
    candidate_embeddings["document_id"], candidate_embeddings["embedding"]
):
    document_info = DocumentInfo(document_id=document_id)
    score = CosineSimilarity.score(query_abstract_embedding, candidate_embedding)

    document_score = DocumentScore(document_info=document_info, score=score)
    cosine_similarity_scores.append(document_score)

query_cosine_similarities = pd.DataFrame(
    {"score": [cosine_similarity_scores]}, index=[-1]
).rename_axis("document_id", axis="index")

query_language_model_data_constructor = QueryLanguageModelDataConstructor(
    response=response,
    query_document_id=-1,
    documents_data=documents_authors_labels_citations_most_cited,
    cosine_similarities=query_cosine_similarities,
)
query_language_model_data = LanguageModelData.from_constructor(
    query_language_model_data_constructor
)
