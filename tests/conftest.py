from pathlib import Path

import pandas as pd
import pytest

from readnext.data import (
    add_citation_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)
from readnext.evaluation.scoring import FeatureWeights
from readnext.inference.attribute_getter import SeenPaperAttributeGetter, UnseenPaperAttributeGetter
from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    LanguageModelData,
    LanguageModelDataConstructor,
)
from readnext.modeling.language_models import LanguageModelChoice
from readnext.utils import (
    EmbeddingsMapping,
    ScoresFrame,
    TokensIdMapping,
    TokensMapping,
    load_df_from_pickle,
    load_object_from_pickle,
)


@pytest.fixture(scope="session")
def root_path() -> Path:
    """Return project root path when pytest is executed from the project root directory."""
    return Path().cwd()


@pytest.fixture(scope="session")
def test_data_size() -> int:
    return 100


# SECTION: Test Data
@pytest.fixture(scope="session")
def test_documents_authors_labels_citations_most_cited(root_path: Path) -> pd.DataFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_documents_authors_labels_citations_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bert_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_bert_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bert_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_bert_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bert_tokenized_abstracts_mapping_most_cited(root_path: Path) -> TokensIdMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_bert_tokenized_abstracts_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bibliographic_coupling_scores_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_bibliographic_coupling_scores_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bm25_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_bm25_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_bm25_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_bm25_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_co_citation_analysis_scores_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_co_citation_analysis_scores_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_fasttext_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_fasttext_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_fasttext_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_fasttext_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_glove_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_glove_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_glove_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_glove_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_longformer_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_longformer_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_longformer_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_longformer_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_longformer_tokenized_abstracts_mapping_most_cited(root_path: Path) -> TokensIdMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_longformer_tokenized_abstracts_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_scibert_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_scibert_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_scibert_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_scibert_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_scibert_tokenized_abstracts_mapping_most_cited(root_path: Path) -> TokensIdMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_scibert_tokenized_abstracts_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_spacy_tokenized_abstracts_mapping_most_cited(root_path: Path) -> TokensMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_spacy_tokenized_abstracts_mapping_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_tfidf_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_tfidf_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_tfidf_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_tfidf_embeddings_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_word2vec_cosine_similarities_most_cited(root_path: Path) -> ScoresFrame:
    return load_df_from_pickle(
        root_path / "tests" / "data" / "test_word2vec_cosine_similarities_most_cited.pkl"
    )


@pytest.fixture(scope="session")
def test_word2vec_embeddings_most_cited(root_path: Path) -> EmbeddingsMapping:
    return load_object_from_pickle(
        root_path / "tests" / "data" / "test_word2vec_embeddings_most_cited.pkl"
    )


# SECTION: Model Data Constructors
@pytest.fixture(scope="session")
def citation_model_data_constructor(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
    test_co_citation_analysis_scores_most_cited: ScoresFrame,
    test_bibliographic_coupling_scores_most_cited: ScoresFrame,
) -> CitationModelDataConstructor:
    query_d3_document_id = 206594692

    return CitationModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=test_documents_authors_labels_citations_most_cited.pipe(
            add_citation_feature_rank_cols
        ).pipe(set_missing_publication_dates_to_max_rank),
        co_citation_analysis_scores=test_co_citation_analysis_scores_most_cited,
        bibliographic_coupling_scores=test_bibliographic_coupling_scores_most_cited,
    )


@pytest.fixture(scope="session")
def language_model_data_constructor(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
    test_bert_cosine_similarities_most_cited: ScoresFrame,
) -> LanguageModelDataConstructor:
    query_d3_document_id = 206594692

    return LanguageModelDataConstructor(
        d3_document_id=query_d3_document_id,
        documents_data=test_documents_authors_labels_citations_most_cited,
        cosine_similarities=test_bert_cosine_similarities_most_cited,
    )


# SECTION: Model Data
@pytest.fixture(scope="session")
def citation_model_data(
    citation_model_data_constructor: CitationModelDataConstructor,
) -> CitationModelData:
    return CitationModelData.from_constructor(citation_model_data_constructor)


@pytest.fixture(scope="session")
def language_model_data(
    language_model_data_constructor: LanguageModelDataConstructor,
) -> LanguageModelData:
    return LanguageModelData.from_constructor(language_model_data_constructor)


# SECTION: Inference
# SUBSECTION: AttributeGetter
@pytest.fixture(scope="session")
def seen_paper_attribute_getter_co_citation_analysis(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return seen_paper_attribute_getter.get_co_citation_analysis_scores()


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_co_citation_analysis(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    unseen_paper_attribute_getter = UnseenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return unseen_paper_attribute_getter.get_co_citation_analysis_scores()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_bibliographic_coupling(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return seen_paper_attribute_getter.get_bibliographic_coupling_scores()


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_bibliographic_coupling(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    unseen_paper_attribute_getter = UnseenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return unseen_paper_attribute_getter.get_bibliographic_coupling_scores()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_tfidf(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_cosine_similarities_tfidf(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    unseen_paper_attribute_getter = UnseenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return unseen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_bm25(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.bm25,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_cosine_similarities_bm25(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    unseen_paper_attribute_getter = UnseenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.bm25,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return unseen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_word2vec(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.word2vec,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_cosine_similarities_word2vec(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    unseen_paper_attribute_getter = UnseenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.word2vec,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return unseen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_glove(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.glove,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_cosine_similarities_glove(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    unseen_paper_attribute_getter = UnseenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.glove,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return unseen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_fasttext(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.fasttext,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_cosine_similarities_fasttext(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    unseen_paper_attribute_getter = UnseenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.fasttext,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return unseen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_bert(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.bert,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_cosine_similarities_bert(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    unseen_paper_attribute_getter = UnseenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.bert,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return unseen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_scibert(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.scibert,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_cosine_similarities_scibert(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    unseen_paper_attribute_getter = UnseenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.scibert,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return unseen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_cosine_similarities_longformer(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.longformer,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return seen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_cosine_similarities_longformer(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> ScoresFrame:
    semantischolar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    unseen_paper_attribute_getter = UnseenPaperAttributeGetter(
        semanticscholar_id=semantischolar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.longformer,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return unseen_paper_attribute_getter.get_cosine_similarities()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_citation_model_data(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> CitationModelData:
    semanticscholar_id = "2c03df8b48bf3fa39054345bafabfeff15bfd11d"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return seen_paper_attribute_getter.get_citation_model_data()


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_citation_model_data(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> CitationModelData:
    semanticscholar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    unseen_paper_attribute_getter = UnseenPaperAttributeGetter(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return unseen_paper_attribute_getter.get_citation_model_data()


@pytest.fixture(scope="session")
def seen_paper_attribute_getter_language_model_data(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> LanguageModelData:
    semanticscholar_id = "2c03df8b48bf3fa39054345bafabfeff15bfd11d"

    seen_paper_attribute_getter = SeenPaperAttributeGetter(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return seen_paper_attribute_getter.get_language_model_data()


@pytest.fixture(scope="session")
def unseen_paper_attribute_getter_language_model_data(
    test_documents_authors_labels_citations_most_cited: pd.DataFrame,
) -> LanguageModelData:
    semanticscholar_id = "8ca62fdf4c276ea3052dc96dcfd8ee96ca425a48"

    unseen_paper_attribute_getter = UnseenPaperAttributeGetter(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=None,
        arxiv_id=None,
        arxiv_url=None,
        language_model_choice=LanguageModelChoice.tfidf,
        feature_weights=FeatureWeights(),
        documents_data=test_documents_authors_labels_citations_most_cited,
    )

    return unseen_paper_attribute_getter.get_language_model_data()
