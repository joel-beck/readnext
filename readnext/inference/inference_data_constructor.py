from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.scoring import HybridScorer
from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    DocumentInfo,
    LanguageModelData,
    LanguageModelDataConstructor,
)
from readnext.modeling.citation_models import (
    add_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)
from readnext.modeling.language_models import LanguageModelChoice
from readnext.utils import (
    get_arxiv_id_from_arxiv_url,
    get_arxiv_url_from_arxiv_id,
    load_df_from_pickle,
)


class NonUniqueError(Exception):
    """Raise when a non-unique document identifier is encountered."""


@dataclass
class DocumentIdentifiers:
    d3_document_id: int
    semanticscholar_url: str
    arxiv_url: str
    paper_title: str


@dataclass
class Features:
    publication_date: pd.Series
    citationcount_document: pd.Series
    citationcount_author: pd.Series
    co_citation_analysis: pd.Series
    bibliographic_coupling: pd.Series
    cosine_similarity: pd.Series


@dataclass
class Ranks:
    publication_date: pd.Series
    citationcount_document: pd.Series
    citationcount_author: pd.Series
    co_citation_analysis: pd.Series
    bibliographic_coupling: pd.Series
    cosine_similarity: pd.DataFrame


@dataclass
class Labels:
    arxiv: pd.Series
    integer: pd.Series


@dataclass(kw_only=True)
class Recommendations:
    citation_to_language_candidates: pd.DataFrame
    citation_to_language: pd.DataFrame
    language_to_citation_candidates: pd.DataFrame
    language_to_citation: pd.DataFrame


@dataclass(kw_only=True)
class InferenceDataConstructor:
    query_id: int | None = None
    semanticscholar_url: str | None = None
    arxiv_url: str | None = None
    paper_title: str | None = None
    language_model_choice: LanguageModelChoice

    _documents_data: pd.DataFrame = field(init=False)
    _co_citation_analysis_scores: pd.DataFrame = field(init=False)
    _bibliographic_coupling_scores: pd.DataFrame = field(init=False)
    _cosine_similarities: pd.DataFrame = field(init=False)
    _citation_model_data: CitationModelData = field(init=False)
    _language_model_data: LanguageModelData = field(init=False)

    def __post_init__(self) -> None:
        """
        Collect all information needed for inference right after initialization.
        """
        self.set_documents_data()
        # this step must occur after self._documents_data is set
        self.set_identifiers()
        self.set_co_citation_analysis_scores()
        self.set_bibliographic_coupling_scores()
        self.set_cosine_similarities()
        self.set_citation_model_data()
        self.set_language_model_data()

    def id_from_semanticscholar_url(self, semanticscholar_url: str) -> int:
        """Retrieve D3 document id from Semanticscholar url."""
        return self._documents_data[
            self._documents_data["semanticscholar_url"] == semanticscholar_url
        ].index.item()

    def semanticscholar_url_from_id(self, id: int) -> str:
        """Retrieve Semanticscholar url from D3 document id."""
        return cast(str, self._documents_data.loc[id, "semanticscholar_url"])

    def id_from_arxiv_url(self, arxiv_url: str) -> int:
        """Retrieve D3 document id from Arxiv url."""
        arxiv_id = get_arxiv_id_from_arxiv_url(arxiv_url)

        return self._documents_data[self._documents_data["arxiv_id"] == arxiv_id].index.item()

    def arxiv_url_from_id(self, id: int) -> str:
        """Retrieve Arxiv url from D3 document id."""
        arxiv_id = cast(str, self._documents_data.loc[id, "arxiv_id"])
        return get_arxiv_url_from_arxiv_id(arxiv_id)

    def id_from_title(self, title: str) -> int:
        """Retrieve D3 document id from Paper title."""
        title_rows = self._documents_data[self._documents_data["title"] == title]
        if len(title_rows) > 1:
            raise NonUniqueError(f"Multiple papers with title {title} found.")

        return title_rows.index.item()

    def title_from_id(self, id: int) -> str:
        """Retrieve Paper title from D3 document id."""
        return cast(str, self._documents_data.loc[id, "title"])

    def set_identifiers(self) -> None:
        if self.query_id is not None:
            self.semanticscholar_url = self.semanticscholar_url_from_id(self.query_id)
            self.arxiv_url = self.arxiv_url_from_id(self.query_id)
            self.paper_title = self.title_from_id(self.query_id)
        elif self.semanticscholar_url is not None:
            self.query_id = self.id_from_semanticscholar_url(self.semanticscholar_url)
            self.arxiv_url = self.arxiv_url_from_id(self.query_id)
            self.paper_title = self.title_from_id(self.query_id)
        elif self.arxiv_url is not None:
            self.query_id = self.id_from_arxiv_url(self.arxiv_url)
            self.semanticscholar_url = self.semanticscholar_url_from_id(self.query_id)
            self.paper_title = self.title_from_id(self.query_id)
        elif self.paper_title is not None:
            self.query_id = self.id_from_title(self.paper_title)
            self.semanticscholar_url = self.semanticscholar_url_from_id(self.query_id)
            self.arxiv_url = self.arxiv_url_from_id(self.query_id)

    def set_documents_data(self) -> None:
        self._documents_data = load_df_from_pickle(
            DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
        ).set_index("document_id")

    def set_co_citation_analysis_scores(self) -> None:
        self._co_citation_analysis_scores = load_df_from_pickle(
            ResultsPaths.citation_models.co_citation_analysis_scores_most_cited_pkl
        )

    def set_bibliographic_coupling_scores(self) -> None:
        self._bibliographic_coupling_scores = load_df_from_pickle(
            ResultsPaths.citation_models.bibliographic_coupling_scores_most_cited_pkl
        )

    def get_cosine_similarities_path(self) -> Path:
        match self.language_model_choice:
            case LanguageModelChoice.tfidf:
                return ResultsPaths.language_models.tfidf_cosine_similarities_most_cited_pkl
            case LanguageModelChoice.bm25:
                return ResultsPaths.language_models.bm25_cosine_similarities_most_cited_pkl
            case LanguageModelChoice.word2vec:
                return ResultsPaths.language_models.word2vec_cosine_similarities_most_cited_pkl
            case LanguageModelChoice.glove:
                return ResultsPaths.language_models.glove_cosine_similarities_most_cited_pkl
            case LanguageModelChoice.fasttext:
                return ResultsPaths.language_models.fasttext_cosine_similarities_most_cited_pkl
            case LanguageModelChoice.bert:
                return ResultsPaths.language_models.bert_cosine_similarities_most_cited_pkl
            case LanguageModelChoice.scibert:
                return ResultsPaths.language_models.scibert_cosine_similarities_most_cited_pkl
            case LanguageModelChoice.longformer:
                return ResultsPaths.language_models.longformer_cosine_similarities_most_cited_pkl
            case _:
                raise ValueError(f"Invalid language model choice: {self.language_model_choice}")

    def set_cosine_similarities(self) -> None:
        cosine_similarities_path = self.get_cosine_similarities_path()
        self._cosine_similarities = load_df_from_pickle(cosine_similarities_path)

    def set_citation_model_data(self) -> None:
        assert self.query_id is not None

        citation_model_data_constructor = CitationModelDataConstructor(
            query_document_id=self.query_id,
            documents_data=self._documents_data.pipe(add_feature_rank_cols).pipe(
                set_missing_publication_dates_to_max_rank
            ),
            co_citation_analysis_scores=self._co_citation_analysis_scores,
            bibliographic_coupling_scores=self._bibliographic_coupling_scores,
        )
        self._citation_model_data = CitationModelData.from_constructor(
            citation_model_data_constructor
        )

    def set_language_model_data(self) -> None:
        assert self.query_id is not None

        language_model_data_constructor = LanguageModelDataConstructor(
            query_document_id=self.query_id,
            documents_data=self._documents_data,
            cosine_similarities=self._cosine_similarities,
        )
        self._language_model_data = LanguageModelData.from_constructor(
            language_model_data_constructor
        )

    def collect_document_identifiers(self) -> DocumentIdentifiers:
        assert self.query_id is not None
        assert self.semanticscholar_url is not None
        assert self.arxiv_url is not None
        assert self.paper_title is not None

        return DocumentIdentifiers(
            d3_document_id=self.query_id,
            semanticscholar_url=self.semanticscholar_url,
            arxiv_url=self.arxiv_url,
            paper_title=self.paper_title,
        )

    def collect_document_info(self) -> DocumentInfo:
        return self._citation_model_data.query_document

    def collect_features(self) -> Features:
        return Features(
            self._citation_model_data.info_matrix["publication_date"],
            self._citation_model_data.info_matrix["citationcount_document"],
            self._citation_model_data.info_matrix["citationcount_author"],
            self._citation_model_data.info_matrix["co_citation_analysis"],
            self._citation_model_data.info_matrix["bibliographic_coupling"],
            self._language_model_data.info_matrix["cosine_similarity"],
        )

    def collect_ranks(self) -> Ranks:
        return Ranks(
            self._citation_model_data.feature_matrix["publication_date_rank"],
            self._citation_model_data.feature_matrix["citationcount_document_rank"],
            self._citation_model_data.feature_matrix["citationcount_author_rank"],
            self._citation_model_data.feature_matrix["co_citation_analysis_rank"],
            self._citation_model_data.feature_matrix["bibliographic_coupling_rank"],
            self._language_model_data.cosine_similarity_ranks,
        )

    def collect_labels(self) -> Labels:
        return Labels(
            self._citation_model_data.info_matrix["arxiv_labels"],
            self._citation_model_data.integer_labels,
        )

    def collect_recommendations(self) -> Recommendations:
        hybrid_scorer = HybridScorer(
            language_model_name=self.language_model_choice.value,
            citation_model_data=self._citation_model_data,
            language_model_data=self._language_model_data,
        )
        hybrid_scorer.recommend()

        return Recommendations(
            citation_to_language_candidates=hybrid_scorer.citation_to_language_candidates,
            citation_to_language=hybrid_scorer.citation_to_language_recommendations,
            language_to_citation_candidates=hybrid_scorer.language_to_citation_candidates,
            language_to_citation=hybrid_scorer.language_to_citation_recommendations,
        )
