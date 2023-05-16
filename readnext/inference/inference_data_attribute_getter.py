from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import pandas as pd

from readnext.config import ResultsPaths
from readnext.evaluation.scoring import FeatureWeights
from readnext.modeling import (
    CitationModelData,
    CitationModelDataConstructor,
    LanguageModelData,
    LanguageModelDataConstructor,
)
from readnext.modeling.citation_models import (
    add_feature_rank_cols,
    set_missing_publication_dates_to_max_rank,
)
from readnext.modeling.language_models import (
    LanguageModelChoice,
    load_cosine_similarities_from_choice,
)
from readnext.utils import (
    get_arxiv_id_from_arxiv_url,
    get_arxiv_url_from_arxiv_id,
    load_df_from_pickle,
)


@dataclass
class DocumentIdentifiers:
    d3_document_id: int
    semanticscholar_url: str
    arxiv_url: str
    paper_title: str


class NonUniqueError(Exception):
    """Raise when a non-unique document identifier is encountered."""


@dataclass(kw_only=True)
class InferenceDataAttributeGetter(ABC):
    """
    Abstract Base class for getting data attributes in the `InferenceDataConstructor`
    class.
    """

    semanticscholar_url: str | None = None
    arxiv_url: str | None = None
    language_model_choice: LanguageModelChoice
    feature_weights: FeatureWeights
    documents_data: pd.DataFrame

    @abstractmethod
    def get_identifiers(self) -> DocumentIdentifiers:
        ...

    @abstractmethod
    def get_co_citation_analysis_scores(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_bibliographic_coupling_scores(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_cosine_similarities(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_citation_model_data(self) -> CitationModelData:
        ...

    @abstractmethod
    def get_language_model_data(self) -> LanguageModelData:
        ...


@dataclass(kw_only=True)
class InferenceDataAttributeGetterSeenPaper(InferenceDataAttributeGetter):
    """Get data attributes for a paper that is contained in the training data."""

    query_document_id: int | None = None
    paper_title: str | None = None

    def id_from_semanticscholar_url(self, semanticscholar_url: str) -> int:
        """Retrieve D3 document id from Semanticscholar url."""
        return self.documents_data[
            self.documents_data["semanticscholar_url"] == semanticscholar_url
        ].index.item()

    def semanticscholar_url_from_id(self, id: int) -> str:
        """Retrieve Semanticscholar url from D3 document id."""
        return cast(str, self.documents_data.loc[id, "semanticscholar_url"])

    def id_from_arxiv_url(self, arxiv_url: str) -> int:
        """Retrieve D3 document id from Arxiv url."""
        arxiv_id = get_arxiv_id_from_arxiv_url(arxiv_url)

        return self.documents_data[self.documents_data["arxiv_id"] == arxiv_id].index.item()

    def arxiv_url_from_id(self, id: int) -> str:
        """Retrieve Arxiv url from D3 document id."""
        arxiv_id = cast(str, self.documents_data.loc[id, "arxiv_id"])
        return get_arxiv_url_from_arxiv_id(arxiv_id)

    def id_from_title(self, title: str) -> int:
        """Retrieve D3 document id from Paper title."""
        title_rows = self.documents_data[self.documents_data["title"] == title]
        if len(title_rows) > 1:
            raise NonUniqueError(f"Multiple papers with title {title} found.")

        return title_rows.index.item()

    def title_from_id(self, id: int) -> str:
        """Retrieve Paper title from D3 document id."""
        return cast(str, self.documents_data.loc[id, "title"])

    def get_identifiers(self) -> DocumentIdentifiers:
        if self.query_document_id is not None:
            semanticscholar_url = self.semanticscholar_url_from_id(self.query_document_id)
            arxiv_url = self.arxiv_url_from_id(self.query_document_id)
            paper_title = self.title_from_id(self.query_document_id)

            return DocumentIdentifiers(
                d3_document_id=self.query_document_id,
                semanticscholar_url=semanticscholar_url,
                arxiv_url=arxiv_url,
                paper_title=paper_title,
            )

        if self.semanticscholar_url is not None:
            query_document_id = self.id_from_semanticscholar_url(self.semanticscholar_url)
            arxiv_url = self.arxiv_url_from_id(query_document_id)
            paper_title = self.title_from_id(query_document_id)

            return DocumentIdentifiers(
                d3_document_id=query_document_id,
                semanticscholar_url=self.semanticscholar_url,
                arxiv_url=arxiv_url,
                paper_title=paper_title,
            )

        if self.arxiv_url is not None:
            query_document_id = self.id_from_arxiv_url(self.arxiv_url)
            semanticscholar_url = self.semanticscholar_url_from_id(query_document_id)
            paper_title = self.title_from_id(query_document_id)

            return DocumentIdentifiers(
                d3_document_id=query_document_id,
                semanticscholar_url=semanticscholar_url,
                arxiv_url=self.arxiv_url,
                paper_title=paper_title,
            )

        if self.paper_title is not None:
            query_document_id = self.id_from_title(self.paper_title)
            semanticscholar_url = self.semanticscholar_url_from_id(query_document_id)
            arxiv_url = self.arxiv_url_from_id(query_document_id)

            return DocumentIdentifiers(
                d3_document_id=query_document_id,
                semanticscholar_url=semanticscholar_url,
                arxiv_url=arxiv_url,
                paper_title=self.paper_title,
            )

        raise ValueError("No identifiers provided.")

    def get_co_citation_analysis_scores(self) -> pd.DataFrame:
        return load_df_from_pickle(
            ResultsPaths.citation_models.co_citation_analysis_scores_most_cited_pkl
        )

    def get_bibliographic_coupling_scores(self) -> pd.DataFrame:
        return load_df_from_pickle(
            ResultsPaths.citation_models.bibliographic_coupling_scores_most_cited_pkl
        )

    def get_cosine_similarities(self) -> pd.DataFrame:
        return load_cosine_similarities_from_choice(self.language_model_choice)

    def get_citation_model_data(self) -> CitationModelData:
        assert self.query_document_id is not None

        citation_model_data_constructor = CitationModelDataConstructor(
            query_document_id=self.query_document_id,
            documents_data=self.documents_data.pipe(add_feature_rank_cols).pipe(
                set_missing_publication_dates_to_max_rank
            ),
            co_citation_analysis_scores=self.get_co_citation_analysis_scores(),
            bibliographic_coupling_scores=self.get_bibliographic_coupling_scores(),
        )
        return CitationModelData.from_constructor(citation_model_data_constructor)

    def get_language_model_data(self) -> LanguageModelData:
        assert self.query_document_id is not None

        language_model_data_constructor = LanguageModelDataConstructor(
            query_document_id=self.query_document_id,
            documents_data=self.documents_data,
            cosine_similarities=self.get_cosine_similarities(),
        )
        return LanguageModelData.from_constructor(language_model_data_constructor)
