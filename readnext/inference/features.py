from dataclasses import dataclass

import polars as pl

from readnext import FeatureWeights
from readnext.utils import generate_frame_repr


@dataclass(kw_only=True)
class Features:
    publication_date: pl.DataFrame
    citationcount_document: pl.DataFrame
    citationcount_author: pl.DataFrame
    co_citation_analysis: pl.DataFrame
    bibliographic_coupling: pl.DataFrame
    cosine_similarity: pl.DataFrame
    feature_weights: FeatureWeights

    def __repr__(self) -> str:
        publication_date_repr = f"publication_date={generate_frame_repr(self.publication_date)}"
        citationcount_document_repr = (
            f"citationcount_document={generate_frame_repr(self.citationcount_document)}"
        )
        citationcount_author_repr = (
            f"citationcount_author={generate_frame_repr(self.citationcount_author)}"
        )
        co_citation_analysis_repr = (
            f"co_citation_analysis={generate_frame_repr(self.co_citation_analysis)}"
        )
        bibliographic_coupling_repr = (
            f"bibliographic_coupling={generate_frame_repr(self.bibliographic_coupling)}"
        )
        cosine_similarity_repr = f"cosine_similarity={generate_frame_repr(self.cosine_similarity)}"
        feature_weights_repr = f"feature_weights={self.feature_weights!r}"

        return (
            f"{self.__class__.__name__}(\n"
            f"  {publication_date_repr},\n"
            f"  {citationcount_document_repr},\n"
            f"  {citationcount_author_repr},\n"
            f"  {co_citation_analysis_repr},\n"
            f"  {bibliographic_coupling_repr},\n"
            f"  {cosine_similarity_repr},\n"
            f"  {feature_weights_repr}\n"
            ")"
        )


@dataclass(kw_only=True)
class Ranks:
    publication_date: pl.DataFrame
    citationcount_document: pl.DataFrame
    citationcount_author: pl.DataFrame
    co_citation_analysis: pl.DataFrame
    bibliographic_coupling: pl.DataFrame

    def __repr__(self) -> str:
        publication_date_repr = f"publication_date={generate_frame_repr(self.publication_date)}"
        citationcount_document_repr = (
            f"citationcount_document={generate_frame_repr(self.citationcount_document)}"
        )
        citationcount_author_repr = (
            f"citationcount_author={generate_frame_repr(self.citationcount_author)}"
        )
        co_citation_analysis_repr = (
            f"co_citation_analysis={generate_frame_repr(self.co_citation_analysis)}"
        )
        bibliographic_coupling_repr = (
            f"bibliographic_coupling={generate_frame_repr(self.bibliographic_coupling)}"
        )

        return (
            f"{self.__class__.__name__}(\n"
            f"  {publication_date_repr},\n"
            f"  {citationcount_document_repr},\n"
            f"  {citationcount_author_repr},\n"
            f"  {co_citation_analysis_repr},\n"
            f"  {bibliographic_coupling_repr},\n"
            ")"
        )


@dataclass(kw_only=True)
class Points:
    publication_date: pl.DataFrame
    citationcount_document: pl.DataFrame
    citationcount_author: pl.DataFrame
    co_citation_analysis: pl.DataFrame
    bibliographic_coupling: pl.DataFrame

    def __repr__(self) -> str:
        publication_date_repr = f"publication_date={generate_frame_repr(self.publication_date)}"
        citationcount_document_repr = (
            f"citationcount_document={generate_frame_repr(self.citationcount_document)}"
        )
        citationcount_author_repr = (
            f"citationcount_author={generate_frame_repr(self.citationcount_author)}"
        )
        co_citation_analysis_repr = (
            f"co_citation_analysis={generate_frame_repr(self.co_citation_analysis)}"
        )
        bibliographic_coupling_repr = (
            f"bibliographic_coupling={generate_frame_repr(self.bibliographic_coupling)}"
        )

        return (
            f"{self.__class__.__name__}(\n"
            f"  {publication_date_repr},\n"
            f"  {citationcount_document_repr},\n"
            f"  {citationcount_author_repr},\n"
            f"  {co_citation_analysis_repr},\n"
            f"  {bibliographic_coupling_repr},\n"
            ")"
        )


@dataclass(kw_only=True)
class Labels:
    arxiv: pl.DataFrame
    integer: pl.DataFrame

    def __repr__(self) -> str:
        arxiv_repr = f"arxiv={generate_frame_repr(self.arxiv)}"
        integer_repr = f"integer={generate_frame_repr(self.integer)}"

        return f"{self.__class__.__name__}(\n  {arxiv_repr},\n  {integer_repr}\n)"


@dataclass(kw_only=True)
class Recommendations:
    citation_to_language_candidates: pl.DataFrame
    citation_to_language: pl.DataFrame
    language_to_citation_candidates: pl.DataFrame
    language_to_citation: pl.DataFrame

    def __repr__(self) -> str:
        citation_to_language_candidates_repr = (
            f"citation_to_language_candidates="
            f"{generate_frame_repr(self.citation_to_language_candidates)}"
        )
        citation_to_language_repr = (
            f"citation_to_language={generate_frame_repr(self.citation_to_language)}"
        )
        language_to_citation_candidates_repr = (
            f"language_to_citation_candidates="
            f"{generate_frame_repr(self.language_to_citation_candidates)}"
        )
        language_to_citation_repr = (
            f"language_to_citation={generate_frame_repr(self.language_to_citation)}"
        )

        return (
            f"{self.__class__.__name__}(\n"
            f"  {citation_to_language_candidates_repr},\n"
            f"  {citation_to_language_repr},\n"
            f"  {language_to_citation_candidates_repr},\n"
            f"  {language_to_citation_repr}\n"
            ")"
        )
