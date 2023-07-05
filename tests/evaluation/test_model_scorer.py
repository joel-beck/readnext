from readnext import FeatureWeights
from readnext.evaluation.metrics import AveragePrecision
from readnext.evaluation.metrics import CountUniqueLabels
from readnext.evaluation.scoring import CitationModelScorer, LanguageModelScorer

import polars as pl


def test_select_top_n_citation(citation_model_scorer: CitationModelScorer) -> None:
    top_n_frame = citation_model_scorer.select_top_n(FeatureWeights(), n=10)
    assert isinstance(top_n_frame, pl.DataFrame)

    assert top_n_frame.width == 2
    assert top_n_frame.columns == ["candidate_d3_document_id", "weighted_points"]

    assert top_n_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert top_n_frame["weighted_points"].dtype == pl.Float32

    assert (top_n_frame["weighted_points"] >= 0).all()
    assert top_n_frame["weighted_points"].is_sorted(descending=True)


def test_display_top_n_citation(citation_model_scorer: CitationModelScorer) -> None:
    recommendations_frame = citation_model_scorer.display_top_n(FeatureWeights(), n=10)
    assert isinstance(recommendations_frame, pl.DataFrame)

    assert recommendations_frame.width == 18
    assert recommendations_frame.columns == [
        "candidate_d3_document_id",
        "weighted_points",
        "title",
        "author",
        "arxiv_labels",
        "integer_label",
        "semanticscholar_url",
        "arxiv_url",
        "publication_date",
        "publication_date_points",
        "citationcount_document",
        "citationcount_document_points",
        "citationcount_author",
        "citationcount_author_points",
        "co_citation_analysis_score",
        "co_citation_analysis_points",
        "bibliographic_coupling_score",
        "bibliographic_coupling_points",
    ]

    assert recommendations_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert recommendations_frame["weighted_points"].dtype == pl.Float32
    assert recommendations_frame["title"].dtype == pl.Utf8
    assert recommendations_frame["author"].dtype == pl.Utf8
    assert recommendations_frame["arxiv_labels"].dtype == pl.List(pl.Utf8)
    assert recommendations_frame["integer_label"].dtype == pl.Int64
    assert recommendations_frame["semanticscholar_url"].dtype == pl.Utf8
    assert recommendations_frame["arxiv_url"].dtype == pl.Utf8
    assert recommendations_frame["publication_date"].dtype == pl.Utf8
    assert recommendations_frame["publication_date_points"].dtype == pl.Float32
    assert recommendations_frame["citationcount_document"].dtype == pl.Int64
    assert recommendations_frame["citationcount_document_points"].dtype == pl.Float32
    assert recommendations_frame["citationcount_author"].dtype == pl.Int64
    assert recommendations_frame["citationcount_author_points"].dtype == pl.Float32
    assert recommendations_frame["co_citation_analysis_score"].dtype == pl.Int64
    assert recommendations_frame["co_citation_analysis_points"].dtype == pl.Float32
    assert recommendations_frame["bibliographic_coupling_score"].dtype == pl.Int64
    assert recommendations_frame["bibliographic_coupling_points"].dtype == pl.Float32

    assert recommendations_frame["weighted_points"].is_sorted(descending=True)


def test_score_top_n_citation(citation_model_scorer: CitationModelScorer) -> None:
    average_precision_score = citation_model_scorer.score_top_n(AveragePrecision())
    assert isinstance(average_precision_score, float)
    assert 0 <= average_precision_score <= 1

    unique_labels_count = citation_model_scorer.score_top_n(CountUniqueLabels())
    assert isinstance(unique_labels_count, int)
    assert unique_labels_count >= 0


def test_select_top_n_language(language_model_scorer: LanguageModelScorer) -> None:
    top_n_frame = language_model_scorer.select_top_n(FeatureWeights(), n=10)
    assert isinstance(top_n_frame, pl.DataFrame)

    assert top_n_frame.width == 2
    assert top_n_frame.columns == ["candidate_d3_document_id", "cosine_similarity"]

    assert top_n_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert top_n_frame["cosine_similarity"].dtype == pl.Float64

    assert (top_n_frame["cosine_similarity"] >= 0).all()
    assert (top_n_frame["cosine_similarity"] <= 1).all()
    assert top_n_frame["cosine_similarity"].is_sorted(descending=True)


def test_display_top_n_language(language_model_scorer: LanguageModelScorer) -> None:
    recommendations_frame = language_model_scorer.display_top_n(FeatureWeights(), n=10)
    assert isinstance(recommendations_frame, pl.DataFrame)

    assert recommendations_frame.width == 9
    assert recommendations_frame.columns == [
        "candidate_d3_document_id",
        "cosine_similarity",
        "title",
        "author",
        "publication_date",
        "arxiv_labels",
        "integer_label",
        "semanticscholar_url",
        "arxiv_url",
    ]

    assert recommendations_frame["candidate_d3_document_id"].dtype == pl.Int64
    assert recommendations_frame["cosine_similarity"].dtype == pl.Float64
    assert recommendations_frame["title"].dtype == pl.Utf8
    assert recommendations_frame["author"].dtype == pl.Utf8
    assert recommendations_frame["publication_date"].dtype == pl.Utf8
    assert recommendations_frame["arxiv_labels"].dtype == pl.List(pl.Utf8)
    assert recommendations_frame["integer_label"].dtype == pl.Int64
    assert recommendations_frame["semanticscholar_url"].dtype == pl.Utf8
    assert recommendations_frame["arxiv_url"].dtype == pl.Utf8

    assert recommendations_frame["cosine_similarity"].is_sorted(descending=True)


def test_score_top_n_language(language_model_scorer: LanguageModelScorer) -> None:
    average_precision_score = language_model_scorer.score_top_n(AveragePrecision())
    assert isinstance(average_precision_score, float)
    assert 0 <= average_precision_score <= 1

    unique_labels_count = language_model_scorer.score_top_n(CountUniqueLabels())
    assert isinstance(unique_labels_count, int)
    assert unique_labels_count >= 0
