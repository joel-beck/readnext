import pandas as pd

from readnext.data.config import DataPaths
from readnext.modeling.citation_models.model_data import (
    CitationModelDataFromId,
    add_feature_rank_cols,
    add_labels,
    eval_top_n,
    select_top_n,
    set_missing_publication_dates_to_max_rank,
)
from readnext.modeling.config import ResultsPaths


def main() -> None:
    input_document_id = 206594692

    documents_authors_labels_citations_most_cited: pd.DataFrame = (
        pd.read_pickle(DataPaths.merged.documents_authors_labels_citations_most_cited_pkl)
        .set_index("document_id")
        .pipe(add_feature_rank_cols)
        .pipe(set_missing_publication_dates_to_max_rank)
    )

    bibliographic_coupling_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.citation_models.bibliographic_coupling_most_cited_pkl
    )

    co_citation_analysis_most_cited: pd.DataFrame = pd.read_pickle(
        ResultsPaths.citation_models.co_citation_analysis_most_cited_pkl
    )

    citation_model_data_from_id = CitationModelDataFromId(
        document_id=input_document_id,
        documents_data=documents_authors_labels_citations_most_cited,
        co_citation_analysis_data=co_citation_analysis_most_cited,
        bibliographic_coupling_data=bibliographic_coupling_most_cited,
    )

    citation_model_data = citation_model_data_from_id.get_model_data()

    citation_model_data.input_document
    citation_model_data.labels
    citation_model_data.feature_matrix

    top_n_publication_date = select_top_n(citation_model_data, "publication_date_rank")
    top_n_citationcount_document = select_top_n(citation_model_data, "citationcount_document_rank")
    top_n_citationcount_author = select_top_n(citation_model_data, "citationcount_author_rank")
    top_n_co_citation_analysis = select_top_n(citation_model_data, "co_citation_analysis_ranks")
    top_n_bibliographic_coupling = select_top_n(citation_model_data, "bibliographic_coupling_ranks")

    top_n_weighted = (
        citation_model_data.feature_matrix.sum(axis=1)
        .sort_values(ascending=True)
        .head(20)
        .rename("weighted_rank")
    )

    add_labels(top_n_publication_date, citation_model_data.labels)
    add_labels(top_n_citationcount_document, citation_model_data.labels)
    add_labels(top_n_citationcount_author, citation_model_data.labels)
    add_labels(top_n_co_citation_analysis, citation_model_data.labels)
    add_labels(top_n_bibliographic_coupling, citation_model_data.labels)
    add_labels(top_n_weighted, citation_model_data.labels)

    eval_top_n(citation_model_data, "publication_date_rank")
    eval_top_n(citation_model_data, "citationcount_document_rank")
    eval_top_n(citation_model_data, "citationcount_author_rank")
    eval_top_n(citation_model_data, "co_citation_analysis_ranks")
    eval_top_n(citation_model_data, "bibliographic_coupling_ranks")


if __name__ == "__main__":
    main()
