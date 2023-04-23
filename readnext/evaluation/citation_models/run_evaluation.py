import pandas as pd

from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.citation_models import (
    ScoringFeature,
    add_feature_rank_cols,
    display_top_n,
    score_top_n,
    set_missing_publication_dates_to_max_rank,
)
from readnext.modeling.citation_models import CitationModelDataFromId


def main() -> None:
    # evaluation for a single input document
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

    print(citation_model_data.input_document)
    display_top_n(citation_model_data, ScoringFeature.weighted, n=10)
    score_top_n(citation_model_data, ScoringFeature.weighted, n=10)


if __name__ == "__main__":
    main()
