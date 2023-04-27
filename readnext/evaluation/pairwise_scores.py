import pandas as pd
from tqdm import tqdm

from readnext.evaluation.metrics import (
    PairwiseMetric,
    cosine_similarity_from_df,
    count_common_citations_from_df,
    count_common_references_from_df,
)
from readnext.modeling import DocumentInfo, DocumentScore


def find_top_n_matches_single_document(
    input_df: pd.DataFrame,
    document_ids: list[int],
    query_document_id: int,
    pairwise_metric: PairwiseMetric,
    n: int,
) -> list[DocumentScore]:
    """
    Find the n documents with the highest pairwise score for a single document.
    """
    scores = []
    for document_id in document_ids:
        if document_id == query_document_id:
            continue
        document_info = DocumentInfo(document_id=document_id)
        score = pairwise_metric(input_df, query_document_id, document_id)
        scores.append(DocumentScore(document_info, score))

    return sorted(scores, key=lambda x: x.score, reverse=True)[:n]


def precompute_pairwise_scores(
    input_df: pd.DataFrame, pairwise_metric: PairwiseMetric, n: int | None
) -> pd.DataFrame:
    if n is None:
        n = len(input_df)

    tqdm.pandas()
    document_ids = input_df["document_id"].tolist()

    return (
        pd.DataFrame(data=document_ids, columns=["document_id"])
        .assign(
            scores=lambda df: df["document_id"].progress_apply(
                lambda query_document_id: find_top_n_matches_single_document(
                    input_df, document_ids, query_document_id, pairwise_metric, n
                )
            )
        )
        .set_index("document_id")
    )


# Set value for `n` higher for co-citation analysis and bibliographic coupling since
# they are features for the weighted linear model. The higher the value for `n`, the
# more observations the weighted model is able to use.
def precompute_co_citations(
    df: pd.DataFrame,
    n: int | None = None,
) -> pd.DataFrame:
    return precompute_pairwise_scores(df, count_common_citations_from_df, n)


def precompute_co_references(
    df: pd.DataFrame,
    n: int | None = None,
) -> pd.DataFrame:
    return precompute_pairwise_scores(df, count_common_references_from_df, n)


def precompute_cosine_similarities(
    df: pd.DataFrame,
    n: int | None = None,
) -> pd.DataFrame:
    return precompute_pairwise_scores(df, cosine_similarity_from_df, n)
