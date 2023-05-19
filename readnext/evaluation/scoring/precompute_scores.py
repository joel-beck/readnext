import pandas as pd
from tqdm import tqdm

from readnext.evaluation.metrics import (
    CosineSimilarity,
    CountCommonCitations,
    CountCommonReferences,
    PairwiseMetric,
)
from readnext.modeling import DocumentInfo, DocumentScore
from readnext.utils import ScoresFrame


def find_top_n_matches_single_document(
    input_df: pd.DataFrame,
    candidate_d3_document_ids: list[int],
    query_d3_document_id: int,
    pairwise_metric: PairwiseMetric,
    n: int,
) -> list[DocumentScore]:
    """
    Find the n documents with the highest pairwise score for a single query document.
    """
    scores = []
    for d3_document_id in candidate_d3_document_ids:
        if d3_document_id == query_d3_document_id:
            continue
        document_info = DocumentInfo(d3_document_id=d3_document_id)
        score = pairwise_metric.from_df(input_df, query_d3_document_id, d3_document_id)
        scores.append(DocumentScore(document_info=document_info, score=score))

    return sorted(scores, key=lambda x: x.score, reverse=True)[:n]


def precompute_pairwise_scores(
    input_df: pd.DataFrame,
    candidate_d3_document_ids: list[int],
    pairwise_metric: PairwiseMetric,
    n: int | None,
) -> ScoresFrame:
    """
    Precompute and store pairwise scores for all documents in a dataframe with one row
    per query document. The scores are stored as a sorted list of `DocumentScore`
    objects.
    """
    tqdm.pandas()

    if n is None:
        n = len(input_df)

    return (
        pd.DataFrame(data=candidate_d3_document_ids, columns=["document_id"])
        .assign(
            # the new scoped dataframe inside the first lambda function must have a
            # different name than the input dataframe since the input dataframe is
            # passed to the `find_top_n_matches_single_document` function and NOT the
            # scoped dataframe!
            scores=lambda new_df: new_df["document_id"].progress_apply(
                lambda query_d3_document_id: find_top_n_matches_single_document(
                    input_df, candidate_d3_document_ids, query_d3_document_id, pairwise_metric, n
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
) -> ScoresFrame:
    """
    Precompute and store pairwise co-citation scores for all documents in a dataframe
    with one row per query document.

    The input dataframe is the full documents data.
    """
    candidate_d3_document_ids = df["document_id"].tolist()
    return precompute_pairwise_scores(df, candidate_d3_document_ids, CountCommonCitations(), n)


def precompute_co_references(
    df: pd.DataFrame,
    n: int | None = None,
) -> ScoresFrame:
    """
    Precmopute and store pairwise co-reference scores for all documents in a dataframe
    with one row per query document.

    The input dataframe is the full documents data.
    """
    candidate_d3_document_ids = df["document_id"].tolist()
    return precompute_pairwise_scores(df, candidate_d3_document_ids, CountCommonReferences(), n)


def precompute_cosine_similarities(
    df: pd.DataFrame,
    n: int | None = None,
) -> ScoresFrame:
    """
    Precompute and store pairwise cosine similarity scores for all documents in a
    dataframe with one row per query document.

    The input dataframe has a single column named `embedding` and the index is named
    `document_id`.
    """
    candidate_d3_document_ids = df.index.tolist()
    return precompute_pairwise_scores(df, candidate_d3_document_ids, CosineSimilarity(), n)
