import polars as pl
from tqdm import tqdm

from readnext.evaluation.metrics import (
    CosineSimilarity,
    CountCommonCitations,
    CountCommonReferences,
    PairwiseMetric,
)
from readnext.modeling import DocumentInfo, DocumentScore
from readnext.utils import ScoresFrame, sort_document_scores


def find_top_n_matches_single_document(
    input_df: pl.DataFrame, query_d3_document_id: int, pairwise_metric: PairwiseMetric, n: int
) -> list[DocumentScore]:
    """
    Find the n documents with the highest pairwise score for a single query document.
    """
    document_scores = []

    for d3_document_id in input_df["document_id"]:
        if d3_document_id == query_d3_document_id:
            continue

        # full documents data (input of `precompute_co_citations` and
        # `precompute_co_references`) has index of type `pl.Index`, while the input of
        # `precompute_cosine_similarities` does not require an extra `.item()` call
        d3_document_id = (
            d3_document_id if isinstance(d3_document_id, int) else d3_document_id.item()
        )

        document_info = DocumentInfo(d3_document_id=d3_document_id)
        score = pairwise_metric.from_df(input_df, query_d3_document_id, d3_document_id)
        document_scores.append(DocumentScore(document_info=document_info, score=score))

    return sort_document_scores(document_scores)[:n]


def precompute_pairwise_scores(
    input_df: pl.DataFrame, pairwise_metric: PairwiseMetric, n: int | None
) -> ScoresFrame:
    """
    Precompute and store pairwise scores for all documents in a dataframe with one row
    per query document. The scores are stored as a sorted list of `DocumentScore`
    objects.
    """
    tqdm.pandas()

    if n is None:
        n = len(input_df)

    return pl.DataFrame(data={"document_id": input_df["document_id"]}).with_columns(
        # the new scoped dataframe inside the first lambda function must have a
        # different name than the input dataframe since the input dataframe is
        # passed to the `find_top_n_matches_single_document` function and NOT the
        # scoped dataframe!
        scores=pl.col("document_id").apply(
            lambda query_d3_document_id: find_top_n_matches_single_document(
                input_df, query_d3_document_id, pairwise_metric, n
            )
        )
    )


# Set value for `n` higher for co-citation analysis and bibliographic coupling since
# they are features for the weighted linear model. The higher the value for `n`, the
# more observations the weighted model is able to use.
def precompute_co_citations(
    df: pl.DataFrame,
    n: int | None = None,
) -> ScoresFrame:
    """
    Precompute and store pairwise co-citation scores for all documents in a dataframe
    with one row per query document.

    The input dataframe is the full documents data.
    """
    return precompute_pairwise_scores(df, CountCommonCitations(), n)


def precompute_co_references(
    df: pl.DataFrame,
    n: int | None = None,
) -> ScoresFrame:
    """
    Precmopute and store pairwise co-reference scores for all documents in a dataframe
    with one row per query document.

    The input dataframe is the full documents data.
    """
    return precompute_pairwise_scores(df, CountCommonReferences(), n)


def precompute_cosine_similarities(
    df: pl.DataFrame,
    n: int | None = None,
) -> ScoresFrame:
    """
    Precompute and store pairwise cosine similarity scores for all documents in a
    dataframe with one row per query document.

    The input dataframe has a single column named `embedding` and the index is named
    `document_id`.
    """
    return precompute_pairwise_scores(df, CosineSimilarity(), n)
