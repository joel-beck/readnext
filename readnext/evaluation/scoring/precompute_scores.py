import polars as pl

from readnext.evaluation.metrics import (
    CosineSimilarity,
    CountCommonCitations,
    CountCommonReferences,
    PairwiseMetric,
)
from readnext.modeling import DocumentInfo, DocumentScore
from readnext.utils import ScoresFrame, setup_progress_bar, sort_document_scores


def find_top_n_matches_single_document(
    input_df: pl.DataFrame, query_d3_document_id: int, pairwise_metric: PairwiseMetric, n: int
) -> pl.DataFrame:
    """
    Find the n documents with the highest pairwise score for a single query document.
    """
    document_score_frames = []

    for d3_document_id in input_df["document_id"]:
        if d3_document_id == query_d3_document_id:
            continue

        document_info = DocumentInfo(d3_document_id=d3_document_id)
        score = pairwise_metric.from_df(input_df, query_d3_document_id, d3_document_id)
        document_score = DocumentScore(document_info=document_info, score=score)
        query_frame = pl.DataFrame(
            {
                "query_d3_document_id": d3_document_id,
                "candidate_d3_document_id": document_score.document_info.d3_document_id,
                "score": document_score.score,
            }
        )
        document_score_frames.append(query_frame)

    return pl.concat(document_score_frames).sort("score", descending=True).head(n)


# TODO: Make this function more efficient, try to use apply() on
# `input_df["document_id"]` instead of looping over the polars Series this is the major
# bottleneck  when precomputing co-citation scores, bibliographic coupling scores and
# cosine similarities
def precompute_pairwise_scores(
    input_df: pl.DataFrame, pairwise_metric: PairwiseMetric, n: int | None
) -> ScoresFrame:
    """
    Precompute and store pairwise scores for all documents in a dataframe with one row
    per query document. The scores are stored as a sorted list of `DocumentScore`
    objects.
    """
    if n is None:
        n = len(input_df)

    scores = []
    with setup_progress_bar() as progress_bar:
        for query_d3_document_id in progress_bar.track(
            input_df["document_id"],
            total=len(input_df["document_id"]),
            description="Computing Scores...",
        ):
            scores.append(
                find_top_n_matches_single_document(
                    input_df, query_d3_document_id, pairwise_metric, n
                )
            )

    return pl.DataFrame({"document_id": input_df["document_id"], "scores": scores})


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
    Precompute and store pairwise co-reference scores for all documents in a dataframe
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
