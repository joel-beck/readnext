import itertools
from collections.abc import Sequence

import polars as pl
from tqdm import tqdm

from readnext.evaluation.metrics import (
    CosineSimilarity,
    CountCommonCitations,
    CountCommonReferences,
    PairwiseMetric,
)
from readnext.utils import ScoresFrame, tqdm_progress_bar_wrapper

# def find_top_n_matches_single_document(
#     input_df: pl.DataFrame, query_d3_document_id: int, pairwise_metric: PairwiseMetric, n: int
# ) -> pl.DataFrame:
#     """
#     Find the n documents with the highest pairwise score for a single query document.
#     """
#     document_score_frames = []

#     for d3_document_id in input_df["d3_document_id"]:
#         if d3_document_id == query_d3_document_id:
#             continue

#         score = pairwise_metric.from_df(input_df, query_d3_document_id, d3_document_id)
#         query_frame = pl.DataFrame(
#             {
#                 "query_d3_document_id": query_d3_document_id,
#                 "candidate_d3_document_id": d3_document_id,
#                 "score": score,
#             }
#         )
#         document_score_frames.append(query_frame)

#     return pl.concat(document_score_frames).sort("score", descending=True).head(n)


def column_combinations_to_frame(
    input_df: pl.DataFrame, combinations_column: str, output_columns: Sequence[str]
) -> pl.DataFrame:
    """
    Create a dataframe with two columns that represent all combinations of a dataframe
    column with itself.
    """
    pairwise_combinations: list[tuple[int, int]] = list(
        itertools.product(
            input_df[combinations_column],
            repeat=2,
        )
    )  # type: ignore

    # remove combinations of values with themselves
    non_matching_combinations = [
        combination_tuple
        for combination_tuple in pairwise_combinations
        if combination_tuple[0] != combination_tuple[1]
    ]

    return pl.from_records(non_matching_combinations, schema=output_columns)


def pairwise_scores_from_columns(
    input_df: pl.DataFrame, combinations_frame: pl.DataFrame, pairwise_metric: PairwiseMetric
) -> pl.DataFrame:
    """
    Add third `score` that is computed from the `query_d3_document_id` and
    `candidate_d3_document_id` columns by means of a given `pairwise_metric`.
    """
    with tqdm(total=len(combinations_frame)) as progress_bar:
        progress_bar.set_description(f"Precomputing {pairwise_metric.__class__.__name__} Scores...")

        return combinations_frame.with_columns(
            score=pl.struct(["query_d3_document_id", "candidate_d3_document_id"]).apply(
                tqdm_progress_bar_wrapper(
                    progress_bar,
                    lambda df: pairwise_metric.from_df(
                        input_df,
                        df["query_d3_document_id"],
                        df["candidate_d3_document_id"],
                    ),
                )
            )
        )


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

    combinations_frame = column_combinations_to_frame(
        input_df,
        combinations_column="d3_document_id",
        output_columns=["query_d3_document_id", "candidate_d3_document_id"],
    )

    combinations_scores_frame = pairwise_scores_from_columns(
        input_df, combinations_frame, pairwise_metric
    ).sort(by=["query_d3_document_id", "score"], descending=[False, True])

    # slice top n scores per query document
    return combinations_scores_frame.groupby("query_d3_document_id").head(n)


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

    The input dataframe has two columns named `d3_document_id` and `embedding`.
    """
    return precompute_pairwise_scores(df, CosineSimilarity(), n)
