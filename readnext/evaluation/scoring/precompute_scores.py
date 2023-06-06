import polars as pl
from tqdm import tqdm

from readnext.config import MagicNumbers
from readnext.evaluation.metrics import (
    CosineSimilarity,
    CountCommonCitations,
    CountCommonReferences,
    PairwiseMetric,
)
from readnext.utils import DocumentsFrame, ScoresFrame, tqdm_progress_bar_wrapper


def generate_id_combinations_frame(documents_frame: DocumentsFrame) -> pl.DataFrame:
    """
    Create a dataframe with two columns `query_d3_document_id` and
    `candidate_d3_document_id` with all combinations of the documents `d3_document_id`
    values.
    """
    query_frame = pl.DataFrame(dict(query_d3_document_id=documents_frame["d3_document_id"]))
    candidate_frame = pl.DataFrame(dict(candidate_d3_document_id=documents_frame["d3_document_id"]))

    return query_frame.join(candidate_frame, how="cross")


def remove_matching_id_rows(id_combinations_frame: pl.DataFrame) -> pl.DataFrame:
    """
    Remove all rows from a dataframe where the `query_d3_document_id` and
    `candidate_d3_document_id` values are identical. Pairwise scores should only be
    computed for two different documents.
    """
    return id_combinations_frame.filter(
        pl.col("query_d3_document_id") != pl.col("candidate_d3_document_id")
    )


def pairwise_scores_from_columns(
    documents_frame: DocumentsFrame,
    id_combinations_frame: pl.DataFrame,
    pairwise_metric: PairwiseMetric,
) -> ScoresFrame:
    """
    Add third `score` that is computed from the `query_d3_document_id` and
    `candidate_d3_document_id` columns by means of a given `pairwise_metric`.
    """
    with tqdm(total=len(id_combinations_frame)) as progress_bar:
        progress_bar.set_description(f"Precomputing {pairwise_metric.__class__.__name__} Scores...")

        return id_combinations_frame.with_columns(
            score=pl.struct(["query_d3_document_id", "candidate_d3_document_id"]).apply(
                tqdm_progress_bar_wrapper(
                    progress_bar,
                    lambda struct: pairwise_metric.from_df(
                        documents_frame,
                        struct["query_d3_document_id"],
                        struct["candidate_d3_document_id"],
                    ),
                )
            )
        )


def precompute_pairwise_scores(
    documents_frame: DocumentsFrame, pairwise_metric: PairwiseMetric, n: int
) -> ScoresFrame:
    """
    Precompute and store pairwise scores for all documents in `ScoresFrame` with three
    columns named `query_d3_document_id`, `candidate_d3_document_id`, and `score`.

    Threshold the number of scores per query document to a given integer value `n` such
    that the output dataframe size grows linearly with the number of documents.
    """
    id_combinations_frame = generate_id_combinations_frame(documents_frame).pipe(
        remove_matching_id_rows
    )

    scores_frame = pairwise_scores_from_columns(
        documents_frame, id_combinations_frame, pairwise_metric
    ).sort(by=["query_d3_document_id", "score"], descending=[False, True])

    # slice top n scores per query document
    return scores_frame.groupby("query_d3_document_id").head(n)


# Set value for `n` higher for co-citation analysis and bibliographic coupling since
# they are features for the weighted linear model. The higher the value for `n`, the
# more observations the weighted model is able to use.
def precompute_co_citations(
    documents_frame: DocumentsFrame,
    n: int = MagicNumbers.scoring_limit,
) -> ScoresFrame:
    """
    Precompute and store pairwise co-citation scores for all documents in a dataframe
    with one row per query document.

    The input dataframe is the full documents data.
    """
    return precompute_pairwise_scores(documents_frame, CountCommonCitations(), n)


def precompute_co_references(
    documents_frame: DocumentsFrame,
    n: int = MagicNumbers.scoring_limit,
) -> ScoresFrame:
    """
    Precompute and store pairwise co-reference scores for all documents in a dataframe
    with one row per query document.

    The input dataframe is the full documents data.
    """
    return precompute_pairwise_scores(documents_frame, CountCommonReferences(), n)


def precompute_cosine_similarities(
    documents_frame: DocumentsFrame,
    n: int = MagicNumbers.scoring_limit,
) -> ScoresFrame:
    """
    Precompute and store pairwise cosine similarity scores for all documents in a
    dataframe with one row per query document.

    The input dataframe has two columns named `d3_document_id` and `embedding`.
    """
    return precompute_pairwise_scores(documents_frame, CosineSimilarity(), n)
