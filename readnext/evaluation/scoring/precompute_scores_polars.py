"""
Precompute co-citation analysis scores, bibliographic coupling scores and cosine
similarity scores.
"""

import polars as pl

from readnext.config import MagicNumbers
from readnext.utils import rich_progress_bar


def construct_combinations_frame(df: pl.LazyFrame, colname: str) -> pl.LazyFrame:
    query_frame = df.select(["d3_document_id", colname]).rename(
        {"d3_document_id": "query_d3_document_id", colname: f"query_{colname}"}
    )
    candidate_frame = df.select(["d3_document_id", colname]).rename(
        {
            "d3_document_id": "candidate_d3_document_id",
            colname: f"candidate_{colname}",
        }
    )
    return query_frame.join(candidate_frame, how="cross")


def remove_matching_ids(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.filter(pl.col("query_d3_document_id") != pl.col("candidate_d3_document_id"))


def add_concatenated_list_values_column(
    df: pl.LazyFrame, query_colname: str, candidate_colname: str
) -> pl.LazyFrame:
    return df.with_columns(
        concatenated=pl.col(query_colname).list.concat(pl.col(candidate_colname))
    )


def add_unique_list_values_column(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(unique=pl.col("concatenated").list.unique())


def add_score_column(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        score=pl.col("concatenated").list.lengths() - pl.col("unique").list.lengths().cast(pl.Int64)
    )


def count_common_values(
    df: pl.LazyFrame, query_colname: str, candidate_colname: str
) -> pl.LazyFrame:
    return (
        df.pipe(add_concatenated_list_values_column, query_colname, candidate_colname)
        .pipe(add_unique_list_values_column)
        .pipe(add_score_column)
    )


def select_output_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.select(["query_d3_document_id", "candidate_d3_document_id", "score"])


def select_highest_scores(df: pl.LazyFrame, n: int) -> pl.LazyFrame:
    return (
        df.sort(by=["query_d3_document_id", "score"], descending=[False, True])
        .groupby("query_d3_document_id")
        .head(n)
    )


def precompute_common_values_slice(
    documents_frame_slice: pl.LazyFrame, colname: str, n: int
) -> pl.DataFrame:
    """
    Compute scores for a slice of the full dataframe.
    """
    query_colname = f"query_{colname}"
    candidate_colname = f"candidate_{colname}"

    combinations_frame = construct_combinations_frame(documents_frame_slice, colname).pipe(
        remove_matching_ids
    )

    return (
        combinations_frame.pipe(count_common_values, query_colname, candidate_colname)
        .pipe(select_output_columns)
        .pipe(select_highest_scores, n)
        .collect()
    )


def precompute_common_values_polars(
    documents_frame: pl.LazyFrame, colname: str, n: int, description: str
) -> pl.DataFrame:
    """
    Compute scores sequentially for slices of the full dataframe and stack the outputs vertically.
    """
    slice_size = 100
    num_rows = documents_frame.collect().height
    num_slices = num_rows // slice_size

    with rich_progress_bar() as progress_bar:
        return pl.concat(
            [
                precompute_common_values_slice(
                    documents_frame.slice(next_index, slice_size), colname, n
                )
                for next_index in progress_bar.track(
                    range(0, num_rows, slice_size),
                    total=num_slices,
                    description=description,
                )
            ]
        )


def precompute_co_citations_polars(
    documents_frame: pl.LazyFrame, n: int = MagicNumbers.scoring_limit
) -> pl.DataFrame:
    """
    Compute co-citation analysis scores sequentially for slices of the full dataframe
    and stack the outputs vertically.
    """
    return precompute_common_values_polars(
        documents_frame, "citations", n, description="Computing Co-Citation Analysis Scores..."
    )


def precompute_co_references_polars(
    documents_frame: pl.LazyFrame, n: int = MagicNumbers.scoring_limit
) -> pl.DataFrame:
    """
    Compute bibliographic coupling scores sequentially for slices of the full dataframe
    and stack the outputs vertically.
    """
    return precompute_common_values_polars(
        documents_frame, "references", n, description="Computing Bibliographic Coupling Scores..."
    )


def explode_embeddings(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.select(["query_embedding", "candidate_embedding"])
        .with_row_count()
        .explode(["query_embedding", "candidate_embedding"])
    )


def norm(expression: pl.Expr) -> pl.Expr:
    return expression.mul(expression).sum().sqrt()


def dot_product(expression_1: pl.Expr, expression_2: pl.Expr) -> pl.Expr:
    return expression_1.dot(expression_2)


def cosine_similarity(expression_1: pl.Expr, expression_2: pl.Expr) -> pl.Expr:
    return dot_product(expression_1, expression_2) / (norm(expression_1) * norm(expression_2))


def compute_cosine_similarities(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.groupby("row_nr", maintain_order=True)
        .agg(score=cosine_similarity(pl.col("query_embedding"), pl.col("candidate_embedding")))
        .drop("row_nr")
    )


def concatenate_ids_and_scores(
    combinations_frame: pl.DataFrame, scores: pl.DataFrame
) -> pl.DataFrame:
    return pl.concat(
        [
            combinations_frame.select(["query_d3_document_id", "candidate_d3_document_id"]),
            scores,
        ],
        how="horizontal",
    )


def precompute_cosine_similarities_slice(
    embeddings_frame_slice: pl.LazyFrame, n: int
) -> pl.LazyFrame:
    """
    Compute cosine similarities for a slice of the full dataframe.
    """
    combinations_frame = construct_combinations_frame(embeddings_frame_slice, "embedding").pipe(
        remove_matching_ids
    )

    scores = combinations_frame.pipe(explode_embeddings).pipe(compute_cosine_similarities)

    return (
        concatenate_ids_and_scores(combinations_frame.collect(), scores.collect())
        .lazy()
        .pipe(select_highest_scores, n)
    )


def precompute_cosine_similarities_polars(
    embeddings_frame: pl.LazyFrame, n: int = MagicNumbers.scoring_limit
) -> pl.DataFrame:
    """
    Compute cosine similarities sequentially for slices of the full dataframe and stack
    the outputs vertically.
    """
    slice_size = 100
    num_rows = embeddings_frame.collect().height
    num_slices = num_rows // slice_size

    with rich_progress_bar() as progress_bar:
        return pl.concat(
            [
                precompute_cosine_similarities_slice(
                    embeddings_frame.slice(next_index, slice_size), n
                ).collect()
                for next_index in progress_bar.track(
                    range(0, num_rows, slice_size),
                    total=num_slices,
                    description="Computing Cosine Similarities...",
                )
            ]
        )
