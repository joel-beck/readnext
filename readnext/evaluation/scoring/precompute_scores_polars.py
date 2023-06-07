"""
Precompute co-citation analysis scores, bibliographic coupling scores and cosine
similarity scores.
"""

import polars as pl

from readnext.config import MagicNumbers
from readnext.utils import ScoresFrame


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


def add_concatenated_list_values_column(df: pl.LazyFrame, colname: str) -> pl.LazyFrame:
    return df.with_columns(
        concatenated=pl.col(f"query_{colname}").list.concat(pl.col(f"candidate_{colname}"))
    )


def add_unique_list_values_column(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(unique=pl.col("concatenated").list.unique())


def add_score_column(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        score=pl.col("concatenated").list.lengths() - pl.col("unique").list.lengths().cast(pl.Int64)
    )


def select_output_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.select(["query_d3_document_id", "candidate_d3_document_id", "score"])


def select_highest_scores(df: pl.LazyFrame, n: int) -> pl.LazyFrame:
    return (
        df.sort(by=["query_d3_document_id", "score"], descending=[False, True])
        .groupby("query_d3_document_id")
        .head(n)
    )


def precompute_values_polars(
    df: pl.LazyFrame, colname: str, n: int = MagicNumbers.scoring_limit
) -> pl.DataFrame:
    combinations_frame = construct_combinations_frame(df, colname).pipe(remove_matching_ids)

    return (
        combinations_frame.pipe(add_concatenated_list_values_column, colname)
        .pipe(add_unique_list_values_column)
        .pipe(add_score_column)
        .pipe(select_output_columns)
        .pipe(select_highest_scores, n)
        .collect()
    )


def precompute_co_citations_polars(
    df: pl.LazyFrame, n: int = MagicNumbers.scoring_limit
) -> pl.DataFrame:
    return precompute_values_polars(df, "citations", n)


def precompute_co_references_polars(
    df: pl.LazyFrame, n: int = MagicNumbers.scoring_limit
) -> pl.DataFrame:
    return precompute_values_polars(df, "references", n)


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
        .agg(
            cosine_similarity(pl.col("query_embedding"), pl.col("candidate_embedding")).alias(
                "score"
            )
        )
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


def precompute_cosine_similarities_polars(
    embeddings_frame: pl.LazyFrame,
    n: int = MagicNumbers.scoring_limit,
) -> ScoresFrame:
    combinations_frame = construct_combinations_frame(embeddings_frame, "embedding").pipe(
        remove_matching_ids
    )

    scores = combinations_frame.pipe(explode_embeddings).pipe(compute_cosine_similarities)

    return (
        concatenate_ids_and_scores(combinations_frame.collect(), scores.collect())
        .lazy()
        .pipe(select_highest_scores, n)
        .collect()
    )
