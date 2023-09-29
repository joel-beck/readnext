"""
Step 2: Metrics Computation

Evaluates all combinations of test documents, language models and the baseline feature
weights plus the selected feature weights candidates from the previous step. Computes
the average precision and counts the unique labels for both hybrid recommender orders
and candidate recommendation lists. Stores the resulting evaluation dataframe in a
parquet file.
"""

import polars as pl

from readnext.config import ResultsPaths
from readnext.data.data_split import DataSplit, load_data_split
from readnext.scripts.evaluation.s1_search_feature_weights import (
    add_scoring_columns,
    construct_combinations_frame,
    string_to_list,
)
from readnext.utils.io import read_df_from_parquet, write_df_to_parquet


def select_top_n_feature_weight_rows(df: pl.DataFrame, n: int) -> pl.DataFrame:
    return df.sort(by="avg_precision_hybrid", descending=True).head(n)


def extract_feature_weights(df: pl.DataFrame) -> list[str]:
    return (
        df.groupby("feature_weights", maintain_order=True)
        .agg(mean_avg_precision_hybrid=pl.col("avg_precision_hybrid").mean())["feature_weights"]
        .to_list()
    )


def get_feature_weight_candidates(
    feature_weights_candidates_frame: pl.DataFrame, num_best_feature_weights: int
) -> list[list[int]]:
    """
    First computes the marginal mean average precision for all feature weights
    aggregating over all documents and language models. Then selects the top n feature
    weights and converts them from strings to lists.
    """
    feature_weights_candidates_strings = extract_feature_weights(feature_weights_candidates_frame)[
        :num_best_feature_weights
    ]

    return [
        string_to_list(feature_weights) for feature_weights in feature_weights_candidates_strings
    ]


def main() -> None:
    query_data_split = DataSplit.TEST
    query_documents_frame = load_data_split(query_data_split)

    num_best_feature_weights = 10

    language_model_candidates = [
        "TFIDF",
        "BM25",
        "WORD2VEC",
        "GLOVE",
        "FASTTEXT",
        "BERT",
        "SCIBERT",
        "LONGFORMER",
    ]

    baseline_feature_weights = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    feature_weights_candidates_frame = read_df_from_parquet(
        ResultsPaths.evaluation.feature_weights_candidates_frame_parquet
    )
    feature_weight_candidates = get_feature_weight_candidates(
        feature_weights_candidates_frame, num_best_feature_weights
    )

    evaluated_feature_weights = baseline_feature_weights + feature_weight_candidates

    evaluation_frame = construct_combinations_frame(
        query_documents_frame, language_model_candidates, evaluated_feature_weights
    ).pipe(add_scoring_columns)

    write_df_to_parquet(evaluation_frame, ResultsPaths.evaluation.evaluation_frame_parquet)


if __name__ == "__main__":
    main()
