from collections.abc import Sequence

import polars as pl

from readnext.config import ResultsPaths
from readnext.utils.io import read_df_from_parquet


def average(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        mean_avg_precision_c_to_l=pl.col("avg_precision_c_to_l").mean(),
        mean_avg_precision_c_to_l_cand=pl.col("avg_precision_c_to_l_cand").mean(),
        mean_avg_precision_l_to_c=pl.col("avg_precision_l_to_c").mean(),
        mean_avg_precision_l_to_c_cand=pl.col("avg_precision_l_to_c_cand").mean(),
        mean_num_unique_labels_c_to_l=pl.col("num_unique_labels_c_to_l").mean(),
        mean_num_unique_labels_c_to_l_cand=pl.col("num_unique_labels_c_to_l_cand").mean(),
        mean_num_unique_labels_l_to_c=pl.col("num_unique_labels_l_to_c").mean(),
        mean_num_unique_labels_l_to_c_cand=pl.col("num_unique_labels_l_to_c_cand").mean(),
    )


def average_by_group(
    evaluation_frame: pl.DataFrame, grouping_columns: Sequence[str]
) -> pl.DataFrame:
    return evaluation_frame.groupby(grouping_columns, maintain_order=True).agg(
        mean_avg_precision_c_to_l=pl.col("avg_precision_c_to_l").mean(),
        mean_avg_precision_c_to_l_cand=pl.col("avg_precision_c_to_l_cand").mean(),
        mean_avg_precision_l_to_c=pl.col("avg_precision_l_to_c").mean(),
        mean_avg_precision_l_to_c_cand=pl.col("avg_precision_l_to_c_cand").mean(),
        mean_num_unique_labels_c_to_l=pl.col("num_unique_labels_c_to_l").mean(),
        mean_num_unique_labels_c_to_l_cand=pl.col("num_unique_labels_c_to_l_cand").mean(),
        mean_num_unique_labels_l_to_c=pl.col("num_unique_labels_l_to_c").mean(),
        mean_num_unique_labels_l_to_c_cand=pl.col("num_unique_labels_l_to_c_cand").mean(),
    )


def compare_language_models(evaluation_frame: pl.DataFrame) -> pl.DataFrame:
    return (
        average_by_group(evaluation_frame, ["language_model"])
        .select(["language_model", "mean_avg_precision_c_to_l", "mean_avg_precision_l_to_c_cand"])
        .sort(by="mean_avg_precision_c_to_l", descending=True)
    )


def compare_feature_weights(evaluation_frame: pl.DataFrame) -> pl.DataFrame:
    return (
        average_by_group(evaluation_frame, ["feature_weights"])
        .select(["feature_weights", "mean_avg_precision_l_to_c", "mean_avg_precision_c_to_l_cand"])
        .sort(by="mean_avg_precision_l_to_c", descending=True)
    )


def compare_language_model_feature_weight_combinations(
    evaluation_frame: pl.DataFrame,
) -> pl.DataFrame:
    return (
        average_by_group(evaluation_frame, ["language_model", "feature_weights"])
        .select(
            [
                "language_model",
                "feature_weights",
                "mean_avg_precision_c_to_l",
                "mean_avg_precision_l_to_c",
            ]
        )
        .sort("mean_avg_precision_c_to_l", descending=True)
    )


def compare_hybridization_strategies(evaluation_frame: pl.DataFrame) -> pl.DataFrame:
    return average(evaluation_frame).select(
        [
            "mean_avg_precision_c_to_l",
            "mean_avg_precision_c_to_l_cand",
            "mean_avg_precision_l_to_c",
            "mean_avg_precision_l_to_c_cand",
        ]
    )


def compare_diversity(evaluation_frame: pl.DataFrame) -> pl.DataFrame:
    """
    Compare number of unique labels for both hybrid recommender orders. Since the number
    keeps the same when moving from the candidate list to the final recommendations, it
    is sufficient to consider the candidate lists.
    """
    return average(evaluation_frame).select(
        ["mean_num_unique_labels_c_to_l_cand", "mean_num_unique_labels_l_to_c_cand"]
    )


def main() -> None:
    evaluation_frame = read_df_from_parquet(ResultsPaths.evaluation.evaluation_frame_parquet)

    # TODO: Why is the mean average precision for the candidate lists constant? Due to
    # small sample size or logic error?
    compare_language_models(evaluation_frame)

    compare_feature_weights(evaluation_frame)

    compare_language_model_feature_weight_combinations(evaluation_frame)

    compare_hybridization_strategies(evaluation_frame)

    compare_diversity(evaluation_frame)

    # TODO: Give the files more descriptive names.


if __name__ == "__main__":
    main()
