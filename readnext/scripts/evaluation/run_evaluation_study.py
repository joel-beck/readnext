import pickle
from collections.abc import Sequence

import polars as pl

from readnext.config import ResultsPaths


def compute_mean_average_precision(
    results_frame: pl.DataFrame, grouping_columns: Sequence[str]
) -> pl.DataFrame:
    return results_frame.groupby(grouping_columns).agg(
        mean_avg_precision_c_to_l=pl.col("avg_precision_c_to_l").mean(),
        mean_avg_precision_c_to_l_cand=pl.col("avg_precision_c_to_l_cand").mean(),
        mean_avg_precision_l_to_c=pl.col("avg_precision_l_to_c").mean(),
        mean_avg_precision_l_to_c_cand=pl.col("avg_precision_l_to_c_cand").mean(),
        mean_num_unique_labels_c_to_l=pl.col("num_unique_labels_c_to_l").mean(),
        mean_num_unique_labels_c_to_l_cand=pl.col("num_unique_labels_c_to_l_cand").mean(),
        mean_num_unique_labels_l_to_c=pl.col("num_unique_labels_l_to_c").mean(),
        mean_num_unique_labels_l_to_c_cand=pl.col("num_unique_labels_l_to_c_cand").mean(),
    )


def main() -> None:
    with ResultsPaths.evaluation.feature_weights_candidates_pkl.open("rb") as f:
        feature_weights_candidates = pickle.load(f)

        # compare language models
        # compute_mean_average_precision(results_frame, ["language_model"])

        # compare feature weights
        # compute_mean_average_precision(results_frame, ["feature_weights"])

        # compare hybridization strategies
        # compute_mean_average_precision(results_frame, ["semanticscholar_id"])

        # compare combinations of language models and feature weights
        # compute_mean_average_precision(results_frame, ["language_model", "feature_weights"])

        # TODO: Implement workaround since polars does not support grouping by list columns, see
        # https://github.com/pola-rs/polars/issues/4175
        # results_frame.groupby(
        #     "language_model",
        #     pl.col("feature_weights").list.eval(pl.element().cast(pl.Utf8)).list.join(","),
        # ).agg(
        #     pl.col("avg_precision_c_to_l").mean(),
        # )


if __name__ == "__main__":
    main()
