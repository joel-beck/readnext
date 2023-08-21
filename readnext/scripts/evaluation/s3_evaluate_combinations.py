"""
Step 3: Final Evaluation

Numerical and graphical analysis of the evaluation results.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from readnext.config import ResultsPaths
from readnext.utils.io import read_df_from_parquet

sns.set_theme(style="whitegrid")


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
        .select(["language_model", "mean_avg_precision_l_to_c_cand", "mean_avg_precision_c_to_l"])
        .sort(by="mean_avg_precision_l_to_c_cand", descending=True)
    )


def plot_language_models(evaluation_frame: pl.DataFrame) -> None:
    plot_df = (
        compare_language_models(evaluation_frame)
        .sort(by="mean_avg_precision_l_to_c_cand", descending=True)
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="mean_avg_precision_l_to_c_cand", y="language_model", data=plot_df, ax=ax)
    ax.set(
        xlabel="Mean Average Precision",
        ylabel="Language Model",
        title="Marginal Mean Average Precision of Language Models",
    )
    plt.show()


def compare_feature_weights(evaluation_frame: pl.DataFrame) -> pl.DataFrame:
    return (
        average_by_group(evaluation_frame, ["feature_weights"])
        .select(["feature_weights", "mean_avg_precision_c_to_l_cand", "mean_avg_precision_l_to_c"])
        .sort(by="mean_avg_precision_c_to_l_cand", descending=True)
    )


def construct_feature_weights_labels(feature_weights_string: str) -> str:
    return "[" + ", ".join(feature_weights_string.split(", ")) + "]"


def plot_feature_weights(evaluation_frame: pl.DataFrame) -> None:
    plot_df = (
        compare_feature_weights(evaluation_frame)
        .sort(by="mean_avg_precision_c_to_l_cand", descending=True)
        .with_columns(
            feature_weight_labels=pl.col("feature_weights").apply(construct_feature_weights_labels)
        )
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="mean_avg_precision_c_to_l_cand", y="feature_weight_labels", data=plot_df, ax=ax)
    ax.set(
        xlabel="Mean Average Precision",
        ylabel="Feature Weights",
        title="Marginal Mean Average Precision of Feature Weights",
    )
    plt.show()


def compare_language_models_feature_weights(
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


def plot_language_models_feature_weights_barplot(
    evaluation_frame: pl.DataFrame,
) -> None:
    plot_df = (
        compare_language_models_feature_weights(evaluation_frame)
        .sort(by="mean_avg_precision_c_to_l", descending=True)
        .with_columns(
            feature_weight_labels=pl.col("feature_weights").apply(construct_feature_weights_labels)
        )
        .to_pandas()
    )

    g: sns.FacetGrid = sns.catplot(
        data=plot_df,
        kind="bar",
        x="mean_avg_precision_c_to_l",
        y="language_model",
        col="feature_weight_labels",
        col_wrap=4,
        height=16,
        aspect=1,
        sharex=False,
        sharey=True,
    )
    g.set_axis_labels(x_var="", y_var="")
    g.set_titles("{col_name}", size=30)
    g.tick_params(labelsize=30)
    # sns.set(font_scale=3)

    plt.suptitle("Mean Average Precision of Feature Weights by Language Model", y=1.02, size=40)
    plt.show()


def plot_language_models_feature_weights_heatmap(
    evaluation_frame: pl.DataFrame,
) -> None:
    column_order = [
        "TFIDF",
        "BM25",
        "WORD2VEC",
        "GLOVE",
        "FASTTEXT",
        "BERT",
        "SCIBERT",
        "LONGFORMER",
    ]

    index_order = [
        "[1,0,0,0,0]",
        "[0,1,0,0,0]",
        "[0,0,1,0,0]",
        "[0,0,0,1,0]",
        "[0,0,0,0,1]",
        "[1,1,1,1,1]",
        "[0,3,17,96,34]",
        "[2,7,12,15,72]",
        "[2,13,18,72,66]",
        "[9,9,14,9,87]",
        "[9,12,13,68,95]",
        "[9,19,20,67,1]",
        "[10,13,19,65,14]",
        "[16,15,5,84,10]",
        "[17,2,18,54,4]",
        "[18,10,6,83,63]",
    ]

    plot_df = (
        compare_language_models_feature_weights(evaluation_frame)
        .with_columns(
            feature_weight_labels=pl.col("feature_weights").apply(construct_feature_weights_labels)
        )
        .to_pandas()
        .pivot_table(
            index="feature_weight_labels",
            columns="language_model",
            values="mean_avg_precision_c_to_l",
            # values="mean_avg_precision_l_to_c",
            aggfunc="mean",
        )
        .reindex(index_order)
        .loc[:, column_order]
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(plot_df, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5, ax=ax)

    ax.set_title("MAP (Citation to Language) of Feature Weights by Language Model", size=16, pad=20)
    ax.set(xlabel="", ylabel="")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.show()


def compare_hybridization_strategies(evaluation_frame: pl.DataFrame) -> pl.DataFrame:
    return average(evaluation_frame).select(
        [
            "mean_avg_precision_c_to_l",
            "mean_avg_precision_c_to_l_cand",
            "mean_avg_precision_l_to_c",
            "mean_avg_precision_l_to_c_cand",
        ]
    )


def plot_hybridization_strategies_barplot(evaluation_frame: pl.DataFrame) -> None:
    label_mapping = {
        "mean_avg_precision_c_to_l": "Citation to Language",
        "mean_avg_precision_c_to_l_cand": "Citation to Language Candidates",
        "mean_avg_precision_l_to_c": "Language to Citation",
        "mean_avg_precision_l_to_c_cand": "Language to Citation Candidates",
    }

    plot_df = (
        compare_hybridization_strategies(evaluation_frame)
        .melt()
        .with_columns(variable=pl.col("variable").map_dict(label_mapping))
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="value", y="variable", data=plot_df, ax=ax)
    ax.set(
        xlabel="Mean Average Precision",
        ylabel="Hybridization Strategy",
        title="Mean Average Precision of Hybridization Strategies",
    )
    plt.show()


def plot_hybridization_strategies_boxplot(evaluation_frame: pl.DataFrame) -> None:
    cols = [
        "avg_precision_c_to_l",
        "avg_precision_c_to_l_cand",
        "avg_precision_l_to_c",
        "avg_precision_l_to_c_cand",
    ]
    strategy_names = {
        "avg_precision_c_to_l": "Citation to Language",
        "avg_precision_c_to_l_cand": "Citation to Language Candidates",
        "avg_precision_l_to_c": "Language to Citation",
        "avg_precision_l_to_c_cand": "Language to Citation Candidates",
    }

    boxplot_data = (
        evaluation_frame.select(cols)
        .melt()
        .with_columns(variable=pl.col("variable").map_dict(strategy_names))
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(y="variable", x="value", data=boxplot_data, palette="pastel", ax=ax)
    ax.set(
        xlabel="Average Precision",
        ylabel="",
        title="Distribution of Average Precision by Hybridization Strategy",
    )

    fig.tight_layout()
    plt.show()


def compare_hybridization_strategies_by_language_model(
    evaluation_frame: pl.DataFrame,
) -> pl.DataFrame:
    return average_by_group(evaluation_frame, ["language_model"]).select(
        [
            "language_model",
            "mean_avg_precision_c_to_l",
            "mean_avg_precision_c_to_l_cand",
            "mean_avg_precision_l_to_c",
            "mean_avg_precision_l_to_c_cand",
        ]
    )


def plot_hybridization_strategies_by_language_model_barplot(
    evaluation_frame: pl.DataFrame,
) -> None:
    label_mapping = {
        "mean_avg_precision_c_to_l": "Citation to Language",
        "mean_avg_precision_c_to_l_cand": "Citation to Language Candidates",
        "mean_avg_precision_l_to_c": "Language to Citation",
        "mean_avg_precision_l_to_c_cand": "Language to Citation Candidates",
    }

    plot_df = (
        compare_hybridization_strategies_by_language_model(evaluation_frame)
        .melt(id_vars=["language_model"])
        .with_columns(variable=pl.col("variable").map_dict(label_mapping))
        .to_pandas()
    )

    g: sns.FacetGrid = sns.catplot(
        data=plot_df,
        kind="bar",
        x="value",
        y="variable",
        col="language_model",
        col_wrap=3,
        height=12,
        aspect=1,
        sharex=True,
        sharey=True,
    )

    g.set_axis_labels(x_var="", y_var="")
    g.set_titles("{col_name}", size=30)
    g.tick_params(labelsize=30)

    plt.suptitle("MAP of Hybridization Strategies by Language Model", y=1.02, size=40)
    plt.subplots_adjust(top=0.95)
    plt.tight_layout()

    plt.show()


def plot_hybridization_strategies_by_language_model_stripplot(
    evaluation_frame: pl.DataFrame,
) -> None:
    label_mapping = {
        "mean_avg_precision_c_to_l": "Citation to Language",
        "mean_avg_precision_c_to_l_cand": "Citation to Language Candidates",
        "mean_avg_precision_l_to_c": "Language to Citation",
        "mean_avg_precision_l_to_c_cand": "Language to Citation Candidates",
    }

    plot_df = (
        compare_hybridization_strategies_by_language_model(evaluation_frame)
        .melt(id_vars=["language_model"])
        .with_columns(variable=pl.col("variable").map_dict(label_mapping))
        .to_pandas()
    )

    g: sns.FacetGrid = sns.catplot(
        data=plot_df,
        kind="strip",
        x="value",
        y="variable",
        hue="variable",
        col="language_model",
        col_wrap=3,
        size=30,
        height=12,
        aspect=1,
        sharex=True,
        sharey=True,
        legend=False,
    )

    g.set_axis_labels(x_var="", y_var="")
    g.set_titles("{col_name}", size=30)
    g.tick_params(labelsize=30)

    plt.suptitle("MAP of Hybridization Strategies by Language Model", y=1.02, size=40)
    plt.subplots_adjust(top=0.95)
    plt.tight_layout()

    plt.show()


def compare_diversity(evaluation_frame: pl.DataFrame) -> pl.DataFrame:
    """
    Compare number of unique labels for both hybrid recommender orders. Since the number
    keeps the same when moving from the candidate list to the final recommendations, it
    is sufficient to consider the candidate lists.
    """
    return average(evaluation_frame).select(
        ["mean_num_unique_labels_c_to_l_cand", "mean_num_unique_labels_l_to_c_cand"]
    )


def plot_diversities_barplot(evaluation_frame: pl.DataFrame) -> None:
    label_mapping = {
        "mean_num_unique_labels_c_to_l_cand": "Citation to Language",
        "mean_num_unique_labels_l_to_c_cand": "Language to Citation",
    }

    plot_df = (
        compare_diversity(evaluation_frame)
        .melt()
        .with_columns(variable=pl.col("variable").map_dict(label_mapping))
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="value", y="variable", data=plot_df, ax=ax)
    ax.set(
        xlabel="Number of Unique Labels",
        ylabel="Hybridization Strategy",
        title="Mean Number of Unique Labels of Hybridization Strategies",
    )
    plt.show()


def plot_diversities_boxplot(evaluation_frame: pl.DataFrame) -> None:
    label_mapping = {
        "num_unique_labels_c_to_l_cand": "Citation to Language",
        "num_unique_labels_l_to_c_cand": "Language to Citation",
    }

    plot_df = (
        evaluation_frame.select(["num_unique_labels_c_to_l_cand", "num_unique_labels_l_to_c_cand"])
        .melt()
        .with_columns(variable=pl.col("variable").map_dict(label_mapping))
        .to_pandas()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="value", y="variable", data=plot_df, ax=ax)
    ax.set(
        xlabel="Number of Unique Labels",
        ylabel="",
        title="Distribution of Unique Labels by Hybridization Strategy",
    )
    plt.show()


def compare_diversity_by_language_model(evaluation_frame: pl.DataFrame) -> pl.DataFrame:
    return average_by_group(evaluation_frame, ["language_model"]).select(
        [
            "language_model",
            "mean_num_unique_labels_c_to_l_cand",
            "mean_num_unique_labels_l_to_c_cand",
        ]
    )


def plot_diversity_by_language_model(evaluation_frame: pl.DataFrame) -> None:
    label_mapping = {
        "mean_num_unique_labels_c_to_l_cand": "Citation to Language",
        "mean_num_unique_labels_l_to_c_cand": "Language to Citation",
    }

    plot_df = (
        compare_diversity_by_language_model(evaluation_frame)
        .melt(id_vars=["language_model"])
        .with_columns(variable=pl.col("variable").map_dict(label_mapping))
        .to_pandas()
    )

    g: sns.FacetGrid = sns.catplot(
        data=plot_df,
        kind="bar",
        x="value",
        y="variable",
        col="language_model",
        col_wrap=3,
        height=12,
        aspect=1,
        sharex=True,
        sharey=True,
    )

    g.set_axis_labels(x_var="", y_var="")
    g.set_titles("{col_name}", size=30)
    g.tick_params(labelsize=30)

    plt.suptitle(
        "Mean Number of Unique Labels of Hybridization Strategies by Language Model",
        y=1.02,
        size=40,
    )
    plt.subplots_adjust(top=0.95)
    plt.tight_layout()

    plt.show()


def main() -> None:
    evaluation_frame = read_df_from_parquet(ResultsPaths.evaluation.evaluation_frame_parquet)

    compare_language_models(evaluation_frame)
    plot_language_models(evaluation_frame)

    compare_feature_weights(evaluation_frame)
    plot_feature_weights(evaluation_frame)

    compare_language_models_feature_weights(evaluation_frame)
    # plot_language_model_feature_weight_combinations_comparison_barplot(evaluation_frame)
    plot_language_models_feature_weights_heatmap(evaluation_frame)

    compare_hybridization_strategies(evaluation_frame)
    # plot_hybridization_strategies_comparison_barplot(evaluation_frame)
    plot_hybridization_strategies_boxplot(evaluation_frame)

    compare_hybridization_strategies_by_language_model(evaluation_frame)
    # plot_hybridization_strategies_by_language_model_barplot(evaluation_frame)
    plot_hybridization_strategies_by_language_model_stripplot(evaluation_frame)

    compare_diversity(evaluation_frame)
    # plot_diversities_barplot(evaluation_frame)
    plot_diversities_boxplot(evaluation_frame)

    # NOTE: No significant variation between language models, do not include in thesis
    # compare_diversity_by_language_model(evaluation_frame)
    # plot_diversity_by_language_model(evaluation_frame)


if __name__ == "__main__":
    main()
