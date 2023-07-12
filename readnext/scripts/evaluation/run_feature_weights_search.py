"""
Computes Average Precision for all combinations of language models and feature weights.
The resulting scores are stored in a polars dataframe and saved to a parquet file.
"""

from collections.abc import Sequence

import polars as pl

from readnext import readnext
from readnext.config import DataPaths, ResultsPaths
from readnext.evaluation.metrics import AveragePrecision, CountUniqueLabels
from readnext.evaluation.scoring import FeatureWeights, FeatureWeightsRanges
from readnext.inference import Recommendations
from readnext.modeling.language_models import LanguageModelChoice
from readnext.utils.aliases import DocumentsFrame
from readnext.utils.io import read_df_from_parquet
from readnext.utils.progress_bar import rich_progress_bar
import pickle


def construct_combinations_frame(
    documents_frame: DocumentsFrame,
    language_model_candidates: Sequence[str],
    feature_weights_candidates: Sequence[Sequence[int]],
) -> pl.DataFrame:
    language_model_candidates_frame = pl.DataFrame(
        {
            "language_model": language_model_candidates,
        }
    )
    feature_weights_candidates_frame = pl.DataFrame(
        {
            "feature_weights": feature_weights_candidates,
        }
    )

    return (
        documents_frame.select("semanticscholar_id")
        .join(language_model_candidates_frame, how="cross")
        .join(feature_weights_candidates_frame, how="cross")
    )


def sample_input_combinations(
    combination_frame: pl.DataFrame, num_samples: int, seed: int
) -> pl.DataFrame:
    """
    Choose random documents from the full dataframe to conduct tests during development.
    """
    return combination_frame.sample(n=num_samples, with_replacement=False, seed=seed)


def retrieve_recommendations(
    semanticscholar_id: str, language_model: str, feature_weights: Sequence[int]
) -> Recommendations:
    result = readnext(
        semanticscholar_id=semanticscholar_id,
        language_model_choice=LanguageModelChoice(language_model),
        feature_weights=FeatureWeights.from_sequence(feature_weights),
        verbose=False,
    )
    return result.recommendations


def compute_average_precision_citation_to_language(
    recommendations: Recommendations,
) -> float:
    return AveragePrecision.from_df(recommendations.citation_to_language)


def compute_average_precision_citation_to_language_candidates(
    recommendations: Recommendations,
) -> float:
    return AveragePrecision.from_df(recommendations.citation_to_language_candidates)


def compute_average_precision_language_to_citation(
    recommendations: Recommendations,
) -> float:
    return AveragePrecision.from_df(recommendations.language_to_citation)


def compute_average_precision_language_to_citation_candidates(
    recommendations: Recommendations,
) -> float:
    return AveragePrecision.from_df(recommendations.language_to_citation_candidates)


def compute_num_unique_labels_citation_to_language(
    recommendations: Recommendations,
) -> int:
    return CountUniqueLabels.from_df(recommendations.citation_to_language)


def compute_num_unique_labels_citation_to_language_candidates(
    recommendations: Recommendations,
) -> int:
    return CountUniqueLabels.from_df(recommendations.citation_to_language_candidates)


def compute_num_unique_labels_language_to_citation(
    recommendations: Recommendations,
) -> int:
    return CountUniqueLabels.from_df(recommendations.language_to_citation)


def compute_num_unique_labels_language_to_citation_candidates(
    recommendations: Recommendations,
) -> int:
    return CountUniqueLabels.from_df(recommendations.language_to_citation_candidates)


def add_scoring_columns(combinations_frame: pl.DataFrame) -> pl.DataFrame:
    avg_precision_c_to_l_list = []
    avg_precision_c_to_l_cand_list = []
    avg_precision_l_to_c_list = []
    avg_precision_l_to_c_cand_list = []
    num_unique_labels_c_to_l_list = []
    num_unique_labels_c_to_l_cand_list = []
    num_unique_labels_l_to_c_list = []
    num_unique_labels_l_to_c_cand_list = []

    with rich_progress_bar() as progress_bar:
        task = progress_bar.add_task(description="Processing...", total=len(combinations_frame))

        for row in combinations_frame.iter_rows(named=True):
            recommendations = retrieve_recommendations(
                semanticscholar_id=row["semanticscholar_id"],
                language_model=row["language_model"],
                feature_weights=row["feature_weights"],
            )

            avg_precision_c_to_l_list.append(
                compute_average_precision_citation_to_language(recommendations)
            )
            avg_precision_c_to_l_cand_list.append(
                compute_average_precision_citation_to_language_candidates(recommendations)
            )
            avg_precision_l_to_c_list.append(
                compute_average_precision_language_to_citation(recommendations)
            )
            avg_precision_l_to_c_cand_list.append(
                compute_average_precision_language_to_citation_candidates(recommendations)
            )
            num_unique_labels_c_to_l_list.append(
                compute_num_unique_labels_citation_to_language(recommendations)
            )
            num_unique_labels_c_to_l_cand_list.append(
                compute_num_unique_labels_citation_to_language_candidates(recommendations)
            )
            num_unique_labels_l_to_c_list.append(
                compute_num_unique_labels_language_to_citation(recommendations)
            )
            num_unique_labels_l_to_c_cand_list.append(
                compute_num_unique_labels_language_to_citation_candidates(recommendations)
            )

            progress_bar.update(task, advance=1)

    return combinations_frame.with_columns(
        avg_precision_c_to_l=pl.Series(avg_precision_c_to_l_list),
        avg_precision_c_to_l_cand=pl.Series(avg_precision_c_to_l_cand_list),
        avg_precision_l_to_c=pl.Series(avg_precision_l_to_c_list),
        avg_precision_l_to_c_cand=pl.Series(avg_precision_l_to_c_cand_list),
        num_unique_labels_c_to_l=pl.Series(num_unique_labels_c_to_l_list),
        num_unique_labels_c_to_l_cand=pl.Series(num_unique_labels_c_to_l_cand_list),
        num_unique_labels_l_to_c=pl.Series(num_unique_labels_l_to_c_list),
        num_unique_labels_l_to_c_cand=pl.Series(num_unique_labels_l_to_c_cand_list),
    )


def add_hybrid_average_precision(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a new columns with the hybrid average precision, i.e. the mean of both hybrid
    order average precisions.
    """
    return df.with_columns(
        avg_precision_hybrid=(pl.col("avg_precision_c_to_l") + pl.col("avg_precision_l_to_c")) / 2,
    )


def select_top_n_feature_weights(df: pl.DataFrame, n: int) -> list[list[int]]:
    return (
        df.groupby("feature_weights")
        .agg(mean_avg_precision_hybrid=pl.col("avg_precision_hybrid").mean())["feature_weights"]
        .head(n)
        .to_list()
    )


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
    num_samples_feature_weights_candidates = 200
    num_samples_input_combinations = 1000
    num_best_feature_weights = 10
    seed = 42

    documents_frame: DocumentsFrame = read_df_from_parquet(DataPaths.merged.documents_frame)

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

    feature_weights_ranges = FeatureWeightsRanges()
    feature_weights_candidates = feature_weights_ranges.sample(
        num_samples=num_samples_feature_weights_candidates
    )

    best_feature_weights = (
        construct_combinations_frame(
            documents_frame, language_model_candidates, feature_weights_candidates
        )
        .pipe(sample_input_combinations, num_samples=num_samples_input_combinations, seed=seed)
        .pipe(add_scoring_columns)
        .pipe(add_hybrid_average_precision)
        .pipe(select_top_n_feature_weights, n=num_best_feature_weights)
    )

    with open(ResultsPaths.evaluation.feature_weights_candidates_pkl, "wb") as file:
        pickle.dump(best_feature_weights, file)

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
