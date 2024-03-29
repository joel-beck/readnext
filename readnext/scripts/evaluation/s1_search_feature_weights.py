"""
Step 1: Hyperparameter Search for Feature Weights

Samples a random subset of feature weights from the given ranges. The ten best
performing feature weights (in terms of "hybrid average precision") are stored and used
for the final evaluation. The query papers for inference are sampled from the validation
set.
"""

from collections.abc import Sequence

import polars as pl

from readnext import readnext
from readnext.config import ResultsPaths
from readnext.data.data_split import DataSplit, load_data_split
from readnext.evaluation.metrics import AveragePrecision, CountUniqueLabels
from readnext.evaluation.scoring import FeatureWeightRanges, FeatureWeights
from readnext.inference import Recommendations
from readnext.modeling.language_models import LanguageModelChoice
from readnext.utils.aliases import DocumentsFrame
from readnext.utils.io import write_df_to_parquet
from readnext.utils.progress_bar import rich_progress_bar


def seq_to_string(sequence: Sequence[int]) -> str:
    return ",".join(str(num) for num in sequence)


def string_to_list(string: str) -> list[int]:
    return [int(num) for num in string.split(",")]


def construct_combinations_frame(
    query_documents_frame: DocumentsFrame,
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
            "feature_weights": [
                seq_to_string(feature_weights_candidate)
                for feature_weights_candidate in feature_weights_candidates
            ],
        }
    )

    return (
        query_documents_frame.select("semanticscholar_id")
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
    semanticscholar_id: str,
    language_model: str,
    feature_weights: Sequence[int],
) -> Recommendations:
    result = readnext(
        semanticscholar_id=semanticscholar_id,
        language_model_choice=LanguageModelChoice(language_model),
        feature_weights=FeatureWeights.from_sequence(feature_weights),
        _verbose=False,
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
                feature_weights=string_to_list(row["feature_weights"]),
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
    ).sort(by="avg_precision_hybrid", descending=True)


def main() -> None:
    query_data_split = DataSplit.VALIDATION
    query_documents_frame = load_data_split(query_data_split)

    seed = 42
    num_samples_feature_weights_candidates = 200
    num_samples_input_combinations = 10_000

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

    feature_weights_ranges = FeatureWeightRanges()
    feature_weights_candidates = feature_weights_ranges.sample(
        num_samples=num_samples_feature_weights_candidates, seed=seed
    )

    feature_weights_candidates_frame = (
        construct_combinations_frame(
            query_documents_frame, language_model_candidates, feature_weights_candidates
        )
        .pipe(sample_input_combinations, num_samples=num_samples_input_combinations, seed=seed)
        .pipe(add_scoring_columns)
        .pipe(add_hybrid_average_precision)
    )

    write_df_to_parquet(
        feature_weights_candidates_frame,
        ResultsPaths.evaluation.feature_weights_candidates_frame_parquet,
    )


if __name__ == "__main__":
    main()
