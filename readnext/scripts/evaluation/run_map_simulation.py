"""
Simulates the Mean Average Precision (MAP) for 10.000 simulated recommendation lists and
different proportions of relevant and irrelevant recommendations. These values are used
as benchmarks for performance comparisons.

Each recommendation list contains values of 0 (irrelevant) and 1 (relevant) for 20
items. The Average Precision (AP) is calculated for each list and the mean of all APs is
returned as the MAP.
"""

from readnext.evaluation.metrics.evaluation_metric import AveragePrecision
import numpy as np


def generate_recommendation_lists(
    recommendation_labels: list[int],
    num_simulations: int,
    length_recommendation_list: int,
    seed: int,
    proportions: list[float],
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    return rng.choice(
        recommendation_labels,
        size=(num_simulations, length_recommendation_list),
        p=proportions,
        replace=True,
    )


def main() -> None:
    num_simulations = 100_000
    length_recommendation_list = 20
    recommendation_labels = [0, 1]
    seed = 42

    recommendation_lists_50_50 = generate_recommendation_lists(
        recommendation_labels,
        num_simulations,
        length_recommendation_list,
        seed,
        proportions=[0.5, 0.5],
    )

    mean_precision_50_50 = AveragePrecision.mean_precision(recommendation_lists_50_50)

    mean_average_precision_50_50 = AveragePrecision.mean_average_precision(
        recommendation_lists_50_50
    )

    print(f"Mean Precision with 50/50 proportions: {mean_precision_50_50:.3f}")
    print(f"Mean Average Precision with 50/50 proportions: {mean_average_precision_50_50:.3f}\n")

    recommendation_lists_20_80 = generate_recommendation_lists(
        recommendation_labels,
        num_simulations,
        length_recommendation_list,
        seed,
        proportions=[0.2, 0.8],
    )

    mean_precision_20_80 = AveragePrecision.mean_precision(recommendation_lists_20_80)

    mean_average_precision_20_80 = AveragePrecision.mean_average_precision(
        recommendation_lists_20_80
    )

    print(f"Mean Precision with 20/80 proportions: {mean_precision_20_80:.3f}")
    print(f"Mean Average Precision with 20/80 proportions: {mean_average_precision_20_80:.3f}\n")

    recommendation_lists_80_20 = generate_recommendation_lists(
        recommendation_labels,
        num_simulations,
        length_recommendation_list,
        seed,
        proportions=[0.8, 0.2],
    )

    mean_precision_80_20 = AveragePrecision.mean_precision(recommendation_lists_80_20)

    mean_average_precision_80_20 = AveragePrecision.mean_average_precision(
        recommendation_lists_80_20
    )

    print(f"Mean Precision with 80/20 proportions: {mean_precision_80_20:.3f}")
    print(f"Mean Average Precision with 80/20 proportions: {mean_average_precision_80_20:.3f}\n")

    # findings:
    # - the values of the NULL model with randomly sampled recommendations strongly
    #   depend on the proportions of relevant and irrelevant recommendations
    #
    # - random MAP value for 50/50 proportions of 0/1, i.e. irrelevant/relevant
    #   recommendations: 0.569
    #
    # - random MAP value for 20/80 proportions of 0/1, i.e. irrelevant/relevant
    #   recommendations: 0.828
    #
    # - random MAP value for 80/20 proportions of 0/1, i.e. irrelevant/relevant
    #   recommendations: 0.308


if __name__ == "__main__":
    main()
