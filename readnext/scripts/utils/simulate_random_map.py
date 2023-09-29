"""
Simulates the Mean Average Precision (MAP) for 10.000 simulated recommendation lists and
different proportions of relevant and irrelevant recommendations. These values are used
as benchmarks for performance comparisons.

Each recommendation list contains values of 0 (irrelevant) and 1 (relevant) for 20
items. The Average Precision (AP) is calculated for each list and the mean of all APs is
returned as the MAP.
"""

import numpy as np

from readnext.evaluation.metrics.evaluation_metric import AveragePrecision


def generate_recommendation_lists(
    recommendation_labels: list[int],
    num_simulations: int,
    seed: int,
    num_recommendations: int,
    proportions: list[float],
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    return rng.choice(
        recommendation_labels,
        size=(num_simulations, num_recommendations),
        p=proportions,
        replace=True,
    )


def map_by_proportion_num_recommendations(
    recommendation_labels: list[int],
    num_simulations: int,
    seed: int,
    num_recommendations: int,
    proportions: list[float],
) -> float:
    recommendation_lists = generate_recommendation_lists(
        recommendation_labels,
        num_simulations,
        seed,
        num_recommendations,
        proportions,
    )

    return AveragePrecision.mean_average_precision(recommendation_lists)


def main() -> None:
    num_simulations = 100_000
    recommendation_labels = [0, 1]
    seed = 42

    num_recommendations_list = [10, 20, 50]

    mean_average_precisions_50_50 = [
        map_by_proportion_num_recommendations(
            recommendation_labels,
            num_simulations,
            seed,
            num_recommendations,
            proportions=[0.5, 0.5],
        )
        for num_recommendations in num_recommendations_list
    ]

    for i in range(len(num_recommendations_list)):
        print(
            f"MAP with 50/50 proportions and {num_recommendations_list[i]} recommendations: "
            f"{mean_average_precisions_50_50[i]: .3f}"
        )
    print()

    mean_average_precisions_20_80 = [
        map_by_proportion_num_recommendations(
            recommendation_labels,
            num_simulations,
            seed,
            num_recommendations,
            proportions=[0.2, 0.8],
        )
        for num_recommendations in num_recommendations_list
    ]

    for i in range(len(num_recommendations_list)):
        print(
            f"MAP with 20/80 proportions and {num_recommendations_list[i]} recommendations: "
            f"{mean_average_precisions_20_80[i]: .3f}"
        )
    print()

    mean_average_precisions_80_20 = [
        map_by_proportion_num_recommendations(
            recommendation_labels,
            num_simulations,
            seed,
            num_recommendations,
            proportions=[0.8, 0.2],
        )
        for num_recommendations in num_recommendations_list
    ]

    for i in range(len(num_recommendations_list)):
        print(
            f"MAP with 80/20 proportions and {num_recommendations_list[i]} recommendations: "
            f"{mean_average_precisions_80_20[i]: .3f}"
        )
    print()

    mean_average_precisions_713_287 = [
        map_by_proportion_num_recommendations(
            recommendation_labels,
            num_simulations,
            seed,
            num_recommendations,
            proportions=[0.713, 0.287],
        )
        for num_recommendations in num_recommendations_list
    ]

    for i in range(len(num_recommendations_list)):
        print(
            f"MAP with 71.3/28.7 proportions and {num_recommendations_list[i]} recommendations: "
            f"{mean_average_precisions_713_287[i]: .3f}"
        )
    print()

    # findings:
    # - the values of the NULL model with randomly sampled recommendations strongly
    #   depend on the proportions of relevant and irrelevant recommendations and the
    #   length of the recommendation list
    #
    # - 50/50 proportions of 0/1, i.e. irrelevant/relevant recommendations:
    #   - MAP for 10 recommendations: 0.608
    #   - MAP for 20 recommendations: 0.569
    #   - MAP for 50 recommendations: 0.536
    #
    # - 20/80 proportions of 0/1, i.e. irrelevant/relevant recommendations:
    #   - MAP for 10 recommendations: 0.843
    #   - MAP for 20 recommendations: 0.828
    #   - MAP for 50 recommendations: 0.815
    #
    # - 80/20 proportions of 0/1, i.e. irrelevant/relevant recommendations:
    #   - MAP for 10 recommendations: 0.350
    #   - MAP for 20 recommendations: 0.308
    #   - MAP for 50 recommendations: 0.257
    #
    # - 71.3/28.7 proportions of 0/1, i.e. irrelevant/relevant recommendations:
    #   - MAP for 10 recommendations: 0.433
    #   - MAP for 20 recommendations: 0.384 <-- this is the value used for the NULL
    #     model in the evaluation
    #   - MAP for 50 recommendations: 0.338


if __name__ == "__main__":
    main()
