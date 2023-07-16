# Evaluation

This section describes the evaluation setup and the results of the conducted experiments.
To reproduce the evaluation steps in this chapter, we recommend reading the [Setup](setup.md) section to ensure that you have downloaded the required data and model files, and set the required environment variables.
In addition, the [Inference](inference.md) section provides an overview of the `readnext()` function and its parameters, which are central to the evaluation process.

## Recap

Recall that the `readnext()` function returns a list of paper recommendations to read next, given a query document.
The output crucially depends on two input parameters:

- The Language Model for the Language Recommender that is used to compute embeddings of the paper abstracts.
- The Feature Weights for the Citation Recommender that determine the relative importance of the publication date, the paper citation count, the author citation count, the co-citation analysis score and the bibliographic coupling score.

For the evaluation, only 'seen' query documents are considered, i.e. documents that are part of the training set.
For these documents their Arxiv Category is known and used as the ground truth to determine their label with respect to the query document: If query and candidate paper share at least one Arxiv Category, the candidate paper is considered a 'relevant' recommendation with a label 1, otherwise it is considered 'irrelevant' with a label 0.

Based on these 0/1 integer labels the Mean Average Precision (MAP) is computed as the primary evaluation metric.
Further, the number of unique labels within the recommendation list is used as a secondary metric to assess the diversity of the recommendations.

## Evaluation Goals

The goal of the evaluation is to provide data-driven answers to the following research questions:

1. Using only the Language Recommender, which Language Model yields the best MAP score?
1. Using only the Citation Recommender, which Feature Weights yield the best MAP score?
1. Is there a dependency between the Language Model and the Feature Weights in the Hybrid Recommender, i.e. do certain Language Models perform better when coupled with certain Feature Weights?
1. Which Hybridization Strategy, i.e. which order of Language Recommender and Citation Recommender in the Hybrid Recommender, yields the best MAP score and leads to more diverse recommendations in terms of unique labels? Does the Hybrid Recommender outperform the Language Recommender and the Citation Recommender in isolation?


## Evaluation Strategy

