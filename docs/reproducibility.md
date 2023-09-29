# Reproducibility

This section provides instructions on how to reproduce the evaluation results presented in the thesis.


## Evaluation

The evaluation strategy and results are described in detail in chapter 4 of the [thesis](https://github.com/joel-beck/msc-thesis/blob/main/thesis/beck-joel_masters-thesis.pdf).

The main findings are:

1. The Bibliographic Coupling feature is the most important feature for the Citation Recommender followed by the Co-Citation Analysis feature. The Paper Citation Count performs worst and is, on average, equally effective as randomly chosen papers from the training corpus.

1. The SciBERT language model performs best for the Language Recommender followed by TF-IDF and BERT. The Longformer model cannot leverage its strength on long documents and performs worst.

1. When using only a single recommender, the Language Recommender outperforms the Citation Recommender.

1. The best hybrid model is the Language -> Citation Hybrid Recommender, i.e. using the Language Recommender first for candidate selection and the Citation Recommender second for re-ranking.

1. Surprisingly, the best overall model is *not* a hybrid model, but rather the Language Recommender with the SciBERT language model alone.
