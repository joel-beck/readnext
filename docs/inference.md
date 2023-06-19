# Inference

The user interface for generating recommendations is designed to be simple and easy to use.
It relies on the top-level `readnext()` function, which takes two required and one optional keyword argument:

- An identifier for the query paper.
This can be the Semanticscholar ID, Semanticscholar URL, Arxiv ID, or Arxiv URL of the paper.
This argument is required and should be provided as a string.

!!! info "Term Definitions"

    - The *Semanticscholar ID* is a 40-digit hexadecimal string at the end of the Semanticscholar URL after the last forward slash.
    For example, the Semanticscholar ID for the URL `https://www.semanticscholar.org/paper/67c4ffa7f9c25e9e0f0b0eac5619070f6a5d143d` is `67c4ffa7f9c25e9e0f0b0eac5619070f6a5d143d`.
    - The *Arxiv ID* is a 4-digit number followed by a dot followed by a 5-digit number at the end of the Arxiv URL after the last forward slash.
    For example, the Arxiv ID for the URL `https://arxiv.org/abs/1234.56789` is `1234.56789`.

- The language model choice for the Language Recommender, which is used to tokenize and embed the query paper's abstract.
This argument is required and should be passed using the `LanguageModelChoice` Enum, which provides autocompletion for all eight available language models.

- The feature weighting for the Citation Recommender.
This argument is optional and is submitted using an instance of the `FeatureWeights` class.
If not specified, the five features (`publication_date`, `citationcount_document`, `citationcount_author`, `co_citation_analysis`, and `bibliographic_coupling`) are given equal weights of one.
Note that the weights are normalized to sum up to one, so the absolute values are irrelevant; only the relative ratios matter.


## Examples

Inference works for both 'seen' and 'unseen' query documents, depending on whether the query document is part of the training corpus or not.

### Seen Query Paper

If the query paper is part of the training corpus, all feature values are precomputed and inference is fast.

Assume that we have just read the groundbreaking paper "Attention is all you need" by Vaswani et al. (2017) and want to find related papers that could be of interest to us.

The following code example illustrates the standard usage of the `readnext()` function to retrieve recommendations.
In this case we use the ArXiV URL as identifier, the FastText language model and assign a custom weighting scheme to the Citation Recommender:

```python
from readnext import readnext, LanguageModelChoice, FeatureWeights

result = readnext(
    # `Attention is all you need` query paper
    arxiv_url="https://arxiv.org/abs/1706.03762",
    language_model_choice=LanguageModelChoice.FASTTEXT,
    feature_weights=FeatureWeights(
        publication_date=1,
        citationcount_document=2,
        citationcount_author=0.5,
        co_citation_analysis=2,
        bibliographic_coupling=2,
    ),
)
```

A message is printed to the console indicating that the query paper is part of the training corpus:

```console
> ╭──────────────────────────────────────────────────╮
> │                                                  │
> │ Query document is contained in the training data │
> │                                                  │
> ╰──────────────────────────────────────────────────╯
```

The return value of the `readnext()` function contains the following attributes:

- `document_identifier`: Contains the identifiers of the query paper.

- `document_info`: Provides information about the query paper.

- `features`: Individual dataframes that include values for `publication_date`, `citationcount_document`, `citationcount_author`, `co_citation_analysis`, `bibliographic_coupling`, `cosine_similarity`, and `feature_weights`.

- `ranks`: Individual dataframes that list the ranks of individual features.

- `points`: Individual dataframes that specify the points of individual features.

- `labels`: Individual dataframes that present the arxiv labels for all candidate papers and binary 0/1 labels related to the query paper.
These binary labels are useful for 'seen' query papers where the arxiv labels of the query paper is known.
For 'unseen' papers this information is not availabels and all binary labels are set to 0.

- `recommendations`: Individual dataframes that offer the top paper recommendations.
Recommendations are calculated for both Hybrid-Recommender orders (Citation -> Language and Language -> Citation) and both the intermediate candidate lists and the final hybrid recommendations.


```python
print(result.recommendations.language_to_citation.head(10))
```

| candidate_d3_document_id | weighted_points | title                                                                                                           | author            | arxiv_labels                        | integer_label | semanticscholar_url                                                            | arxiv_url                        | publication_date | publication_date_points | citationcount_document | citationcount_document_points | citationcount_author | citationcount_author_points | co_citation_analysis_score | co_citation_analysis_points | bibliographic_coupling_score | bibliographic_coupling_points |
| -----------------------: | --------------: | :-------------------------------------------------------------------------------------------------------------- | :---------------- | :---------------------------------- | ------------: | :----------------------------------------------------------------------------- | :------------------------------- | :--------------- | ----------------------: | ---------------------: | ----------------------------: | -------------------: | --------------------------: | -------------------------: | --------------------------: | ---------------------------: | ----------------------------: |
|                 11212020 |            80.3 | Neural Machine Translation by Jointly Learning to Align and Translate                                           | Yoshua Bengio     | ['cs.CL' 'cs.LG' 'cs.NE' 'stat.ML'] |             1 | https://www.semanticscholar.org/paper/fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5 | https://arxiv.org/abs/1409.0473  | 2014-09-01       |                       0 |                  19996 |                            88 |               372099 |                         100 |                         45 |                          93 |                            4 |                            95 |
|                  7961699 |            70.9 | Sequence to Sequence Learning with Neural Networks                                                              | Ilya Sutskever    | ['cs.CL' 'cs.LG']                   |             1 | https://www.semanticscholar.org/paper/cea967b59209c6be22829699f05b8b1ac4dc092d | https://arxiv.org/abs/1409.3215  | 2014-09-10       |                       0 |                  15342 |                            83 |               234717 |                           0 |                         25 |                          86 |                            5 |                            97 |
|                  1629541 |            58.9 | Fully convolutional networks for semantic segmentation                                                          | Trevor Darrell    | ['cs.CV']                           |             0 | https://www.semanticscholar.org/paper/317aee7fc081f2b137a85c4f20129007fd8e717e | https://arxiv.org/abs/1411.4038  | 2014-11-14       |                       0 |                  25471 |                            91 |               142117 |                           0 |                         20 |                          81 |                            0 |                            49 |
|                206593880 |            57.9 | Rethinking the Inception Architecture for Computer Vision                                                       | Christian Szegedy | ['cs.CV']                           |             0 | https://www.semanticscholar.org/paper/23ffaa0fe06eae05817f527a47ac3291077f9e58 | https://arxiv.org/abs/1512.00567 | 2015-12-02       |                       0 |                  16562 |                            85 |               128072 |                           0 |                         21 |                          83 |                            0 |                            49 |
|                 10716717 |            56.8 | Feature Pyramid Networks for Object Detection                                                                   | Kaiming He        | ['cs.CV']                           |             0 | https://www.semanticscholar.org/paper/b9b4e05faa194e5022edd9eb9dd07e3d675c2b36 | https://arxiv.org/abs/1612.03144 | 2016-12-09       |                       0 |                  10198 |                            70 |               251467 |                           0 |                         14 |                          71 |                            1 |                            72 |
|                  6287870 |            55.7 | TensorFlow: A system for large-scale machine learning                                                           | J. Dean           | ['cs.DC' 'cs.AI']                   |             0 | https://www.semanticscholar.org/paper/46200b99c40e8586c8a0f588488ab6414119fb28 | https://arxiv.org/abs/1605.08695 | 2016-05-27       |                       0 |                  13266 |                            77 |               115104 |                           0 |                          4 |                          33 |                            7 |                            99 |
|                  3429309 |            52.8 | DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs | A. Yuille         | ['cs.CV']                           |             0 | https://www.semanticscholar.org/paper/cab372bc3824780cce20d9dd1c22d4df39ed081a | https://arxiv.org/abs/1606.00915 | 2016-06-02       |                       0 |                   9963 |                            69 |                64894 |                           0 |                          9 |                          57 |                            1 |                            72 |
|                  4555207 |            52.8 | MobileNetV2: Inverted Residuals and Linear Bottlenecks                                                          | Liang-Chieh Chen  | ['cs.CV']                           |             0 | https://www.semanticscholar.org/paper/dd9cfe7124c734f5a6fc90227d541d3dbcd72ba4 | https://arxiv.org/abs/1801.04381 | 2018-01-13       |                       0 |                   7925 |                            56 |                39316 |                           0 |                         10 |                          60 |                            2 |                            82 |
|                 13740328 |            52.5 | Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification                     | Kaiming He        | ['cs.CV' 'cs.AI' 'cs.LG']           |             1 | https://www.semanticscholar.org/paper/d6f2f611da110b5b5061731be3fc4c7f45d8ee23 | https://arxiv.org/abs/1502.01852 | 2015-02-06       |                       0 |                  12933 |                            76 |               251467 |                           0 |                          6 |                          49 |                            1 |                            72 |
|                225039882 |            52.3 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale                                      | Jakob Uszkoreit   | ['cs.CV' 'cs.AI' 'cs.LG']           |             1 | https://www.semanticscholar.org/paper/268d347e8a55b5eb82fb5e7d2f800e33c75ab18a | https://arxiv.org/abs/2010.11929 | 2020-10-22       |                       0 |                   5519 |                            16 |                51813 |                           0 |                        185 |                          98 |                            2 |                            82 |


Thus, the top recommendation is the paper "Neural Machine Translation by Jointly Learning to Align and Translate" by Yoshua Bengio.

Recall that the second Recommender of the hybrid structure is responsible for re-ranking the candidate list.
Thus, the *selection* of the recommendations above is performed by the Language Recommender based on cosine similarity whereas the *order* of the final recommendations is determined by the Citation Recommender based on the weighted points score in the second column.

!!! tip

    In situations where the co-citation analysis and bibliographic coupling scores do not differ much between the top candidate papers, the **author citation count** tends to influence the result heavily!
    To obtain balanced results, it is recommended to downscale the `citationcount_author` weight compared to some of the other features.


### Continue the Flow

We have read the previously recommended paper, but want to dive even deeper into the literature.
Note that the Semanticscholar URL and the ArXiV URL of the recommendations are contained in the output table.
Thus, they can be immediately used as identifiers for a new query!

In this case, we use the `SciBERT` language model and the default feature weights of 1 for each feature to demonstrate the minimal set of inputs of the `readnext()` function:

```python
from readnext import readnext, LanguageModelChoice, FeatureWeights

result = readnext(
    # `Attention is all you need` query paper
    arxiv_url="https://arxiv.org/abs/1706.03762",
    language_model_choice=LanguageModelChoice.FASTTEXT,
    feature_weights=FeatureWeights(
        publication_date=1,
        citationcount_document=2,
        citationcount_author=0.5,
        co_citation_analysis=2,
        bibliographic_coupling=2,
    ),
)

# extract one of the paper identifiers from the previous top recommendation
semanticscholar_url = result.recommendations.citation_to_language[0, "semanticscholar_url"]

result_seen_query = readnext(
    semanticscholar_url=semanticscholar_url,
    language_model_choice=LanguageModelChoice.SCIBERT,
)
```

Let's first take a look at our new query paper:

```python
print(result_seen_query.document_info)
```

```console
> Document 11212020
> ---------------------
> Title: Neural Machine Translation by Jointly Learning to Align and Translate
> Author: Yoshua Bengio
> Publication Date: 2014-09-01
> Arxiv Labels: ['cs.CL', 'cs.LG', 'cs.NE', 'stat.ML']
> Semanticscholar URL: https://www.semanticscholar.org/paper/fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5
> Arxiv URL: https://arxiv.org/abs/1409.0473
```

In contrast to the previous example, we now choose the Citation -> Language Hybrid-Recommender order.

In this case, the rows are sorted in descending order by the **cosine similarity** between the query paper and the candidate papers since the re-ranking step is performed by the Language Recommender.

For brevity we limit the output to the top three recommendations:


```python
print(result_seen_query.recommendations.citation_to_language.head(3))
```

| candidate_d3_document_id | cosine_similarity | title                                                                                         | author                 | publication_date | arxiv_labels                        | integer_label | semanticscholar_url                                                            | arxiv_url                        |
| -----------------------: | ----------------: | :-------------------------------------------------------------------------------------------- | :--------------------- | :--------------- | :---------------------------------- | ------------: | :----------------------------------------------------------------------------- | :------------------------------- |
|                  7961699 |             0.959 | Sequence to Sequence Learning with Neural Networks                                            | Ilya Sutskever         | 2014-09-10       | ['cs.CL' 'cs.LG']                   |             1 | https://www.semanticscholar.org/paper/cea967b59209c6be22829699f05b8b1ac4dc092d | https://arxiv.org/abs/1409.3215  |
|                  5590763 |            0.9537 | Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation | Yoshua Bengio          | 2014-06-03       | ['cs.CL' 'cs.LG' 'cs.NE' 'stat.ML'] |             1 | https://www.semanticscholar.org/paper/0b544dfe355a5070b60986319a3f51fb45d1348e | https://arxiv.org/abs/1406.1078  |
|                  1998416 |            0.9467 | Effective Approaches to Attention-based Neural Machine Translation                            | Christopher D. Manning | 2015-08-17       | ['cs.CL']                           |             1 | https://www.semanticscholar.org/paper/93499a7c7f699b6630a86fad964536f9423bb6d0 | https://arxiv.org/abs/1508.04025 |


Hence, we might read the paper "Sequence to Sequence Learning with Neural Networks" by Ilya Sutskever et al. next.


!!! tip

    If you are interested in the additional Citation Recommender feature values that were used to generate the candidate list, you can access them via the `recommendations.citation_to_language_candidates` attribute of the `result_seen_query` object.



### Unseen Query Paper

If the query paper is not part of the training corpus, the inference step takes longer since tokenization, embedding and the computation of co-citation analysis, bibliographic coupling and cosine similarity scores has to be performed from scratch.

However, apart from a longer waiting time, **the user does not have to care about** if the query paper is part of the training corpus or not since the user interface remains the same!

As an example, we fetch recommendations for the "GPT-4 Technical Report" paper by OpenAI. This paper is too recent to be part of the training corpus.

Due to its recency, it might not have been cited that often, so we lower the weight of the `co_citation_analysis` feature. Further, we increase the `publication_date` weight and decrease the `citationcount_author` weight.
For the Language Recommender we use the `GloVe` model to embed the paper abstract.

Note that we only need to specify the weights for the features we want to change from the default value of 1

```python
from readnext import readnext, LanguageModelChoice, FeatureWeights

result_unseen_query = readnext(
    arxiv_url="https://arxiv.org/abs/2303.08774",
    language_model_choice=LanguageModelChoice.GLOVE,
    feature_weights=FeatureWeights(
        publication_date=4,
        citationcount_author=0.2,
        co_citation_analysis=0.2,
    ),
)
```

The console output informs us that the query paper is not part of the training corpus and provides some progress updates for the ongoing computations:

```console
> ╭──────────────────────────────────────────────────────╮
> │                                                      │
> │ Query document is not contained in the training data │
> │                                                      │
> ╰──────────────────────────────────────────────────────╯

> Loading training corpus................. ✅ (0.07 seconds)
> Tokenizing query abstract............... ✅ (0.41 seconds)
> Loading pretrained Glove model.......... ✅ (26.38 seconds)
> Embedding query abstract................ ✅ (0.00 seconds)
> Loading pretrained embeddings........... ✅ (0.19 seconds)
> Computing cosine similarities........... ✅ (0.05 seconds)
```

The time distribution differs between the language models. For `GloVe`, loading the large pretrained model into memory allocates by far the most time.

Now, we generate the recommendations candidate list with the Language Recommender and re-rank the candidates with the Citation Recommender.
Since the second recommender of the hybrid structure is the Citation Recommender, the output is again sorted by the weighted points score of the individual features:

```python
print(result_unseen_query.recommendations.language_to_citation.head(3))
```

| candidate_d3_document_id | weighted_points | title                                                            | author          | arxiv_labels      | integer_label | semanticscholar_url                                                            | arxiv_url                        | publication_date | publication_date_points | citationcount_document | citationcount_document_points | citationcount_author | citationcount_author_points | co_citation_analysis_score | co_citation_analysis_points | bibliographic_coupling_score | bibliographic_coupling_points |
| -----------------------: | --------------: | :--------------------------------------------------------------- | :-------------- | :---------------- | ------------: | :----------------------------------------------------------------------------- | :------------------------------- | :--------------- | ----------------------: | ---------------------: | ----------------------------: | -------------------: | --------------------------: | -------------------------: | --------------------------: | ---------------------------: | ----------------------------: |
|                247951931 |              80 | PaLM: Scaling Language Modeling with Pathways                    | Noam M. Shazeer | ['cs.CL']         |             0 | https://www.semanticscholar.org/paper/094ff971d6a8b8ff870946c9b3ce5aa173617bfb | https://arxiv.org/abs/2204.02311 | 2022-04-05       |                      99 |                    145 |                             0 |                51316 |                           0 |                         72 |                          99 |                           77 |                            96 |
|                230435736 |            14.6 | The Pile: An 800GB Dataset of Diverse Text for Language Modeling | Jason Phang     | ['cs.CL']         |             0 | https://www.semanticscholar.org/paper/db1afe3b3cd4cd90e41fbba65d3075dd5aebb61e | https://arxiv.org/abs/2101.00027 | 2020-12-31       |                      19 |                    154 |                             0 |                 1303 |                           0 |                         17 |                          86 |                           48 |                             0 |
|                227239228 |            13.3 | Pre-Trained Image Processing Transformer                         | W. Gao          | ['cs.CV' 'cs.LG'] |             0 | https://www.semanticscholar.org/paper/43cb4886a8056d5005702edbc51be327542b2124 | https://arxiv.org/abs/2012.00364 | 2020-12-01       |                       5 |                    379 |                             0 |                13361 |                           0 |                          1 |                           0 |                           52 |                            65 |


The top recommendation introducing the PaLM language model as a competitor to the GPT family seems quite reasonable.

Note that the `integer_label` column is not informative for unseen query papers and only kept for consistency.
Since no arxiv labels are available for unseen query papers they can not intersect with the arxiv labels of the candidates such that all values of the `integer_label` column are set to 0.


!!! tip

    If you are interested in the cosine similarity values that were used to generate the candidate list, you can access them via the `recommendations.language_to_citation_candidates` attribute of the `result_unseen_query` object.


## Input Validation

The `pydantic` library is used for basic input validation.
For invalid user inputs the command fails early before any computations are performed with an informative error message.

The following checks are performed:

- The Semanticscholar ID must be a 40-character hexadecimal string.

- The Semanticscholar URL must be a valid URL starting with `https://www.semanticscholar.org/paper/`.

- The Arxiv ID must start with 4 digits followed by a dot followed by 5 more
digits (e.g. `1234.56789`).

- The Arxiv URL must be a valid URL starting with `https://arxiv.org/abs/`.

- At least one of the four query paper identifiers must be provided.

- The feature weights must be non-negative numeric values.


For example, the following command fails because we assigned a negative weight to the `publication_date` feature:

```python
from readnext import readnext, LanguageModelChoice, FeatureWeights

result = readnext(
    arxiv_id="2101.03041",
    language_model_choice=LanguageModelChoice.BM25,
    feature_weights=FeatureWeights(publication_date=-1),
)
```

```console
pydantic.error_wrappers.ValidationError: 1 validation error for FeatureWeights
publication_date
  ensure this value is greater than or equal to 0 (type=value_error.number.not_ge; limit_value=0)
```
