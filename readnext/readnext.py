from pydantic import HttpUrl

from readnext import FeatureWeights, LanguageModelChoice
from readnext.inference import InferenceData, InferenceDataConstructor
from readnext.utils.transformers_logging import suppress_transformers_logging


def readnext(
    *,
    semanticscholar_id: str | None = None,
    semanticscholar_url: HttpUrl | str | None = None,
    arxiv_id: str | None = None,
    arxiv_url: HttpUrl | str | None = None,
    language_model_choice: LanguageModelChoice = LanguageModelChoice.SCIBERT,
    feature_weights: FeatureWeights = FeatureWeights(
        publication_date=0,
        citationcount_document=0,
        citationcount_author=0,
        co_citation_analysis=0,
        bibliographic_coupling=1,
    ),
    _verbose: bool = True,
) -> InferenceData:
    """
    Generates paper recommendations based on a specified query paper.

    The function can handle both a 'seen' query paper (present in the training data) or
    an 'unseen' query paper (not present in the training data). For 'seen' papers, the
    function quickly retrieves the recommendation through a simple lookup. However, for
    'unseen' papers, the function requires on-the-fly tokenization, embedding, and
    computation of cosine similarity, which slows down the inference process.

    The function expects the following input arguments: - A paper identifier (required):
    This can be either `semanticscholar_id`,
      `semanticscholar_url`, `arxiv_id`, or `arxiv_url`.
    - `language_model_choice` (required): This argument determines the language model to
      be used for the Language Recommender. The default value `SciBERT` indicated the
      best performance in our experiments (See chapter 4.2.3 of the thesis:
      https://github.com/joel-beck/msc-thesis/blob/main/thesis/beck-joel_masters-thesis.pdf)
    - `feature_weights` (optional): These weights influence the citation features and
      global document features for the Citation Recommender. The default values using
      only the bibliographic coupling feature indicated the best performance in our
      experiments (See chapter 4.2.2 of the thesis:
      https://github.com/joel-beck/msc-thesis/blob/main/thesis/beck-joel_masters-thesis.pdf)
    - `_verbose` (optional): If set to `True`, the function prints status and progress
      messages to the console.

    The function returns an `InferenceData` object that includes the following
    attributes: - `document_identifier`: Contains the identifiers of the query paper. -
    `document_information`: Provides information about the query paper. - `features`:
    Individual dataframes that include values for `publication_date`,
      `citationcount_document`, `citationcount_author`, `co_citation_analysis`,
      `bibliographic_coupling`, `cosine_similarity`, and `feature_weights`.
    - `ranks`: Individual dataframes that list the ranks of individual features.
    - `points`: Individual dataframes that specify the points of individual features.
    - `labels`: Individual dataframes that present the arxiv labels for all candidate
      papers and binary 0/1 labels related to the query paper. These binary labels are
      useful for 'seen' query papers where the arxiv labels of the query paper is known.
      For 'unseen' papers this information is not availabels and all binary labels are
      set to 0.
    - `recommendations`: Individual dataframes that offer the top paper recommendations.
      Recommendations are calculated for both Hybrid-Recommender orders (Citation ->
      Language and Language -> Citation), and this includes both the intermediate
      candidate lists and the final hybrid recommendations.
    """

    suppress_transformers_logging()

    constructor = InferenceDataConstructor(
        semanticscholar_id=semanticscholar_id,
        semanticscholar_url=semanticscholar_url,
        arxiv_id=arxiv_id,
        arxiv_url=arxiv_url,
        language_model_choice=language_model_choice,
        feature_weights=feature_weights,
        verbose=_verbose,
    )

    return InferenceData.from_constructor(constructor)
