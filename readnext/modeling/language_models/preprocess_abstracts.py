import pickle

import pandas as pd
import spacy

from readnext.data.config import DataPaths
from readnext.modeling.config import ModelVersions, ResultsPaths
from readnext.modeling.language_models.document_preprocessor import (
    # BERTPreprocessor,
    SpacyPreprocessor,
    documents_info_from_df,
)


def main() -> None:
    documents_authors_labels_citations_most_cited = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )

    documents_info = documents_info_from_df(documents_authors_labels_citations_most_cited)

    # requires downloading the model first with `python -m spacy download
    # <model_version>` from the command line
    spacy_model = spacy.load(ModelVersions.spacy)
    spacy_preprocessor = SpacyPreprocessor(documents_info, spacy_model)
    spacy_tokenized_abstracts = spacy_preprocessor.tokenize()

    with ResultsPaths.language_models.spacy_preprocessing_most_cited.open("wb") as f:
        pickle.dump(spacy_tokenized_abstracts, f)

    # TODO: Add BERT Preprocessing


if __name__ == "__main__":
    main()
