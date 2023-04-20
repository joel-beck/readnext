import pickle

import pandas as pd
import spacy

from readnext.config import DataPaths, ModelVersions, ResultsPaths
from readnext.modeling.language_models import (
    # BERTTokenizer,
    SpacyTokenizer,
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
    spacy_tokenizer = SpacyTokenizer(documents_info, spacy_model)
    spacy_tokenized_abstracts = spacy_tokenizer.tokenize()

    with ResultsPaths.language_models.spacy_tokenized_abstracts_most_cited.open("wb") as f:
        pickle.dump(spacy_tokenized_abstracts, f)

    # TODO: Add BERT Tokenizer


if __name__ == "__main__":
    main()
