"""Tokenize paper abstracts in order to pass it to embedding models."""

import pandas as pd
import spacy
from transformers import BertTokenizerFast

from readnext.config import DataPaths, ModelVersions, ResultsPaths
from readnext.modeling import documents_info_from_df
from readnext.modeling.language_models import (
    BERTTokenizer,
    SpacyTokenizer,
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
    spacy_tokenizer.save_tokens_mapping(
        ResultsPaths.language_models.spacy_tokenized_abstracts_mapping_most_cited_pkl,
        spacy_tokenized_abstracts,
    )

    bert_tokenizer_transformers = BertTokenizerFast.from_pretrained(ModelVersions.bert)
    bert_tokenizer = BERTTokenizer(documents_info, bert_tokenizer_transformers)
    bert_tokenized_abstracts = bert_tokenizer.tokenize()
    bert_tokenizer.save_tokens_mapping(
        ResultsPaths.language_models.bert_tokenized_abstracts_mapping_most_cited_pt,
        bert_tokenized_abstracts,
    )

    scibert_tokenizer_transformers = BertTokenizerFast.from_pretrained(ModelVersions.scibert)
    scibert_tokenizer = BERTTokenizer(documents_info, scibert_tokenizer_transformers)
    scibert_tokenized_abstracts = scibert_tokenizer.tokenize()
    scibert_tokenizer.save_tokens_mapping(
        ResultsPaths.language_models.scibert_tokenized_abstracts_mapping_most_cited_pt,
        scibert_tokenized_abstracts,
    )


if __name__ == "__main__":
    main()
