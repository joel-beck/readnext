"""Tokenize paper abstracts in order to pass it to embedding models."""

import pandas as pd
import spacy
from transformers import BertTokenizerFast, LongformerTokenizerFast

from readnext.config import DataPaths, ModelVersions, ResultsPaths
from readnext.modeling import documents_info_from_df
from readnext.modeling.language_models import BERTTokenizer, LongformerTokenizer, SpacyTokenizer


def main() -> None:
    documents_authors_labels_citations_most_cited = pd.read_pickle(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    documents_authors_labels_citations_most_cited = (
        documents_authors_labels_citations_most_cited.head(1000)
    )

    documents_info = documents_info_from_df(documents_authors_labels_citations_most_cited)

    spacy_model = spacy.load(ModelVersions.spacy)
    spacy_tokenizer = SpacyTokenizer(documents_info, spacy_model)
    spacy_tokenized_abstracts = spacy_tokenizer.tokenize()
    spacy_tokenizer.save_tokens_mapping(
        ResultsPaths.language_models.spacy_tokenized_abstracts_mapping_most_cited_pkl,
        spacy_tokenized_abstracts,
    )

    bert_tokenizer_transformers = BertTokenizerFast.from_pretrained(
        ModelVersions.bert, do_lower_case=True, clean_text=True
    )
    bert_tokenizer = BERTTokenizer(documents_info, bert_tokenizer_transformers)
    bert_tokenized_abstracts = bert_tokenizer.tokenize()
    bert_tokenizer.save_tokens_mapping(
        ResultsPaths.language_models.bert_tokenized_abstracts_mapping_most_cited_pkl,
        bert_tokenized_abstracts,
    )

    scibert_tokenizer_transformers = BertTokenizerFast.from_pretrained(ModelVersions.scibert)
    scibert_tokenizer = BERTTokenizer(documents_info, scibert_tokenizer_transformers)
    scibert_tokenized_abstracts = scibert_tokenizer.tokenize()
    scibert_tokenizer.save_tokens_mapping(
        ResultsPaths.language_models.scibert_tokenized_abstracts_mapping_most_cited_pkl,
        scibert_tokenized_abstracts,
    )

    longformer_tokenizer_transformers = LongformerTokenizerFast.from_pretrained(
        ModelVersions.longformer
    )
    longformer_tokenizer = LongformerTokenizer(documents_info, longformer_tokenizer_transformers)
    longformer_tokenized_abstracts = longformer_tokenizer.tokenize()
    longformer_tokenizer.save_tokens_mapping(
        ResultsPaths.language_models.longformer_tokenized_abstracts_mapping_most_cited_pkl,
        longformer_tokenized_abstracts,
    )


if __name__ == "__main__":
    main()
