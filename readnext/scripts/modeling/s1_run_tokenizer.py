"""Tokenize paper abstracts in order to pass it to embedding models."""

import spacy
from transformers import BertTokenizerFast, LongformerTokenizerFast

from readnext.config import DataPaths, ModelVersions, ResultsPaths
from readnext.modeling.language_models import BERTTokenizer, LongformerTokenizer, SpacyTokenizer
from readnext.utils import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    documents_data = read_df_from_parquet(DataPaths.merged.documents_data)
    # NOTE: Remove to train on full data
    documents_data = documents_data.head(1000)

    spacy_model = spacy.load(ModelVersions.spacy)
    spacy_tokenizer = SpacyTokenizer(documents_data, spacy_model)
    spacy_tokens_frame = spacy_tokenizer.tokenize()
    write_df_to_parquet(
        spacy_tokens_frame,
        ResultsPaths.language_models.spacy_tokenized_abstracts_parquet,
    )

    bert_tokenizer_transformers = BertTokenizerFast.from_pretrained(
        ModelVersions.bert, do_lower_case=True, clean_text=True
    )
    bert_tokenizer = BERTTokenizer(documents_data, bert_tokenizer_transformers)
    bert_tokens_frame = bert_tokenizer.tokenize()
    write_df_to_parquet(
        bert_tokens_frame,
        ResultsPaths.language_models.bert_tokenized_abstracts_parquet,
    )

    scibert_tokenizer_transformers = BertTokenizerFast.from_pretrained(
        ModelVersions.scibert, do_lower_case=True, clean_text=True
    )
    scibert_tokenizer = BERTTokenizer(documents_data, scibert_tokenizer_transformers)
    scibert_tokens_frame = scibert_tokenizer.tokenize()
    write_df_to_parquet(
        scibert_tokens_frame,
        ResultsPaths.language_models.scibert_tokenized_abstracts_parquet,
    )

    longformer_tokenizer_transformers = LongformerTokenizerFast.from_pretrained(
        ModelVersions.longformer
    )
    longformer_tokenizer = LongformerTokenizer(documents_data, longformer_tokenizer_transformers)
    longformer_tokens_frame = longformer_tokenizer.tokenize()
    write_df_to_parquet(
        longformer_tokens_frame,
        ResultsPaths.language_models.longformer_tokenized_abstracts_parquet,
    )


if __name__ == "__main__":
    main()
