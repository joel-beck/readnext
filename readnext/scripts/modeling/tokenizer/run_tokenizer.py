"""
Tokenize abstracts for all documents to pass the tokenized abstracts to the embedding
models.
"""
import spacy
from transformers import BertTokenizerFast, LongformerTokenizerFast

from readnext.config import DataPaths, ModelVersions, ResultsPaths
from readnext.modeling.language_models import BERTTokenizer, LongformerTokenizer, SpacyTokenizer
from readnext.utils.io import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    documents_frame = read_df_from_parquet(DataPaths.merged.documents_frame)

    spacy_model = spacy.load(ModelVersions.spacy)
    spacy_tokenizer = SpacyTokenizer(spacy_model)
    spacy_tokens_frame = spacy_tokenizer.tokenize(documents_frame)
    write_df_to_parquet(
        spacy_tokens_frame,
        ResultsPaths.language_models.spacy_tokens_frame_parquet,
    )

    bert_tokenizer_transformers = BertTokenizerFast.from_pretrained(
        ModelVersions.bert, do_lower_case=True, clean_text=True
    )
    bert_tokenizer = BERTTokenizer(bert_tokenizer_transformers)
    bert_token_ids_frame = bert_tokenizer.tokenize(documents_frame)
    write_df_to_parquet(
        bert_token_ids_frame,
        ResultsPaths.language_models.bert_token_ids_frame_parquet,
    )

    scibert_tokenizer_transformers = BertTokenizerFast.from_pretrained(
        ModelVersions.scibert, do_lower_case=True, clean_text=True
    )
    scibert_tokenizer = BERTTokenizer(scibert_tokenizer_transformers)
    scibert_token_ids_frame = scibert_tokenizer.tokenize(documents_frame)
    write_df_to_parquet(
        scibert_token_ids_frame,
        ResultsPaths.language_models.scibert_token_ids_frame_parquet,
    )

    longformer_tokenizer_transformers = LongformerTokenizerFast.from_pretrained(
        ModelVersions.longformer
    )
    longformer_tokenizer = LongformerTokenizer(longformer_tokenizer_transformers)
    longformer_token_ids_frame = longformer_tokenizer.tokenize(documents_frame)
    write_df_to_parquet(
        longformer_token_ids_frame,
        ResultsPaths.language_models.longformer_token_ids_frame_parquet,
    )


if __name__ == "__main__":
    main()
