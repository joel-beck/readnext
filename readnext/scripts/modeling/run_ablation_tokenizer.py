"""
Analyze the percentage of paper abstracts that are tokenized into more than the maximum
number of tokens of 512 allowed by the BERT tokenizer. These are the abstracts that are
truncated by the tokenizer and thus lose information.
"""

import numpy as np
from transformers import BertTokenizerFast

from readnext.config import DataPaths, ModelVersions
from readnext.modeling import documents_info_from_df
from readnext.utils import read_df_from_parquet, suppress_transformers_logging


def main() -> None:
    suppress_transformers_logging()

    documents_authors_labels_citations_most_cited = read_df_from_parquet(
        DataPaths.merged.documents_authors_labels_citations_most_cited_pkl
    )
    documents_info = documents_info_from_df(documents_authors_labels_citations_most_cited)

    bert_tokenizer_transformers = BertTokenizerFast.from_pretrained(ModelVersions.bert)

    token_ids_lengths = []

    for abstract in documents_info.abstracts:
        token_ids = bert_tokenizer_transformers(
            abstract, max_length=None, truncation=False, padding=True
        )["input_ids"]
        token_ids_lengths.append(len(token_ids))

    fraction_abstracts_above_max_length = np.mean(np.array(token_ids_lengths) > 512)

    print(f"Shortest abstract: {min(token_ids_lengths)} tokens")
    print(f"Longest abstract: {max(token_ids_lengths)} tokens")
    print(f"Average abstract length: {np.mean(token_ids_lengths):.2f} tokens")
    print(
        "Percentage of abstracts above maximum length of 512 tokens: "
        f"{fraction_abstracts_above_max_length * 100:.2f}%"
    )
    print()

    min_index = np.argmin(token_ids_lengths)
    print(f"Shortest Abstract: {documents_info.abstracts[min_index]}")

    max_index = np.argmax(token_ids_lengths)
    print(f"Longest Abstract: {documents_info.abstracts[max_index]}")


if __name__ == "__main__":
    main()
