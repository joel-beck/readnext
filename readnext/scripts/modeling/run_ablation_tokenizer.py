"""
Analyze the percentage of paper abstracts that are tokenized into more than the maximum
number of tokens of 512 allowed by the BERT tokenizer. These are the abstracts that are
truncated by the tokenizer and thus lose information.
"""

import numpy as np
from transformers import BertTokenizerFast

from readnext.config import DataPaths, ModelVersions
from readnext.utils import read_df_from_parquet, setup_progress_bar, suppress_transformers_logging


def main() -> None:
    suppress_transformers_logging()

    documents_frame = read_df_from_parquet(DataPaths.merged.documents_frame)
    abstracts = documents_frame["abstract"]

    bert_tokenizer_transformers = BertTokenizerFast.from_pretrained(ModelVersions.bert)

    token_ids_lengths = []

    with setup_progress_bar() as progress_bar:
        for abstract in progress_bar.track(
            abstracts, total=len(abstracts), description="Tokenizing..."
        ):
            token_ids = bert_tokenizer_transformers(
                abstract, max_length=None, truncation=False, padding=False
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

    min_index = np.argmin(token_ids_lengths).item()
    print(f"Shortest Abstract: {abstracts[min_index]}")

    max_index = np.argmax(token_ids_lengths).item()
    print(f"Longest Abstract: {abstracts[max_index]}")


if __name__ == "__main__":
    main()
