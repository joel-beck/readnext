"""
Preprocess all document chunks. Add ranks for the document's citation count and
influential citation count. Keep only documents where the arxiv id is provided - to
merge arxiv tags as labels by the arxiv id later on.
"""

from pathlib import Path

import pandas as pd

from readnext.data.config import DataPaths
from readnext.data.preprocessing_utils import add_rank


def flatten_list_of_dicts(list_of_dicts: list[dict], key: str) -> list[str]:
    return [d[key] for d in list_of_dicts] if list_of_dicts is not None else []


def remove_duplicates(list_: list[str]) -> list[str]:
    return list(set(list_))


def extract_arxiv_id(df: pd.DataFrame) -> pd.Series:
    # keep only dictionaries where arxiv id is not None
    return df["externalids"].apply(lambda d: d["ArXiv"])


def filter_by_tag(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    return df.loc[df["tags"].apply(lambda x: tag in x)]


def preprocess_document_chunk(filepath: Path, chunk_index: int) -> None:
    documents_chunk: pd.DataFrame = pd.read_pickle(filepath)

    documents_ranked = documents_chunk.assign(
        citationcount_rank=add_rank(documents_chunk["citationcount"]),
        influentialcitationcount_rank=add_rank(documents_chunk["influentialcitationcount"]),
    )

    documents_long_format = (
        documents_ranked.assign(
            author_id=lambda df: df["authors"].apply(
                lambda x: flatten_list_of_dicts(x, "authorId")
            ),
            author_name=lambda df: df["authors"].apply(lambda x: flatten_list_of_dicts(x, "name")),
            tags=lambda df: df["s2fieldsofstudy"]
            .apply(lambda x: flatten_list_of_dicts(x, "category"))
            .apply(remove_duplicates),
        )
        .explode(["author_id", "author_name"])
        .dropna(subset=["author_id", "author_name"])
        .astype({"author_id": "int"})
        .convert_dtypes()
        .pipe(filter_by_tag, tag="Computer Science")
    )

    documents_long_format_arxiv = documents_long_format.assign(
        arxiv_id=extract_arxiv_id(documents_long_format)
    ).loc[lambda df: df["arxiv_id"].notna()]

    documents_long_format_arxiv.to_pickle(
        f"{DataPaths.d3.documents.preprocessed_chunks_stem}_{chunk_index}.pkl"
    )


def main() -> None:
    filename_pattern = "*_documents_chunks_*.pkl"

    for chunk_index, filepath in enumerate(
        DataPaths.d3.documents.preprocessed_chunks_stem.parent.glob(filename_pattern), 1
    ):
        print(f"Preprocessing file {filepath.name}")
        preprocess_document_chunk(filepath, chunk_index)


if __name__ == "__main__":
    main()
