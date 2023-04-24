from dataclasses import dataclass
from typing import overload

import pandas as pd
from typing_extensions import Self


@dataclass
class DocumentInfo:
    document_id: int
    title: str
    abstract: str


@dataclass
class DocumentsInfo:
    documents_info: list[DocumentInfo]

    def __post_init__(self) -> None:
        self.document_ids = [document_info.document_id for document_info in self.documents_info]
        self.titles = [document_info.title for document_info in self.documents_info]
        self.abstracts = [document_info.abstract for document_info in self.documents_info]

    def __len__(self) -> int:
        return len(self.documents_info)

    @overload
    def __getitem__(self, index: int) -> DocumentInfo:
        ...

    @overload
    def __getitem__(self, index: slice) -> Self:
        ...

    def __getitem__(self, index: int | slice) -> DocumentInfo | Self:
        # return single document info for integer index
        if isinstance(index, int):
            return self.documents_info[index]
        # return list of document infos for slice index
        return self.__class__(self.documents_info[index])


@dataclass
class DocumentStatistics:
    similarity: float
    document_info: DocumentInfo


def documents_info_from_df(df: pd.DataFrame) -> DocumentsInfo:
    document_ids = df["document_id"].tolist()
    titles = df["title"].tolist()
    abstracts = df["abstract"].tolist()

    return DocumentsInfo(
        [
            DocumentInfo(document_id, title, abstract)
            for document_id, title, abstract in zip(document_ids, titles, abstracts)
        ]
    )
