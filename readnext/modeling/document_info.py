import json
from dataclasses import dataclass, field
from typing import TypedDict, overload

import dacite
import polars as pl
from typing_extensions import Self


class DocumentInfoDict(TypedDict):
    d3_document_id: int
    title: str
    author: str
    arxiv_labels: list[str]
    abstract: str


@dataclass(kw_only=True)
class DocumentInfo:
    """Collects information about a single document/paper."""

    d3_document_id: int
    title: str = ""
    author: str = ""
    arxiv_labels: list[str] = field(default_factory=list)
    abstract: str = ""

    def __repr__(self) -> str:
        return (
            f"DocumentInfo(\n"
            f"  d3_document_id={self.d3_document_id},\n"
            f"  title={self.title},\n"
            f"  author={self.author},\n"
            f"  arxiv_labels={self.arxiv_labels},\n"
            f"  abstract={self.abstract}\n"
            ")"
        )

    def __str__(self) -> str:
        return (
            f"Document {self.d3_document_id}\n"
            "---------------------\n"
            f"Title: {self.title}\n"
            f"Author: {self.author}\n"
            f"Arxiv Labels: {self.arxiv_labels}"
        )

    def to_dict(self) -> DocumentInfoDict:
        return {
            "d3_document_id": self.d3_document_id,
            "title": self.title,
            "author": self.author,
            "arxiv_labels": self.arxiv_labels,
            "abstract": self.abstract,
        }


@dataclass
class DocumentsInfo:
    """Represents a collection of multiple documents/papers."""

    documents_info: list[DocumentInfo]

    def __post_init__(self) -> None:
        self.d3_document_ids = [
            document_info.d3_document_id for document_info in self.documents_info
        ]
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


class DocumentScoreDict(TypedDict):
    document_info: DocumentInfoDict
    score: float


# defined here instead of in readnext.evaluation to avoid circular imports
@dataclass(kw_only=True)
class DocumentScore:
    """
    Represents a document and its corresponding score. Depending on the context, the
    score can be e.g. a similarity score or an average precision score.
    """

    document_info: DocumentInfo
    score: float

    def to_dict(self) -> DocumentScoreDict:
        return {"document_info": self.document_info.to_dict(), "score": self.score}

    def serialize(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def deserialize(cls, serialized: str) -> Self:
        return dacite.from_dict(cls, json.loads(serialized))


def documents_info_from_df(df: pl.DataFrame) -> DocumentsInfo:
    """
    Generate a `DocumentsInfo` instance from the input documents dataframe, which
    contains `document_id`, `title`, and `abstract` columns.
    """
    document_ids = df["document_id"].to_list()
    titles = df["title"].to_list()
    abstracts = df["abstract"].to_list()

    return DocumentsInfo(
        [
            DocumentInfo(d3_document_id=d3_document_id, title=title, abstract=abstract)
            for d3_document_id, title, abstract in zip(document_ids, titles, abstracts)
        ]
    )
