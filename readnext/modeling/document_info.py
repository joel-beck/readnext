from dataclasses import dataclass, field


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
