from dataclasses import dataclass


@dataclass
class DocumentInfo:
    document_id: int
    title: str
    author: str
    arxiv_labels: list[str]

    def __str__(self) -> str:
        return (
            f"Document {self.document_id}\n"
            "---------------------\n"
            f"Title: {self.title}\n"
            f"Author: {self.author}\n"
            f"Arxiv Labels: {self.arxiv_labels}"
        )


# defined here instead of in readnext.evaluation to avoid circular imports
@dataclass
class DocumentScore:
    document_id: int
    score: float
