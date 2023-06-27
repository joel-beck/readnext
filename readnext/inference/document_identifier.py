from dataclasses import dataclass


@dataclass(kw_only=True)
class DocumentIdentifier:
    d3_document_id: int
    semanticscholar_id: str
    semanticscholar_url: str
    arxiv_id: str
    arxiv_url: str

    def __repr__(self) -> str:
        return (
            f"DocumentIdentifier(\n"
            f"  d3_document_id={self.d3_document_id},\n"
            f"  semanticscholar_id={self.semanticscholar_id},\n"
            f"  semanticscholar_url={self.semanticscholar_url},\n"
            f"  arxiv_id={self.arxiv_id},\n"
            f"  arxiv_url={self.arxiv_url}\n"
            ")"
        )
