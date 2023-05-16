from dataclasses import dataclass


@dataclass(kw_only=True)
class DocumentIdentifier:
    semanticscholar_id: str
    semanticscholar_url: str
    arxiv_id: str
    arxiv_url: str
