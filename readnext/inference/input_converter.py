from dataclasses import dataclass
from typing import cast

import polars as pl

from readnext.utils.convert_id_urls import (
    get_arxiv_id_from_arxiv_url,
    get_arxiv_url_from_arxiv_id,
    get_semanticscholar_id_from_semanticscholar_url,
    get_semanticscholar_url_from_semanticscholar_id,
)
from readnext.utils.aliases import DocumentsFrame
from readnext.utils.repr import generate_frame_repr


@dataclass
class InferenceDataInputConverter:
    """Converts input to `InferenceDataConstructor` from and to D3 document ID."""

    documents_frame: DocumentsFrame

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({generate_frame_repr(self.documents_frame)})"

    def get_d3_document_id_from_semanticscholar_url(self, semanticscholar_url: str) -> int:
        """Retrieve D3 document id from Semanticscholar url."""
        return (
            self.documents_frame.filter(pl.col("semanticscholar_url") == semanticscholar_url)
            .select("d3_document_id")
            .item()
        )

    def get_d3_document_id_from_semanticscholar_id(self, semanticscholar_id: str) -> int:
        """Retrieve D3 document id from Semanticscholar id."""
        semanticscholar_url = get_semanticscholar_url_from_semanticscholar_id(semanticscholar_id)
        return self.get_d3_document_id_from_semanticscholar_url(semanticscholar_url)

    def get_d3_document_id_from_arxiv_id(self, arxiv_id: str) -> int:
        """Retrieve D3 document id from Arxiv id."""

        return (
            self.documents_frame.filter(pl.col("arxiv_id") == arxiv_id)
            .select("d3_document_id")
            .item()
        )

    def get_d3_document_id_from_arxiv_url(self, arxiv_url: str) -> int:
        """Retrieve D3 document id from Arxiv url."""
        arxiv_id = get_arxiv_id_from_arxiv_url(arxiv_url)

        return self.get_d3_document_id_from_arxiv_id(arxiv_id)

    def get_semanticscholar_url_from_d3_document_id(self, d3_document_id: int) -> str:
        """Retrieve Semanticscholar url from D3 document id."""
        return cast(
            str,
            self.documents_frame.filter(pl.col("d3_document_id") == d3_document_id)
            .select("semanticscholar_url")
            .item(),
        )

    def get_semanticscholar_id_from_d3_document_id(self, d3_document_id: int) -> str:
        """Retrieve Semanticscholar id from D3 document id."""
        semanticscholar_url = self.get_semanticscholar_url_from_d3_document_id(d3_document_id)
        return get_semanticscholar_id_from_semanticscholar_url(semanticscholar_url)

    def get_arxiv_id_from_d3_document_id(self, d3_document_id: int) -> str:
        """Retrieve Arxiv id from D3 document id."""
        return cast(
            str,
            self.documents_frame.filter(pl.col("d3_document_id") == d3_document_id)
            .select("arxiv_id")
            .item(),
        )

    def get_arxiv_url_from_d3_document_id(self, d3_document_id: int) -> str:
        """Retrieve Arxiv url from D3 document id."""
        arxiv_id = self.get_arxiv_id_from_d3_document_id(d3_document_id)
        return get_arxiv_url_from_arxiv_id(arxiv_id)
