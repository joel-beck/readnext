from dataclasses import dataclass
from typing import cast

import pandas as pd

from readnext.utils import (
    get_arxiv_id_from_arxiv_url,
    get_arxiv_url_from_arxiv_id,
    get_semanticscholar_id_from_semanticscholar_url,
    get_semanticscholar_url_from_semanticscholar_id,
)


@dataclass
class InferenceDataInputConverter:
    """Converts input to `InferenceDataConstructor` from and to D3 document ID."""

    documents_data: pd.DataFrame

    def get_query_id_from_semanticscholar_url(self, semanticscholar_url: str) -> int:
        """Retrieve D3 document id from Semanticscholar url."""
        return self.documents_data[
            self.documents_data["semanticscholar_url"] == semanticscholar_url
        ].index.item()

    def get_query_id_from_semanticscholar_id(self, semanticscholar_id: str) -> int:
        """Retrieve D3 document id from Semanticscholar id."""
        semanticscholar_url = get_semanticscholar_url_from_semanticscholar_id(semanticscholar_id)
        return self.get_query_id_from_semanticscholar_url(semanticscholar_url)

    def get_query_id_from_arxiv_id(self, arxiv_id: str) -> int:
        """Retrieve D3 document id from Arxiv id."""

        return self.documents_data[self.documents_data["arxiv_id"] == arxiv_id].index.item()

    def get_query_id_from_arxiv_url(self, arxiv_url: str) -> int:
        """Retrieve D3 document id from Arxiv url."""
        arxiv_id = get_arxiv_id_from_arxiv_url(arxiv_url)

        return self.get_query_id_from_arxiv_id(arxiv_id)

    def get_semanticscholar_url_from_query_id(self, query_id: int) -> str:
        """Retrieve Semanticscholar url from D3 document id."""
        return cast(str, self.documents_data.loc[query_id, "semanticscholar_url"])

    def get_semanticscholar_id_from_query_id(self, query_id: int) -> str:
        """Retrieve Semanticscholar id from D3 document id."""
        semanticscholar_url = self.get_semanticscholar_url_from_query_id(query_id)
        return get_semanticscholar_id_from_semanticscholar_url(semanticscholar_url)

    def get_arxiv_id_from_query_id(self, query_id: int) -> str:
        """Retrieve Arxiv id from D3 document id."""
        return cast(str, self.documents_data.loc[query_id, "arxiv_id"])

    def get_arxiv_url_from_query_id(self, query_id: int) -> str:
        """Retrieve Arxiv url from D3 document id."""
        arxiv_id = self.get_arxiv_id_from_query_id(query_id)
        return get_arxiv_url_from_arxiv_id(arxiv_id)