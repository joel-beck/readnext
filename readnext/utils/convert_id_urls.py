def get_semanticscholar_id_from_semanticscholar_url(semanticscholar_url: str) -> str:
    """
    Retrieve the semanticscholar paper ID from the semanticscholar URL. The
    semanticscholar paper ID is the last part of the URL after the final forward slash.
    """
    # handles None and empty string
    return semanticscholar_url.rsplit("/", 1)[-1] if semanticscholar_url else ""


def get_semanticscholar_url_from_semanticscholar_id(semanticscholar_id: str | None) -> str:
    """
    Retrieve the semanticscholar URL from the semanticscholar paper ID. Return an empty
    string if the paper ID is None.
    """
    # handles None and empty string
    return (
        f"https://www.semanticscholar.org/paper/{semanticscholar_id}" if semanticscholar_id else ""
    )


def get_arxiv_id_from_arxiv_url(arxiv_url: str) -> str:
    """
    Retrieve the arxiv paper ID from the arxiv URL. The arxiv paper ID is the last part
    of the URL after the final forward slash.
    """
    # handles None and empty string
    return arxiv_url.rsplit("/", 1)[-1] if arxiv_url else ""


def get_arxiv_url_from_arxiv_id(arxiv_id: str | None) -> str:
    """
    Retrieve the arxiv URL from the arxiv paper ID. Return an empty string if the paper
    ID is None.
    """
    # handles None and empty string
    return f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
