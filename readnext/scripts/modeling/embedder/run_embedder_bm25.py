"""
Generate embedding frames of document abstracts with BM25.
"""

from readnext.config import ResultsPaths
from readnext.modeling.language_models import BM25Embedder
from readnext.utils.io import read_df_from_parquet, write_df_to_parquet


def main() -> None:
    spacy_tokens_frame = read_df_from_parquet(
        ResultsPaths.language_models.spacy_tokens_frame_parquet
    )

    bm25_embedder = BM25Embedder(tokens_frame=spacy_tokens_frame)
    bm25_embeddings_frame = bm25_embedder.compute_embeddings_frame()

    write_df_to_parquet(
        bm25_embeddings_frame,
        ResultsPaths.language_models.bm25_embeddings_frame_parquet,
    )


if __name__ == "__main__":
    main()
