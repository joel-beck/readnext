"""
Generate embedding frames of document abstracts with Longformer.
"""

from readnext.config import ResultsPaths
from readnext.modeling.language_models import (
    LanguageModelChoice,
    LongformerEmbedder,
    load_language_model,
)
from readnext.utils.io import read_df_from_parquet, write_df_to_parquet
from readnext.utils.transformers_logging import suppress_transformers_logging


def main() -> None:
    suppress_transformers_logging()

    longformer_token_ids_frame = read_df_from_parquet(
        ResultsPaths.language_models.longformer_token_ids_frame_parquet
    )

    longformer_model = load_language_model(LanguageModelChoice.LONGFORMER)
    longformer_embedder = LongformerEmbedder(
        token_ids_frame=longformer_token_ids_frame, torch_model=longformer_model
    )
    longformer_embeddings_frame = longformer_embedder.compute_embeddings_frame()

    write_df_to_parquet(
        longformer_embeddings_frame,
        ResultsPaths.language_models.longformer_embeddings_frame_parquet,
    )


if __name__ == "__main__":
    main()
