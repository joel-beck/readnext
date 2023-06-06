"""
Generate embedding frames of document abstracts with Longformer.
"""

from transformers import LongformerModel

from readnext.config import ModelVersions, ResultsPaths
from readnext.modeling.language_models import LongformerEmbedder
from readnext.utils import read_df_from_parquet, suppress_transformers_logging, write_df_to_parquet


def main() -> None:
    suppress_transformers_logging()

    longformer_token_ids_frame = read_df_from_parquet(
        ResultsPaths.language_models.longformer_tokenized_abstracts_parquet
    )

    longformer_model = LongformerModel.from_pretrained(ModelVersions.longformer)  # type: ignore
    longformer_embedder = LongformerEmbedder(
        token_ids_frame=longformer_token_ids_frame,
        torch_model=longformer_model,  # type: ignore
    )
    longformer_embeddings_frame = longformer_embedder.compute_embeddings_frame()

    write_df_to_parquet(
        longformer_embeddings_frame,
        ResultsPaths.language_models.longformer_embeddings_parquet,
    )


if __name__ == "__main__":
    main()
