"""
Generate embedding frames of document abstracts with BERT.
"""

from transformers import BertModel

from readnext.config import ModelVersions, ResultsPaths
from readnext.modeling.language_models import BERTEmbedder
from readnext.utils.io import read_df_from_parquet, write_df_to_parquet
from readnext.utils.logging import suppress_transformers_logging
from readnext.utils.torch_device import get_torch_device


def main() -> None:
    suppress_transformers_logging()

    bert_token_ids_frame = read_df_from_parquet(
        ResultsPaths.language_models.bert_token_ids_frame_parquet
    )

    bert_model = BertModel.from_pretrained(ModelVersions.bert)
    bert_embedder = BERTEmbedder(
        token_ids_frame=bert_token_ids_frame,
        torch_model=bert_model,  # type: ignore
        device=get_torch_device(),
    )
    bert_embeddings_frame = bert_embedder.compute_embeddings_frame()

    write_df_to_parquet(
        bert_embeddings_frame,
        ResultsPaths.language_models.bert_embeddings_frame_parquet,
    )


if __name__ == "__main__":
    main()
