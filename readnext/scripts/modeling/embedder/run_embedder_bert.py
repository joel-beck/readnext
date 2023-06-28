"""
Generate embedding frames of document abstracts with BERT.
"""

from readnext.config import ResultsPaths
from readnext.modeling.language_models import BERTEmbedder, LanguageModelChoice, load_language_model
from readnext.utils.io import read_df_from_parquet, write_df_to_parquet
from readnext.utils.transformers_logging import suppress_transformers_logging


def main() -> None:
    suppress_transformers_logging()

    bert_token_ids_frame = read_df_from_parquet(
        ResultsPaths.language_models.bert_token_ids_frame_parquet
    )

    bert_model = load_language_model(LanguageModelChoice.BERT)
    bert_embedder = BERTEmbedder(token_ids_frame=bert_token_ids_frame, torch_model=bert_model)
    bert_embeddings_frame = bert_embedder.compute_embeddings_frame()

    write_df_to_parquet(
        bert_embeddings_frame,
        ResultsPaths.language_models.bert_embeddings_frame_parquet,
    )


if __name__ == "__main__":
    main()
