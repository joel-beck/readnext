from readnext.modeling.language_models.model_choice import (
    LanguageModelChoice,
    get_cosine_similarities_path_from_choice,
    get_embeddings_path_from_choice,
)
from readnext.utils import EmbeddingsFrame, ScoresFrame, read_df_from_parquet, status_update


def load_cosine_similarities_from_choice(language_model_choice: LanguageModelChoice) -> ScoresFrame:
    cosine_similarities_path = get_cosine_similarities_path_from_choice(language_model_choice)
    return read_df_from_parquet(cosine_similarities_path)


@status_update("Loading pretrained embeddings")
def load_embeddings_from_choice(language_model_choice: LanguageModelChoice) -> EmbeddingsFrame:
    embeddings_path = get_embeddings_path_from_choice(language_model_choice)
    return read_df_from_parquet(embeddings_path)
