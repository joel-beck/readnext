from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model
from gensim.models.keyedvectors import load_word2vec_format
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel

from readnext.config import ModelPaths, ModelVersions, ResultsPaths
from readnext.modeling.language_models import (
    BERTEmbedder,
    BERTTokenizer,
    FastTextEmbedder,
    SpacyTokenizer,
    TFIDFEmbedder,
    Word2VecEmbedder,
    embeddings_mapping_to_frame,
)
from readnext.utils import save_df_to_pickle, slice_mapping


def main() -> None:
    spacy_tokens_list_mapping = SpacyTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.spacy_tokenized_abstracts_most_cited_pkl
    )
    # NOTE: Remove to train on full data
    spacy_tokens_list_mapping = slice_mapping(spacy_tokens_list_mapping, size=100)

    spacy_tokens_string_mapping = SpacyTokenizer.strings_from_tokens(spacy_tokens_list_mapping)

    bert_tokens_tensor_mapping = BERTTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.bert_tokenized_abstracts_most_cited_pt
    )
    # NOTE: Remove to train on full data
    bert_tokens_tensor_mapping = slice_mapping(bert_tokens_tensor_mapping, size=100)

    scibert_tokens_tensor_mapping = BERTTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.scibert_tokenized_abstracts_most_cited_pt
    )
    # NOTE: Remove to train on full data
    scibert_tokens_tensor_mapping = slice_mapping(scibert_tokens_tensor_mapping, size=100)

    tfidf_model = TfidfVectorizer()
    tfidf_embedder = TFIDFEmbedder(tfidf_model)
    tfidf_embeddings_mapping = tfidf_embedder.compute_embeddings_mapping(
        spacy_tokens_string_mapping
    )
    save_df_to_pickle(
        embeddings_mapping_to_frame(tfidf_embeddings_mapping),
        ResultsPaths.language_models.tfidf_embeddings_mapping_most_cited_pkl,
    )

    # requires pre-downloaded model from gensim data repository:
    # https://github.com/RaRe-Technologies/gensim-data
    #
    # download and save model locally with the commands
    # `import gensim.downloader as api`
    # `api.load(ModelVersions.word2vec, return_path=True)`
    #
    # then unzip the model file and move it to the local `models` directory
    word2vec_model: KeyedVectors = load_word2vec_format(ModelPaths.word2vec, binary=True)
    word2vec_embedder = Word2VecEmbedder(word2vec_model)
    word2vec_embeddings_mapping = word2vec_embedder.compute_embeddings_mapping(
        spacy_tokens_list_mapping
    )
    save_df_to_pickle(
        embeddings_mapping_to_frame(word2vec_embeddings_mapping),
        ResultsPaths.language_models.word2vec_embeddings_mapping_most_cited_pkl,
    )

    # requires pre-downloaded model from fasttext website:
    # https://fasttext.cc/docs/en/crawl-vectors.html#models
    fasttext_model = load_facebook_model(ModelPaths.fasttext)
    fasttext_embedder = FastTextEmbedder(fasttext_model)
    fasttext_embeddings_mapping = fasttext_embedder.compute_embeddings_mapping(
        spacy_tokens_list_mapping
    )
    save_df_to_pickle(
        embeddings_mapping_to_frame(fasttext_embeddings_mapping),
        ResultsPaths.language_models.fasttext_embeddings_mapping_most_cited_pkl,
    )

    # takes roughly an hour for 10.000 documents
    bert_model = BertModel.from_pretrained(ModelVersions.bert)  # type: ignore
    bert_embedder = BERTEmbedder(bert_model)  # type: ignore
    bert_embeddings_mapping = bert_embedder.compute_embeddings_mapping(bert_tokens_tensor_mapping)
    save_df_to_pickle(
        embeddings_mapping_to_frame(bert_embeddings_mapping),
        ResultsPaths.language_models.bert_embeddings_mapping_most_cited_pkl,
    )

    scibert_model = BertModel.from_pretrained(ModelVersions.scibert)  # type: ignore
    scibert_embedder = BERTEmbedder(scibert_model)  # type: ignore
    scibert_embeddings = scibert_embedder.compute_embeddings_mapping(scibert_tokens_tensor_mapping)
    save_df_to_pickle(
        embeddings_mapping_to_frame(scibert_embeddings),
        ResultsPaths.language_models.scibert_embeddings_mapping_most_cited_pkl,
    )


if __name__ == "__main__":
    main()
