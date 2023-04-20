import pickle

from gensim.models.fasttext import load_facebook_model
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel

from readnext.config import ModelPaths, ModelVersions, ResultsPaths
from readnext.modeling.language_models import (
    BERTEmbedder,
    DocumentsTokensList,
    FastTextEmbedder,
    SpacyTokenizer,
    TFIDFEmbedder,
)


def main() -> None:
    with ResultsPaths.language_models.spacy_tokenized_abstracts_most_cited.open("rb") as f:
        spacy_tokens_list: DocumentsTokensList = pickle.load(f)

    # NOTE: Remove to train on full dataset
    spacy_tokens_list = spacy_tokens_list[:100]
    spacy_tokens_string = SpacyTokenizer.strings_from_tokens(spacy_tokens_list)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_embedder = TFIDFEmbedder(tfidf_vectorizer)
    tfidf_embeddings = tfidf_embedder.compute_embeddings(spacy_tokens_string)
    # sparse vector embeddings of dimension 2728
    print(f"Shape of TF-IDF Embeddings: {tfidf_embeddings.shape}")

    # requires pre-downloaded model from fasttext website:
    # https://fasttext.cc/docs/en/crawl-vectors.html#models
    fasttext_model = load_facebook_model(ModelPaths.fasttext)
    fasttext_embedder = FastTextEmbedder(fasttext_model)
    fasttext_embeddings = fasttext_embedder.compute_embeddings(spacy_tokens_list)
    # dense vector embeddings of dimension 300
    print(f"Shape of FastText Embeddings: {fasttext_embeddings.shape}")

    scibert_model = BertModel.from_pretrained(ModelVersions.scibert)  # type: ignore
    scibert_embedder = BERTEmbedder(scibert_model)  # type: ignore
    print(scibert_embedder)


if __name__ == "__main__":
    main()
