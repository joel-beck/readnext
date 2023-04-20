from gensim.models.fasttext import load_facebook_model
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel

from readnext.config import ModelPaths, ModelVersions, ResultsPaths
from readnext.modeling.language_models.modeling import (
    BERTEmbedder,
    FastTextEmbedder,
    TFIDFEmbedder,
    save_embeddings,
)
from readnext.modeling.language_models.preprocessing import BERTTokenizer, SpacyTokenizer


def main() -> None:
    spacy_tokens_list = SpacyTokenizer.load_tokens(
        ResultsPaths.language_models.spacy_tokenized_abstracts_most_cited_pkl
    )
    spacy_tokens_string = SpacyTokenizer.strings_from_tokens(spacy_tokens_list)

    bert_tokens_tensor = BERTTokenizer.load_tokens(
        ResultsPaths.language_models.bert_tokenized_abstracts_most_cited_pth
    )

    scibert_tokens_tensor = BERTTokenizer.load_tokens(
        ResultsPaths.language_models.scibert_tokenized_abstracts_most_cited_pth
    )

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_embedder = TFIDFEmbedder(tfidf_vectorizer)
    tfidf_embeddings = tfidf_embedder.compute_embeddings(spacy_tokens_string)
    save_embeddings(ResultsPaths.language_models.tfidf_embeddings_most_cited_npy, tfidf_embeddings)

    # requires pre-downloaded model from fasttext website:
    # https://fasttext.cc/docs/en/crawl-vectors.html#models
    fasttext_model = load_facebook_model(ModelPaths.fasttext)
    fasttext_embedder = FastTextEmbedder(fasttext_model)
    fasttext_embeddings = fasttext_embedder.compute_embeddings(spacy_tokens_list)
    save_embeddings(
        ResultsPaths.language_models.fasttext_embeddings_most_cited_npy, fasttext_embeddings
    )

    bert_model = BertModel.from_pretrained(ModelVersions.bert)  # type: ignore
    bert_embedder = BERTEmbedder(bert_model)  # type: ignore
    bert_embeddings = bert_embedder.compute_embeddings(bert_tokens_tensor)
    save_embeddings(ResultsPaths.language_models.bert_embeddings_most_cited_npy, bert_embeddings)

    scibert_model = BertModel.from_pretrained(ModelVersions.scibert)  # type: ignore
    scibert_embedder = BERTEmbedder(scibert_model)  # type: ignore
    scibert_embeddings = scibert_embedder.compute_embeddings(scibert_tokens_tensor)
    save_embeddings(
        ResultsPaths.language_models.scibert_embeddings_most_cited_npy, scibert_embeddings
    )

    # sparse vector embeddings of dimension 2728
    print(f"Shape of TF-IDF Embeddings: {tfidf_embeddings.shape}")
    # dense vector embeddings of dimension 300
    print(f"Shape of FastText Embeddings: {fasttext_embeddings.shape}")
    # dense vector embeddings of dimension 768
    print(f"Shape of BERT Embeddings: {bert_embeddings.shape}")
    # dense vector embeddings of dimension 768
    print(f"Shape of SciBERT Embeddings: {scibert_embeddings.shape}")


if __name__ == "__main__":
    main()
