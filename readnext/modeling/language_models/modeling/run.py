from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model
from gensim.models.keyedvectors import load_word2vec_format
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel

from readnext.config import ModelPaths, ModelVersions, ResultsPaths
from readnext.modeling.language_models.modeling import (
    BERTEmbedder,
    BM25Embedder,
    FastTextEmbedder,
    TFIDFEmbedder,
    Word2VecEmbedder,
    save_embeddings_mapping,
)
from readnext.modeling.language_models.preprocessing import BERTTokenizer, SpacyTokenizer


def main() -> None:
    spacy_tokens_list_mapping = SpacyTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.spacy_tokenized_abstracts_most_cited_pkl
    )
    spacy_tokens_string_mapping = SpacyTokenizer.strings_from_tokens(spacy_tokens_list_mapping)

    bert_tokens_tensor_mapping = BERTTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.bert_tokenized_abstracts_most_cited_pt
    )
    scibert_tokens_tensor_mapping = BERTTokenizer.load_tokens_mapping(
        ResultsPaths.language_models.scibert_tokenized_abstracts_most_cited_pt
    )

    tfidf_model = TfidfVectorizer()
    tfidf_embedder = TFIDFEmbedder(tfidf_model)
    tfidf_embeddings_mapping = tfidf_embedder.compute_embeddings_mapping(
        spacy_tokens_string_mapping
    )

    save_embeddings_mapping(
        ResultsPaths.language_models.tfidf_embeddings_most_cited_npy, tfidf_embeddings_mapping
    )

    # TODO: See TODO in `embedder.py`
    bm25_model = BM25Okapi(spacy_tokens_list_mapping)
    bm25_embedder = BM25Embedder(bm25_model)
    bm25_embeddings_mapping = bm25_embedder.compute_embeddings_mapping(spacy_tokens_list_mapping)
    len(bm25_embeddings_mapping)
    # save_embeddings(ResultsPaths.language_models.bm25_embeddings_most_cited_npy, bm25_embeddings)

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
    save_embeddings_mapping(
        ResultsPaths.language_models.word2vec_embeddings_most_cited_npy, word2vec_embeddings_mapping
    )

    # requires pre-downloaded model from fasttext website:
    # https://fasttext.cc/docs/en/crawl-vectors.html#models
    fasttext_model = load_facebook_model(ModelPaths.fasttext)
    fasttext_embedder = FastTextEmbedder(fasttext_model)
    fasttext_embeddings_mapping = fasttext_embedder.compute_embeddings_mapping(
        spacy_tokens_list_mapping
    )
    save_embeddings_mapping(
        ResultsPaths.language_models.fasttext_embeddings_most_cited_npy, fasttext_embeddings_mapping
    )

    bert_model = BertModel.from_pretrained(ModelVersions.bert)  # type: ignore
    bert_embedder = BERTEmbedder(bert_model)  # type: ignore
    bert_embeddings_mapping = bert_embedder.compute_embeddings_mapping(bert_tokens_tensor_mapping)
    save_embeddings_mapping(
        ResultsPaths.language_models.bert_embeddings_most_cited_npy, bert_embeddings_mapping
    )

    scibert_model = BertModel.from_pretrained(ModelVersions.scibert)  # type: ignore
    scibert_embedder = BERTEmbedder(scibert_model)  # type: ignore
    scibert_embeddings = scibert_embedder.compute_embeddings_mapping(scibert_tokens_tensor_mapping)
    save_embeddings_mapping(
        ResultsPaths.language_models.scibert_embeddings_most_cited_npy, scibert_embeddings
    )

    # sparse vector embeddings of dimension 2728
    print(f"Dimension of TF-IDF Embeddings: {list(tfidf_embeddings_mapping.values())[0].shape}")
    # dense vector embeddings of dimension 300
    print(f"Dimension of Word2Vec Embeddings: {len(word2vec_embeddings_mapping)}")
    # dense vector embeddings of dimension 300
    print(f"Dimension of FastText Embeddings: {len(fasttext_embeddings_mapping)}")
    # dense vector embeddings of dimension 768
    print(f"Dimension of BERT Embeddings: {len(bert_embeddings_mapping)}")
    # dense vector embeddings of dimension 768
    print(f"Dimension of SciBERT Embeddings: {len(scibert_embeddings)}")


if __name__ == "__main__":
    main()
