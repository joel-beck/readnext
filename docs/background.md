# Background

The hybrid recommender system of this project is built on top of citation-based methods for quantifying document similarity as well as well-known language models from the general NLP domain.

This section provides an overview of the citation-based methods and language models that are used in this project.
These are *co-citation analysis* and *bibliographic coupling* for the citation-based methods, and *TF-IDF*, *BM25*, *Word2Vec*, *GloVe*, *FastText*, *BERT*, *SciBERT* and *Longformer* for the language models.

## Citation Models

Citation-based methods have been widely used to detect document similarity in the scientific literature. These methods leverage the citation relationships between documents to determine their similarity, based on the assumption that papers citing the same or similar sources tend to discuss related topics. Two of the most common citation-based approaches are co-citation analysis(Small, 1973)[^1] and bibliographic coupling (Kessler, 1963)[^2]. Enhanced methods, such as Citation Proximity Analysis (CPA) (Gipp & Beel, 2009)[^3] and section-based bibliographic coupling (Habib & Afzal, 2019)[^4], take the citation position into account for a more fine-grained analysis. However, in this project, we will only consider position-independent co-citation analysis and bibliographic coupling, as we have access to only the paper abstracts and not the full text.

[^1]: Small, H. (1973). Co-citation in the scientific literature: A new measure of the relationship between two documents. Journal of the American Society for Information Science, 24(4), 265–269. https://doi.org/10.1002/asi.4630240406

[^2]: Kessler, M. M. (1963). Bibliographic coupling between scientific papers. American Documentation, 14(1), 10–25. https://doi.org/10.1002/asi.5090140103

[^3]: Gipp, B., & Beel, J. (2009). Citation Proximity Analysis (CPA) – A new approach for identifying related work based on Co-Citation Analysis.

[^4]: Habib, R., & Afzal, M. T. (2019). Sections-based bibliographic coupling for research paper recommendation. Scientometrics, 119(2), 643–656. https://doi.org/10.1007/s11192-019-03053-8

### Co-Citation Analysis

Co-citation analysis measures the similarity between two documents based on the number of times they are cited together by other documents - here, these are called *citing papers*.
The underlying assumption is that if two documents are frequently cited together, they are likely to be related in some way.
Hence, the focus of co-citation analysis is on the *incoming links* to the two query documents.

More specifically, the co-citation analysis score between two query documents is the number of other documents that each cite both of the query documents.
The higher the co-citation score, the more similar the two query documents are considered to be.
The score is not impacted by the position of the citation in the citing document, i.e., the score is the same if the two query documents are cited within the same sentence or in completely different sections of the citing document.

### Bibliographic Coupling

Bibliographic coupling is another citation-based method for assessing document similarity. In contrast to co-citation analysis, bibliographic coupling measures the similarity between two documents based on the number of references they share in common - here, these are called *cited papers*. The rationale is that if two documents cite the same or similar sources, they likely address related topics or problems.
Hence, the focus of bibliographic coupling is on the *outgoing links* from the two query documents.

More specifically, the bibliographic coupling score between two query documents is the number of other documents that are cited by both query documents.
The higher the bibliographic coupling score, the more similar the two query documents are considered to be.
The score is not impacted by the position of the citation in the query documents, i.e., the score is the same if the two query documents cite the same paper in the same sentence or in completely different sections of the query documents.

## Language Models

Language models have been extensively used in various natural language processing (NLP) tasks, such as text classification, sentiment analysis, and information retrieval[^5]. An important application of language models is generating numeric embedding vectors from text, which can be utilized in different machine learning models for various purposes. These embedding vectors are compact numerical representations of words, phrases, or sentences, capturing the semantic meaning in a high-dimensional space. In this chapter, we will discuss several language models and methods to generate embedding vectors for text, particularly focusing on their application in recommender systems for computer science papers.

The discussed language models will be grouped into three categories: Keyword-based models, Static embedding models, and Contextual embedding models. For each model, we will emphasize its unique features, improvements, and differences compared to previous models.

[^5]: Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. http://www.deeplearningbook.org

### Keyword-based Models

#### TF-IDF

Term Frequency-Inverse Document Frequency (TF-IDF) is a widely used technique for information retrieval and text mining (Salton & McGill, 1983)[^6]. It is a statistical measure that quantifies the importance of a word within a document relative to its frequency in a collection of documents, called a corpus. The main idea behind TF-IDF is that the more frequent a word is in a particular document and the less frequent it is in the corpus, the more important it is for that document.

TF-IDF is a simple yet effective method for generating embedding vectors from text, mainly for document retrieval purposes. However, it does not capture semantic relationships between words, limiting its ability to model complex textual structures.

[^6]: Salton, G., & McGill, M. J. (1983). Introduction to Modern Information Retrieval. McGraw-Hill.

#### BM25

BM25 (Best Matching 25) is an improvement over the TF-IDF model, designed for ranking documents in information retrieval systems (Robertson & Zaragoza, 2009)[^7]. It uses a probabilistic approach to model the relevance of a document to a query, considering both term frequency and inverse document frequency. The primary advantage of BM25 over TF-IDF is its more sophisticated term weighting scheme, which takes document length and term frequency saturation into account.

Despite these improvements, BM25 still lacks the ability to capture semantic relationships between words, similar to TF-IDF.

[^7]: Robertson, S. E., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends® in Information Retrieval, 3(4), 333–389. https://doi.org/10.1561/1500000019

### Static Embedding Models

#### Word2Vec

Word2Vec is a popular word embedding model introduced by Mikolov et al. (2013)[^8] that generates fixed-size continuous vector representations of words based on their context in the text. It captures semantic and syntactic relationships between words by using a shallow neural network trained on large text corpora. Word2Vec comes in two main architectures: Continuous Bag of Words (CBOW) and Skip-Gram.

The main advantage of Word2Vec over keyword-based models is its ability to capture semantic relationships between words in the vector space, enabling more accurate representation of textual meaning. However, Word2Vec embeddings are context-independent, meaning that each word has a single vector representation regardless of its context, which can be limiting for capturing nuances in meaning.

[^8]: Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. arXiv preprint arXiv:1301.3781.

#### GloVe

GloVe (Global Vectors for Word Representation) is another static word embedding model, introduced by Pennington, Socher, and Manning (2014)[^9]. It combines the advantages of both count-based and predictive methods for generating word embeddings. GloVe is trained on global word-word co-occurrence statistics from a large corpus, creating dense vector representations that capture linear substructures of the word vector space.

Compared to Word2Vec, GloVe offers improved performance on various NLP tasks due to its combination of global and local context information. However, it still suffers from the same context-independent limitation as Word2Vec.

[^9]: Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1532–1543.

#### FastText

FastText, developed by Bojanowski et al. (2017)[^10], is another static word embedding model that extends the Word2Vec approach. The key innovation of FastText is its ability to represent words as the sum of their subword embeddings, allowing it to capture morphological information and generate embeddings for out-of-vocabulary words. This makes FastText particularly suitable for languages with rich morphology or for handling rare words and misspellings.

Although FastText improves upon Word2Vec and GloVe by incorporating subword information, it still generates context-independent embeddings, which can be limiting for capturing the full range of semantic relationships between words.

[^10]: Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. Transactions of the Association for Computational Linguistics, 5, 135–146.

### Contextual Embedding Models

#### BERT

Bidirectional Encoder Representations from Transformers (BERT), introduced by Devlin et al. (2018)[^11], is a breakthrough in the field of NLP. BERT is a pre-trained deep bidirectional Transformer model that generates context-dependent embeddings, capturing the meaning of words based on their context in a sentence. The model is trained on a large corpus using unsupervised learning techniques, such as masked language modeling and next sentence prediction.

Compared to previous static embedding models, BERT's context-dependent embeddings offer a significant improvement in capturing complex semantic relationships between words, leading to state-of-the-art performance on a wide range of NLP tasks.

[^11]: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

#### SciBERT

SciBERT, introduced by Beltagy et al. (2019)[^12], is a variant of BERT specifically designed for scientific text. It is pre-trained on a large corpus of scientific publications, which allows it to capture domain-specific knowledge and terminology better than the general-domain BERT model. This makes SciBERT particularly suitable for tasks related to scientific literature, such as recommending computer science papers.

By leveraging domain-specific pre-training, SciBERT provides improved performance on scientific NLP tasks compared to the general-domain BERT model.

[^12]: Beltagy, I., Lo, K., & Cohan, A. (2019). SciBERT: A Pretrained Language Model for Scientific Text. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 3615–3620.

#### Longformer

Longformer, developed by Beltagy et al. (2020)[^13], is another contextual embedding model based on the Transformer architecture. It is designed to handle longer documents by using a novel self-attention mechanism called "sliding window attention," which scales linearly with sequence length. This allows Longformer to efficiently process and generate embeddings for documents that are significantly longer than those typically handled by BERT and other Transformer-based models.

Longformer's ability to handle longer documents makes it particularly suitable for tasks involving large-scale text analysis, such as generating embeddings for entire research papers, where capturing the full context is essential.

[^13]: Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. arXiv preprint arXiv:2004.05150.
