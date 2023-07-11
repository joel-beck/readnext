# Customization

This chapter guides you through the process of adding custom components to the
`readnext` package tailored to your needs. While certain parts of the codebase
are not intended for modification to maintain lean code (e.g., the features of
the Citation Recommender), others are designed for easy extension.

We utilize the [Strategy Pattern](https://refactoring.guru/design-patterns/strategy)
to ensure adherence to the open-closed principle: New components can be added
without modifying any of the existing code.

In this chapter, we present three hands-on examples where customizing the
`readnext` package is particularly straightforward. We will:

- Add a new tokenizer for the Language Recommender.
- Add a new language model for the Language Recommender.
- Add a new evaluation metric.

By implementing a predefined interface, the new components can serve as
**drop-in replacements** for the existing ones throughout the codebase.


## Adding Evaluation Metrics

All evaluation metrics for a list of recommendations inherit from the abstract
`EvaluationMetric` class.

!!! note "Source Code"

    Explore the source code of the `EvaluationMetric` class
    [here](https://github.com/joel-beck/readnext/blob/main/readnext/evaluation/metrics/evaluation_metric.py).

To comply with the interface of the parent class, all evaluation metrics must
implement the `score()` and `from_df()` methods:

- The `score()` method accepts any generic
  [Sequence type](https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes)
  or polars Series of integers or strings as input and computes a scalar value
  from it.
- The `from_df()` method takes the original documents dataframe with label
  columns `arxiv_labels` and `integer_label` as input and applies the
  `score()` method to one of these columns.


### Example

As an illustrative example, we implement the new `FirstRelevant` evaluation
metric that takes a list of 0/1 for irrelevant/relevant recommendations as input
and returns the position of the first relevant recommendation.

Thus, lower (positive) values are better: If the top recommendation is indeed
relevant, the score is 1, if the second recommendation is relevant but the first
is not, the score is 2, and so on. If none of the recommendations are relevant,
the score is set to 0.

The following code snippet shows the implementation of the `FirstRelevant` class:

```python
from dataclasses import dataclass

from readnext.evaluation.metrics import EvaluationMetric
from readnext.utils.aliases import DocumentsFrame

@dataclass
class FirstRelevant(EvaluationMetric):
    """
    Computes the position of the first relevant item in a list of integer
    recommendations.
    """

    @classmethod
    def score(cls, label_list: IntegerLabelList) -> int:
        if not len(label_list):
            return 0

        for k, label in enumerate(label_list, 1):
            if label == 1:
                return k

        return 0

    @classmethod
    def from_df(cls, df: DocumentsFrame) -> int:
        return cls.score(df["integer_label"])
```

The `score()` method first checks if the provided sequence is empty (in which
case it returns early) and then uses the native `enumerate()` function to check
if any label is equal to one. If so, this label position is returned. If not,
the score is set to 0.

!!! info "Info"

    The `score()` method is one of the rare cases where we could have used the
    `else` keyword of a `for` loop: If the loop is not exited early, i.e. if no
    label is equal to 1, the `else` block is executed and returns 0.

The `from_df()` method simply calls the `score()` method on the `integer_label`
column of the documents dataframe.

Our new `FirstRelevant` metric can now be used anywhere in the codebase in place
of the existing `AveragePrecision` and `CountUniqueLabels` evaluation metrics.


## Adding Tokenizers

In contrast to the case for evaluation metrics, tokenizers can inherit from two
different abstract base classes, depending on whether they tokenize text into
string tokens (such as `spacy`) or into integer token ids (such as all
tokenizers from the `transformers` library):

1. If they tokenize text into string tokens, they inherit from the `Tokenizer`
   class. To comply with the interface of the parent class, they must implement
   the `tokenize_single_document()` method, which takes a single string as input
   and returns a list of string tokens. Then, they inherit the `tokenize()`
   method which is used to construct a polars dataframe with a `tokens` column
   that contains the tokenized documents.
2. If they tokenize text into integer token ids, they inherit from the
   `TorchTokenizer` class. Similar to the case above, all child classes must
   implement a `tokenize_into_ids()` method that either takes a single string
   as input and returns a list of integer token ids or a list of strings as input
   and returns a list of lists of integer token ids. Then, they inherit the
   `tokenize()` method which is used to construct a polars dataframe with a
   `token_ids` column that contains the tokenized documents.

!!! note "Source Code"

    Take a look at the source code of the `Tokenizer` class
    [here](https://github.com/joel-beck/readnext/blob/main/readnext/modeling/language_models/tokenizer.py)
    and the source code of the `TorchTokenizer` class
    [here](https://github.com/joel-beck/readnext/blob/main/readnext/modeling/language_models/tokenizer_torch.py).


### Example

As an example, we add the `NaiveTokenizer` that simply converts all characters
to lowercase and splits the text at whitespace characters.

Since the `NaiveTokenizer` tokenizes text into string tokens, it inherits from
the `Tokenizer` class and must implement the `tokenize_single_document()` method:

```python
from dataclasses import dataclass

from readnext.utils.aliases import Tokens
from readnext.modeling.language_models import Tokenizer

@dataclass
class NaiveTokenizer(Tokenizer):
    """
    Naive tokenizer that splits text at whitespace characters and converts all
    characters to lowercase.
    """

    def tokenize_single_document(self, document: str) -> Tokens:
        return document.lower().split()
```

The new `NaiveTokenizer` can now be used anywhere in the codebase in place of the
existing `SpacyTokenizer`.


## Adding Language Models

Similar to Tokenizers, Language Models can inherit from two different abstract
base classes, depending on whether they are a `transformers` model or not:

- If they are not a `transformers` model, they inherit from the `Embedder` class.
  In this case, they must implement the `compute_embedding_single_document()`
  and `compute_embeddings_frame()` methods. The former takes a `Tokens`
  (a list of strings) as input and returns an `Embedding` (a list of floats).
  The latter uses the `tokens_frame` from the instance initialization and returns
  a polars DataFrame with two columns named `d3_document_id` and `embedding`.
- If they are a `transformers` model, they inherit from the `TorchEmbedder`
  class. In this case, they do not have to implement any methods in the child
  class since the interface for all `transformers` models is identical. For the
  purpose of correct typing, any child class must only define the `torch_model`
  attribute with the correct type annotation.

!!! note "Source Code"

    Take a look at the source code of the `Embedder` class
    [here](https://github.com/joel-beck/readnext/blob/main/readnext/modeling/language_models/embedder.py)
    and the source code of the `TorchEmbedder` class
    [here](https://github.com/joel-beck/readnext/blob/main/readnext/modeling/language_models/embedder_torch.py).


### Example

As an example, we add the `DummyEmbedder` that simply returns a list of zeros
for each document. To simulate a 300-dimensional embedding, we set the length of
the list to 300.

Since the `DummyEmbedder` is not a `transformers` model, it inherits from the
`Embedder` class and must implement the `compute_embedding_single_document()`
and `compute_embeddings_frame()` methods:

```python
from dataclasses import dataclass

from readnext.utils.aliases import EmbeddingsFrame, Tokens
from readnext.modeling.language_models import Embedder

@dataclass
class DummyEmbedder(Embedder):
    """
    Dummy embedder that returns a list of zeros for each document.
    """

    def compute_embedding_single_document(self, document_tokens: Tokens) -> Embedding:
        # although the `document_tokens` argument is not used, it is necessary
        # to implement the interface properly
        return [0] * 300

    def compute_embeddings_frame(self) -> EmbeddingsFrame:
        return self.tokens_frame.with_columns(
            embedding=pl.Series([[0] * 300] * self.tokens_frame.height)
        ).drop("tokens")
```

The new `DummyEmbedder` can now be used anywhere in the codebase in place of any
existing embedder that is not based on a `transformers` model. This includes the
`TFIDFEmbedder`, `BM25Embedder`, `Word2VecEmbedder`, `GloveEmbedder`, and
`FastTextEmbedder` classes.
