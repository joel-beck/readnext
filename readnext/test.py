from time import perf_counter

import polars as pl

from readnext.config import ResultsPaths

start = perf_counter()

fasttext_embeddings = pl.read_parquet(
    ResultsPaths.language_models.fasttext_embeddings_parquet,
)

fasttext_embeddings = fasttext_embeddings.lazy()

query_frame = fasttext_embeddings.select(["d3_document_id", "embedding"]).rename(
    {"d3_document_id": "query_d3_document_id", "embedding": "query_embedding"}
)
candidate_frame = fasttext_embeddings.select(["d3_document_id", "embedding"]).rename(
    {"d3_document_id": "candidate_d3_document_id", "embedding": "candidate_embedding"}
)

# TODO: Error since mltiplication not supperted for python lists
result = (
    query_frame.join(candidate_frame, how="cross")
    .with_columns(
        dot_product=pl.col("query_embedding").dot(pl.col("candidate_embedding")),
        query_embedding_norm=pl.col("query_embedding").mul(pl.col("query_embedding")).sum().sqrt(),
        candidate_embedding_norm=pl.col("candidate_embedding").mul(
            pl.col("candidate_embedding").sum().sqrt()
        ),
    )
    .with_columns(
        score=pl.col("dot_product")
        / (pl.col("query_embedding_norm") * pl.col("candidate_embedding_norm"))
    )
    .select(["query_d3_document_id", "candidate_d3_document_id", "score"])
    .sort(by=["query_d3_document_id", "score"], descending=[False, True])
    .groupby("query_d3_document_id")
    .head(100)
)

stop = perf_counter()

# print the execution time formatted as MM:SS
print(f"{(stop - start) / 60:02.0f}:{(stop - start) % 60:02.0f}")
