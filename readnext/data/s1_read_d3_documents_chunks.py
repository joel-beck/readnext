"""
Read documents metadata from the D3 dataset. The data can be downloaded from
https://zenodo.org/record/7071698#.ZA7jbC8w2Lc. This project uses version 2.1, published
on 2022-11-25.

Since the documents dataset is too large to process at once, it is split into chunks of
100,000 documents
"""

import pandas as pd

from readnext.config import DataPaths
from readnext.utils import setup_progress_bar


def write_out_chunked_dataframe(dataframe_chunks: list[pd.DataFrame], file_index: int) -> None:
    print("\nConcatenating dataframes...")
    dataframe = pd.concat(dataframe_chunks)

    file_path = f"{DataPaths.d3.documents.chunks_stem}_{file_index}.pkl"
    print(f"Writing to {file_path}\n\n{'-' * 50}\n")
    dataframe.to_pickle(file_path)


def main() -> None:
    chunksize = 100_000
    chunks_per_file = 10

    # split entire documents file into chunks
    json_reader_chunks = pd.read_json(
        DataPaths.d3.documents.raw_json, lines=True, chunksize=chunksize
    )

    dataframe_chunks = []
    file_index = 1

    with setup_progress_bar() as progress_bar:
        for i, chunk in progress_bar.track(enumerate(json_reader_chunks, 1)):
            dataframe_chunks.append(chunk)
            print(f"Read {len(dataframe_chunks) * chunksize} documents")

            if i % chunks_per_file == 0:
                write_out_chunked_dataframe(dataframe_chunks, file_index)

                # remove already processed chunks and increment file index
                dataframe_chunks = []
                file_index += 1


if __name__ == "__main__":
    main()
