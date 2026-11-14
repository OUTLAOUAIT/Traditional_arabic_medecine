"""Create the self-provided-vector collection used by the BGE-M3 pipeline."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
)

from src.weaviate_connection import connect_weaviate


DEFAULT_COLLECTION_NAME = "TibbChunksBgeM3"


def setup_weaviate_collection() -> str:
    load_dotenv()
    collection_name = os.getenv("WEAVIATE_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)

    with connect_weaviate() as client:
        if not client.is_ready():
            raise RuntimeError(
                "Weaviate is not ready. Check the Cloud endpoint/API key or start the local service."
            )

        if client.collections.exists(collection_name):
            print(f"{collection_name} already exists.")
            return collection_name

        client.collections.create(
            name=collection_name,
            # Let Weaviate Cloud's free-tier profile choose its supported index.
            # Local Weaviate uses its default HNSW index.
            vector_config=Configure.Vectors.self_provided(name="dense"),
            properties=[
                Property(
                    name="mongo_chunk_id",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(
                    name="fasl_id",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
                Property(
                    name="source_url",
                    data_type=DataType.TEXT,
                    index_searchable=False,
                ),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(
                    name="chunking_version",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(name="embedding_input_hash", data_type=DataType.TEXT),
                Property(name="embedding_model", data_type=DataType.TEXT),
                Property(name="embedding_token_count", data_type=DataType.INT),
            ],
        )

    print(f"Created {collection_name}.")
    return collection_name


if __name__ == "__main__":
    setup_weaviate_collection()
