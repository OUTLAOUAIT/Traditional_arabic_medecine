"""Generate BGE-M3 vectors for MongoDB chunks and store/search them in Weaviate."""

from __future__ import annotations

import hashlib
import os
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from weaviate.classes.query import MetadataQuery

from src.weaviate_connection import connect_weaviate


DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_COLLECTION_NAME = "TibbChunksBgeM3"
VECTOR_NAME = "dense"


def _embedding_input(title: str, text: str) -> str:
    return f"{title}\n\n{text}"


def _object_uuid(mongo_chunk_id: Any) -> UUID:
    """Return the same Weaviate UUID for a given MongoDB chunk ID."""
    return uuid5(NAMESPACE_URL, f"traditional-arabic-medicine:chunk:{mongo_chunk_id}")


def _input_hash(model_name: str, embedding_input: str) -> str:
    return hashlib.sha256(
        f"{model_name}\0{embedding_input}".encode("utf-8")
    ).hexdigest()


def _load_model(model_name: str) -> SentenceTransformer:
    # The first call downloads the model from Hugging Face; later runs use its cache.
    return SentenceTransformer(model_name)


def sync_mongodb_chunks_to_weaviate(
    *,
    mongo_chunks_collection_name: str | None = None,
    weaviate_collection_name: str | None = None,
    model_name: str | None = None,
    batch_size: int | None = None,
    model: SentenceTransformer | None = None,
) -> dict[str, int]:
    """Embed MongoDB chunks and idempotently upsert their vectors into Weaviate.

    MongoDB remains the source of truth. Stable UUIDs ensure that rerunning this
    function replaces the matching Weaviate objects instead of duplicating them.
    """
    load_dotenv()
    mongo_uri = os.getenv("MONGODB_CONNECTION_STRING")
    database_name = os.getenv("DATA_BASE")
    mongo_collection_name = (
        mongo_chunks_collection_name
        or os.getenv("CHUNKS_COLLECTION_NAME")
        or "chunks"
    )
    destination_collection_name = (
        weaviate_collection_name
        or os.getenv("WEAVIATE_COLLECTION_NAME")
        or DEFAULT_COLLECTION_NAME
    )
    selected_model_name = model_name or os.getenv(
        "EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME
    )
    selected_batch_size = (
        batch_size
        if batch_size is not None
        else int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    )

    if not mongo_uri or not database_name:
        raise RuntimeError("MONGODB_CONNECTION_STRING and DATA_BASE must be configured in .env")
    if selected_batch_size <= 0:
        raise ValueError("Embedding batch size must be a positive integer")

    if model is None:
        model = _load_model(selected_model_name)

    with MongoClient(mongo_uri, serverSelectionTimeoutMS=10000) as mongo_client:
        mongo_collection = mongo_client[database_name][mongo_collection_name]
        chunks = list(mongo_collection.find({}).sort([("fasl_id", 1), ("chunk_index", 1)]))

    if not chunks:
        return {"chunks_read": 0, "vectors_written": 0}

    with connect_weaviate() as weaviate_client:
        if not weaviate_client.is_ready():
            raise RuntimeError(
                "Weaviate is not ready. Check the Cloud endpoint/API key or start the local service."
            )
        if not weaviate_client.collections.exists(destination_collection_name):
            raise RuntimeError(
                f"Weaviate collection '{destination_collection_name}' does not exist. "
                "Create it first with `python setup_weaviate.py`."
            )

        collection = weaviate_client.collections.get(destination_collection_name)
        written = 0
        with collection.batch.fixed_size(batch_size=selected_batch_size) as batch:
            for offset in range(0, len(chunks), selected_batch_size):
                chunk_batch = chunks[offset : offset + selected_batch_size]
                inputs = [
                    _embedding_input(str(chunk["title"]), str(chunk["text"]))
                    for chunk in chunk_batch
                ]
                vectors = model.encode(
                    inputs,
                    batch_size=selected_batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

                for chunk, embedding_input, vector in zip(chunk_batch, inputs, vectors):
                    title = str(chunk["title"])
                    text = str(chunk["text"])
                    properties = {
                        "mongo_chunk_id": str(chunk["_id"]),
                        "fasl_id": str(chunk["fasl_id"]),
                        "title": title,
                        "text": text,
                        "source_url": str(chunk.get("source_url", "")),
                        "chunk_index": int(chunk["chunk_index"]),
                        "chunking_version": str(chunk.get("chunking_version", "v1")),
                        "embedding_input_hash": _input_hash(
                            selected_model_name, embedding_input
                        ),
                        "embedding_model": selected_model_name,
                        "embedding_token_count": int(chunk["embedding_token_count"]),
                    }
                    batch.add_object(
                        properties=properties,
                        uuid=_object_uuid(chunk["_id"]),
                        vector={VECTOR_NAME: vector.tolist()},
                    )
                    written += 1

        failed_objects = collection.batch.failed_objects
        if failed_objects:
            first_error = failed_objects[0].message
            raise RuntimeError(
                f"Weaviate failed to store {len(failed_objects)} chunks. "
                f"First error: {first_error}"
            )

    return {"chunks_read": len(chunks), "vectors_written": written}


def search_chunks(
    query: str,
    *,
    limit: int = 5,
    unique_fasls: bool = True,
    weaviate_collection_name: str | None = None,
    model_name: str | None = None,
    model: SentenceTransformer | None = None,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Return nearest BGE-M3 chunks, optionally keeping only the best per fasl.

    By default, results contain at most one chunk per fasl so one long fasl
    cannot crowd other relevant fasls out of the result list. Set
    ``unique_fasls=False`` to retrieve several chunks from the same fasl.

    Pass an open ``client`` (e.g. from a long-running API) to reuse its
    connection; it is then not checked or closed here. Otherwise a connection
    is opened, checked, and closed for this one search.
    """
    if not query.strip():
        raise ValueError("Search query cannot be empty")
    if limit <= 0:
        raise ValueError("Search limit must be a positive integer")

    load_dotenv()
    destination_collection_name = (
        weaviate_collection_name
        or os.getenv("WEAVIATE_COLLECTION_NAME")
        or DEFAULT_COLLECTION_NAME
    )
    selected_model_name = model_name or os.getenv(
        "EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME
    )
    if model is None:
        model = _load_model(selected_model_name)

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].tolist()

    if client is not None:
        return _nearest_chunks(
            client, destination_collection_name, query_vector, limit, unique_fasls
        )

    with connect_weaviate() as own_client:
        check_collection(own_client, destination_collection_name)
        return _nearest_chunks(
            own_client, destination_collection_name, query_vector, limit, unique_fasls
        )


def check_collection(client: Any, collection_name: str) -> None:
    """Raise a clear error if Weaviate is down or the collection is missing."""
    if not client.is_ready():
        raise RuntimeError(
            "Weaviate is not ready. Check the Cloud endpoint/API key or start the local service."
        )
    if not client.collections.exists(collection_name):
        raise RuntimeError(
            f"Weaviate collection '{collection_name}' does not exist. "
            "Create it first with `python setup_weaviate.py`."
        )


def _nearest_chunks(
    client: Any,
    collection_name: str,
    query_vector: list[float],
    limit: int,
    unique_fasls: bool,
) -> list[dict[str, Any]]:
    collection = client.collections.get(collection_name)
    candidate_limit = min(max(limit * 10, limit), 1000) if unique_fasls else limit
    response = collection.query.near_vector(
        near_vector=query_vector,
        target_vector=VECTOR_NAME,
        limit=candidate_limit,
        return_metadata=MetadataQuery(distance=True),
    )
    results = []
    seen_fasls: set[str] = set()
    for obj in response.objects:
        result = {
            **obj.properties,
            "distance": obj.metadata.distance if obj.metadata else None,
        }
        if unique_fasls:
            fasl_id = str(result.get("fasl_id", ""))
            if fasl_id in seen_fasls:
                continue
            seen_fasls.add(fasl_id)
        results.append(result)
        if len(results) == limit:
            break
    return results


if __name__ == "__main__":
    print(sync_mongodb_chunks_to_weaviate())
