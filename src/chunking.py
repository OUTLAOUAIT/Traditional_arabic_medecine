"""Chapter-aware chunking and MongoDB ingestion for Arabic fasls."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from transformers import AutoTokenizer


EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNKING_VERSION = "v1"
CHUNK_MAX_TOKENS = 2048
OVERLAP_TARGET_TOKENS = 100

_PARAGRAPH_SEPARATOR = re.compile(r"\r?\n[ \t]*\r?\n+")
_TERMINAL_PUNCTUATION = frozenset(".!?؟۔…؛")
_QUOTE_PAIRS = {"«": "»", "“": "”", "‘": "’", "﴿": "﴾", '"': '"'}


@dataclass
class _GeneratedChunks:
    chunk_texts: list[str]
    core_texts: list[str]
    overlaps: list[str]
    token_counts: list[int]


def count_embedding_tokens(title: str, text: str, tokenizer: Any) -> int:
    """Count the complete, special-token-wrapped BGE-M3 embedding input."""
    embedding_input = f"{title}\n\n{text}"
    return len(tokenizer.encode(embedding_input, add_special_tokens=True))


def _paragraph_parts(text: str) -> list[tuple[str, str]]:
    """Return paragraph bodies and their exact following separators."""
    parts: list[tuple[str, str]] = []
    start = 0
    for match in _PARAGRAPH_SEPARATOR.finditer(text):
        body = text[start:match.start()]
        if body:
            parts.append((body, match.group(0)))
        elif parts:
            previous_body, previous_separator = parts[-1]
            parts[-1] = (previous_body, previous_separator + match.group(0))
        start = match.end()
    if start < len(text):
        parts.append((text[start:], ""))
    return parts


def _sentence_parts(text: str) -> list[str]:
    """Split at sentence endings outside quotation spans, retaining spacing."""
    if not text:
        return []

    pieces: list[str] = []
    quote_stack: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in _QUOTE_PAIRS:
            closing = _QUOTE_PAIRS[char]
            if char == '"' and quote_stack and quote_stack[-1] == '"':
                quote_stack.pop()
            else:
                quote_stack.append(closing)
            index += 1
            continue
        if quote_stack and char == quote_stack[-1]:
            quote_stack.pop()
            index += 1
            continue
        if char not in _TERMINAL_PUNCTUATION:
            index += 1
            continue

        end = index + 1
        while end < len(text) and text[end] in _TERMINAL_PUNCTUATION:
            end += 1

        remaining_quotes = list(quote_stack)
        quote_end = end
        while quote_end < len(text) and remaining_quotes and text[quote_end] == remaining_quotes[-1]:
            remaining_quotes.pop()
            quote_end += 1

        # Keep punctuation inside a quotation together through its closing quote.
        boundary_end = quote_end if not remaining_quotes else end
        if not remaining_quotes:
            while boundary_end < len(text) and text[boundary_end].isspace():
                boundary_end += 1
            if boundary_end > start:
                pieces.append(text[start:boundary_end])
                start = boundary_end
                index = boundary_end
                quote_stack.clear()
                continue
        index = end

    if start < len(text):
        pieces.append(text[start:])
    return [piece for piece in pieces if piece]


def _split_oversized_word(word: str, title: str, tokenizer: Any, max_tokens: int) -> list[str]:
    """Use BGE subword offsets if a single whitespace-delimited word is too large."""
    encoded = tokenizer(word, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [end for start, end in encoded["offset_mapping"] if end > start]
    pieces: list[str] = []
    start = 0
    while start < len(word):
        fitting_end = None
        for end in offsets:
            if end <= start:
                continue
            if count_embedding_tokens(title, word[start:end], tokenizer) <= max_tokens:
                fitting_end = end
            else:
                break
        if fitting_end is None:
            raise ValueError("A single tokenizer unit exceeds the configured chunk limit")
        pieces.append(word[start:fitting_end])
        start = fitting_end
    return pieces


def _split_at_word_boundaries(
    text: str, title: str, tokenizer: Any, max_tokens: int
) -> list[str]:
    """Split oversized text at whitespace-delimited word boundaries."""
    words = list(re.finditer(r"\S+", text))
    if not words:
        if count_embedding_tokens(title, text, tokenizer) > max_tokens:
            raise ValueError("Whitespace-only text exceeds the configured chunk limit")
        return [text]

    pieces: list[str] = []
    start = 0
    word_index = 0
    while word_index < len(words):
        fitting_end = None
        current_index = word_index
        while current_index < len(words):
            candidate_end = words[current_index].end()
            candidate = text[start:candidate_end]
            if count_embedding_tokens(title, candidate, tokenizer) > max_tokens:
                break
            fitting_end = candidate_end
            current_index += 1

        if fitting_end is not None:
            if current_index == len(words):
                # Keep any final whitespace as part of the final piece.
                pieces.append(text[start:])
                break
            pieces.append(text[start:fitting_end])
            start = fitting_end
            word_index = current_index
            continue

        # No complete word fits: use the tokenizer's exact subword boundaries.
        word = words[word_index]
        prefix = text[start:word.start()]
        subpieces = _split_oversized_word(word.group(0), title, tokenizer, max_tokens)
        if prefix:
            if count_embedding_tokens(title, prefix + subpieces[0], tokenizer) <= max_tokens:
                subpieces[0] = prefix + subpieces[0]
            else:
                pieces.append(prefix)
        for subpiece in subpieces[:-1]:
            pieces.append(subpiece)
        start = word.end()
        word_index += 1
        # Keep processing later words with the current accumulated tail.
        if subpieces:
            pieces.append(subpieces[-1])

    if start < len(text) and (not pieces or "".join(pieces) != text):
        # Normally consumed by the final complete-word branch; this handles a
        # trailing substring after a subword fallback without dropping text.
        consumed = sum(len(piece) for piece in pieces)
        pieces.append(text[consumed:])
    return pieces


def _pack_core_chunks(
    title: str, text: str, tokenizer: Any, max_tokens: int
) -> list[list[str]]:
    if count_embedding_tokens(title, "", tokenizer) > max_tokens:
        raise ValueError(f"Title alone exceeds the {max_tokens}-token limit")
    if not text.strip():
        raise ValueError("Cannot create an empty chunk from an empty fasl")

    chunks: list[list[str]] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append(list(current))
            current.clear()

    for body, separator in _paragraph_parts(text):
        paragraph = body + separator
        paragraph_tokens = count_embedding_tokens(title, paragraph, tokenizer)
        if paragraph_tokens > max_tokens:
            raise ValueError(
                "A complete paragraph exceeds the configured embedding limit "
                f"({paragraph_tokens} > {max_tokens} tokens); refusing to split it"
            )
        if count_embedding_tokens(title, "".join(current) + paragraph, tokenizer) <= max_tokens:
            current.append(paragraph)
        else:
            # Start the next chunk at the beginning of a paragraph, even when
            # that leaves unused room in the current chunk.
            flush()
            current.append(paragraph)

    flush()

    # Rebalance a very short final core by moving preceding complete units when
    # both resulting embedding inputs remain within the hard token limit.
    if len(chunks) > 1:
        short_threshold = max_tokens // 4
        while len(chunks[-2]) > 1 and count_embedding_tokens(
            title, "".join(chunks[-1]), tokenizer
        ) < short_threshold:
            moved = chunks[-2][-1]
            candidate_previous = "".join(chunks[-2][:-1])
            candidate_last = moved + "".join(chunks[-1])
            if (
                count_embedding_tokens(title, candidate_previous, tokenizer) > max_tokens
                or count_embedding_tokens(title, candidate_last, tokenizer) > max_tokens
            ):
                break
            chunks[-2].pop()
            chunks[-1].insert(0, moved)

    return chunks


def _overlap_candidates(
    previous_core: str,
    next_core: str,
    title: str,
    tokenizer: Any,
    max_tokens: int,
    target: int,
):
    # Overlap is made only from complete trailing paragraphs. If the previous
    # chunk has no paragraph-sized suffix near the target (or there's no room),
    # omit overlap rather than begin a chunk mid-paragraph.
    paragraphs = _paragraph_parts(previous_core)
    candidates = [""]
    suffix = ""
    for body, separator in reversed(paragraphs):
        suffix = body + separator + suffix
        candidates.append(suffix)

    fitting = []
    for candidate in candidates:
        token_count = len(tokenizer.encode(candidate, add_special_tokens=False))
        if token_count > max(target * 2, target + 128):
            continue
        if count_embedding_tokens(title, candidate + next_core, tokenizer) > max_tokens:
            continue
        fitting.append((abs(token_count - target), candidate))
    return min(fitting)[1] if fitting else ""


def _generate_chunks(
    fasl: dict[str, Any],
    tokenizer: Any,
    *,
    max_tokens: int,
    overlap_target_tokens: int,
) -> _GeneratedChunks:
    title = str(fasl["fasl_title"])
    source_text = str(fasl["fasl_content"])
    core_parts = _pack_core_chunks(title, source_text, tokenizer, max_tokens)
    core_texts = ["".join(parts) for parts in core_parts]
    if "".join(core_texts) != source_text:
        raise ValueError(f"Chunk cores do not cover fasl {fasl.get('_id')} exactly")

    chunk_texts: list[str] = []
    overlaps: list[str] = []
    token_counts: list[int] = []
    for index, core_text in enumerate(core_texts):
        overlap = ""
        if index:
            overlap = _overlap_candidates(
                core_texts[index - 1],
                core_text,
                title,
                tokenizer,
                max_tokens,
                overlap_target_tokens,
            )
        chunk_text = overlap + core_text
        token_count = count_embedding_tokens(title, chunk_text, tokenizer)
        if not chunk_text.strip():
            raise ValueError(f"Generated an empty chunk for fasl {fasl.get('_id')}")
        if token_count > max_tokens:
            raise ValueError(f"Chunk for fasl {fasl.get('_id')} exceeds {max_tokens} tokens")
        chunk_texts.append(chunk_text)
        overlaps.append(overlap)
        token_counts.append(token_count)

    # Removing the intentional overlap prefixes must recover the exact source.
    reconstructed = "".join(
        chunk_text[len(overlap):] for chunk_text, overlap in zip(chunk_texts, overlaps)
    )
    if reconstructed != source_text:
        raise ValueError(f"Generated chunks fail source coverage for fasl {fasl.get('_id')}")

    return _GeneratedChunks(chunk_texts, core_texts, overlaps, token_counts)


def ingest_chunks_to_mongodb(
    *,
    fasls_collection_name: str | None = None,
    chunks_collection_name: str = "chunks",
    model_name: str = EMBEDDING_MODEL,
    chunking_version: str = CHUNKING_VERSION,
    chunk_max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_target_tokens: int = OVERLAP_TARGET_TOKENS,
    tokenizer: Any | None = None,
) -> dict[str, int]:
    """Chunk every fasl and idempotently upsert chunks into MongoDB.

    Stale chunks are deleted per fasl/version only after all replacement chunks
    for that fasl have been generated, validated, and upserted successfully.
    """
    if chunk_max_tokens <= 0 or overlap_target_tokens < 0:
        raise ValueError("Chunk limits must be positive and overlap non-negative")

    load_dotenv()
    connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    database_name = os.getenv("DATA_BASE")
    source_name = fasls_collection_name or os.getenv("FASLS_COLLECTION_NAME") or os.getenv("COLLECTION_NAME") or "fasls"
    target_name = os.getenv("CHUNKS_COLLECTION_NAME") or chunks_collection_name
    if not connection_string or not database_name:
        raise RuntimeError("MONGODB_CONNECTION_STRING and DATA_BASE must be configured in .env")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, local_files_only=True
        )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast BGE-M3 tokenizer is required for exact token counts")

    with MongoClient(connection_string, serverSelectionTimeoutMS=10000) as client:
        database = client[database_name]
        if source_name not in database.list_collection_names():
            raise RuntimeError(
                f"Fasls collection '{source_name}' does not exist in database '{database_name}'"
            )
        fasls_collection = database[source_name]
        chunks_collection = database[target_name]
        fasls = list(fasls_collection.find({}))
        chunks_collection.create_index(
            [("fasl_id", 1), ("chunking_version", 1), ("chunk_index", 1)],
            unique=True,
            name="fasl_version_chunk_unique",
        )

        # Generate and validate every fasl before writing any of its chunks.
        generated = []
        for fasl in fasls:
            result = _generate_chunks(
                fasl,
                tokenizer,
                max_tokens=chunk_max_tokens,
                overlap_target_tokens=overlap_target_tokens,
            )
            generated.append((fasl, result))

        upserted = 0
        modified = 0
        stale_deleted = 0
        chunks_written = 0
        for fasl, result in generated:
            fasl_id = fasl["_id"]
            operations = []
            for index, (text, token_count) in enumerate(zip(result.chunk_texts, result.token_counts)):
                fields = {
                    "fasl_id": fasl_id,
                    "chunk_index": index,
                    "title": fasl["fasl_title"],
                    "source_url": fasl.get("fasl_url", ""),
                    "text": text,
                    "embedding_token_count": token_count,
                    "embedding_model": model_name,
                    "chunking_version": chunking_version,
                    "chunk_max_tokens": chunk_max_tokens,
                    "overlap_target_tokens": overlap_target_tokens,
                }
                operations.append(
                    UpdateOne(
                        {
                            "fasl_id": fasl_id,
                            "chunking_version": chunking_version,
                            "chunk_index": index,
                        },
                        {"$set": fields, "$setOnInsert": {"_id": ObjectId()}},
                        upsert=True,
                    )
                )

            result_write = chunks_collection.bulk_write(operations, ordered=True)
            upserted += result_write.upserted_count
            modified += result_write.modified_count
            chunks_written += len(operations)

            # Only remove stale siblings after replacement writes succeeded.
            stale_deleted += chunks_collection.delete_many(
                {
                    "fasl_id": fasl_id,
                    "chunking_version": chunking_version,
                    "chunk_index": {"$gte": len(operations)},
                }
            ).deleted_count

    return {
        "fasls_processed": len(fasls),
        "chunks_written": chunks_written,
        "chunks_upserted": upserted,
        "chunks_modified": modified,
        "stale_chunks_deleted": stale_deleted,
    }


if __name__ == "__main__":
    print(ingest_chunks_to_mongodb())
