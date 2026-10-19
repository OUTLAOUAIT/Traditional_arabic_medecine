"""Utilities for turning the raw Arabic source text into fasls."""

from __future__ import annotations

import re
import hashlib
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from pymongo import MongoClient, ReplaceOne
import requests
from bs4 import BeautifulSoup
from pyarabic.araby import tokenize as tokenize_arabic

from schemas.data_schema import FaslSchema


_FASL_START = re.compile(r"^(?:فصل|فصول)(?:\s|$)|^في\s+هديه(?:\s|$)")
_LETTER_ENTRY = re.compile(r"^(?:حرف|الحرف)\s+\S+\s*$")
_WIKISOURCE_ARTICLE_URL = "https://ar.wikisource.org/wiki/الطب_النبوي"


def build_fasl_url(fasl_title: str) -> str:
    """Compose a Wikisource article URL with the fasl title as its fragment."""
    anchor = quote(fasl_title.strip().replace(" ", "_"), safe="_-")
    return f"{_WIKISOURCE_ARTICLE_URL}#{anchor}"


def normalize_fasl_text(text: str) -> str:
    """Create a conservative Arabic search copy while retaining paragraphs.

    Removes Arabic combining diacritics and tatweel, collapses whitespace
    within each paragraph, and preserves paragraph boundaries and punctuation.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_chars = []
    for char in text:
        if char == "\u0640":  # Arabic tatweel
            continue
        if unicodedata.category(char).startswith("M") and "ARABIC" in unicodedata.name(char, ""):
            continue
        normalized_chars.append(char)

    paragraphs = re.split(r"\n\s*\n+", "".join(normalized_chars).strip())
    paragraphs = [re.sub(r"\s+", " ", paragraph).strip() for paragraph in paragraphs]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def count_arabic_tokens(text: str) -> int:
    """Count Arabic-aware word and punctuation tokens with PyArabic."""
    return len(tokenize_arabic(text))


def resolve_fasl_urls(
    fasls: list[FaslSchema],
    *,
    timeout: int = 30,
) -> tuple[int, int]:
    """Set canonical heading URLs, using the article URL when no heading exists.

    Returns the number of fasls linked to a section heading and the number
    using the base article URL as a fallback.
    """
    response = requests.get(
        "https://ar.wikisource.org/w/api.php",
        params={
            "action": "parse",
            "page": "الطب النبوي",
            "prop": "text",
            "format": "json",
            "formatversion": "2",
        },
        headers={"User-Agent": "TraditionalArabicMedicineParser/1.0 (research)"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"MediaWiki API error: {payload['error']}")

    soup = BeautifulSoup(payload.get("parse", {}).get("text", ""), "html.parser")
    heading_anchors: dict[str, list[str]] = {}
    for heading in soup.find_all(["h2", "h3", "h4", "h5", "h6"]):
        anchor = heading.get("id")
        normalized_title = _normalize_heading(heading.get_text(" ", strip=True))
        if anchor and normalized_title:
            heading_anchors.setdefault(normalized_title, []).append(anchor)

    used_anchors: set[str] = set()
    section_links = 0
    article_fallbacks = 0
    for fasl in fasls:
        candidates = heading_anchors.get(_normalize_heading(fasl.fasl_title), [])
        if not candidates:
            _, separator, letter_title = fasl.fasl_title.rpartition(" : ")
            if separator and _LETTER_ENTRY.fullmatch(letter_title):
                candidates = heading_anchors.get(_normalize_heading(letter_title), [])
        anchor = next((item for item in candidates if item not in used_anchors), None)
        if anchor:
            # MediaWiki section links use underscores for spaces in fragments.
            anchor = anchor.replace(" ", "_")
            fasl.fasl_url = f"{_WIKISOURCE_ARTICLE_URL}#{quote(anchor, safe='_-')}"
            used_anchors.add(anchor)
            section_links += 1
        else:
            fasl.fasl_url = _WIKISOURCE_ARTICLE_URL
            article_fallbacks += 1
    return section_links, article_fallbacks


def split_fasls(text: str) -> list[FaslSchema]:
    """Split text into fasls when a paragraph begins with a fasl marker.

    A new fasl starts at a paragraph beginning with ``فصل``, ``فصول``, or
    ``في هديه``. Two-word alphabetic headings beginning with ``حرف`` or
    ``الحرف`` also start fasls, prefixed by their parent fasl title. Text
    before the first such paragraph is treated as the article preface.
    """
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\r?\n\s*\r?\n", text.strip())
        if paragraph.strip()
    ]

    fasl_paragraphs: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_paragraphs: list[str] = []
    parent_title: str | None = None

    def flush_current() -> None:
        if current_title is not None and current_paragraphs:
            fasl_paragraphs.append((current_title, list(current_paragraphs)))

    for paragraph in paragraphs:
        first_line = paragraph.splitlines()[0].strip()
        if _FASL_START.match(first_line):
            flush_current()
            current_title = first_line
            parent_title = first_line
            current_paragraphs = [paragraph]
        elif _LETTER_ENTRY.fullmatch(first_line) and parent_title:
            flush_current()
            current_title = f"{parent_title} : {first_line}"
            # Store the combined title in place of the short letter heading.
            current_paragraphs = [current_title + paragraph[len(first_line):]]
        elif current_title is not None:
            current_paragraphs.append(paragraph)

    flush_current()

    created_at = datetime.now(timezone.utc)
    fasls = []
    for title, content_paragraphs in fasl_paragraphs:
        content = "\n\n".join(content_paragraphs)
        normalized_content = normalize_fasl_text(content)
        fasls.append(
            FaslSchema(
                fasl_content=content,
                normalized_fasl=normalized_content,
                number_tokens=count_arabic_tokens(normalized_content),
                fasl_title=title,
                fasl_url=build_fasl_url(title),
                created_at=created_at,
            )
        )
    return fasls


def split_raw_data_file(
    path: str | Path = "data/raw_data.txt",
) -> list[FaslSchema]:
    """Read a UTF-8 raw data file and return its fasls."""
    return split_fasls(Path(path).read_text(encoding="utf-8"))


def save_fasls_to_mongodb(fasls: list[FaslSchema]) -> int:
    """Upsert fasls into the database and collection configured in ``.env``.

    Uses ``MONGODB_CONNECTION_STRING``, ``DATA_BASE``, and
    ``COLLECTION_NAME``. Each fasl is stored as one MongoDB document. A stable
    hash of its content is used as ``_id`` so rerunning the save won't create
    duplicate documents for unchanged fasls. Before saving, it resolves each
    URL against Wikisource's live heading IDs and uses the article URL as a
    fallback when the fasl title has no matching heading.

    Returns the number of documents newly inserted by the upsert operation.
    """
    resolve_fasl_urls(fasls)
    load_dotenv()
    connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    database_name = os.getenv("DATA_BASE")
    collection_name = os.getenv("COLLECTION_NAME")
    missing = [
        name
        for name, value in (
            ("MONGODB_CONNECTION_STRING", connection_string),
            ("DATA_BASE", database_name),
            ("COLLECTION_NAME", collection_name),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing MongoDB configuration in .env: {', '.join(missing)}")

    operations = []
    for fasl in fasls:
        document = fasl.model_dump(mode="python")
        document["_id"] = hashlib.sha256(
            fasl.fasl_content.encode("utf-8")
        ).hexdigest()
        operations.append(
            ReplaceOne({"_id": document["_id"]}, document, upsert=True)
        )

    if not operations:
        return 0

    with MongoClient(connection_string, serverSelectionTimeoutMS=5000) as client:
        result = client[database_name][collection_name].bulk_write(
            operations, ordered=False
        )
    return result.upserted_count


def _normalize_heading(value: str) -> str:
    """Normalize displayed headings for comparison with raw fasl titles."""
    value = BeautifulSoup(value.replace("_", " "), "html.parser").get_text(" ")
    value = value.replace("ﷺ", "صلى الله عليه وسلم")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split()).casefold()
