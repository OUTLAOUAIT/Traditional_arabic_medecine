"""Fetch readable text from a Wikisource article via the MediaWiki API."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup


def scrape_wikisource_text(
    url: str,
    *,
    section: str | None = None,
    timeout: int = 30,
) -> str:
    """Return an article's text, or a selected section's text.

    Args:
        url: Wikisource article URL. URL fragments are ignored so the full
            article is returned by default.
        section: Optional section title or anchor. When omitted, the full
            article is returned.
        timeout: HTTP request timeout in seconds.

    Install dependencies with ``python -m pip install requests beautifulsoup4``.
    """
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Expected an absolute Wikisource URL, got: {url!r}")

    api_url = f"{parsed_url.scheme}://{parsed_url.netloc}/w/api.php"
    page_title = unquote(parsed_url.path.rsplit("/", 1)[-1]).replace("_", " ")
    session = requests.Session()
    session.headers.update({"User-Agent": "WikisourceTextScraper/1.0 (personal research)"})

    def api_get(params: dict[str, str]) -> dict:
        response = session.get(api_url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"MediaWiki API error: {payload['error']}")
        return payload

    base_params = {
        "action": "parse",
        "page": page_title,
        "prop": "text|sections",
        "format": "json",
        "formatversion": "2",
    }

    if section:
        parsed = api_get(base_params)
        sections = parsed.get("parse", {}).get("sections", [])
        wanted = _normalize_heading(section)
        match = next(
            (
                item
                for item in sections
                if wanted
                in {
                    _normalize_heading(item.get("anchor", "")),
                    _normalize_heading(item.get("line", "")),
                }
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Could not find section {section!r} in {url}")
        section_params = {**base_params, "section": str(match["index"])}
        html = api_get(section_params).get("parse", {}).get("text", "")
        heading = match.get("line", "")
    else:
        parsed = api_get(base_params)
        html = parsed.get("parse", {}).get("text", "")
        heading = ""

    paragraphs = _extract_paragraphs(html)
    if heading and (not paragraphs or paragraphs[0] != heading):
        paragraphs.insert(0, heading)
    return "\n\n".join(paragraphs)


def scrape_wikisource_paragraphs(
    url: str,
    *,
    section: str | None = None,
    timeout: int = 30,
) -> list[str]:
    """Return the article as a list of paragraphs and headings."""
    text = scrape_wikisource_text(url, section=section, timeout=timeout)
    return [block for block in text.split("\n\n") if block.strip()]


def save_wikisource_text(
    url: str,
    output_path: str | Path = "الطب_النبوي.txt",
    *,
    section: str | None = None,
    timeout: int = 30,
) -> Path:
    """Save the article as UTF-8 with a blank line between paragraphs."""
    paragraphs = scrape_wikisource_paragraphs(
        url, section=section, timeout=timeout
    )
    path = Path(output_path)
    path.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    return path


def _extract_paragraphs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.select(
        "sup.reference, .mw-editsection, .navbox, .metadata, "
        ".noprint, table.toc, .toc, script, style"
    ):
        element.decompose()

    blocks = soup.find_all(
        ["p", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"]
    )
    paragraphs = []
    for block in blocks:
        text = re.sub(r"\s+", " ", block.get_text(" ", strip=True)).strip()
        if text and (not paragraphs or paragraphs[-1] != text):
            paragraphs.append(text)
    return paragraphs


def _normalize_heading(value: str) -> str:
    """Normalize MediaWiki anchors and heading text for matching."""
    value = unquote(value).replace("_", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip().casefold()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "url",
        nargs="?",
        default="https://ar.wikisource.org/wiki/الطب_النبوي",
        help="Wikisource article URL",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="الطب_النبوي.txt",
        help="UTF-8 output file (default: الطب_النبوي.txt)",
    )
    args = parser.parse_args()
    saved_path = save_wikisource_text(args.url, args.output)
    print("Saved article text file.")
