# Traditional Arabic Medicine Text Pipeline

Tools for retrieving the Arabic Wikisource text of *الطب النبوي*, splitting it into fasls (sections), and creating tokenizer-counted chunks for search and retrieval.

## Requirements

- Python 3.10 or newer
- A MongoDB database for ingestion
- Network access to Wikisource when scraping or resolving section URLs
- The `BAAI/bge-m3` tokenizer downloaded locally before chunk ingestion

Install dependencies in a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Configuration

Copy `.env.example` to `.env` and set the MongoDB connection URI. The `.env` file is ignored by Git.

```powershell
Copy-Item .env.example .env
```

Configuration variables:

| Variable | Purpose |
| --- | --- |
| `MONGODB_CONNECTION_STRING` | MongoDB connection URI |
| `DATA_BASE` | MongoDB database name |
| `COLLECTION_NAME` | Fasls collection name |
| `CHUNKS_COLLECTION_NAME` | Destination collection for chunks (defaults to `chunks`) |
| `FASLS_COLLECTION_NAME` | Optional source collection override for chunk ingestion |

## Scrape the Wikisource article

The scraper retrieves paragraphs and headings through the MediaWiki API and writes UTF-8 text separated by blank lines:

```powershell
python scrape_wikisource.py "https://ar.wikisource.org/wiki/الطب_النبوي" data/raw_data.txt
```

The scraper functions are also available from `scrape_wikisource.py` if you want to fetch one section or handle the text in Python.

## Split and save fasls

`src/data_processing.py` splits `data/raw_data.txt` at paragraph starts matching the configured Arabic section headings. It also normalizes a search copy, counts Arabic word and punctuation tokens with PyArabic, and builds Wikisource URLs for each fasl.

Save the resulting fasls to the configured MongoDB collection:

```powershell
python -c "from src.data_processing import split_raw_data_file, save_fasls_to_mongodb; fasls = split_raw_data_file(); print(save_fasls_to_mongodb(fasls))"
```

The fasl save resolves section links against Wikisource, so it needs network access. Documents are upserted using a stable content hash as `_id`.

## Create retrieval chunks

Download the tokenizer once (the ingestion command uses the local cache):

```powershell
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('BAAI/bge-m3', use_fast=True)"
```

Then generate and upsert chunks:

```powershell
python -m src.chunking
```

Chunking measures the complete embedding input (`title + blank line + text`) with the `BAAI/bge-m3` tokenizer and enforces a 2,048-token maximum. It packs consecutive complete paragraphs, allows shorter chunks when a paragraph does not fit, and uses complete trailing paragraphs for overlap where they fit. It raises an error rather than splitting an oversized paragraph. Each fasl gets at least one chunk, including fasls that fit as a single chunk. Ingestion creates a unique index on `(fasl_id, chunking_version, chunk_index)`, upserts chunk records, and removes stale indexes after replacement writes succeed.

Fasl documents remain in the fasls collection; generated chunks are stored separately in the chunks collection.

## Project layout

```text
.
├── data/
│   ├── raw_data.txt       # Source text, paragraph separated
│   └── references.txt
├── schemas/
│   └── data_schema.py     # Pydantic fasl schema
├── src/
│   ├── data_processing.py # Normalization, fasl splitting, MongoDB upserts
│   └── chunking.py        # BGE-M3 paragraph chunking and MongoDB upserts
├── scrape_wikisource.py
├── requirements.txt
└── .env.example
```

## Initialize Git

This workspace has been initialized as a local Git repository. After reviewing the files you want to version, stage them and make the first commit:

```powershell
git add .
git status
git commit -m "Initial commit"
```
