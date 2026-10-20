"""Create a Weaviate client for Cloud or the local Docker service."""

from __future__ import annotations

import os

import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth


def connect_weaviate():
    """Connect to Weaviate Cloud when credentials are set, otherwise use local."""
    load_dotenv()
    cluster_url = os.getenv("WEAVIATE_URL", "").strip()
    api_key = os.getenv("WEAVIATE_API_KEY", "").strip()

    if cluster_url or api_key:
        if not cluster_url or not api_key:
            raise RuntimeError(
                "Set both WEAVIATE_URL and WEAVIATE_API_KEY to use Weaviate Cloud."
            )
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=Auth.api_key(api_key),
        )

    return weaviate.connect_to_local()
