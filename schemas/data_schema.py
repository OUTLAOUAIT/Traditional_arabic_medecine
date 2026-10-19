"""Pydantic schema for one fasl (section) of the source text."""

from datetime import datetime

from pydantic import BaseModel


class FaslSchema(BaseModel):
    fasl_content: str
    normalized_fasl: str
    number_tokens: int
    fasl_title: str
    fasl_url: str
    created_at: datetime
