"""HTTP API that streams grounded answers to the web front end.

Run locally:  uvicorn src.api:app --reload
The front end in ``frontend/`` is served at ``/`` by the same process.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from src.rag import DEFAULT_GEMINI_MODEL, stream_answer
from src.weaviate_store import DEFAULT_COLLECTION_NAME, DEFAULT_MODEL_NAME, check_collection


logger = logging.getLogger("tibb.api")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
PRESET_QUESTIONS = [
    "كيف تكون العلاقة بين التداوي والقدر؟",
    "ما هدي النبي ﷺ في التداوي والأمر به؟",
    "ما علاج المفؤود كما ورد في الكتاب؟",
]
GENERIC_ERROR = "تعذّر إكمال الإجابة. يرجى المحاولة مرة أخرى."
RATE_LIMIT_MESSAGE = "لقد أرسلت أسئلة كثيرة في وقت قصير. انتظر دقيقة ثم حاول مجدداً."


@dataclass
class Resources:
    """Long-lived objects shared by every request."""

    embedding_model: Any
    gemini_client: Any
    weaviate_client: Any
    gemini_model_name: str
    # One query embedding + Weaviate search at a time keeps memory predictable.
    retrieval_lock: threading.Lock


def load_resources() -> Resources:
    """Load BGE-M3 once and open long-lived Gemini and Weaviate clients."""
    from google import genai
    from sentence_transformers import SentenceTransformer

    from src.weaviate_connection import connect_weaviate

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in the environment or .env file.")

    weaviate_client = connect_weaviate()
    check_collection(
        weaviate_client, os.getenv("WEAVIATE_COLLECTION_NAME") or DEFAULT_COLLECTION_NAME
    )
    return Resources(
        embedding_model=SentenceTransformer(
            os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME), device="cpu"
        ),
        gemini_client=genai.Client(api_key=api_key),
        weaviate_client=weaviate_client,
        gemini_model_name=os.getenv("GEMINI_MODEL_NAME", DEFAULT_GEMINI_MODEL),
        retrieval_lock=threading.Lock(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_dotenv()
    # Tests (or another host process) may provide resources up front.
    owns_resources = getattr(app.state, "resources", None) is None
    if owns_resources:
        app.state.resources = load_resources()
    try:
        yield
    finally:
        if owns_resources:
            close = getattr(app.state.resources.weaviate_client, "close", None)
            if callable(close):
                close()
            app.state.resources = None


class RateLimiter:
    """Small in-memory sliding-window limit per client IP (single process)."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.max_requests <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_seconds:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            return True


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=40)


def _sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app() -> FastAPI:
    load_dotenv()
    app = FastAPI(title="Tibb Nabawi RAG API", lifespan=lifespan)
    limiter = RateLimiter(int(os.getenv("RATE_LIMIT_PER_MINUTE", "10")))

    # Only needed when the front end is served from another origin.
    allowed_origins = [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {"presets": PRESET_QUESTIONS}

    @app.post("/api/ask")
    async def ask(body: AskRequest, request: Request) -> StreamingResponse:
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Question cannot be empty")
        client_ip = request.client.host if request.client else "unknown"
        if not limiter.allow(client_ip):
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)

        resources: Resources = request.app.state.resources
        events = stream_answer(
            question,
            embedding_model=resources.embedding_model,
            gemini_client=resources.gemini_client,
            conversation_history=[message.model_dump() for message in body.history],
            model_name=resources.gemini_model_name,
            weaviate_client=resources.weaviate_client,
            retrieval_lock=resources.retrieval_lock,
            limit=5,
        )

        async def event_stream() -> AsyncIterator[str]:
            try:
                async for event in iterate_in_threadpool(events):
                    if await request.is_disconnected():
                        break  # Stop: the reader went away, stop pulling from Gemini.
                    yield _sse(event)
            except Exception:
                logger.exception("Answer generation failed")
                yield _sse({"type": "error", "message": GENERIC_ERROR})
            finally:
                try:
                    events.close()
                except ValueError:
                    # Still running in a worker thread (request was cancelled
                    # mid-chunk); it ends on its own after that chunk.
                    pass

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
