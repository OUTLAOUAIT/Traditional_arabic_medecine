"""Streamlit chat interface for the Traditional Arabic Medicine RAG app."""

from __future__ import annotations

import os
import re
from urllib.parse import quote, urlsplit

import streamlit as st
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from src.rag import DEFAULT_GEMINI_MODEL, answer_question
from src.weaviate_store import DEFAULT_MODEL_NAME


PRESET_QUESTIONS = (
    "كيف تكون العلاقة بين التداوي والقدر؟",
    "ما هدي النبي ﷺ في التداوي والأمر به؟",
    "ما علاج المفؤود كما ورد في الكتاب؟",
)


st.set_page_config(
    page_title="الطب النبوي | اسأل عن النص",
    page_icon="📚",
    layout="centered",
)


def _load_settings() -> None:
    """Load local .env values and override them with hosted Streamlit secrets."""
    load_dotenv()
    for name in (
        "WEAVIATE_URL",
        "WEAVIATE_API_KEY",
        "WEAVIATE_COLLECTION_NAME",
        "EMBEDDING_MODEL_NAME",
        "GEMINI_API_KEY",
        "GEMINI_MODEL_NAME",
    ):
        try:
            value = st.secrets.get(name)
        except (FileNotFoundError, KeyError):
            value = None
        if value:
            os.environ[name] = str(value)


@st.cache_resource(show_spinner="Loading the BGE-M3 query encoder…")
def _get_embedding_model(model_name: str) -> SentenceTransformer:
    # The deployed app only embeds user queries; bulk document embedding stays in Colab.
    return SentenceTransformer(model_name, device="cpu")


@st.cache_resource
def _get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in Streamlit secrets or your local .env file.")
    return genai.Client(api_key=api_key)


def _valid_source_url(value: object) -> str | None:
    """Allow source links only to the project's Wikisource host."""
    url = str(value or "").strip()
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "ar.wikisource.org":
        return None
    return quote(url, safe=":/?#&=%")


def _linked_answer(answer: str, sources: list[dict]) -> tuple[str, list[dict]]:
    """Renumber cited sources by first appearance while retaining internal IDs."""
    by_id = {str(source.get("id", "")): source for source in sources}
    display_ids: dict[str, str] = {}
    displayed_sources: list[dict] = []
    answer = re.sub(r"\\([\[\]*_])", r"\1", answer)

    def replace_citation(match: re.Match[str]) -> str:
        source_id = match.group(1)
        source = by_id.get(source_id)
        if source is None:
            return match.group(0)
        if source_id not in display_ids:
            display_id = str(len(display_ids) + 1)
            display_ids[source_id] = display_id
            displayed_sources.append({**source, "display_id": display_id})
        display_id = display_ids[source_id]
        url = _valid_source_url(source.get("source_url"))
        return f"[[{display_id}]]({url})" if url else f"[{display_id}]"

    return re.sub(r"\[(S\d+)\]", replace_citation, answer), displayed_sources


def _render_sources(sources: list[dict]) -> None:
    """Show cited chapter titles and their original links."""
    if not sources:
        return
    with st.expander("المصادر"):
        for source in sources:
            url = _valid_source_url(source.get("source_url"))
            label = f"[{source.get('display_id', source['id'])}] {source['title']}"
            if url:
                st.markdown(f"- [{label}]({url})")
            else:
                st.markdown(f"- {label}")


def _render_followups(questions: list[str], key_prefix: str) -> str | None:
    """Render up to three one-click follow-up questions."""
    for index, question in enumerate(questions[:3]):
        if st.button(
            question,
            key=f"followup_{key_prefix}_{index}",
            use_container_width=True,
        ):
            return question
    return None


def main() -> None:
    _load_settings()
    st.title("الطب النبوي")
    st.caption("اسأل عن محتوى الكتاب، وستظهر الإجابة مع روابط المقاطع المستخدمة.")
    st.markdown(
        """
        <style>
        /* Center the Arabic page heading and its subtitle. */
        [data-testid="stHeading"],
        [data-testid="stCaptionContainer"] {
            direction: rtl;
            text-align: center;
        }

        /* Keep the document shell LTR so the browser scrollbar stays on the right. */
        [data-testid="stAppViewContainer"] main {
            direction: ltr;
        }

        /* Arabic answer flow, including Markdown lists and inline citations. */
        [data-testid="stChatMessage"],
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] :is(p, h1, h2, h3, h4, li, blockquote) {
            direction: rtl;
            text-align: right;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] :is(ul, ol) {
            direction: rtl;
            padding-right: 1.5rem;
            padding-left: 0;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li::marker {
            direction: ltr;
            unicode-bidi: isolate;
        }

        [data-testid="stChatInput"] textarea {
            direction: rtl;
            text-align: right;
        }

        [data-testid="stExpander"] {
            direction: rtl;
            text-align: right;
        }

        [data-testid="stButton"] button {
            direction: rtl;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    preset_question = None
    if not st.session_state.messages:
        for index, preset in enumerate(PRESET_QUESTIONS):
            if st.button(preset, key=f"preset_question_{index}", use_container_width=True):
                preset_question = preset

    history_placeholder = st.empty()
    with history_placeholder.container():
        selected_followup = None
        for message_index, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    linked_answer, displayed_sources = _linked_answer(
                        message["answer"], message.get("sources", [])
                    )
                    st.markdown(linked_answer, unsafe_allow_html=False)
                    _render_sources(displayed_sources)
                    followup = _render_followups(
                        message.get("follow_up_questions", []), str(message_index)
                    )
                    if followup:
                        selected_followup = followup
                else:
                    st.markdown(message["content"])

    typed_question = st.chat_input("اكتب سؤالك بالعربية…")
    question = selected_followup or preset_question or typed_question
    if not question:
        return

    # Clear stale transcript elements before the blocking retrieval/generation call.
    history_placeholder.empty()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
            embedding_model = _get_embedding_model(model_name)
            client = _get_gemini_client()
            with st.spinner("أبحث في النص وأعدّ الإجابة…"):
                result = answer_question(
                    question,
                    embedding_model=embedding_model,
                    gemini_client=client,
                    conversation_history=st.session_state.messages[:-1],
                    model_name=os.getenv("GEMINI_MODEL_NAME", DEFAULT_GEMINI_MODEL),
                    limit=5,
                )
            answer = result["answer"]
            linked_answer, displayed_sources = _linked_answer(answer, result["sources"])
            st.markdown(linked_answer, unsafe_allow_html=False)
            _render_sources(displayed_sources)
            follow_up_questions = result.get("follow_up_questions", [])[:3]
            _render_followups(follow_up_questions, str(len(st.session_state.messages)))
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "answer": answer,
                    "sources": result["sources"],
                    "follow_up_questions": follow_up_questions,
                }
            )
        except Exception as exc:
            st.error(f"تعذّر إكمال الإجابة: {exc}")


main()
