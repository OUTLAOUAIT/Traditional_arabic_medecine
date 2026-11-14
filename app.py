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


def _queue_question(question: str) -> None:
    """Button callback: runs before the rerun, so the click is known up front."""
    st.session_state.pending_question = question


def _render_question_buttons(questions: list[str] | tuple[str, ...], key_prefix: str) -> None:
    """Render one-click question buttons (presets or follow-ups)."""
    for index, question in enumerate(questions[:3]):
        st.button(
            question,
            key=f"{key_prefix}_{index}",
            use_container_width=True,
            on_click=_queue_question,
            args=(question,),
        )


def _render_assistant_message(message: dict) -> None:
    linked_answer, displayed_sources = _linked_answer(
        message["answer"], message.get("sources", [])
    )
    st.markdown(linked_answer, unsafe_allow_html=False)
    _render_sources(displayed_sources)


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

    # The chat input is pinned to the bottom of the page regardless of call order,
    # so read it first: the transcript below then knows whether a question is pending.
    typed_question = st.chat_input("اكتب سؤالك بالعربية…")
    question = st.session_state.pop("pending_question", None) or typed_question

    messages = st.session_state.messages
    if not messages and not question:
        _render_question_buttons(PRESET_QUESTIONS, "preset_question")

    # Keep the whole conversation on screen, oldest first.
    for message_index, message in enumerate(messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant_message(message)
                # Always draw this slot so it overwrites last run's buttons at once;
                # otherwise Streamlit keeps showing them greyed out until the run ends.
                followup_slot = st.empty()
                # Like ChatGPT/Claude: suggestions only under the latest answer,
                # and hidden while a new question is being answered.
                if message_index == len(messages) - 1 and not question:
                    with followup_slot.container():
                        _render_question_buttons(
                            message.get("follow_up_questions", []),
                            f"followup_{message_index}",
                        )
            else:
                st.markdown(message["content"])

    if not question:
        return

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
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "follow_up_questions": result.get("follow_up_questions", [])[:3],
                }
            )
        except Exception as exc:
            st.error(f"تعذّر إكمال الإجابة: {exc}")
            return

    # Redraw from session state so the new answer and its follow-up buttons
    # render through the same transcript path as every earlier message.
    st.rerun()


main()
