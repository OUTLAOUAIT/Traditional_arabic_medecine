"""Run a small, title-checked Arabic retrieval evaluation against Weaviate."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.weaviate_store import DEFAULT_MODEL_NAME, _load_model, search_chunks


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    expected_title_fragment: str


CASES = [
    RetrievalCase("كيف تكون العلاقة بين التداوي والقدر؟", "التداوي والأمر به"),
    RetrievalCase("هل يجوز استعمال دواء محرم للعلاج؟", "المنع من التداوي بالمحرمات"),
    RetrievalCase("كيف كان يعالج النبي الحمى؟", "علاج الحمى"),
    RetrievalCase("ما علاج الإسهال واستطلاق البطن؟", "علاج استطلاق البطن"),
    RetrievalCase("كيف يتصرف الناس عند انتشار الطاعون؟", "الطاعون وعلاجه"),
    RetrievalCase("ما علاج داء الاستسقاء؟", "داء الاستسقاء وعلاجه"),
    RetrievalCase("ما فوائد الحجامة؟", "منافع الحجامة"),
    RetrievalCase("ما مواضع الحجامة وأوقاتها؟", "مواضع الحجامة وأوقاتها"),
    RetrievalCase("كيف يعالج عرق النسا؟", "علاج عرق النسا"),
    RetrievalCase("ما علاج الصداع النصفي والشقيقة؟", "علاج الصداع والشقيقة"),
    RetrievalCase("كيف عالج النبي الصرع؟", "علاج الصرع"),
    RetrievalCase("ما علاج حكة الجسم وما يسبب القمل؟", "حكة الجسم وما يولد القمل"),
    RetrievalCase("كيف يعالج الجرح؟", "علاج الجرح"),
    RetrievalCase("ما هدي النبي في علاج الرمد؟", "علاج الرمد"),
    RetrievalCase("كيف يعالج حزن المصيبة؟", "حر المصيبة وحزنها"),
    RetrievalCase("ما الرقية التي تقرأ على اللديغ؟", "رقية اللديغ بالفاتحة"),
    RetrievalCase("كيف تعالج لدغة العقرب بالرقية؟", "لدغة العقرب بالرقية"),
    RetrievalCase("ما هديه في العلاج بالعسل والحجامة والكي؟", "شرب العسل والحجامة والكي"),
]


def evaluate_retrieval(
    *,
    cases: list[RetrievalCase] = CASES,
    limit: int = 5,
    model: SentenceTransformer | None = None,
) -> dict[str, int]:
    """Print top matches and title-match-at-k counts for manual review.

    A title match is only a lightweight relevance signal. Review the printed
    text excerpts before treating a case as a true retrieval success.
    """
    if model is None:
        load_dotenv()
        model = _load_model(os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME))

    hits = 0
    for index, case in enumerate(cases, start=1):
        results = search_chunks(case.query, limit=limit, model=model)
        matching_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if case.expected_title_fragment in str(result.get("title", ""))
            ),
            None,
        )
        if matching_rank is not None:
            hits += 1

        status = f"HIT@{matching_rank}" if matching_rank is not None else "REVIEW"
        print(f"{index:02}. {status} | Query: {case.query}")
        print(f"    Expected title contains: {case.expected_title_fragment}")
        for rank, result in enumerate(results[:3], start=1):
            excerpt = " ".join(str(result.get("text", "")).split())[:160]
            distance = result.get("distance")
            distance_text = (
                f"{distance:.4f}"
                if isinstance(distance, (int, float))
                else "n/a"
            )
            print(
                f"    {rank}. {result.get('title', '')} | "
                f"chunk {result.get('chunk_index', '?')} | "
                f"distance {distance_text}"
            )
            print(f"       {excerpt}")

    summary = {"cases": len(cases), f"title_hit_at_{limit}": hits}
    print(f"\nTitle match summary: {hits}/{len(cases)} within top {limit}.")
    print("Review each excerpt; a title match is not a complete relevance judgment.")
    return summary


if __name__ == "__main__":
    evaluate_retrieval()
