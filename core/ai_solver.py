import os
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, List

import streamlit as st
from openai import OpenAI

from core.parser import URLData


def get_openai_key() -> str:
    return os.getenv("OPENAI_KEY") or st.secrets["OPENAI_KEY"]


OPENAI_KEY = get_openai_key()


@lru_cache
def get_openai_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=get_openai_key(),
    )


def generate_completion(
    messages: List[Dict[str, str]], temperatura: float = 1.0
) -> str:
    client = get_openai_client()
    completion = client.chat.completions.create(
        model="x-ai/grok-4-fast:free",
        messages=messages,
        temperature=temperatura,
    )
    return completion.choices[0].message.content


def prepare_doc_for_prompt(doc: URLData) -> Dict[str, Any]:
    doc_dict = asdict(doc)
    return {k: v for k, v in doc_dict.items() if k not in ("word_count", "url", "text")}


def prepare_competitors_for_prompt(docs: List[URLData]) -> List[Dict[str, Any]]:
    return [
        {k: v for k, v in asdict(doc).items() if k not in ("word_count", "url", "text")}
        for doc in docs
    ]


def analyze_results(
    MY_DOCUMENT: URLData,
    semantic_gaps: Dict[str, Any],
    keyword_list: List[str],
    zone_relevance: Dict[str, Any],
    temperatura: float = 1.0,
    struct: bool = False,
) -> str:
    doc_dict = prepare_doc_for_prompt(MY_DOCUMENT)
    semantic_gaps_for_message = {k: v for k, v in semantic_gaps.items() if k != "text"}
    add_struct_instruction = (
        "Сформируй идеальную структуру для документа" if struct else ""
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Ты выступаешь в роли SEO-эксперта. "
                "У тебя есть результаты анализа моей страницы и страниц конкурентов. "
                "Твоя задача: выявить недостающие темы и дать рекомендации по зонам (title, h1, подзаголовки, первые 500 символов, структуры, ссылки, slug, текст). "
                "Отвечай структурированно."
            ),
        },
        {
            "role": "user",
            "content": f"""
            Ключевые слова: {keyword_list}

            Мой документ: {doc_dict}

            Анализ зон: {zone_relevance}

            Семантические разрывы: {semantic_gaps_for_message}

            Сформируй рекомендации:
            1. Какие зоны сильные?
            2. Какие зоны слабые?
            3. Какие темы/подзаголовки/списки/таблицы внедрить?
            4. Пример формулировок (коротко, только идеи).
            5. Ответ дай на русском.

            {add_struct_instruction}
        """,
        },
    ]
    return generate_completion(messages, temperatura)


def create_new_page(
    competitors: List[URLData],
    keyword_list: List[str],
    temperatura: float = 1.0,
) -> str:
    competitors_prepared = prepare_competitors_for_prompt(competitors)

    messages = [
        {
            "role": "system",
            "content": (
                "Ты выступаешь в роли SEO-эксперта. "
                "У тебя есть список ключевых слов и страницы конкурентов из ТОПа. "
                "Твоя задача: составить ТЗ для новой страницы."
            ),
        },
        {
            "role": "user",
            "content": f"""
            Ключевые слова: {keyword_list}

            Конкуренты: {competitors_prepared}

            Сформируй рекомендации:
            1. Title (с учётом SEO).
            2. Meta description.
            3. H1.
            4. Структура документа.
            5. Списки и таблицы — нужны ли, где разместить?
            6. Стиль и акценты.
            7. Ответ дай на русском.
        """,
        },
    ]
    return generate_completion(messages, temperatura)
