import os
import streamlit as st
from dataclasses import asdict

from openai import OpenAI

OPENAI_KEY = os.getenv("OPENAI_KEY")
if not OPENAI_KEY:
    OPENAI_KEY = st.secrets["OPENAI_KEY"]


def analyze_results(MY_DOCUMENT, semantic_gaps, keyword_list, zone_relevance):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENAI_KEY,
    )

    doc_dict = asdict(MY_DOCUMENT)
    exclude_fields = ["word_count", "url", "text"]
    for field in exclude_fields:
        doc_dict.pop(field, None)

    semantic_gaps_for_message = semantic_gaps.copy()
    semantic_gaps_for_message.pop("text")

    messages = [
        {
            "role": "system",
            "content": (
                "Ты выступаешь в роли SEO-эксперта. "
                "У тебя есть результаты семантического анализа моей страницы и страниц конкурентов. "
                "Твоя задача: выявить недостающие темы, подсказать улучшения по каждому блоку (title, h1, подзаголовки, первые 500 символов, "
                "структуры — списки и таблицы, ссылки, slug, основной текст). "
                "Отвечай структурированно: сначала зона, затем рекомендации."
            ),
        },
        {
            "role": "user",
            "content": f"""
            Ключевые слова: {keyword_list}

            Мой документ (по зонам):
            {doc_dict}

            Анализ моего документа (по зонам):
            {zone_relevance}

            Результаты анализа (сравнение с конкурентами):
            {semantic_gaps_for_message}

            Сформируй рекомендации:
            1. Какие зоны у меня сильные (лучше конкурентов)?
            2. Какие зоны слабые и что нужно добавить или изменить?
            3. Какие темы/подзаголовки/списки/таблицы желательно внедрить?
            4. Пример формулировок (без длинного текста, только идеи).
            5. Ответ дай на русском языке.
            """,
        },
    ]

    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1:free", messages=messages
    )

    return completion.choices[0].message.content
