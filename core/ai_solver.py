import os
from dataclasses import asdict
import streamlit as st
from openai import OpenAI
from typing import Any, Dict, List


# Получение ключа OpenAI
def get_openai_key() -> str:
    return os.getenv("OPENAI_KEY") or st.secrets["OPENAI_KEY"]


OPENAI_KEY = get_openai_key()


def analyze_results(
    MY_DOCUMENT: Any,
    semantic_gaps: Dict[str, Any],
    keyword_list: List[str],
    zone_relevance: Dict[str, Any],
    temperatura: float = 1.0,
    struct: bool = False,
) -> str:
    """
    Формирует рекомендации по улучшению SEO документа на основе анализа конкурентов и семантических разрывов.
    """

    # Добавляем инструкцию про структуру, если нужно
    add_struct_instruction = (
        "Сформируй идеальную структуру для документа" if struct else ""
    )

    # Инициализация клиента OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENAI_KEY,
    )

    # Подготовка документа без больших полей
    doc_dict = asdict(MY_DOCUMENT)
    for field in ["word_count", "url", "text"]:
        doc_dict.pop(field, None)

    # Подготовка семантических разрывов без текста
    semantic_gaps_for_message = semantic_gaps.copy()
    semantic_gaps_for_message.pop("text", None)

    # Формирование сообщений для модели
    messages = [
        {
            "role": "system",
            "content": (
                "Ты выступаешь в роли SEO-эксперта. "
                "У тебя есть результаты семантического анализа моей страницы и страниц конкурентов. "
                "Твоя задача: выявить недостающие темы, подсказать улучшения по каждому блоку "
                "(title, h1, подзаголовки, первые 500 символов, структуры — списки и таблицы, ссылки, slug, основной текст). "
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
                
                {add_struct_instruction}
                """,
        },
    ]

    # Запрос к модели
    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=messages,
        temperature=temperatura,
    )

    return completion.choices[0].message.content
