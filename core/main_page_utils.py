import pandas as pd
import streamlit as st

from core.pipline import analyze

FAQ_TEXT = """
    #### ❓ FAQ: Как работает программа
    
    **1. Что делает пользователь на старте?**  
    Пользователь вводит:
    - свою страницу,
    - страницы конкурентов из ТОПа,
    - список поисковых запросов.
    
    ---
    
    **2. Как обрабатываются страницы?**  
    Программа парсит страницы и выделяет контент по зонам:
    - `title`  
    - `h1`  
    - `first_500_chars_after_h1`  
    - `text`  
    - `structures` (списки и таблицы)  
    - `slug`  
    - `subheadings`
    
    ---
    
    **3. Как оценивается соответствие контента?**  
    Для каждой зоны трансформерная модель вычисляет эмбеддинги и сравнивает усреднённый вектор конкурентов с вектором зоны анализируемой страницы.  
    Это позволяет понять, насколько контент по каждой зоне соответствует ТОПу.
    
    ---
    
    **4. Что происходит дальше?**  
    Программа проходит по конкурентам и ищет **самые релевантные фрагменты** в каждой зоне.
    
    ---
    
    **5. Как сравниваются фрагменты?**  
    Каждый фрагмент оценивается по трём метрикам:
    - `keywords_sim` — сходство с эмбеддингом ключевых слов,  
    - `my_doc_kw_sim` — релевантность зоны ключам,  
    - `my_doc_sim_zone` — сходство с аналогичной зоной моего документа,  
    - `my_doc_sim_full` — сходство с полным текстом моего документа.  
    
    Так выявляются **семантические разрывы**.
    
    ---
    
    **6. Как формируются рекомендации?**  
    Все результаты анализа передаются в **LLM**, которая на основе данных формирует рекомендации для улучшения контента.
    
    """


def clean_input(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def run_analysis(
    my_domain,
    competitors,
    keywords,
    user_agent,
    exclude_tags_list,
    temperatura,
    struct,
    new_page,
):
    if not new_page:
        results = analyze(
            my_domain,
            competitors,
            keywords,
            user_agent,
            exclude_tags_list,
            temperatura,
            struct,
        )
        return {
            "zone_relevance": results['zone_relevance'],
            "semantics_gaps": results['semantics_gaps'],
            "results": results['results'],
            "new_page": False,
        }
    else:
        results = analyze(
            my_domain,
            competitors,
            keywords,
            user_agent,
            exclude_tags_list,
            temperatura,
            struct,
            new_page,
        )
        return {"results": results, "new_page": True}


def display_results(zone_relevance, semantics_gaps, results):
    st.subheader("Зональная релевантность ТОПу")
    df = pd.DataFrame.from_dict(zone_relevance, orient="index", columns=["relevance"])
    st.bar_chart(df.sort_values("relevance", ascending=True))

    st.subheader("Семантические разрывы")
    st.json(semantics_gaps)

    st.subheader("Итоговый анализ")
    st.markdown(results, unsafe_allow_html=True)
