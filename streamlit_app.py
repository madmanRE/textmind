import streamlit as st
from core.pipline import analyze
import pandas as pd

st.title("📊 TextMind 1.0")

with st.form("analyze_form"):
    st.write("Настройка данных для анализа")

    my_domain = st.text_input("Моя страница")
    competitors = st.text_area("Конкуренты (по одному на строку)")
    keywords = st.text_area("Ключевые слова (по одному на строку)")

    submitted = st.form_submit_button("Анализировать")

if submitted:
    my_domain = my_domain.strip()
    competitors = [c.strip() for c in competitors.splitlines() if c.strip()]
    keywords = [k.strip() for k in keywords.splitlines() if k.strip()]

    with st.spinner("Анализ выполняется, подождите..."):
        zone_relevance, semantics_gaps, results = analyze(my_domain, competitors, keywords)

    st.success("Анализ завершён ✅")


    st.subheader("Зональная релевантность ТОПу")
    df = pd.DataFrame.from_dict(zone_relevance, orient="index", columns=["relevance"])
    df = df.sort_values("relevance", ascending=True)
    st.bar_chart(df)

    st.subheader("Семантические разрывы")
    st.json(semantics_gaps)

    st.subheader("Итоговый анализ")
    st.markdown(results, unsafe_allow_html=True)

