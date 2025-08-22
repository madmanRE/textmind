import streamlit as st

from core.main_page_utils import FAQ_TEXT, clean_input, display_results, run_analysis

st.title("📊 TextMind 1.0")

with st.expander("📘 Инструкция по использованию"):
    st.markdown(FAQ_TEXT)

with st.form("analyze_form"):
    st.write("Настройки")

    new_page = st.checkbox("Создать новую страницу")

    my_domain = None if new_page else st.text_input("Моя страница")
    competitors = st.text_area("Конкуренты (по одному на строку)")
    keywords = st.text_area("Ключевые слова (по одному на строку)")

    with st.expander("Экспертные настройки"):
        user_agent = st.text_input(
            "User-Agent",
            value=(
                "Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/W.X.Y.Z Mobile Safari/537.36 "
                "(compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
            ),
        )

        expose_tags = st.text_area(
            "Исключаемые теги (по одному на строку)",
            value="\n".join(["script", "style", "noscript", "footer", "header", "nav"]),
        )

        temperatura = st.slider(
            "Температура генерации", min_value=0.0, max_value=2.0, value=1.0, step=0.1
        )

        struct = st.checkbox("Формировать структуру")

    submitted = st.form_submit_button("Анализировать")

if submitted:
    form_data = {
        "my_domain": my_domain.strip() if my_domain else None,
        "competitors": clean_input(competitors),
        "keywords": clean_input(keywords),
        "user_agent": user_agent.strip(),
        "exclude_tags_list": clean_input(expose_tags),
        "temperatura": temperatura,
        "struct": struct,
        "new_page": new_page,
    }

    with st.spinner("Анализ выполняется, подождите..."):
        analysis = run_analysis(**form_data)

    st.success("Анализ завершён ✅")

    if analysis["new_page"]:
        st.subheader("Итоговый анализ")
        st.markdown(analysis["results"]["results"], unsafe_allow_html=True)
    else:
        display_results(
            analysis["zone_relevance"], analysis["semantics_gaps"], analysis["results"]
        )
