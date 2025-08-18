import streamlit as st
import pandas as pd
import plotly.express as px
import json, ast
import numpy as np

st.set_page_config(page_title="DS Analytics Dashboard", layout="wide")
st.title("DS Analytics Dashboard — Streamlit")

# ---------- Utils ----------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.copy()


def parse_array_field(x):
    if x is None:
        return []
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass
    if isinstance(x, (list, tuple, np.ndarray)):
        return [str(i).strip() for i in x if str(i).strip() and str(i).strip() not in {"nan", "None"}]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    arr = parser(s)
                    if isinstance(arr, list):
                        return [str(i).strip() for i in arr if str(i).strip()]
                except Exception:
                    pass
        return [i.strip() for i in s.split(",") if i.strip()]
    return [str(x).strip()] if str(x).strip() else []


@st.cache_data(show_spinner=False)
def build_value_counts(df: pd.DataFrame, field: str, top_n: int = 20):
    if field not in df.columns:
        return pd.Series([], dtype=int)
    vc = df[field].dropna().value_counts()
    return vc.head(top_n)


@st.cache_data(show_spinner=False)
def build_array_counts(df: pd.DataFrame, field: str, top_n: int = 20):
    if field not in df.columns:
        return pd.Series([], dtype=int)
    items = []
    for x in df[field].dropna():
        items.extend(parse_array_field(x))
    if not items:
        return pd.Series([], dtype=int)
    vc = pd.Series(items).value_counts()
    return vc.head(top_n)


# ---------- Load ----------
df = load_data("data/ds_vacancies.parquet")

# Safer accessors for DS columns (fallback to names)
DS_COLUMNS = {
    "specialization": df.columns[13] if len(df.columns) > 13 else "specialization",
    "programming_languages": df.columns[14] if len(df.columns) > 14 else "programming_languages",
    "ml_libraries": df.columns[15] if len(df.columns) > 15 else "ml_libraries",
    "visualization": df.columns[16] if len(df.columns) > 16 else "visualization",
    "data_processing": df.columns[17] if len(df.columns) > 17 else "data_processing",
    "nlp_tools": df.columns[18] if len(df.columns) > 18 else "nlp_tools",
    "cv_tools": df.columns[19] if len(df.columns) > 19 else "cv_tools",
    "mlops_tools": df.columns[20] if len(df.columns) > 20 else "mlops_tools",
    "business_domains": df.columns[21] if len(df.columns) > 21 else "business_domains",
    "level": df.columns[22] if len(df.columns) > 22 else "level",
    "seniority": df.columns[23] if len(df.columns) > 23 else "seniority",
    "job_type": df.columns[24] if len(df.columns) > 24 else "job_type",
    "category": df.columns[25] if len(df.columns) > 25 else "category",
}

# Salary normalization (quick version)
@st.cache_data(show_spinner=False)
def normalize_salary(df: pd.DataFrame) -> pd.DataFrame:
    rates = {"RUR": 1, "RUB": 1, "USD": 95, "EUR": 105, "KZT": 0.2, "UZS": 0.0075,
             "BYR": 0.035, "UAH": 2.5, "KGS": 1.1, "AZN": 55}
    d = df.copy()
    if not {"salary_from", "salary_to", "salary_currency", "salary_gross"}.issubset(d.columns):
        d["salary_from_rub"] = np.nan
        d["salary_to_rub"] = np.nan
        return d
    cur = d["salary_currency"].fillna("RUR").map(rates).fillna(1)
    d["salary_from_rub"] = d["salary_from"].fillna(0) * cur
    d["salary_to_rub"] = d["salary_to"].fillna(0) * cur
    gross_mask = d.get("salary_gross", 0).fillna(0).astype(float) == 1.0
    d.loc[gross_mask, ["salary_from_rub", "salary_to_rub"]] *= 0.87
    # bounds
    for c in ("salary_from_rub", "salary_to_rub"):
        d.loc[(d[c] < 1000) | (d[c] > 10_000_000), c] = np.nan
    return d


df = normalize_salary(df)

# ---------- Sidebar Filters ----------
with st.sidebar:
    st.subheader("Фильтры")
    
    # General filters section
    with st.expander("Общие фильтры", expanded=True):
        # Category and job type
        category_opts = build_array_counts(df, DS_COLUMNS["category"], top_n=50).index.tolist()
        job_type_opts = build_array_counts(df, DS_COLUMNS["job_type"], top_n=50).index.tolist()
        level_opts = build_array_counts(df, DS_COLUMNS["level"], top_n=50).index.tolist()
        seniority_opts = build_array_counts(df, DS_COLUMNS["seniority"], top_n=50).index.tolist()
        
        category_sel = st.multiselect("Категория", category_opts)
        job_type_sel = st.multiselect("Тип вакансии", job_type_opts)
        level_sel = st.multiselect("Уровень позиции", level_opts)
        seniority_sel = st.multiselect("Сеньорность", seniority_opts)
    
    # Basic filters section
    with st.expander("Базовые фильтры", expanded=True):
        area_vals = sorted([v for v in df.get("area_name", pd.Series(dtype=str)).dropna().unique()])
        experience_vals = sorted([v for v in df.get("experience_name", pd.Series(dtype=str)).dropna().unique()])
        employer_vals = sorted([v for v in df.get("employer_name", pd.Series(dtype=str)).dropna().unique()][:100])  # Limit for performance

        area_sel = st.multiselect("География", area_vals)
        exp_sel = st.multiselect("Опыт работы", experience_vals)
        employer_sel = st.multiselect("Компания", employer_vals)

        # Salary slider
        sal = df["salary_from_rub"].dropna()
        s_min = int(sal.min()) if len(sal) else 0
        s_max = int(sal.max()) if len(sal) else 1_000_000
        salary_sel = st.slider("Зарплата от (₽)", min_value=s_min, max_value=s_max, value=(s_min, s_max))

    # Technical skills section
    with st.expander("Технические навыки", expanded=False):
        specialization_opts = build_array_counts(df, DS_COLUMNS["specialization"], top_n=50).index.tolist()
        prog_opts = build_array_counts(df, DS_COLUMNS["programming_languages"], top_n=50).index.tolist()
        mllib_opts = build_array_counts(df, DS_COLUMNS["ml_libraries"], top_n=50).index.tolist()
        skill_opts = build_array_counts(df, "key_skills", top_n=50).index.tolist()

        specialization_sel = st.multiselect("Специализация", specialization_opts)
        prog_sel = st.multiselect("Языки программирования", prog_opts)
        mllib_sel = st.multiselect("ML библиотеки", mllib_opts)
        skill_sel = st.multiselect("Общие навыки", skill_opts)

    # Data tools section  
    with st.expander("Инструменты обработки данных", expanded=False):
        data_proc_opts = build_array_counts(df, DS_COLUMNS["data_processing"], top_n=50).index.tolist()
        viz_opts = build_array_counts(df, DS_COLUMNS["visualization"], top_n=50).index.tolist()
        
        data_proc_sel = st.multiselect("Обработка данных", data_proc_opts)
        viz_sel = st.multiselect("Визуализация", viz_opts)

    # AI/ML tools section
    with st.expander("AI/ML инструменты", expanded=False):
        nlp_opts = build_array_counts(df, DS_COLUMNS["nlp_tools"], top_n=50).index.tolist()
        cv_opts = build_array_counts(df, DS_COLUMNS["cv_tools"], top_n=50).index.tolist()
        mlops_opts = build_array_counts(df, DS_COLUMNS["mlops_tools"], top_n=50).index.tolist()
        
        nlp_sel = st.multiselect("NLP инструменты", nlp_opts)
        cv_sel = st.multiselect("Computer Vision", cv_opts)
        mlops_sel = st.multiselect("MLOps инструменты", mlops_opts)

    # Business section
    with st.expander("Бизнес и организация", expanded=False):
        business_opts = build_array_counts(df, DS_COLUMNS["business_domains"], top_n=50).index.tolist()
        format_opts = build_array_counts(df, "work_format", top_n=20).index.tolist() if "work_format" in df.columns else []
        
        business_sel = st.multiselect("Бизнес-домены", business_opts)
        if format_opts:
            format_sel = st.multiselect("Формат работы", format_opts)
        else:
            format_sel = []

    st.divider()
    if st.button("Сбросить все фильтры"):
        st.rerun()

# ---------- Filtering ----------
def contains_any(parsed_list, targets):
    return any(t in parsed_list for t in targets)

@st.cache_data(show_spinner=False)
def apply_filters(df: pd.DataFrame, **filters) -> pd.DataFrame:
    """Apply all filters efficiently with caching"""
    filtered = df.copy()
    
    # Simple string filters
    for field, values in [
        ("area_name", filters.get("area_sel", [])),
        ("experience_name", filters.get("exp_sel", [])),
        ("employer_name", filters.get("employer_sel", []))
    ]:
        if values:
            filtered = filtered[filtered.get(field, pd.Series()).isin(values)]
    
    # Salary filter
    salary_range = filters.get("salary_sel")
    if salary_range and salary_range != (filters.get("s_min", 0), filters.get("s_max", 1000000)):
        filtered = filtered[
            (filtered["salary_from_rub"].notna()) &
            (filtered["salary_from_rub"] >= salary_range[0]) &
            (filtered["salary_from_rub"] <= salary_range[1])
        ]
    
    # Array filters
    array_filters = [
        (DS_COLUMNS["category"], filters.get("category_sel", [])),
        (DS_COLUMNS["job_type"], filters.get("job_type_sel", [])),
        (DS_COLUMNS["level"], filters.get("level_sel", [])),
        (DS_COLUMNS["seniority"], filters.get("seniority_sel", [])),
        (DS_COLUMNS["specialization"], filters.get("specialization_sel", [])),
        (DS_COLUMNS["programming_languages"], filters.get("prog_sel", [])),
        (DS_COLUMNS["ml_libraries"], filters.get("mllib_sel", [])),
        ("key_skills", filters.get("skill_sel", [])),
        (DS_COLUMNS["data_processing"], filters.get("data_proc_sel", [])),
        (DS_COLUMNS["visualization"], filters.get("viz_sel", [])),
        (DS_COLUMNS["nlp_tools"], filters.get("nlp_sel", [])),
        (DS_COLUMNS["cv_tools"], filters.get("cv_sel", [])),
        (DS_COLUMNS["mlops_tools"], filters.get("mlops_sel", [])),
        (DS_COLUMNS["business_domains"], filters.get("business_sel", [])),
    ]
    
    # Add work_format if it exists
    if "work_format" in df.columns:
        array_filters.append(("work_format", filters.get("format_sel", [])))
    
    for field, values in array_filters:
        if values and field in filtered.columns:
            mask = filtered[field].apply(lambda x: contains_any(parse_array_field(x), values))
            filtered = filtered[mask]
    
    return filtered

# Apply all filters
filtered = apply_filters(
    df,
    area_sel=area_sel,
    exp_sel=exp_sel, 
    employer_sel=employer_sel,
    salary_sel=salary_sel,
    s_min=s_min,
    s_max=s_max,
    category_sel=category_sel,
    job_type_sel=job_type_sel,
    level_sel=level_sel,
    seniority_sel=seniority_sel,
    specialization_sel=specialization_sel,
    prog_sel=prog_sel,
    mllib_sel=mllib_sel,
    skill_sel=skill_sel,
    data_proc_sel=data_proc_sel,
    viz_sel=viz_sel,
    nlp_sel=nlp_sel,
    cv_sel=cv_sel,
    mlops_sel=mlops_sel,
    business_sel=business_sel,
    format_sel=format_sel
)

# ---------- Enhanced KPIs ----------
st.subheader("Ключевые метрики")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Всего вакансий", 
        value=f"{len(filtered):,}".replace(",", " "),
        delta=f"{((len(filtered)/len(df))*100):.1f}% от общего"
    )

with col2:
    s = filtered["salary_from_rub"].dropna()
    if len(s):
        avg_sal = int(s.mean())
        st.metric("Средняя зарплата", f"{avg_sal:,} ₽".replace(",", " "))
    else:
        st.metric("Средняя зарплата", "—")

with col3:
    if len(s):
        med_sal = int(s.median())
        st.metric("Медианная зарплата", f"{med_sal:,} ₽".replace(",", " "))
    else:
        st.metric("Медианная зарплата", "—")

with col4:
    # Most popular skill
    if len(filtered) > 0:
        top_skill = build_array_counts(filtered, "key_skills", 1)
        skill_name = top_skill.index[0] if len(top_skill) > 0 else "—"
        skill_count = top_skill.iloc[0] if len(top_skill) > 0 else 0
        st.metric("Топ навык", skill_name, f"{skill_count} упоминаний")
    else:
        st.metric("Топ навык", "—")

with col5:
    # Most popular location
    if len(filtered) > 0 and "area_name" in filtered.columns:
        top_area = filtered["area_name"].value_counts().head(1)
        area_name = top_area.index[0] if len(top_area) > 0 else "—"
        area_count = top_area.iloc[0] if len(top_area) > 0 else 0
        st.metric("Топ регион", area_name, f"{area_count} вакансий")
    else:
        st.metric("Топ регион", "—")

# ---------- Enhanced Tabs ----------
tabs = st.tabs([
    "Общий обзор", "География", "Технические навыки", "AI/ML инструменты", 
    "Анализ зарплат", "Компании и домены", "Корреляции"
])

with tabs[0]:  # Общий обзор
    st.subheader("Обзор вакансий")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Job types and categories
        if category_sel or job_type_sel or not (category_sel or job_type_sel):  # Show if filtered or no filter
            vc_cat = build_array_counts(filtered, DS_COLUMNS["category"], 15)
            if len(vc_cat):
                fig = px.pie(values=vc_cat.values, names=vc_cat.index, title="Распределение по категориям")
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Experience distribution
        vc_exp = build_value_counts(filtered, "experience_name", 10)
        if len(vc_exp):
            fig = px.pie(values=vc_exp.values, names=vc_exp.index, title="Распределение по опыту")
            st.plotly_chart(fig, use_container_width=True)

with tabs[1]:  # География
    st.subheader("Географическое распределение")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        vc = build_value_counts(filtered, "area_name", 20)
        if len(vc):
            fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="Вакансии по городам")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных для отображения")
    
    with col2:
        # Top cities stats
        if len(vc) > 0:
            st.write("**Топ-5 городов:**")
            for i, (city, count) in enumerate(vc.head(5).items()):
                st.write(f"{i+1}. {city}: **{count}** вакансий")

with tabs[2]:  # Технические навыки
    st.subheader("Технические навыки и языки программирования")
    
    # Create sub-tabs for different skill categories
    skill_tabs = st.tabs(["Общие навыки", "Языки программирования", "Специализации"])
    
    with skill_tabs[0]:
        vc = build_array_counts(filtered, "key_skills", 25)
        if len(vc):
            fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="Топ навыков")
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных для отображения")
    
    with skill_tabs[1]:
        vc = build_array_counts(filtered, DS_COLUMNS["programming_languages"], 20)
        if len(vc):
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="Языки программирования")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                # Programming languages pie chart
                fig_pie = px.pie(values=vc.head(8).values, names=vc.head(8).index, title="Топ-8 языков")
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Нет данных для отображения")
            
    with skill_tabs[2]:
        vc = build_array_counts(filtered, DS_COLUMNS["specialization"], 15)
        if len(vc):
            fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="DS Специализации")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных для отображения")

with tabs[3]:  # AI/ML инструменты
    st.subheader("AI/ML инструменты и библиотеки")
    
    ml_tabs = st.tabs(["ML библиотеки", "NLP инструменты", "Computer Vision", "MLOps"])
    
    with ml_tabs[0]:
        vc = build_array_counts(filtered, DS_COLUMNS["ml_libraries"], 20)
        if len(vc):
            fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="ML библиотеки")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных для отображения")
            
    with ml_tabs[1]:
        vc = build_array_counts(filtered, DS_COLUMNS["nlp_tools"], 15)
        if len(vc):
            fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="NLP инструменты")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных для отображения")
            
    with ml_tabs[2]:
        vc = build_array_counts(filtered, DS_COLUMNS["cv_tools"], 15)
        if len(vc):
            fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="Computer Vision инструменты")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных для отображения")
            
    with ml_tabs[3]:
        vc = build_array_counts(filtered, DS_COLUMNS["mlops_tools"], 15)
        if len(vc):
            fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="MLOps инструменты")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных для отображения")

with tabs[4]:  # Анализ зарплат
    st.subheader("Анализ зарплат")
    
    salary_data = filtered[(filtered["salary_from_rub"].notna())]
    
    if len(salary_data) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Salary by experience
            if "experience_name" in salary_data.columns:
                agg = salary_data.groupby("experience_name")["salary_from_rub"].agg(["mean", "median", "count"]).reset_index()
                agg = agg[agg["count"] >= 3]  # Filter out groups with too few samples
                if len(agg) > 0:
                    fig = px.bar(agg, x="experience_name", y=["mean", "median"], 
                               barmode="group", title="Зарплата по опыту работы")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Salary distribution histogram
            fig = px.histogram(salary_data, x="salary_from_rub", nbins=30, 
                             title="Распределение зарплат")
            fig.update_layout(
                xaxis_title="Зарплата (₽)",
                yaxis_title="Количество вакансий"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Salary by top cities
        if "area_name" in salary_data.columns:
            top_cities = salary_data["area_name"].value_counts().head(10).index
            city_sal = salary_data[salary_data["area_name"].isin(top_cities)]
            if len(city_sal) > 0:
                agg_city = city_sal.groupby("area_name")["salary_from_rub"].agg(["mean", "count"]).reset_index()
                agg_city = agg_city[agg_city["count"] >= 5]
                if len(agg_city) > 0:
                    fig = px.bar(agg_city, x="area_name", y="mean", 
                               title="Средняя зарплата по городам (минимум 5 вакансий)")
                    fig.update_layout(
                        xaxis_title="Город",
                        yaxis_title="Средняя зарплата (₽)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных о зарплатах для анализа")

with tabs[5]:  # Компании и домены
    st.subheader("Компании и бизнес-домены")
    
    col1, col2 = st.columns(2)
    
    with col1:
        vc = build_value_counts(filtered, "employer_name", 20)
        if len(vc):
            fig = px.bar(x=vc.values, y=vc.index, orientation="h", 
                        title="Топ компаний по количеству вакансий")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных о компаниях")
    
    with col2:
        vc_domains = build_array_counts(filtered, DS_COLUMNS["business_domains"], 15)
        if len(vc_domains):
            fig = px.pie(values=vc_domains.values, names=vc_domains.index, 
                        title="Бизнес-домены")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Нет данных о бизнес-доменах")

with tabs[6]:  # Корреляции
    st.subheader("Анализ корреляций")
    st.info("Интерактивный анализ взаимосвязей между технологиями и зарплатами")
    
    if len(filtered) > 0:
        # Technology co-occurrence analysis
        st.write("**Совместное использование технологий:**")
        
        # Get top technologies
        top_langs = build_array_counts(filtered, DS_COLUMNS["programming_languages"], 10)
        top_libs = build_array_counts(filtered, DS_COLUMNS["ml_libraries"], 10)
        
        if len(top_langs) > 0 and len(top_libs) > 0:
            # Create a simple co-occurrence matrix
            import numpy as np
            
            lang_names = top_langs.index.tolist()
            lib_names = top_libs.index.tolist()
            
            cooc_matrix = []
            for lang in lang_names[:5]:  # Limit for performance
                row = []
                for lib in lib_names[:5]:
                    # Count co-occurrences
                    lang_mask = filtered[DS_COLUMNS["programming_languages"]].apply(
                        lambda x: lang in parse_array_field(x)
                    )
                    lib_mask = filtered[DS_COLUMNS["ml_libraries"]].apply(
                        lambda x: lib in parse_array_field(x)
                    )
                    cooc = (lang_mask & lib_mask).sum()
                    row.append(cooc)
                cooc_matrix.append(row)
            
            if cooc_matrix:
                fig = px.imshow(cooc_matrix, 
                              x=lib_names[:5], y=lang_names[:5],
                              title="Совместное использование: Языки × ML библиотеки",
                              aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Выберите фильтры для анализа корреляций")

st.caption("Made with Streamlit. Кэширование включено (st.cache_data).")
