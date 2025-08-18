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
    # Basic filters
    area_vals = sorted([v for v in df.get("area_name", pd.Series(dtype=str)).dropna().unique()])
    experience_vals = sorted([v for v in df.get("experience_name", pd.Series(dtype=str)).dropna().unique()])
    employer_vals = sorted([v for v in df.get("employer_name", pd.Series(dtype=str)).dropna().unique()])

    area_sel = st.multiselect("География", area_vals)
    exp_sel = st.multiselect("Опыт работы", experience_vals)
    employer_sel = st.multiselect("Компания", employer_vals)

    # Salary slider
    sal = df["salary_from_rub"].dropna()
    s_min = int(sal.min()) if len(sal) else 0
    s_max = int(sal.max()) if len(sal) else 1_000_000
    salary_sel = st.slider("Зарплата от (₽)", min_value=s_min, max_value=s_max, value=(s_min, s_max))

    # Array filters (example)
    prog_opts = build_array_counts(df, DS_COLUMNS["programming_languages"], top_n=50).index.tolist()
    mllib_opts = build_array_counts(df, DS_COLUMNS["ml_libraries"], top_n=50).index.tolist()
    skill_opts = build_array_counts(df, "key_skills", top_n=50).index.tolist()

    prog_sel = st.multiselect("Языки программирования", prog_opts)
    mllib_sel = st.multiselect("ML библиотеки", mllib_opts)
    skill_sel = st.multiselect("Навыки", skill_opts)

    if st.button("Сбросить фильтры"):
        st.experimental_rerun()

# ---------- Filtering ----------
filtered = df.copy()
if area_sel:
    filtered = filtered[filtered.get("area_name").isin(area_sel)]
if exp_sel:
    filtered = filtered[filtered.get("experience_name").isin(exp_sel)]
if employer_sel:
    filtered = filtered[filtered.get("employer_name").isin(employer_sel)]

# Salary filter (apply only if range narrowed)
if salary_sel != (s_min, s_max):
    filtered = filtered[(filtered["salary_from_rub"].notna()) &
                        (filtered["salary_from_rub"] >= salary_sel[0]) &
                        (filtered["salary_from_rub"] <= salary_sel[1])]

# Array contains filter
 def contains_any(parsed_list, targets):
    return any(t in parsed_list for t in targets)

if prog_sel:
    mask = filtered[DS_COLUMNS["programming_languages"]].apply(lambda x: contains_any(parse_array_field(x), prog_sel))
    filtered = filtered[mask]
if mllib_sel:
    mask = filtered[DS_COLUMNS["ml_libraries"]].apply(lambda x: contains_any(parse_array_field(x), mllib_sel))
    filtered = filtered[mask]
if skill_sel:
    mask = filtered["key_skills"].apply(lambda x: contains_any(parse_array_field(x), skill_sel))
    filtered = filtered[mask]

# ---------- KPIs ----------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Вакансий", len(filtered))
with col2:
    s = filtered["salary_from_rub"].dropna()
    st.metric("Средняя зарплата", f"{int(s.mean()):,} ₽".replace(",", " ") if len(s) else "—")
with col3:
    st.metric("Медианная зарплата", f"{int(s.median()):,} ₽".replace(",", " ") if len(s) else "—")

# ---------- Tabs ----------
t1, t2, t3, t4, t5, t6 = st.tabs([
    "География", "Навыки", "Языки", "ML библиотеки", "Компании", "Зарплата vs Опыт"
])

with t1:
    vc = build_value_counts(filtered, "area_name", 20)
    if len(vc):
        fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="Вакансии по городам")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных для отображения")

with t2:
    vc = build_array_counts(filtered, "key_skills", 25)
    if len(vc):
        fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="Топ навыков")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных для отображения")

with t3:
    vc = build_array_counts(filtered, DS_COLUMNS["programming_languages"], 25)
    if len(vc):
        fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="Языки программирования")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных для отображения")

with t4:
    vc = build_array_counts(filtered, DS_COLUMNS["ml_libraries"], 25)
    if len(vc):
        fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="ML библиотеки")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных для отображения")

with t5:
    vc = build_value_counts(filtered, "employer_name", 30)
    if len(vc):
        fig = px.bar(x=vc.values, y=vc.index, orientation="h", title="Топ компаний по числу вакансий")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных для отображения")

with t6:
    d = filtered[(filtered["salary_from_rub"].notna()) & (filtered["experience_name"].notna())]
    if len(d):
        agg = d.groupby("experience_name")["salary_from_rub"].agg(["mean", "median"]).reset_index()
        fig = px.bar(agg, x="experience_name", y=["mean", "median"], barmode="group", title="Зарплата по опыту")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных о зарплатах")

st.caption("Made with Streamlit. Кэширование включено (st.cache_data).")
