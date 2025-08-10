import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import ast
import json
import numpy as np

# Конфигурация страницы
st.set_page_config(
    page_title="HR Analytics Dashboard", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Функция для загрузки данных с кешированием
@st.cache_data
def load_data():
    """Загружает данные из parquet файла"""
    return pd.read_parquet('data/vacancies.parquet')

# Конфигурация всех типов графиков и фильтров
CHART_CONFIG = {
    'company': {
        'title': 'Типы компаний',
        'chart_func': lambda df: create_single_field_chart(df, 'company_type', 'Распределение вакансий по типам компаний'),
        'filter_field': 'company_type'
    },
    'domain': {
        'title': 'Бизнес-домены', 
        'chart_func': lambda df: create_single_field_chart(df, 'business_domain', 'Топ бизнес-домены', top_n=15),
        'filter_field': 'business_domain'
    },
    'skills': {
        'title': 'Навыки и технологии',
        'chart_func': lambda df: create_array_field_chart(df, 'key_skills', 'Топ навыки и технологии', normalize_skill),
        'filter_field': 'key_skills'
    },
    'fe_framework': {
        'title': 'Фронтенд-фреймворки',
        'chart_func': lambda df: create_array_field_chart(df, 'fe_framework', 'Топ фронтенд-фреймворки'),
        'filter_field': 'fe_framework'
    },
    'state_mgmt': {
        'title': 'Управление состоянием',
        'chart_func': lambda df: create_array_field_chart(df, 'state_mgmt', 'Библиотеки управления состоянием'),
        'filter_field': 'state_mgmt'
    },
    'styling': {
        'title': 'Стилизация',
        'chart_func': lambda df: create_array_field_chart(df, 'styling', 'Технологии стилизации'),
        'filter_field': 'styling'
    },
    'testing': {
        'title': 'Тестирование',
        'chart_func': lambda df: create_array_field_chart(df, 'testing', 'Фреймворки тестирования'),
        'filter_field': 'testing'
    },
    'api_proto': {
        'title': 'API протоколы',
        'chart_func': lambda df: create_array_field_chart(df, 'api_proto', 'API протоколы и форматы'),
        'filter_field': 'api_proto'
    },
    'experience': {
        'title': 'Опыт работы',
        'chart_func': lambda df: create_single_field_chart(df, 'experience_name', 'Распределение вакансий по категориям опыта'),
        'filter_field': 'experience_name'
    },
    'employers': {
        'title': 'Компании',
        'chart_func': lambda df: create_single_field_chart(df, 'employer_name', 'Топ компании по количеству вакансий', top_n=20),
        'filter_field': 'employer_name'
    },
    'salary': {
        'title': 'Зарплаты',
        'chart_func': lambda df: create_salary_experience_chart(df),
        'filter_field': None
    }
}

# Фильтры для sidebar
FILTERS_CONFIG = [
    {'id': 'company-filter', 'label': 'Тип компании', 'field': 'company_type'},
    {'id': 'domain-filter', 'label': 'Бизнес-домен', 'field': 'business_domain'},
    {'id': 'experience-filter', 'label': 'Опыт работы', 'field': 'experience_name'},
    {'id': 'employer-filter', 'label': 'Компания', 'field': 'employer_name'},
    {'id': 'skills-filter', 'label': 'Навыки и технологии', 'field': 'key_skills', 'type': 'array'},
    {'id': 'fe-framework-filter', 'label': 'Фронтенд-фреймворки', 'field': 'fe_framework', 'type': 'array'},
    {'id': 'state-mgmt-filter', 'label': 'Управление состоянием', 'field': 'state_mgmt', 'type': 'array'},
    {'id': 'styling-filter', 'label': 'Стилизация', 'field': 'styling', 'type': 'array'},
    {'id': 'testing-filter', 'label': 'Тестирование', 'field': 'testing', 'type': 'array'},
    {'id': 'api-proto-filter', 'label': 'API протоколы', 'field': 'api_proto', 'type': 'array'}
]

# Универсальная функция для парсинга массивных полей из разных форматов
def parse_array_field(array_data):
    """
    Парсит массивы из различных форматов данных:
    - numpy arrays
    - Python lists
    - JSON strings  
    - comma-separated strings
    - literal eval strings
    
    Args:
        array_data: данные любого формата
        
    Returns:
        list: список строк с элементами
    """
    if array_data is None:
        return []
    
    try:
        if pd.isna(array_data):
            return []
    except (TypeError, ValueError):
        pass
    
    try:
        if isinstance(array_data, np.ndarray):
            if array_data.size == 0:
                return []
            items = []
            for item in array_data:
                item_str = str(item).strip()
                if item_str and item_str != 'nan' and item_str != 'None':
                    items.append(item_str)
            return items
        
        if isinstance(array_data, (list, tuple)):
            items = []
            for item in array_data:
                item_str = str(item).strip()
                if item_str and item_str != 'nan' and item_str != 'None':
                    items.append(item_str)
            return items
        
        if isinstance(array_data, str):
            array_data = array_data.strip()
            
            if not array_data:
                return []
            
            if array_data.startswith('[') and array_data.endswith(']'):
                try:
                    parsed = json.loads(array_data)
                    if isinstance(parsed, list):
                        items = []
                        for item in parsed:
                            item_str = str(item).strip()
                            if item_str and item_str != 'nan' and item_str != 'None':
                                items.append(item_str)
                        return items
                except (json.JSONDecodeError, ValueError):
                    pass
                
                try:
                    parsed = ast.literal_eval(array_data)
                    if isinstance(parsed, list):
                        items = []
                        for item in parsed:
                            item_str = str(item).strip()
                            if item_str and item_str != 'nan' and item_str != 'None':
                                items.append(item_str)
                        return items
                except (SyntaxError, ValueError):
                    pass
            
            items = [s.strip() for s in array_data.split(',')]
            return [item for item in items if item]
        
        return [str(array_data).strip()] if str(array_data).strip() else []
        
    except Exception as e:
        print(f"Warning: Error parsing array data {array_data}: {e}")
        return []

# Функция нормализации навыков
def normalize_skill(skill):
    """
    Нормализует название навыка:
    - убирает лишние пробелы
    - приводит к единому написанию популярных навыков
    """
    if not skill or not isinstance(skill, str):
        return None
    
    skill = skill.strip()
    if not skill:
        return None
    
    skill_normalization = {
        'react': 'React',
        'reactjs': 'React', 
        'react.js': 'React',
        'javascript': 'JavaScript',
        'js': 'JavaScript',
        'typescript': 'TypeScript',
        'ts': 'TypeScript',
        'vue': 'Vue.js',
        'vuejs': 'Vue.js',
        'vue.js': 'Vue.js',
        'nodejs': 'Node.js',
        'node.js': 'Node.js',
        'node': 'Node.js',
        'nextjs': 'Next.js',
        'next.js': 'Next.js',
        'next': 'Next.js',
        'html5': 'HTML',
        'css3': 'CSS',
        'restapi': 'REST API',
        'rest api': 'REST API',
        'api': 'API',
        'github': 'Git',
    }
    
    skill_lower = skill.lower()
    return skill_normalization.get(skill_lower, skill)

# Функция для нормализации зарплат в рубли
def normalize_salary(row):
    try:
        salary_from = row['salary_from'] 
        salary_to = row['salary_to']
        currency = row['salary_currency']
        is_gross = row['salary_gross']
        
        if pd.isna(salary_from) or salary_from <= 0:
            return pd.Series({'salary_from_rub': None, 'salary_to_rub': None})
        
        exchange_rates = {
            'RUR': 1,
            'USD': 95,
            'EUR': 105,
            'KZT': 0.2,
            'UZS': 0.0075,
            'BYR': 0.035,
            'UAH': 2.5,
            'KGS': 1.1,
            'AZN': 55
        }
        
        rate = exchange_rates.get(currency, 1) if pd.notna(currency) else 1
        
        salary_from_rub = float(salary_from * rate) if pd.notna(salary_from) and salary_from > 0 else None
        salary_to_rub = float(salary_to * rate) if pd.notna(salary_to) and salary_to > 0 else None
        
        if pd.notna(is_gross) and float(is_gross) == 1.0:
            if salary_from_rub:
                salary_from_rub = float(salary_from_rub * 0.87)
            if salary_to_rub:
                salary_to_rub = float(salary_to_rub * 0.87)
        
        if salary_from_rub and (salary_from_rub < 1000 or salary_from_rub > 10000000):
            salary_from_rub = None
        if salary_to_rub and (salary_to_rub < 1000 or salary_to_rub > 10000000):
            salary_to_rub = None
            
        return pd.Series({
            'salary_from_rub': salary_from_rub,
            'salary_to_rub': salary_to_rub
        })
    except Exception as e:
        return pd.Series({'salary_from_rub': None, 'salary_to_rub': None})

# Функция для получения уникальных значений из массивного поля
def get_unique_array_values(df, field_name, limit=50):
    """Получает уникальные значения из массивного поля"""
    if field_name not in df.columns:
        return []
    
    all_values = []
    for array_data in df[field_name].dropna():
        parsed_items = parse_array_field(array_data)
        all_values.extend(parsed_items)
    
    if not all_values:
        return []
    
    values_counts = pd.Series(all_values).value_counts()
    return values_counts.head(limit).index.tolist()

# Функция для создания графиков по одиночным полям
def create_single_field_chart(filtered_df, field_name, title, top_n=None):
    """Универсальная функция для создания графиков по одиночным полям"""
    try:
        if field_name not in filtered_df.columns:
            return create_empty_chart(title, "Поле не найдено")
        
        field_counts = filtered_df[field_name].dropna().value_counts()
        if top_n:
            field_counts = field_counts.head(top_n)
        
        if len(field_counts) == 0:
            return create_empty_chart(title, "Нет данных")
        
        fig = px.bar(
            x=field_counts.values,
            y=field_counts.index,
            orientation='h',
            title=title,
            labels={'x': 'Количество вакансий', 'y': field_name.replace('_', ' ').title()}
        )
        
        fig.update_layout(height=500)
        return fig
        
    except Exception as e:
        print(f"Error in create_single_field_chart for {field_name}: {e}")
        return create_empty_chart(title, "Ошибка обработки данных")

# Вспомогательная функция для пустых графиков
def create_empty_chart(title, message):
    """Создает пустой график с сообщением"""
    fig = px.bar(x=[0], y=[message], orientation='h', title=title)
    fig.update_layout(height=500)
    return fig

# Функция для создания графиков по массивным полям
def create_array_field_chart(filtered_df, field_name, title, normalizer_func=None, top_n=50):
    """
    Универсальная функция для создания графиков по массивным полям.
    """
    try:
        if field_name not in filtered_df.columns:
            return create_empty_chart(title, "Поле не найдено")
        
        all_items = []
        
        for array_data in filtered_df[field_name]:
            parsed_items = parse_array_field(array_data)
            if normalizer_func:
                normalized_items = [normalizer_func(item) for item in parsed_items]
                valid_items = [item for item in normalized_items if item is not None and len(str(item)) > 1]
            else:
                valid_items = [item for item in parsed_items if item and len(str(item).strip()) > 1]
            all_items.extend(valid_items)
        
        if not all_items:
            return create_empty_chart(title, "Нет данных")
        
        items_counts = pd.Series(all_items).value_counts().head(top_n)
        
        # Убираем фильтрацию по минимальной частоте для полной статистики
        # min_frequency = max(1, len(filtered_df) // 200)
        # items_counts = items_counts[items_counts >= min_frequency]
        
        if len(items_counts) == 0:
            return create_empty_chart(title, "Слишком мало данных")
        
        fig = px.bar(
            x=items_counts.values,
            y=items_counts.index,
            orientation='h',
            title=title,
            labels={'x': 'Частота упоминания', 'y': 'Элементы'},
            color=items_counts.values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            xaxis_title="Количество вакансий",
            yaxis_title=field_name.replace('_', ' ').title(),
            font=dict(size=12)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in create_array_field_chart for {field_name}: {e}")
        return create_empty_chart(title, "Ошибка обработки данных")

# Функция для создания графика зарплат по опыту
def create_salary_experience_chart(filtered_df):
    try:
        valid_data = filtered_df[
            (filtered_df['salary_from_rub'].notna()) & 
            (filtered_df['experience_name'].notna()) &
            (filtered_df['experience_name'] != '')
        ]
        
        if len(valid_data) == 0:
            return create_empty_chart("Средняя зарплата по опыту работы", "Нет данных")
        
        salary_by_exp = valid_data.groupby('experience_name').agg({
            'salary_from_rub': 'mean',
            'salary_to_rub': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        if not salary_by_exp['salary_from_rub'].isna().all():
            fig.add_trace(go.Bar(
                name='Зарплата от',
                x=salary_by_exp['experience_name'],
                y=salary_by_exp['salary_from_rub'].fillna(0),
                yaxis='y',
                offsetgroup=1
            ))
        
        if not salary_by_exp['salary_to_rub'].isna().all():
            fig.add_trace(go.Bar(
                name='Зарплата до', 
                x=salary_by_exp['experience_name'],
                y=salary_by_exp['salary_to_rub'].fillna(0),
                yaxis='y',
                offsetgroup=2
            ))
        
        fig.update_layout(
            title="Средняя зарплата по опыту работы",
            xaxis_title="Опыт работы",
            yaxis_title="Средняя зарплата (₽)",
            height=500,
            barmode='group'
        )
        
        return fig
    except Exception as e:
        print(f"Error in create_salary_experience_chart: {e}")
        return create_empty_chart("Ошибка загрузки данных", "Ошибка")

# Функция для фильтрации данных
def filter_data(df, filters):
    """Применяет все фильтры к DataFrame"""
    filtered_df = df.copy()
    
    for filter_key, filter_values in filters.items():
        if filter_values and len(filter_values) > 0:
            if filter_key in ['salary_from', 'salary_to']:
                continue
            
            # Найдем конфигурацию фильтра
            filter_config = None
            for config in FILTERS_CONFIG:
                if config['field'] == filter_key:
                    filter_config = config
                    break
            
            if filter_config:
                if filter_config.get('type') == 'array':
                    filtered_df = filter_by_multiple_array_values(filtered_df, filter_key, filter_values)
                else:
                    filtered_df = filtered_df[filtered_df[filter_key].isin(filter_values)]
    
    # Специальная обработка фильтра зарплат
    if 'salary_range' in filters and filters['salary_range']:
        min_salary, max_salary = filters['salary_range']
        # Включаем записи без зарплаты ИЛИ с зарплатой в диапазоне
        salary_filter = (
            (filtered_df['salary_from_rub'].isna()) |
            ((filtered_df['salary_from_rub'].notna()) &
             (filtered_df['salary_from_rub'] >= min_salary) & 
             (filtered_df['salary_from_rub'] <= max_salary))
        )
        filtered_df = filtered_df[salary_filter]
    
    return filtered_df

# Функция для фильтрации по множественным значениям в массивном поле
def filter_by_multiple_array_values(df_to_filter, field_name, target_values):
    """Фильтрует DataFrame по множественным значениям в массивном поле"""
    if not target_values or len(target_values) == 0 or field_name not in df_to_filter.columns:
        return df_to_filter
    
    mask = []
    for array_data in df_to_filter[field_name]:
        parsed_items = parse_array_field(array_data)
        has_match = any(value in parsed_items for value in target_values)
        mask.append(has_match)
    
    return df_to_filter[mask]

# Основная логика приложения
def main():
    # Загружаем данные
    df = load_data()
    
    # Применяем нормализацию зарплат
    if 'salary_from_rub' not in df.columns or 'salary_to_rub' not in df.columns:
        df[['salary_from_rub', 'salary_to_rub']] = df.apply(normalize_salary, axis=1)
    
    # Вычисляем границы для слайдера зарплат
    salary_data = df['salary_from_rub'].dropna()
    if len(salary_data) > 0:
        SALARY_MIN = max(0, int(salary_data.min()))
        SALARY_MAX = int(salary_data.max())
    else:
        SALARY_MIN = 0
        SALARY_MAX = 1000000
    
    # Боковая панель с фильтрами
    st.sidebar.header("🔍 Фильтры")
    
    filters = {}
    
    # Создаем фильтры в боковой панели
    for filter_config in FILTERS_CONFIG:
        field = filter_config['field']
        label = filter_config['label']
        filter_type = filter_config.get('type', 'single')
        
        st.sidebar.subheader(label)
        
        if filter_type == 'array':
            # Для массивных полей получаем уникальные значения
            options = get_unique_array_values(df, field, limit=30)
            if options:
                selected = st.sidebar.multiselect(
                    f"Выберите {label.lower()}:",
                    options=options,
                    key=f"{field}_filter"
                )
                filters[field] = selected
        else:
            # Для обычных полей
            unique_values = df[field].dropna().unique().tolist()[:50]  # Ограничиваем количество
            if unique_values:
                selected = st.sidebar.multiselect(
                    f"Выберите {label.lower()}:",
                    options=sorted(unique_values),
                    key=f"{field}_filter"
                )
                filters[field] = selected
    
    # Фильтр зарплат
    st.sidebar.subheader("💰 Диапазон зарплат (₽)")
    salary_range = st.sidebar.slider(
        "Выберите диапазон:",
        min_value=SALARY_MIN,
        max_value=SALARY_MAX,
        value=(SALARY_MIN, SALARY_MAX),
        format="%d"
    )
    filters['salary_range'] = salary_range
    
    # Кнопка сброса фильтров
    if st.sidebar.button("🔄 Сбросить все фильтры"):
        st.rerun()
    
    # Применяем фильтры
    filtered_df = filter_data(df, filters)
    
    # Статистика в боковой панели
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Статистика")
    st.sidebar.metric("Всего вакансий", len(df))
    st.sidebar.metric("Отфильтровано", len(filtered_df))
    
    # Статистика по зарплатам
    salary_data = filtered_df['salary_from_rub'].dropna()
    if len(salary_data) > 0:
        avg_salary = salary_data.mean()
        median_salary = salary_data.median()
        st.sidebar.metric("Средняя зарплата", f"{avg_salary:,.0f} ₽")
        st.sidebar.metric("Медианная зарплата", f"{median_salary:,.0f} ₽")
    
    # Основной контент - система табов
    tab_names = [config['title'] for config in CHART_CONFIG.values()]
    tabs = st.tabs(tab_names)
    
    for i, (key, config) in enumerate(CHART_CONFIG.items()):
        with tabs[i]:
            try:
                chart = config['chart_func'](filtered_df)
                st.plotly_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка при создании графика: {str(e)}")
                st.info("Попробуйте изменить фильтры или обновить страницу")

if __name__ == "__main__":
    main()