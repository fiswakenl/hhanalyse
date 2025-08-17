import dash
from dash import dcc, html, Input, Output, callback_context, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
import ast
import json
import numpy as np

# Загрузка данных
df = pd.read_parquet('data/ds_vacancies.parquet')

# Mapping DS column indices to readable names for easier reference
DS_COLUMNS = {
    'programming_languages': df.columns[14],  # Языки программирования
    'ml_libraries': df.columns[15],           # ML библиотеки
    'visualization': df.columns[16],          # Визуализация
    'data_processing': df.columns[17],        # Обработка данных
    'nlp_tools': df.columns[18],             # NLP инструменты
    'cv_tools': df.columns[19],              # CV инструменты
    'mlops_tools': df.columns[20],           # MLOps инструменты
}

# Конфигурация всех типов графиков и фильтров
CHART_CONFIG = {
    'skills': {
        'title': 'Общие навыки',
        'tab_id': 'skills-tab',
        'chart_func': lambda df: create_array_field_chart(df, 'key_skills', 'Топ навыки и технологии', normalize_skill),
        'filter_field': 'key_skills'
    },
    'programming': {
        'title': 'Языки программирования',
        'tab_id': 'programming-tab',
        'chart_func': lambda df: create_array_field_chart(df, DS_COLUMNS['programming_languages'], 'Языки программирования'),
        'filter_field': DS_COLUMNS['programming_languages']
    },
    'ml_libraries': {
        'title': 'ML библиотеки',
        'tab_id': 'ml-libraries-tab',
        'chart_func': lambda df: create_array_field_chart(df, DS_COLUMNS['ml_libraries'], 'ML библиотеки и фреймворки'),
        'filter_field': DS_COLUMNS['ml_libraries']
    },
    'visualization': {
        'title': 'Визуализация',
        'tab_id': 'visualization-tab',
        'chart_func': lambda df: create_array_field_chart(df, DS_COLUMNS['visualization'], 'Инструменты визуализации'),
        'filter_field': DS_COLUMNS['visualization']
    },
    'data_processing': {
        'title': 'Обработка данных',
        'tab_id': 'data-processing-tab',
        'chart_func': lambda df: create_array_field_chart(df, DS_COLUMNS['data_processing'], 'Библиотеки обработки данных'),
        'filter_field': DS_COLUMNS['data_processing']
    },
    'nlp_tools': {
        'title': 'NLP инструменты',
        'tab_id': 'nlp-tools-tab',
        'chart_func': lambda df: create_array_field_chart(df, DS_COLUMNS['nlp_tools'], 'NLP библиотеки и инструменты'),
        'filter_field': DS_COLUMNS['nlp_tools']
    },
    'cv_tools': {
        'title': 'Computer Vision',
        'tab_id': 'cv-tools-tab',
        'chart_func': lambda df: create_array_field_chart(df, DS_COLUMNS['cv_tools'], 'Computer Vision инструменты'),
        'filter_field': DS_COLUMNS['cv_tools']
    },
    'mlops_tools': {
        'title': 'MLOps инструменты',
        'tab_id': 'mlops-tools-tab',
        'chart_func': lambda df: create_array_field_chart(df, DS_COLUMNS['mlops_tools'], 'MLOps платформы и инструменты'),
        'filter_field': DS_COLUMNS['mlops_tools']
    },
    'experience': {
        'title': 'Опыт работы',
        'tab_id': 'experience-tab',
        'chart_func': lambda df: create_single_field_chart(df, 'experience_name', 'Распределение вакансий по категориям опыта'),
        'filter_field': 'experience_name'
    },
    'employers': {
        'title': 'Компании',
        'tab_id': 'employers-tab',
        'chart_func': lambda df: create_single_field_chart(df, 'employer_name', 'Топ компании по количеству DS вакансий', top_n=20),
        'filter_field': 'employer_name'
    },
    'work_format': {
        'title': 'Формат работы',
        'tab_id': 'work-format-tab',
        'chart_func': lambda df: create_array_field_chart(df, 'work_format', 'Форматы работы'),
        'filter_field': 'work_format'
    },
    'salary': {
        'title': 'Зарплаты',
        'tab_id': 'salary-tab', 
        'chart_func': lambda df: create_salary_experience_chart(df),
        'filter_field': None
    }
}


# Фильтры для sidebar
FILTERS_CONFIG = [
    {'id': 'experience-filter', 'label': 'Опыт работы', 'field': 'experience_name'},
    {'id': 'employer-filter', 'label': 'Компания', 'field': 'employer_name'},
    {'id': 'company-vacancy-count-filter', 'label': 'Количество вакансий у компании', 'field': 'employer_name', 'type': 'vacancy_count'},
    {'id': 'skills-filter', 'label': 'Общие навыки', 'field': 'key_skills', 'type': 'array'},
    {'id': 'programming-filter', 'label': 'Языки программирования', 'field': DS_COLUMNS['programming_languages'], 'type': 'array'},
    {'id': 'ml-libraries-filter', 'label': 'ML библиотеки', 'field': DS_COLUMNS['ml_libraries'], 'type': 'array'},
    {'id': 'visualization-filter', 'label': 'Визуализация', 'field': DS_COLUMNS['visualization'], 'type': 'array'},
    {'id': 'data-processing-filter', 'label': 'Обработка данных', 'field': DS_COLUMNS['data_processing'], 'type': 'array'},
    {'id': 'nlp-tools-filter', 'label': 'NLP инструменты', 'field': DS_COLUMNS['nlp_tools'], 'type': 'array'},
    {'id': 'cv-tools-filter', 'label': 'Computer Vision', 'field': DS_COLUMNS['cv_tools'], 'type': 'array'},
    {'id': 'mlops-tools-filter', 'label': 'MLOps инструменты', 'field': DS_COLUMNS['mlops_tools'], 'type': 'array'},
    {'id': 'work-format-filter', 'label': 'Формат работы', 'field': 'work_format', 'type': 'array'}
]

# Функция для получения уникальных значений из массивного поля
def get_unique_array_values(field_name, limit=50):
    """Получает уникальные значения из массивного поля"""
    if field_name not in df.columns:
        return []
    
    all_values = []
    for array_data in df[field_name].dropna():
        parsed_items = parse_array_field(array_data)
        all_values.extend(parsed_items)
    
    if not all_values:
        return []
    
    # Возвращаем топ-значения по частоте
    values_counts = pd.Series(all_values).value_counts()
    return values_counts.head(limit).index.tolist()

# Функция для получения опций с подсчетом для массивных полей
def get_array_field_options_with_counts(field_name, filtered_data=None):
    """Получает опции для массивного поля с подсчетом популярности"""
    data_source = filtered_data if filtered_data is not None else df
    
    if field_name not in data_source.columns:
        return []
    
    all_values = []
    for array_data in data_source[field_name].dropna():
        parsed_items = parse_array_field(array_data)
        all_values.extend(parsed_items)
    
    if not all_values:
        return []
    
    # Подсчитываем частоту и сортируем по убыванию (без лимита)
    values_counts = pd.Series(all_values).value_counts()
    
    options = []
    for value, count in values_counts.items():
        if pd.notna(value) and str(value).strip() != '' and str(value) != 'nan':
            options.append({
                "label": f"{value} ({count})",
                "value": str(value)
            })
    
    return options

# Функция для получения опций с подсчетом для обычных полей
def get_single_field_options_with_counts(field_name, filtered_data=None):
    """Получает опции для обычного поля с подсчетом популярности"""
    data_source = filtered_data if filtered_data is not None else df
    
    if field_name not in data_source.columns:
        return []
    
    # Подсчитываем частоту и сортируем по убыванию
    values_counts = data_source[field_name].dropna().value_counts()
    
    options = []
    for value, count in values_counts.items():
        if pd.notna(value) and str(value).strip() != '' and str(value) != 'nan':
            options.append({
                "label": f"{value} ({count})",
                "value": str(value)
            })
    
    return options

# Функция для получения опций фильтра по количеству вакансий у компании
def get_company_vacancy_count_options(filtered_data=None):
    """Получает опции фильтра по количеству вакансий у компании"""
    data_source = filtered_data if filtered_data is not None else df
    
    # Подсчитываем количество вакансий у каждой компании
    company_counts = data_source['employer_name'].value_counts()
    
    # Создаем диапазоны количества вакансий
    ranges = [
        {"label": "1 вакансия", "value": "1"},
        {"label": "2-3 вакансии", "value": "2-3"},
        {"label": "4-5 вакансий", "value": "4-5"},
        {"label": "6-10 вакансий", "value": "6-10"},
        {"label": "11-20 вакансий", "value": "11-20"},
        {"label": "21-50 вакансий", "value": "21-50"},
        {"label": "50+ вакансий", "value": "50+"}
    ]
    
    # Подсчитываем количество компаний в каждом диапазоне
    options_with_counts = []
    for range_info in ranges:
        range_value = range_info["value"]
        count = 0
        
        if range_value == "1":
            count = (company_counts == 1).sum()
        elif range_value == "2-3":
            count = ((company_counts >= 2) & (company_counts <= 3)).sum()
        elif range_value == "4-5":
            count = ((company_counts >= 4) & (company_counts <= 5)).sum()
        elif range_value == "6-10":
            count = ((company_counts >= 6) & (company_counts <= 10)).sum()
        elif range_value == "11-20":
            count = ((company_counts >= 11) & (company_counts <= 20)).sum()
        elif range_value == "21-50":
            count = ((company_counts >= 21) & (company_counts <= 50)).sum()
        elif range_value == "50+":
            count = (company_counts > 50).sum()
        
        if count > 0:
            options_with_counts.append({
                "label": f"{range_info['label']} ({count} компаний)",
                "value": range_value
            })
    
    return options_with_counts

# Генерация фильтров
def generate_filters():
    """Генерирует все фильтры для sidebar"""
    filters = []
    
    for filter_config in FILTERS_CONFIG:
        field = filter_config['field']
        filter_type = filter_config.get('type', 'single')
        
        if filter_type == 'array':
            # Для массивных полей получаем значения с их частотой
            options = get_array_field_options_with_counts(field)
        elif filter_type == 'vacancy_count':
            # Для фильтра по количеству вакансий у компании
            options = get_company_vacancy_count_options()
        else:
            # Для обычных полей получаем значения с их частотой
            options = get_single_field_options_with_counts(field)
        
        filters.extend([
            html.H6(filter_config['label']),
            dcc.Dropdown(
                id=filter_config['id'],
                options=options,
                value=[],  # Пустой список по умолчанию для multiselect
                multi=True,  # Включаем множественный выбор
                searchable=True,  # Включаем поиск для удобства навигации
                placeholder=f"Выберите {filter_config['label'].lower()}..."
            ),
            html.Br()
        ])
    
    return filters

# Генерация фильтров с динамическими опциями на основе отфильтрованных данных
def generate_dynamic_filter_options(filtered_df):
    """Генерирует опции для всех фильтров на основе отфильтрованных данных"""
    filter_options = {}
    
    for filter_config in FILTERS_CONFIG:
        field = filter_config['field']
        filter_type = filter_config.get('type', 'single')
        filter_id = filter_config['id']
        
        if filter_type == 'array':
            # Для массивных полей получаем значения с их частотой
            options = get_array_field_options_with_counts(field, filtered_df)
        elif filter_type == 'vacancy_count':
            # Для фильтра по количеству вакансий у компании
            options = get_company_vacancy_count_options(filtered_df)
        else:
            # Для обычных полей получаем значения с их частотой
            options = get_single_field_options_with_counts(field, filtered_df)
        
        filter_options[filter_id] = options
    
    return filter_options

# Генерация табов
def generate_tabs():
    """Генерирует все табы из конфигурации"""
    tabs = []
    for config in CHART_CONFIG.values():
        tabs.append(dbc.Tab(label=config['title'], tab_id=config['tab_id']))
    return tabs

# Универсальная генерация контента табов
def generate_tab_content(active_tab, filtered_df):
    """Генерирует контент для активного таба на основе конфигурации"""
    
    # Находим конфигурацию для активного таба
    config = None
    for key, cfg in CHART_CONFIG.items():
        if cfg['tab_id'] == active_tab:
            config = cfg
            break
    
    if not config:
        return html.Div("Таб не найден")
    
    # Генерация графика по конфигурации
    chart = config['chart_func'](filtered_df)
    chart_id = f"{active_tab.replace('-tab', '')}-chart"
    
    return dbc.Row([
        dbc.Col([
            dcc.Graph(figure=chart, id=chart_id)
        ], width=12)
    ])

# Универсальная функция для одиночных полей (company_type, business_domain)
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
    # Проверяем на None и NaN с учетом numpy arrays
    if array_data is None:
        return []
    
    # Для numpy arrays проверяем через try/except
    try:
        if pd.isna(array_data):
            return []
    except (TypeError, ValueError):
        # Если pd.isna не работает (например, для numpy array), продолжаем
        pass
    
    try:
        # Если это numpy array
        if isinstance(array_data, np.ndarray):
            if array_data.size == 0:
                return []
            # Фильтруем пустые значения и пустые строки
            items = []
            for item in array_data:
                item_str = str(item).strip()
                if item_str and item_str != 'nan' and item_str != 'None':
                    items.append(item_str)
            return items
        
        # Если это уже список
        if isinstance(array_data, (list, tuple)):
            items = []
            for item in array_data:
                item_str = str(item).strip()
                if item_str and item_str != 'nan' and item_str != 'None':
                    items.append(item_str)
            return items
        
        # Если это строка
        if isinstance(array_data, str):
            array_data = array_data.strip()
            
            if not array_data:
                return []
            
            # Попытка парсинга как JSON массив
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
                
                # Попытка парсинга как Python literal
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
            
            # Разделение по запятым как fallback
            items = [s.strip() for s in array_data.split(',')]
            return [item for item in items if item]
        
        # Для других типов - попытка конвертации в строку
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
    
    # Словарь для нормализации популярных навыков
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
        'github': 'Git',  # GitHub часто означает знание Git
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
        
        # Если нет данных о зарплате от, возвращаем None
        if pd.isna(salary_from) or salary_from <= 0:
            return pd.Series({'salary_from_rub': None, 'salary_to_rub': None})
        
        # Курсы валют (актуализированы для DS данных)
        exchange_rates = {
            'RUR': 1,      # Рубли - базовая валюта
            'USD': 95,     # Доллары США
            'EUR': 105,    # Евро
            'KZT': 0.2,    # Казахстанский тенге
            'UZS': 0.0075, # Узбекский сум
            'BYR': 0.035,  # Белорусский рубль
            'UAH': 2.5,    # Украинская гривна
            'KGS': 1.1,    # Киргизский сом
            'AZN': 55      # Азербайджанский манат
        }
        
        # Проверяем валюту и используем курс, по умолчанию рубли
        rate = exchange_rates.get(currency, 1) if pd.notna(currency) else 1
        
        # Конвертируем в рубли с проверкой типов
        salary_from_rub = float(salary_from * rate) if pd.notna(salary_from) and salary_from > 0 else None
        salary_to_rub = float(salary_to * rate) if pd.notna(salary_to) and salary_to > 0 else None
        
        # Если зарплата gross (с налогами), то чистая ~= gross * 0.87
        if pd.notna(is_gross) and float(is_gross) == 1.0:
            if salary_from_rub:
                salary_from_rub = float(salary_from_rub * 0.87)
            if salary_to_rub:
                salary_to_rub = float(salary_to_rub * 0.87)
        
        # Ограничиваем разумными значениями (от 1000 до 10млн рублей)
        if salary_from_rub and (salary_from_rub < 1000 or salary_from_rub > 10000000):
            salary_from_rub = None
        if salary_to_rub and (salary_to_rub < 1000 or salary_to_rub > 10000000):
            salary_to_rub = None
            
        return pd.Series({
            'salary_from_rub': salary_from_rub,
            'salary_to_rub': salary_to_rub
        })
    except Exception as e:
        # В случае ошибки возвращаем None
        return pd.Series({'salary_from_rub': None, 'salary_to_rub': None})

# Применяем нормализацию
df[['salary_from_rub', 'salary_to_rub']] = df.apply(normalize_salary, axis=1)

# Вычисляем безопасные границы для слайдера зарплат
salary_data = df['salary_from_rub'].dropna()
if len(salary_data) > 0:
    SALARY_MIN = max(0, int(salary_data.min()))
    SALARY_MAX = int(salary_data.max())
else:
    # Дефолтные значения если нет данных о зарплатах
    SALARY_MIN = 0
    SALARY_MAX = 1000000

# Инициализация приложения
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "DS Analytics Dashboard"

# Стили
CARD_STYLE = {
    "box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.18)",
    "margin-bottom": "24px",
    "padding": "16px"
}

# Layout приложения  
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("DS Analytics Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        # Sidebar с фильтрами
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Фильтры"),
                dbc.CardBody([
                    *generate_filters(),
                    
                    html.H6("Диапазон зарплат (₽)"),
                    dcc.RangeSlider(
                        id="salary-filter",
                        min=SALARY_MIN,
                        max=SALARY_MAX,
                        value=[SALARY_MIN, SALARY_MAX],
                        marks={
                            SALARY_MIN: f"{int(SALARY_MIN/1000)}k",
                            SALARY_MAX: f"{int(SALARY_MAX/1000)}k"
                        } if SALARY_MAX > SALARY_MIN else {},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Br(),
                    
                    
                    dbc.Button("Сбросить все фильтры", id="reset-filters", color="secondary", size="sm"),
                    html.Hr(),
                    
                    html.Div(id="filter-stats")
                ])
            ], style=CARD_STYLE)
        ], width=3),
        
        # Основной контент
        dbc.Col([
            # Табы
            dbc.Tabs(generate_tabs(), id="tabs", active_tab="company-tab"),
            
            html.Br(),
            
            # Содержимое табов
            html.Div(id="tab-content")
        ], width=9)
    ])
    
], fluid=True)

# Старые функции создания графиков заменены универсальными

def create_array_field_chart(filtered_df, field_name, title, normalizer_func=None, top_n=20):
    """
    Универсальная функция для создания графиков по массивным полям.
    
    Args:
        filtered_df: отфильтрованный DataFrame
        field_name: название поля для анализа (например, 'key_skills', 'fe_framework')
        title: заголовок графика
        normalizer_func: функция для нормализации значений (необязательно)
        top_n: количество топ-элементов для отображения
    """
    try:
        # Проверяем наличие поля в данных
        if field_name not in filtered_df.columns:
            fig = px.bar(
                x=[0], y=["Поле не найдено"],
                orientation='h',
                title=title
            )
            fig.update_layout(height=500)
            return fig
        
        # Обработка массивных данных
        all_items = []
        
        for array_data in filtered_df[field_name]:
            parsed_items = parse_array_field(array_data)
            # Применяем нормализацию если есть функция
            if normalizer_func:
                normalized_items = [normalizer_func(item) for item in parsed_items]
                # Фильтрация пустых значений
                valid_items = [item for item in normalized_items if item is not None and len(str(item)) > 1]
            else:
                # Простая фильтрация без нормализации
                valid_items = [item for item in parsed_items if item and len(str(item).strip()) > 1]
            all_items.extend(valid_items)
        
        if not all_items:
            # Если нет данных, создаем пустой график
            fig = px.bar(
                x=[0], y=["Нет данных"],
                orientation='h',
                title=title
            )
            fig.update_layout(height=500)
            return fig
        
        # Подсчет частоты элементов
        items_counts = pd.Series(all_items).value_counts().head(top_n)
        
        # Фильтрация элементов с минимальной частотой (убираем редкие)
        min_frequency = max(1, len(filtered_df) // 50)  # Минимум 2% от общего количества записей
        items_counts = items_counts[items_counts >= min_frequency]
        
        if len(items_counts) == 0:
            fig = px.bar(
                x=[0], y=["Слишком мало данных"],
                orientation='h',
                title=title
            )
            fig.update_layout(height=500)
            return fig
        
        # Создание графика
        fig = px.bar(
            x=items_counts.values,
            y=items_counts.index,
            orientation='h',
            title=f"{title} (мин. {min_frequency} упоминаний)",
            labels={'x': 'Частота упоминания', 'y': 'Элементы'},
            color=items_counts.values,
            color_continuous_scale='viridis'
        )
        
        # Улучшение внешнего вида - ФИКСИРОВАННАЯ высота чтобы избежать увеличения
        fig.update_layout(
            height=500,  # Фиксированная высота
            showlegend=False,
            xaxis_title="Количество вакансий",
            yaxis_title=field_name.replace('_', ' ').title(),
            font=dict(size=12)
        )
        
        # Обновление hover информации
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Упоминаний: %{x}<br>Процент: %{customdata:.1f}%<extra></extra>",
            customdata=(items_counts.values / len(filtered_df) * 100)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in create_array_field_chart for {field_name}: {e}")
        # В случае ошибки возвращаем базовый график с ошибкой
        fig = px.bar(
            x=[0], y=["Ошибка обработки данных"],
            orientation='h', 
            title=title
        )
        fig.update_layout(height=500)
        return fig

def create_skills_chart(filtered_df):
    """Создание графика навыков (обертка для обратной совместимости)"""
    return create_array_field_chart(filtered_df, 'key_skills', 'Топ навыки и технологии', normalize_skill)

def create_salary_experience_chart(filtered_df):
    try:
        # Фильтруем данные с валидными зарплатами и опытом
        valid_data = filtered_df[
            (filtered_df['salary_from_rub'].notna()) & 
            (filtered_df['experience_name'].notna()) &
            (filtered_df['experience_name'] != '')
        ]
        
        if len(valid_data) == 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Нет данных"], y=[0], name="Нет данных"))
            fig.update_layout(
                title="Средняя зарплата по опыту работы",
                xaxis_title="Опыт работы",
                yaxis_title="Средняя зарплата (₽)",
                height=500
            )
            return fig
        
        # Средняя зарплата по опыту
        salary_by_exp = valid_data.groupby('experience_name').agg({
            'salary_from_rub': 'mean',
            'salary_to_rub': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        # Добавляем зарплату от только если есть данные
        if not salary_by_exp['salary_from_rub'].isna().all():
            fig.add_trace(go.Bar(
                name='Зарплата от',
                x=salary_by_exp['experience_name'],
                y=salary_by_exp['salary_from_rub'].fillna(0),
                yaxis='y',
                offsetgroup=1
            ))
        
        # Добавляем зарплату до только если есть данные
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
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Ошибка"], y=[0], name="Ошибка загрузки"))
        fig.update_layout(title="Ошибка загрузки данных", height=500)
        return fig


# Функция для фильтрации данных
def filter_data(experience, employer, salary_range, company_vacancy_count=None, skills_filter=None, programming_filter=None, 
                ml_libraries_filter=None, visualization_filter=None, data_processing_filter=None, nlp_tools_filter=None, 
                cv_tools_filter=None, mlops_tools_filter=None, work_format_filter=None):
    try:
        filtered_df = df.copy()
        
        # Фильтрация по обычным полям
        if experience and len(experience) > 0:
            filtered_df = filtered_df[filtered_df['experience_name'].isin(experience)]
            
        if employer and len(employer) > 0:
            filtered_df = filtered_df[filtered_df['employer_name'].isin(employer)]
        
        # Фильтрация по количеству вакансий у компании
        if company_vacancy_count and len(company_vacancy_count) > 0:
            filtered_df = filter_by_company_vacancy_count(filtered_df, company_vacancy_count)
            
        if salary_range and len(salary_range) == 2 and salary_range[0] is not None and salary_range[1] is not None:
            # Применяем фильтр зарплат только если диапазон отличается от полного диапазона
            # Это позволяет показывать все вакансии (включая без зарплаты) при дефолтных настройках
            if salary_range != [SALARY_MIN, SALARY_MAX]:
                # Фильтруем только записи с валидными зарплатами в указанном диапазоне
                salary_filter = (
                    (filtered_df['salary_from_rub'].notna()) &
                    (filtered_df['salary_from_rub'] >= salary_range[0]) & 
                    (filtered_df['salary_from_rub'] <= salary_range[1])
                )
                filtered_df = filtered_df[salary_filter]
        
        # Фильтрация по массивным полям (multiselect)
        if skills_filter and len(skills_filter) > 0:
            filtered_df = filter_by_multiple_array_values(filtered_df, 'key_skills', skills_filter)
        
        if programming_filter and len(programming_filter) > 0:
            filtered_df = filter_by_multiple_array_values(filtered_df, DS_COLUMNS['programming_languages'], programming_filter)
            
        if ml_libraries_filter and len(ml_libraries_filter) > 0:
            filtered_df = filter_by_multiple_array_values(filtered_df, DS_COLUMNS['ml_libraries'], ml_libraries_filter)
            
        if visualization_filter and len(visualization_filter) > 0:
            filtered_df = filter_by_multiple_array_values(filtered_df, DS_COLUMNS['visualization'], visualization_filter)
            
        if data_processing_filter and len(data_processing_filter) > 0:
            filtered_df = filter_by_multiple_array_values(filtered_df, DS_COLUMNS['data_processing'], data_processing_filter)
            
        if nlp_tools_filter and len(nlp_tools_filter) > 0:
            filtered_df = filter_by_multiple_array_values(filtered_df, DS_COLUMNS['nlp_tools'], nlp_tools_filter)
            
        if cv_tools_filter and len(cv_tools_filter) > 0:
            filtered_df = filter_by_multiple_array_values(filtered_df, DS_COLUMNS['cv_tools'], cv_tools_filter)
            
        if mlops_tools_filter and len(mlops_tools_filter) > 0:
            filtered_df = filter_by_multiple_array_values(filtered_df, DS_COLUMNS['mlops_tools'], mlops_tools_filter)
            
        if work_format_filter and len(work_format_filter) > 0:
            filtered_df = filter_by_multiple_array_values(filtered_df, 'work_format', work_format_filter)
        
        return filtered_df
    except Exception as e:
        print(f"Error in filter_data: {e}")
        return df.copy()  # Возвращаем исходные данные в случае ошибки

# Функция для фильтрации по массивному полю
def filter_by_array_field(df_to_filter, field_name, target_value):
    """Фильтрует DataFrame по значению в массивном поле"""
    if not target_value or field_name not in df_to_filter.columns:
        return df_to_filter
    
    # Создаем маску для строк, содержащих нужное значение
    mask = []
    for array_data in df_to_filter[field_name]:
        parsed_items = parse_array_field(array_data)
        mask.append(target_value in parsed_items)
    
    return df_to_filter[mask]

# Функция для фильтрации по множественным значениям в массивном поле
def filter_by_multiple_array_values(df_to_filter, field_name, target_values):
    """Фильтрует DataFrame по множественным значениям в массивном поле"""
    if not target_values or len(target_values) == 0 or field_name not in df_to_filter.columns:
        return df_to_filter
    
    # Создаем маску для строк, содержащих хотя бы одно из нужных значений
    mask = []
    for array_data in df_to_filter[field_name]:
        parsed_items = parse_array_field(array_data)
        # Проверяем, есть ли пересечение между parsed_items и target_values
        has_match = any(value in parsed_items for value in target_values)
        mask.append(has_match)
    
    return df_to_filter[mask]

# Функция для фильтрации по количеству вакансий у компании
def filter_by_company_vacancy_count(df_to_filter, vacancy_count_ranges):
    """Фильтрует DataFrame по количеству вакансий у компаний"""
    if not vacancy_count_ranges or len(vacancy_count_ranges) == 0:
        return df_to_filter
    
    # Подсчитываем количество вакансий у каждой компании
    company_counts = df['employer_name'].value_counts()
    
    # Определяем компании, попадающие в выбранные диапазоны
    companies_to_include = set()
    
    for range_value in vacancy_count_ranges:
        if range_value == "1":
            companies = company_counts[company_counts == 1].index.tolist()
        elif range_value == "2-3":
            companies = company_counts[(company_counts >= 2) & (company_counts <= 3)].index.tolist()
        elif range_value == "4-5":
            companies = company_counts[(company_counts >= 4) & (company_counts <= 5)].index.tolist()
        elif range_value == "6-10":
            companies = company_counts[(company_counts >= 6) & (company_counts <= 10)].index.tolist()
        elif range_value == "11-20":
            companies = company_counts[(company_counts >= 11) & (company_counts <= 20)].index.tolist()
        elif range_value == "21-50":
            companies = company_counts[(company_counts >= 21) & (company_counts <= 50)].index.tolist()
        elif range_value == "50+":
            companies = company_counts[company_counts > 50].index.tolist()
        else:
            companies = []
        
        companies_to_include.update(companies)
    
    # Фильтруем DataFrame по компаниям из выбранных диапазонов
    return df_to_filter[df_to_filter['employer_name'].isin(companies_to_include)]

# Callback для обновления контента и статистики
@app.callback(
    [Output("tab-content", "children"),
     Output("filter-stats", "children")],
    [Input("tabs", "active_tab"),
     Input("experience-filter", "value"),
     Input("employer-filter", "value"),
     Input("company-vacancy-count-filter", "value"),
     Input("salary-filter", "value"),
     Input("skills-filter", "value"),
     Input("programming-filter", "value"),
     Input("ml-libraries-filter", "value"),
     Input("visualization-filter", "value"),
     Input("data-processing-filter", "value"),
     Input("nlp-tools-filter", "value"),
     Input("cv-tools-filter", "value"),
     Input("mlops-tools-filter", "value"),
     Input("work-format-filter", "value"),
     Input("reset-filters", "n_clicks")],
    prevent_initial_call=False
)
def update_content(active_tab, experience, employer, company_vacancy_count, salary_range, skills_filter, 
                  programming_filter, ml_libraries_filter, visualization_filter, data_processing_filter, 
                  nlp_tools_filter, cv_tools_filter, mlops_tools_filter, work_format_filter, reset_clicks):
    try:
        ctx = callback_context
        
        # Если нажата кнопка сброса, сбрасываем фильтры
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-filters.n_clicks':
            experience = []
            employer = []
            company_vacancy_count = []
            salary_range = [SALARY_MIN, SALARY_MAX]
            skills_filter = []
            programming_filter = []
            ml_libraries_filter = []
            visualization_filter = []
            data_processing_filter = []
            nlp_tools_filter = []
            cv_tools_filter = []
            mlops_tools_filter = []
            work_format_filter = []
        
        # Фильтруем данные
        filtered_df = filter_data(experience, employer, salary_range, company_vacancy_count, skills_filter, 
                                programming_filter, ml_libraries_filter, visualization_filter, data_processing_filter,
                                nlp_tools_filter, cv_tools_filter, mlops_tools_filter, work_format_filter)
        
        # Статистика фильтрации с защитой от ошибок
        try:
            salary_data = filtered_df['salary_from_rub'].dropna()
            if len(salary_data) > 0:
                avg_salary = salary_data.mean()
                median_salary = salary_data.median()
                avg_salary_text = f"Средняя зарплата: {avg_salary:,.0f} ₽"
                median_salary_text = f"Медианная зарплата: {median_salary:,.0f} ₽"
            else:
                avg_salary_text = "Нет данных о зарплате"
                median_salary_text = ""
        except:
            avg_salary_text = "Нет данных о зарплате"
            median_salary_text = ""
        
        stats = [
            html.H6("Статистика", className="text-primary"),
            html.P(f"Отфильтровано: {len(filtered_df)} из {len(df)} вакансий"),
            html.P(avg_salary_text),
        ]
        
        # Добавляем медианную зарплату если есть данные
        if median_salary_text:
            stats.append(html.P(median_salary_text))
        
        # Универсальная генерация контента по конфигурации
        content = generate_tab_content(active_tab, filtered_df)
        
        return content, stats
    
    except Exception as e:
        print(f"Error in update_content: {e}")
        # В случае ошибки возвращаем дефолтный контент
        error_content = html.Div([
            html.H4("Ошибка загрузки данных"),
            html.P("Произошла ошибка при обработке данных. Попробуйте обновить страницу.")
        ])
        error_stats = [html.P("Ошибка загрузки статистики")]
        return error_content, error_stats

# Callback для сброса фильтров
@app.callback(
    [Output("experience-filter", "value"),
     Output("employer-filter", "value"),
     Output("company-vacancy-count-filter", "value"),
     Output("salary-filter", "value"),
     Output("skills-filter", "value"),
     Output("programming-filter", "value"),
     Output("ml-libraries-filter", "value"),
     Output("visualization-filter", "value"),
     Output("data-processing-filter", "value"),
     Output("nlp-tools-filter", "value"),
     Output("cv-tools-filter", "value"),
     Output("mlops-tools-filter", "value"),
     Output("work-format-filter", "value")],
    Input("reset-filters", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    if n_clicks:
        return [], [], [], [SALARY_MIN, SALARY_MAX], [], [], [], [], [], [], [], [], []
    return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update)

# Callback для обновления опций в фильтрах на основе отфильтрованных данных
@app.callback(
    [Output("experience-filter", "options"),
     Output("employer-filter", "options"),
     Output("company-vacancy-count-filter", "options"),
     Output("skills-filter", "options"),
     Output("programming-filter", "options"),
     Output("ml-libraries-filter", "options"),
     Output("visualization-filter", "options"),
     Output("data-processing-filter", "options"),
     Output("nlp-tools-filter", "options"),
     Output("cv-tools-filter", "options"),
     Output("mlops-tools-filter", "options"),
     Output("work-format-filter", "options")],
    [Input("experience-filter", "value"),
     Input("employer-filter", "value"),
     Input("company-vacancy-count-filter", "value"),
     Input("salary-filter", "value"),
     Input("skills-filter", "value"),
     Input("programming-filter", "value"),
     Input("ml-libraries-filter", "value"),
     Input("visualization-filter", "value"),
     Input("data-processing-filter", "value"),
     Input("nlp-tools-filter", "value"),
     Input("cv-tools-filter", "value"),
     Input("mlops-tools-filter", "value"),
     Input("work-format-filter", "value")],
    prevent_initial_call=False
)
def update_filter_options(experience, employer, company_vacancy_count, salary_range, 
                         skills_filter, programming_filter, ml_libraries_filter, visualization_filter, 
                         data_processing_filter, nlp_tools_filter, cv_tools_filter, mlops_tools_filter, work_format_filter):
    try:
        # Определяем какой фильтр был изменен
        ctx = callback_context
        triggered_id = None
        if ctx.triggered:
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Создаем отфильтрованные данные
        filtered_df = filter_data(experience, employer, salary_range, company_vacancy_count, 
                                skills_filter, programming_filter, ml_libraries_filter, visualization_filter, 
                                data_processing_filter, nlp_tools_filter, cv_tools_filter, mlops_tools_filter, work_format_filter)
        
        # Получаем актуальные опции для всех фильтров
        filter_options = generate_dynamic_filter_options(filtered_df)
        
        # Определяем какие опции возвращать (не обновляем активно изменяемый фильтр)
        def get_options_or_no_update(filter_id):
            if triggered_id == filter_id:
                return dash.no_update
            return filter_options.get(filter_id, [])
        
        return (
            get_options_or_no_update("experience-filter"),
            get_options_or_no_update("employer-filter"),
            get_options_or_no_update("company-vacancy-count-filter"),
            get_options_or_no_update("skills-filter"),
            get_options_or_no_update("programming-filter"),
            get_options_or_no_update("ml-libraries-filter"),
            get_options_or_no_update("visualization-filter"),
            get_options_or_no_update("data-processing-filter"),
            get_options_or_no_update("nlp-tools-filter"),
            get_options_or_no_update("cv-tools-filter"),
            get_options_or_no_update("mlops-tools-filter"),
            get_options_or_no_update("work-format-filter")
        )
    except Exception as e:
        print(f"Error in update_filter_options: {e}")
        # В случае ошибки возвращаем опции на основе всех данных
        return (
            get_single_field_options_with_counts('experience_name'),
            get_single_field_options_with_counts('employer_name'),
            get_company_vacancy_count_options(),
            get_array_field_options_with_counts('key_skills'),
            get_array_field_options_with_counts(DS_COLUMNS['programming_languages']),
            get_array_field_options_with_counts(DS_COLUMNS['ml_libraries']),
            get_array_field_options_with_counts(DS_COLUMNS['visualization']),
            get_array_field_options_with_counts(DS_COLUMNS['data_processing']),
            get_array_field_options_with_counts(DS_COLUMNS['nlp_tools']),
            get_array_field_options_with_counts(DS_COLUMNS['cv_tools']),
            get_array_field_options_with_counts(DS_COLUMNS['mlops_tools']),
            get_array_field_options_with_counts('work_format')
        )

if __name__ == "__main__":
    app.run(debug=True, port=8050)