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
df = pd.read_parquet('data/vacancies.parquet')

# Конфигурация всех типов графиков и фильтров
CHART_CONFIG = {
    'company': {
        'title': 'Типы компаний',
        'tab_id': 'company-tab',
        'chart_func': lambda df: create_single_field_chart(df, 'company_type', 'Распределение вакансий по типам компаний'),
        'filter_field': 'company_type'
    },
    'domain': {
        'title': 'Бизнес-домены', 
        'tab_id': 'domain-tab',
        'chart_func': lambda df: create_single_field_chart(df, 'business_domain', 'Топ бизнес-домены', top_n=15),
        'filter_field': 'business_domain'
    },
    'skills': {
        'title': 'Навыки и технологии',
        'tab_id': 'skills-tab',
        'chart_func': lambda df: create_array_field_chart(df, 'key_skills', 'Топ навыки и технологии', normalize_skill),
        'filter_field': None  # Особый случай - используем отдельную фильтрацию
    },
    'arrays': {
        'title': 'Массивы данных',
        'tab_id': 'arrays-tab',
        'chart_func': 'dynamic',  # Динамическая функция
        'filter_field': None
    },
    'salary': {
        'title': 'Зарплаты',
        'tab_id': 'salary-tab', 
        'chart_func': lambda df: create_salary_experience_chart(df),
        'filter_field': None
    }
}

# Список массивных полей доступных для анализа
ARRAY_FIELDS = {
    'key_skills': 'Навыки и технологии',
    'fe_framework': 'Фронтенд-фреймворки',
    'state_mgmt': 'Управление состоянием',
    'styling': 'Стилизация',
    'testing': 'Тестирование',
    'api_proto': 'API протоколы'
}

# Фильтры для sidebar
FILTERS_CONFIG = [
    {'id': 'company-filter', 'label': 'Тип компании', 'field': 'company_type'},
    {'id': 'domain-filter', 'label': 'Бизнес-домен', 'field': 'business_domain'},
    {'id': 'experience-filter', 'label': 'Опыт работы', 'field': 'experience_name'}
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

# Генерация фильтров
def generate_filters():
    """Генерирует все фильтры для sidebar"""
    filters = []
    
    for filter_config in FILTERS_CONFIG:
        field = filter_config['field']
        unique_values = df[field].dropna().unique()
        
        filters.extend([
            html.H6(filter_config['label']),
            dcc.Dropdown(
                id=filter_config['id'],
                options=[{"label": "Все", "value": "all"}] + 
                        [{"label": str(val), "value": str(val)} for val in unique_values 
                         if pd.notna(val) and str(val).strip() != '' and str(val) != 'nan'],
                value="all",
                clearable=False
            ),
            html.Br()
        ])
    
    return filters

# Генерация табов
def generate_tabs():
    """Генерирует все табы из конфигурации"""
    tabs = []
    for config in CHART_CONFIG.values():
        tabs.append(dbc.Tab(label=config['title'], tab_id=config['tab_id']))
    return tabs

# Универсальная генерация контента табов
def generate_tab_content(active_tab, filtered_df, array_field=None):
    """Генерирует контент для активного таба на основе конфигурации"""
    
    # Находим конфигурацию для активного таба
    config = None
    for key, cfg in CHART_CONFIG.items():
        if cfg['tab_id'] == active_tab:
            config = cfg
            break
    
    if not config:
        return html.Div("Таб не найден")
    
    # Специальная обработка для массивных данных
    if active_tab == "arrays-tab":
        if array_field and array_field in ARRAY_FIELDS:
            field_title = ARRAY_FIELDS[array_field]
            chart = create_array_field_chart(filtered_df, array_field, field_title)
            chart_id = "arrays-chart"
        else:
            return dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H4("Анализ массивных полей"),
                        html.P("Выберите поле для анализа в фильтрах слева."),
                        html.Ul([
                            html.Li(f"{label} ({field})") for field, label in ARRAY_FIELDS.items()
                        ])
                    ])
                ], width=12)
            ])
    else:
        # Обычная генерация графика по конфигурации
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
        
        # Курсы валют (примерные)
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
app.title = "HR Analytics Dashboard"

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
            html.H1("HR Analytics Dashboard", className="text-center mb-4"),
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
                    
                    html.H6("Фильтр по массивным полям"),
                    html.Div([
                        dcc.Dropdown(
                            id="array-field-selector",
                            options=[{"label": label, "value": field} for field, label in ARRAY_FIELDS.items()],
                            value="key_skills",
                            clearable=False,
                            placeholder="Выберите поле"
                        ),
                        html.Br(),
                        dcc.Dropdown(
                            id="array-value-filter",
                            options=[],
                            value=None,
                            placeholder="Выберите значение для фильтрации"
                        )
                    ]),
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

# Функция для фильтрации данных
def filter_data(company_type, domain, experience, salary_range, array_field=None, array_value=None):
    try:
        filtered_df = df.copy()
        
        if company_type and company_type != "all":
            filtered_df = filtered_df[filtered_df['company_type'] == company_type]
        
        if domain and domain != "all":
            filtered_df = filtered_df[filtered_df['business_domain'] == domain]
            
        if experience and experience != "all":
            filtered_df = filtered_df[filtered_df['experience_name'] == experience]
            
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
        
        # Фильтрация по массивному полю
        if array_field and array_value:
            filtered_df = filter_by_array_field(filtered_df, array_field, array_value)
        
        return filtered_df
    except Exception as e:
        print(f"Error in filter_data: {e}")
        return df.copy()  # Возвращаем исходные данные в случае ошибки

# Callback для обновления значений в dropdown массивных полей
@app.callback(
    Output("array-value-filter", "options"),
    Input("array-field-selector", "value"),
    prevent_initial_call=False
)
def update_array_values(selected_field):
    if not selected_field:
        return []
    
    values = get_unique_array_values(selected_field)
    return [{"label": value, "value": value} for value in values]

# Callback для обновления контента и статистики
@app.callback(
    [Output("tab-content", "children"),
     Output("filter-stats", "children")],
    [Input("tabs", "active_tab"),
     Input("company-filter", "value"),
     Input("domain-filter", "value"),
     Input("experience-filter", "value"),
     Input("salary-filter", "value"),
     Input("array-field-selector", "value"),
     Input("array-value-filter", "value"),
     Input("reset-filters", "n_clicks")],
    prevent_initial_call=False
)
def update_content(active_tab, company_type, domain, experience, salary_range, array_field, array_value, reset_clicks):
    try:
        ctx = callback_context
        
        # Если нажата кнопка сброса, сбрасываем фильтры
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-filters.n_clicks':
            company_type = "all"
            domain = "all"
            experience = "all"
            salary_range = [SALARY_MIN, SALARY_MAX]
            array_field = "key_skills"
            array_value = None
        
        # Фильтруем данные
        filtered_df = filter_data(company_type, domain, experience, salary_range, array_field, array_value)
        
        # Статистика фильтрации с защитой от ошибок
        try:
            avg_salary = filtered_df['salary_from_rub'].mean()
            avg_salary_text = f"Средняя зарплата: {avg_salary:,.0f} ₽" if pd.notna(avg_salary) else "Нет данных о зарплате"
        except:
            avg_salary_text = "Нет данных о зарплате"
        
        stats = [
            html.H6("Статистика", className="text-primary"),
            html.P(f"Отфильтровано: {len(filtered_df)} из {len(df)} вакансий"),
            html.P(avg_salary_text)
        ]
        
        # Универсальная генерация контента по конфигурации
        content = generate_tab_content(active_tab, filtered_df, array_field)
        
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
    [Output("company-filter", "value"),
     Output("domain-filter", "value"),
     Output("experience-filter", "value"),
     Output("salary-filter", "value"),
     Output("array-field-selector", "value"),
     Output("array-value-filter", "value")],
    Input("reset-filters", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    if n_clicks:
        return "all", "all", "all", [SALARY_MIN, SALARY_MAX], "key_skills", None
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

if __name__ == "__main__":
    app.run(debug=True, port=8050)