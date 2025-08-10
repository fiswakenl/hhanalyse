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

# Универсальная функция для парсинга навыков из разных форматов
def parse_skills(skills_data):
    """
    Парсит навыки из различных форматов данных:
    - numpy arrays
    - Python lists
    - JSON strings  
    - comma-separated strings
    - literal eval strings
    
    Args:
        skills_data: данные любого формата
        
    Returns:
        list: список строк с навыками
    """
    # Проверяем на None и NaN с учетом numpy arrays
    if skills_data is None:
        return []
    
    # Для numpy arrays проверяем через try/except
    try:
        if pd.isna(skills_data):
            return []
    except (TypeError, ValueError):
        # Если pd.isna не работает (например, для numpy array), продолжаем
        pass
    
    try:
        # Если это numpy array
        if isinstance(skills_data, np.ndarray):
            if skills_data.size == 0:
                return []
            # Фильтруем пустые значения и пустые строки
            skills = []
            for skill in skills_data:
                skill_str = str(skill).strip()
                if skill_str and skill_str != 'nan' and skill_str != 'None':
                    skills.append(skill_str)
            return skills
        
        # Если это уже список
        if isinstance(skills_data, (list, tuple)):
            skills = []
            for skill in skills_data:
                skill_str = str(skill).strip()
                if skill_str and skill_str != 'nan' and skill_str != 'None':
                    skills.append(skill_str)
            return skills
        
        # Если это строка
        if isinstance(skills_data, str):
            skills_data = skills_data.strip()
            
            if not skills_data:
                return []
            
            # Попытка парсинга как JSON массив
            if skills_data.startswith('[') and skills_data.endswith(']'):
                try:
                    parsed = json.loads(skills_data)
                    if isinstance(parsed, list):
                        skills = []
                        for skill in parsed:
                            skill_str = str(skill).strip()
                            if skill_str and skill_str != 'nan' and skill_str != 'None':
                                skills.append(skill_str)
                        return skills
                except (json.JSONDecodeError, ValueError):
                    pass
                
                # Попытка парсинга как Python literal
                try:
                    parsed = ast.literal_eval(skills_data)
                    if isinstance(parsed, list):
                        skills = []
                        for skill in parsed:
                            skill_str = str(skill).strip()
                            if skill_str and skill_str != 'nan' and skill_str != 'None':
                                skills.append(skill_str)
                        return skills
                except (SyntaxError, ValueError):
                    pass
            
            # Разделение по запятым как fallback
            skills = [s.strip() for s in skills_data.split(',')]
            return [skill for skill in skills if skill]
        
        # Для других типов - попытка конвертации в строку
        return [str(skills_data).strip()] if str(skills_data).strip() else []
        
    except Exception as e:
        print(f"Warning: Error parsing skills {skills_data}: {e}")
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
                    html.H6("Тип компании"),
                    dcc.Dropdown(
                        id="company-filter",
                        options=[{"label": "Все", "value": "all"}] + 
                                [{"label": str(ct), "value": str(ct)} for ct in df['company_type'].dropna().unique() 
                                 if pd.notna(ct) and str(ct).strip() != '' and str(ct) != 'nan'],
                        value="all",
                        clearable=False
                    ),
                    html.Br(),
                    
                    html.H6("Бизнес-домен"),
                    dcc.Dropdown(
                        id="domain-filter", 
                        options=[{"label": "Все", "value": "all"}] + 
                                [{"label": str(bd), "value": str(bd)} for bd in df['business_domain'].dropna().unique() 
                                 if pd.notna(bd) and str(bd).strip() != '' and str(bd) != 'nan'],
                        value="all",
                        clearable=False
                    ),
                    html.Br(),
                    
                    html.H6("Опыт работы"),
                    dcc.Dropdown(
                        id="experience-filter",
                        options=[{"label": "Все", "value": "all"}] + 
                                [{"label": str(exp), "value": str(exp)} for exp in df['experience_name'].dropna().unique() 
                                 if pd.notna(exp) and str(exp).strip() != '' and str(exp) != 'nan'],
                        value="all",
                        clearable=False
                    ),
                    html.Br(),
                    
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
            dbc.Tabs([
                dbc.Tab(label="Типы компаний", tab_id="company-tab"),
                dbc.Tab(label="Бизнес-домены", tab_id="domain-tab"), 
                dbc.Tab(label="Навыки и технологии", tab_id="skills-tab"),
                dbc.Tab(label="Зарплаты", tab_id="salary-tab"),
            ], id="tabs", active_tab="company-tab"),
            
            html.Br(),
            
            # Содержимое табов
            html.Div(id="tab-content")
        ], width=9)
    ])
    
], fluid=True)

# Функции для создания графиков
def create_company_type_chart(filtered_df):
    try:
        company_counts = filtered_df['company_type'].dropna().value_counts()
        
        if len(company_counts) == 0:
            fig = px.bar(
                x=[0], y=["Нет данных"],
                orientation='h',
                title="Распределение вакансий по типам компаний"
            )
        else:
            fig = px.bar(
                x=company_counts.values,
                y=company_counts.index,
                orientation='h',
                title="Распределение вакансий по типам компаний",
                labels={'x': 'Количество вакансий', 'y': 'Тип компании'}
            )
        
        fig.update_layout(height=500)
        return fig
    except Exception as e:
        print(f"Error in create_company_type_chart: {e}")
        fig = px.bar(x=[0], y=["Ошибка загрузки"], orientation='h', title="Ошибка")
        return fig

def create_business_domain_chart(filtered_df):
    try:
        domain_counts = filtered_df['business_domain'].dropna().value_counts().head(15)
        
        if len(domain_counts) == 0:
            fig = px.bar(
                x=[0], y=["Нет данных"],
                orientation='h',
                title="Топ бизнес-домены"
            )
        else:
            fig = px.bar(
                x=domain_counts.values,
                y=domain_counts.index,
                orientation='h',
                title="Топ бизнес-домены",
                labels={'x': 'Количество вакансий', 'y': 'Бизнес-домен'}
            )
        
        fig.update_layout(height=500)
        return fig
    except Exception as e:
        print(f"Error in create_business_domain_chart: {e}")
        fig = px.bar(x=[0], y=["Ошибка загрузки"], orientation='h', title="Ошибка")
        return fig

def create_skills_chart(filtered_df):
    try:
        # Обработка key_skills с новой универсальной функцией парсинга
        all_skills = []
        
        for skills_data in filtered_df['key_skills']:
            parsed_skills = parse_skills(skills_data)
            # Нормализация навыков
            normalized_skills = [normalize_skill(skill) for skill in parsed_skills]
            # Фильтрация пустых значений
            valid_skills = [skill for skill in normalized_skills if skill is not None and len(skill) > 1]
            all_skills.extend(valid_skills)
        
        if not all_skills:
            # Если нет навыков, создаем пустой график
            fig = px.bar(
                x=[0], y=["Нет данных"],
                orientation='h',
                title="Топ навыки и технологии"
            )
            fig.update_layout(height=500)
            return fig
        
        # Подсчет частоты навыков
        skills_counts = pd.Series(all_skills).value_counts().head(20)  # Увеличил до 20 для лучшего анализа
        
        # Фильтрация навыков с минимальной частотой (убираем редкие навыки)
        min_frequency = max(1, len(filtered_df) // 50)  # Минимум 2% от общего количества вакансий
        skills_counts = skills_counts[skills_counts >= min_frequency]
        
        if len(skills_counts) == 0:
            fig = px.bar(
                x=[0], y=["Слишком мало данных"],
                orientation='h',
                title="Топ навыки и технологии"
            )
            fig.update_layout(height=500)
            return fig
        
        # Создание графика
        fig = px.bar(
            x=skills_counts.values,
            y=skills_counts.index,
            orientation='h',
            title=f"Топ навыки и технологии (мин. {min_frequency} упоминаний)",
            labels={'x': 'Частота упоминания', 'y': 'Навыки'},
            color=skills_counts.values,
            color_continuous_scale='viridis'
        )
        
        # Улучшение внешнего вида
        fig.update_layout(
            height=max(500, len(skills_counts) * 25),  # Динамическая высота
            showlegend=False,
            xaxis_title="Количество вакансий",
            yaxis_title="Навыки",
            font=dict(size=12)
        )
        
        # Обновление hover информации
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>Упоминаний: %{x}<br>Процент: %{customdata:.1f}%<extra></extra>",
            customdata=(skills_counts.values / len(filtered_df) * 100)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in create_skills_chart: {e}")
        # В случае ошибки возвращаем базовый график с ошибкой
        fig = px.bar(
            x=[0], y=["Ошибка обработки данных"],
            orientation='h', 
            title="Топ навыки и технологии"
        )
        fig.update_layout(height=500)
        return fig

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
def filter_data(company_type, domain, experience, salary_range):
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
        
        return filtered_df
    except Exception as e:
        print(f"Error in filter_data: {e}")
        return df.copy()  # Возвращаем исходные данные в случае ошибки

# Callback для обновления контента и статистики
@app.callback(
    [Output("tab-content", "children"),
     Output("filter-stats", "children")],
    [Input("tabs", "active_tab"),
     Input("company-filter", "value"),
     Input("domain-filter", "value"),
     Input("experience-filter", "value"),
     Input("salary-filter", "value"),
     Input("reset-filters", "n_clicks")],
    prevent_initial_call=False
)
def update_content(active_tab, company_type, domain, experience, salary_range, reset_clicks):
    try:
        ctx = callback_context
        
        # Если нажата кнопка сброса, сбрасываем фильтры
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'reset-filters.n_clicks':
            company_type = "all"
            domain = "all"
            experience = "all"
            salary_range = [SALARY_MIN, SALARY_MAX]
        
        # Фильтруем данные
        filtered_df = filter_data(company_type, domain, experience, salary_range)
        
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
        
        # Контент таба в зависимости от активного таба
        if active_tab == "company-tab":
            content = dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=create_company_type_chart(filtered_df), id="company-chart")
                ], width=12)
            ])
        
        elif active_tab == "domain-tab":
            content = dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=create_business_domain_chart(filtered_df), id="domain-chart")
                ], width=12)
            ])
        
        elif active_tab == "skills-tab":
            content = dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=create_skills_chart(filtered_df), id="skills-chart")
                ], width=12)
            ])
        
        elif active_tab == "salary-tab":
            content = dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=create_salary_experience_chart(filtered_df), id="salary-chart")
                ], width=12)
            ])
        else:
            content = html.Div("Выберите таб")
        
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
     Output("salary-filter", "value")],
    Input("reset-filters", "n_clicks"),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    if n_clicks:
        return "all", "all", "all", [SALARY_MIN, SALARY_MAX]
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

if __name__ == "__main__":
    app.run(debug=True, port=8050)