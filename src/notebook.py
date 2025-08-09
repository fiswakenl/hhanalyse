# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, Any

# Функция категоризации компаний
def categorize_company(company_name: str) -> str:
    company_name = company_name.lower()
    
    # Определение категорий
    categories = {
        'E-commerce': ['uzum', 'ozon', 'wildberries', 'technodom'],
        'Банки/FinTech': ['сбербанк', 'тинькофф', 'альфа', 'банк', 'bank'],
        'IT Продуктовые': ['yandex', 'яндекс', '1с', 'крок', 'pyrus'],
        'Образование': ['university', 'институт', 'skolkovo'],
        'Аутсорс/Консалтинг': ['innowise', 'epam', 'ibs', 'luxoft', 'redlab'],
        'Телеком': ['а1', 'мтс', 'мегафон', 'билайн'],
        'Госкорпорации': ['дом.рф', 'сбер', 'росатом']
    }
    
    # Проверка вхождения
    for category, keywords in categories.items():
        if any(keyword in company_name for keyword in keywords):
            return category
    
    return 'Startups/Small'

# %%
# Читаем CSV файл с вакансиями
df = pd.read_csv('vacancies_20250808_181820.csv', encoding='utf-8')

# Предобработка данных
df.dropna(subset=['Компания'], inplace=True)
df['company_category'] = df['Компания'].apply(categorize_company)

# Конвертация зарплат в рубли
exchange_rates = {'RUR': 1.0, 'USD': 95.0, 'UZS': 0.0075}

def convert_to_rub(row):
    if pd.isna(row['Зарплата от']) or pd.isna(row['Валюта']):
        return np.nan
    return float(row['Зарплата от']) * exchange_rates.get(row['Валюта'], 1.0)

df['salary_rub'] = df.apply(convert_to_rub, axis=1)

# Основные метрики
print(f"Загружено {len(df)} записей")
print(f"Распределение по категориям компаний:")
company_categories = df['company_category'].value_counts()
print(company_categories)

# %%
# Визуализация и углубленный анализ

# 1. Boxplot зарплат по категориям компаний
plt.figure(figsize=(12, 6))
sns.boxplot(x='company_category', y='salary_rub', data=df, palette='Set2')
plt.title('Распределение зарплат по категориям компаний')
plt.xlabel('Категория компании')
plt.ylabel('Зарплата (руб.)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.close()

# 2. Pie chart распределения вакансий
plt.figure(figsize=(10, 10))
plt.pie(company_categories.values, labels=company_categories.index, autopct='%1.1f%%', startangle=90)
plt.title('Распределение вакансий по категориям компаний')
plt.axis('equal')
plt.close()

# 3. Анализ популярных технологий по категориям
technology_columns = ['Название']  # Добавьте нужные колонки

technology_keywords = {
    'Frontend': ['react', 'vue', 'angular', 'frontend'],
    'Backend': ['node', 'python', 'java', 'backend'],
    'Mobile': ['android', 'ios', 'swift', 'kotlin'],
    'Cloud': ['aws', 'azure', 'gcp', 'cloud'],
    'Fullstack': ['fullstack', 'full stack']
}

tech_category_matrix = {}
for tech_name, keywords in technology_keywords.items():
    tech_category_matrix[tech_name] = {}
    for category in company_categories.index:
        category_df = df[df['company_category'] == category]
        tech_count = category_df[technology_columns[0]].str.contains('|'.join(keywords), case=False).sum()
        tech_category_matrix[tech_name][category] = tech_count

# Визуализация технологий vs категорий
plt.figure(figsize=(12, 8))
tech_matrix_df = pd.DataFrame(tech_category_matrix)
sns.heatmap(tech_matrix_df, annot=True, cmap='YlGnBu', fmt='g')
plt.title('Технологии vs Категории компаний')
plt.tight_layout()
plt.close()

# 4. Медианные зарплаты по категориям
median_salaries = df.groupby('company_category')['salary_rub'].median()
print("\nМедианные зарплаты по категориям (руб.):")
print(median_salaries)

# 5. Сохранение результатов
results_df = df[['ID', 'Компания', 'company_category', 'salary_rub']]
tech_analysis_columns = technology_columns  # Расширьте по необходимости
for col in tech_analysis_columns:
    results_df[f'{col}_technologies'] = df[col]

results_df.to_csv('company_category_analysis.csv', index=False)
print("\nРезультаты сохранены в company_category_analysis.csv")

# %%
# Проверяем уникальность по ID
print(f"Всего записей: {len(df)}")
print(f"Уникальных ID: {df['ID'].nunique()}")
print(f"Дублирующихся записей: {len(df) - df['ID'].nunique()}")

# Показываем дубликаты если есть
duplicates = df[df.duplicated(subset=['ID'], keep=False)]
if len(duplicates) > 0:
    print(f"\nНайдено {len(duplicates)} дублирующихся записей:")
    print(duplicates[['ID', 'Название', 'Компания']].head(10))

# %%
# Удаляем дубликаты по ID, оставляя первое вхождение
df_unique = df.drop_duplicates(subset=['ID'], keep='first')
print(f"После удаления дубликатов: {len(df_unique)} записей")

# Проверяем наличие пустых названий вакансий или компаний
empty_names = df_unique[df_unique['Название'].isna() | df_unique['Компания'].isna()]
print(f"Записей с пустыми названиями/компаниями: {len(empty_names)}")


# %%
# Сохраняем уникальные названия вакансий с количеством в txt файл
job_titles = df_unique['Название'].value_counts()
print(f"Всего уникальных названий: {len(job_titles)}")

# Сохраняем в файл с количеством в нужном формате
with open('unique_job_titles.txt', 'w', encoding='utf-8') as f:
    f.write("[\n")
    for i, (title, count) in enumerate(job_titles.items()):
        if i == len(job_titles) - 1:
            f.write(f"    '{title}' # {count}\n")
        else:
            f.write(f"    '{title}' # {count},\n")
    f.write("]\n")

print("Список с количеством сохранен в файл unique_job_titles.txt")
print("Можете отредактировать файл, удалив ненужные строки")
print("\n" + "="*60)
print("СЛЕДУЮЩИЕ ШАГИ:")
print("1. Откройте файл unique_job_titles.txt")
print("2. Скопируйте нужные названия вакансий")
print("3. Создайте файл unique_job_titles_filter.txt с отобранными названиями")
print("4. Запустите следующие ячейки для фильтрации и анализа")
print("="*60)

# %%
# ШАГИ АНАЛИЗА:
# 1. Сначала запустите ячейки выше, чтобы получить файл unique_job_titles.txt
# 2. Скопируйте нужные названия из unique_job_titles.txt в файл unique_job_titles_filter.txt
# 3. Запустите эту ячейку для фильтрации
# 4. Запустите ячейки анализа ниже

# %%
# Фильтруем вакансии по списку из файла unique_job_titles_filter.txt
import re
from datetime import datetime

# Читаем файл с названиями вакансий для фильтра
try:
    with open('unique_job_titles_filter.txt', 'r', encoding='utf-8') as f:
        filter_content = f.read()
    
    # Парсим названия из формата Python списка
    # Находим все строки вида 'название вакансии' с учетом запятых в конце
    job_titles_pattern = r"'([^']+)'"
    job_titles_filter = re.findall(job_titles_pattern, filter_content)
    
    # Убираем лишние символы и запятые
    job_titles_filter = [title.rstrip(' ,').strip() for title in job_titles_filter if title.strip()]
    
    print(f"Загружено {len(job_titles_filter)} названий для фильтра")
    print("Первые 10 названий:")
    for i, title in enumerate(job_titles_filter[:10]):
        print(f"  {i+1}. {title}")
    
    # Проверим, есть ли точные совпадения
    print("\n=== Отладка фильтрации ===")
    print("Пример названий из фильтра:")
    for i in range(min(5, len(job_titles_filter))):
        print(f"  '{job_titles_filter[i]}'")
    
    print("\nПример названий из DataFrame:")
    sample_df_titles = df_unique['Название'].head(10).tolist()
    for title in sample_df_titles:
        print(f"  '{title}'")
    
    # Проверим совпадения
    matches = df_unique['Название'].isin(job_titles_filter)
    print(f"\nНайдено точных совпадений: {matches.sum()}")
    
    # Фильтруем DataFrame по списку названий
    filtered_df = df_unique[df_unique['Название'].isin(job_titles_filter)]
    
    # Выводим статистику
    print(f"\n=== Статистика фильтрации ===")
    print(f"Всего вакансий в базе: {len(df_unique)}")
    print(f"Названий в фильтре: {len(job_titles_filter)}")
    print(f"Найдено вакансий: {len(filtered_df)}")
    print(f"Процент покрытия: {len(filtered_df)/len(df_unique)*100:.1f}%")
    
    # Сохраняем отфильтрованные данные в CSV
    if len(filtered_df) > 0:
        # Показываем статистику по найденным названиям
        found_titles = filtered_df['Название'].value_counts()
        print(f"\nНайдено вакансий по {len(found_titles)} уникальным названиям")
        print("Топ-10 найденных названий:")
        print(found_titles.head(10))
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'filtered_vacancies_{timestamp}.csv'
        filtered_df.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"\nОтфильтрованные данные сохранены в файл: {output_filename}")
    else:
        print("\nНичего не найдено! Возможные причины:")
        print("1. Названия в фильтре не совпадают точно с названиями в CSV")
        print("2. Проблема с кодировкой или форматированием")
        print("3. Неправильный парсинг файла фильтра")
    
except FileNotFoundError:
    print("Файл unique_job_titles_filter.txt не найден!")
except Exception as e:
    print(f"Ошибка при обработке: {e}")

# %%
# === БЛОК АНАЛИЗА ОТФИЛЬТРОВАННЫХ ДАННЫХ ===
# Запускайте эти ячейки только после создания filtered_vacancies_*.csv

# %%
# 1. Загрузка отфильтрованных данных
import glob

try:
    # Находим последний созданный файл фильтрованных вакансий
    filtered_files = glob.glob('filtered_vacancies_*.csv')
    if filtered_files:
        latest_file = max(filtered_files, key=lambda x: x.split('_')[-1].replace('.csv', ''))
        filtered_df = pd.read_csv(latest_file)
        print(f"Загружен файл: {latest_file}")
        print(f"Количество отфильтрованных вакансий: {len(filtered_df)}")
    else:
        print("Файлы с отфильтрованными вакансиями не найдены!")
        print("Сначала запустите ячейку с фильтрацией выше")
        
except Exception as e:
    print(f"Ошибка при загрузке отфильтрованных данных: {e}")

# %%
# 2. Анализ по названиям вакансий
if 'filtered_df' in locals():
    print("=== АНАЛИЗ ПО НАЗВАНИЯМ ВАКАНСИЙ ===")
    job_title_counts = filtered_df['Название'].value_counts()
    
    print(f"Уникальных названий: {len(job_title_counts)}")
    print(f"Топ-20 самых популярных названий:")
    for i, (title, count) in enumerate(job_title_counts.head(20).items(), 1):
        print(f"{i:2d}. {title:<60} | {count:3d} ваканси{'я' if count == 1 else 'й' if count < 5 else 'й'}")
    
    # Группировка по ключевым технологиям
    print(f"\n=== ГРУППИРОВКА ПО ТЕХНОЛОГИЯМ ===")
    tech_analysis = {
        'React': filtered_df[filtered_df['Название'].str.contains('React', case=False, na=False)],
        'Vue': filtered_df[filtered_df['Название'].str.contains('Vue', case=False, na=False)],
        'Angular': filtered_df[filtered_df['Название'].str.contains('Angular', case=False, na=False)],
        'Node.js': filtered_df[filtered_df['Название'].str.contains('Node', case=False, na=False)],
        'JavaScript': filtered_df[filtered_df['Название'].str.contains('JavaScript|JS-', case=False, na=False)],
        'Frontend (общие)': filtered_df[filtered_df['Название'].str.contains('Frontend|Front-end|Фронтенд', case=False, na=False)],
        'HTML/CSS': filtered_df[filtered_df['Название'].str.contains('HTML|верстальщик', case=False, na=False)]
    }
    
    for tech, df_tech in tech_analysis.items():
        if len(df_tech) > 0:
            print(f"{tech:<20} | {len(df_tech):3d} вакансий ({len(df_tech)/len(filtered_df)*100:.1f}%)")

# %%
# 3. Анализ по компаниям
if 'filtered_df' in locals():
    print("=== АНАЛИЗ ПО КОМПАНИЯМ ===")
    company_counts = filtered_df['Компания'].value_counts()
    
    print(f"Количество уникальных компаний: {len(company_counts)}")
    print(f"Топ-20 компаний по количеству вакансий:")
    
    for i, (company, count) in enumerate(company_counts.head(20).items(), 1):
        print(f"{i:2d}. {company:<50} | {count:3d} ваканси{'я' if count == 1 else 'й' if count < 5 else 'й'}")
    
    # Анализ распределения по размеру компаний
    print(f"\n=== РАСПРЕДЕЛЕНИЕ ПО АКТИВНОСТИ КОМПАНИЙ ===")
    vacancy_bins = [1, 2, 5, 10, 20, float('inf')]
    labels = ['1 вакансия', '2-4 вакансии', '5-9 вакансий', '10-19 вакансий', '20+ вакансий']
    
    company_activity = pd.cut(company_counts, bins=vacancy_bins, labels=labels, right=False)
    activity_distribution = company_activity.value_counts()
    
    for label, count in activity_distribution.items():
        percentage = count / len(company_counts) * 100
        print(f"{label:<20} | {count:3d} компаний ({percentage:.1f}%)")

# %%
# 4. Анализ зарплат с конвертацией валют
if 'filtered_df' in locals():
    print("=== АНАЛИЗ ЗАРПЛАТ ===")
    
    # Курсы валют к рублю
    exchange_rates = {
        'RUR': 1.0,
        'USD': 95.0,
        'UZS': 0.0075
    }
    
    # Функция конвертации в рубли
    def convert_to_rub(amount, currency):
        if pd.isna(amount) or pd.isna(currency):
            return None
        return float(amount) * exchange_rates.get(currency, 1.0)
    
    # Проверяем наличие колонок с зарплатами
    salary_columns = [col for col in filtered_df.columns if 'зарплат' in col.lower() or 'salary' in col.lower()]
    print(f"Найденные колонки с зарплатами: {salary_columns}")
    
    # Проверим колонки 'Зарплата от' и 'Зарплата до'
    if 'Зарплата от' in filtered_df.columns or 'Зарплата до' in filtered_df.columns:
        # Конвертируем зарплаты в рубли
        currency_col = filtered_df['Валюта'] if 'Валюта' in filtered_df.columns else None
        
        salary_from_rub = None
        salary_to_rub = None
        
        if 'Зарплата от' in filtered_df.columns and currency_col is not None:
            salary_from_rub = filtered_df.apply(lambda row: convert_to_rub(row['Зарплата от'], row['Валюта']), axis=1)
        
        if 'Зарплата до' in filtered_df.columns and currency_col is not None:
            salary_to_rub = filtered_df.apply(lambda row: convert_to_rub(row['Зарплата до'], row['Валюта']), axis=1)
        
        # Анализ зарплат в рублях
        if salary_from_rub is not None:
            valid_salary_from = salary_from_rub.dropna()
            
            if len(valid_salary_from) > 0:
                print(f"Вакансий с указанной зарплатой 'от': {len(valid_salary_from)} ({len(valid_salary_from)/len(filtered_df)*100:.1f}%)")
                print(f"Медианная зарплата 'от': {valid_salary_from.median():,.0f} руб.")
                print(f"Средняя зарплата 'от': {valid_salary_from.mean():,.0f} руб.")
                print(f"Минимальная зарплата 'от': {valid_salary_from.min():,.0f} руб.")
                print(f"Максимальная зарплата 'от': {valid_salary_from.max():,.0f} руб.")
                
                # Распределение по диапазонам
                salary_bins = [0, 50000, 100000, 150000, 200000, 300000, float('inf')]
                salary_labels = ['<50k', '50-100k', '100-150k', '150-200k', '200-300k', '>300k']
                salary_distribution = pd.cut(valid_salary_from, bins=salary_bins, labels=salary_labels, right=False)
                
                print(f"\nРаспределение зарплат 'от' по диапазонам:")
                for label, count in salary_distribution.value_counts().sort_index().items():
                    percentage = count / len(valid_salary_from) * 100
                    print(f"{label:<10} | {count:3d} вакансий ({percentage:.1f}%)")
            else:
                print("Нет валидных данных по зарплате 'от'")
        
        if salary_to_rub is not None:
            valid_salary_to = salary_to_rub.dropna()
            
            if len(valid_salary_to) > 0:
                print(f"\nВакансий с указанной зарплатой 'до': {len(valid_salary_to)} ({len(valid_salary_to)/len(filtered_df)*100:.1f}%)")
                print(f"Медианная зарплата 'до': {valid_salary_to.median():,.0f} руб.")
    else:
        print("Колонки с зарплатами не найдены в данных")
    
    # Анализ по валютам (если есть колонка)
    if 'Валюта' in filtered_df.columns:
        currency_counts = filtered_df['Валюта'].value_counts()
        print(f"\nРаспределение по валютам:")
        for currency, count in currency_counts.items():
            print(f"{currency}: {count} вакансий ({count/len(filtered_df)*100:.1f}%)")

# %%
# 5. Анализ географии (города и страны)
if 'filtered_df' in locals():
    print("=== АНАЛИЗ ПО ГОРОДАМ И СТРАНАМ ===")
    
    # Функция определения страны по городу и валюте
    def get_country(city, currency):
        if pd.isna(city):
            city = ""
        city = str(city).lower()
        
        # По городам
        if any(uz_city in city for uz_city in ['ташкент', 'самарканд', 'бухара', 'андижан', 'наманган', 'фергана']):
            return 'Узбекистан'
        elif any(kz_city in city for kz_city in ['астана', 'алматы', 'шымкент', 'караганда', 'актобе']):
            return 'Казахстан'
        elif any(by_city in city for by_city in ['минск', 'гомель', 'могилев', 'витебск', 'гродно']):
            return 'Беларусь'
        elif any(ua_city in city for ua_city in ['киев', 'харьков', 'одесса', 'днепр', 'львов']):
            return 'Украина'
        else:
            # По валютам
            if currency == 'UZS':
                return 'Узбекистан'
            elif currency == 'USD' and any(word in city for word in ['ташкент', 'астана']):
                return 'Узбекистан' if 'ташкент' in city else 'Казахстан'
            else:
                return 'Россия'
    
    # Ищем колонки с информацией о местоположении
    location_columns = [col for col in filtered_df.columns if any(word in col.lower() for word in ['город', 'регион', 'location', 'адрес'])]
    print(f"Найденные колонки с местоположением: {location_columns}")
    
    # Определяем страны
    if 'Город' in filtered_df.columns:
        currency_col = filtered_df['Валюта'] if 'Валюта' in filtered_df.columns else None
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['Страна'] = filtered_df_copy.apply(
            lambda row: get_country(row.get('Город'), row.get('Валюта')), axis=1
        )
        
        # Анализ по странам
        country_counts = filtered_df_copy['Страна'].value_counts()
        print(f"\n=== РАСПРЕДЕЛЕНИЕ ПО СТРАНАМ ===")
        print(f"Всего стран: {len(country_counts)}")
        for country, count in country_counts.items():
            percentage = count / len(filtered_df_copy) * 100
            print(f"{country:<15} | {count:3d} вакансий ({percentage:.1f}%)")
    
    # Анализируем по первой найденной колонке с городом
    city_column = None
    for col in filtered_df.columns:
        if 'город' in col.lower() or col in ['Город', 'City']:
            city_column = col
            break
    
    if city_column:
        city_counts = filtered_df[city_column].value_counts()
        print(f"\nВсего городов: {len(city_counts)}")
        print(f"Топ-15 городов по количеству вакансий:")
        
        for i, (city, count) in enumerate(city_counts.head(15).items(), 1):
            percentage = count / len(filtered_df) * 100
            print(f"{i:2d}. {city:<30} | {count:3d} вакансий ({percentage:.1f}%)")
        
        # Анализ распределения по количеству вакансий в городах
        print(f"\n=== РАСПРЕДЕЛЕНИЕ АКТИВНОСТИ ПО ГОРОДАМ ===")
        city_bins = [1, 2, 5, 10, 20, 50, float('inf')]
        city_labels = ['1 вакансия', '2-4 вакансии', '5-9 вакансий', '10-19 вакансий', '20-49 вакансий', '50+ вакансий']
        
        city_activity = pd.cut(city_counts, bins=city_bins, labels=city_labels, right=False)
        city_distribution = city_activity.value_counts()
        
        for label, count in city_distribution.items():
            percentage = count / len(city_counts) * 100
            print(f"{label:<20} | {count:3d} городов ({percentage:.1f}%)")
    else:
        print("Колонка с городами не найдена")
        
        # Попробуем найти информацию о городах в других колонках
        for col in ['Адрес', 'Address', 'Location']:
            if col in filtered_df.columns:
                print(f"\nПробуем извлечь города из колонки '{col}':")
                sample_locations = filtered_df[col].dropna().head(10)
                for loc in sample_locations:
                    print(f"  {loc}")
                break

# %%
# 6. Анализ опыта/уровня разработчиков
if 'filtered_df' in locals():
    print("=== АНАЛИЗ ПО ОПЫТУ/УРОВНЮ ===")
    
    # Расширенный анализ уровней с зарплатами и технологиями
    level_analysis = {
        'Junior/Стажер': {
            'df': filtered_df[filtered_df['Название'].str.contains(
                'Junior|junior|Стажер|стажер|Trainee|trainee', case=False, na=False)],
            'salary_stats': {},
            'tech_stats': {}
        },
        'Middle': {
            'df': filtered_df[filtered_df['Название'].str.contains(
                'Middle|middle|Мидл|мидл', case=False, na=False)],
            'salary_stats': {},
            'tech_stats': {}
        },
        'Senior': {
            'df': filtered_df[filtered_df['Название'].str.contains(
                'Senior|senior|Старший|старший|Ведущий|ведущий', case=False, na=False)],
            'salary_stats': {},
            'tech_stats': {}
        },
        'Lead/Team Lead': {
            'df': filtered_df[filtered_df['Название'].str.contains(
                'Lead|lead|Лид|лид|Team Lead|team lead|Tech Lead|tech lead|Главный|главный', case=False, na=False)],
            'salary_stats': {},
            'tech_stats': {}
        },
        'Без указания уровня': {
            'df': filtered_df[~filtered_df['Название'].str.contains(
                'Junior|junior|Middle|middle|Senior|senior|Lead|lead|Стажер|стажер|Trainee|trainee|Старший|старший|Ведущий|ведущий|Главный|главный|Мидл|мидл|Лид|лид', case=False, na=False)],
            'salary_stats': {},
            'tech_stats': {}
        }
    }
    
    # Технологии для анализа
    tech_keywords = {
        'React': 'React',
        'Vue': 'Vue', 
        'Angular': 'Angular',
        'Node.js': 'Node',
        'Python': 'Python',
        'JavaScript/JS': 'JavaScript|JS',
        'TypeScript': 'TypeScript',
        'Java': 'Java'
    }
    
    print("Распределение по уровням:")
    total_analyzed = 0
    
    for level, level_data in level_analysis.items():
        df_level = level_data['df']
        count = len(df_level)
        percentage = count / len(filtered_df) * 100
        total_analyzed += count
        print(f"{level:<25} | {count:3d} вакансий ({percentage:.1f}%)")
        
        # Анализ зарплат для уровня
        if 'salary_from_rub' in locals() and 'Зарплата от' in filtered_df.columns:
            level_salaries = df_level.apply(lambda row: convert_to_rub(row['Зарплата от'], row['Валюта']), axis=1)
            level_salaries_cleaned = level_salaries.dropna()
            
            if len(level_salaries_cleaned) > 0:
                level_data['salary_stats'] = {
                    'median': level_salaries_cleaned.median(),
                    'mean': level_salaries_cleaned.mean(),
                    'min': level_salaries_cleaned.min(),
                    'max': level_salaries_cleaned.max()
                }
                print(f"  Зарплаты для {level}:")
                print(f"    Медиана: {level_data['salary_stats']['median']:,.0f} руб.")
                print(f"    Среднее: {level_data['salary_stats']['mean']:,.0f} руб.")
        
        # Анализ технологий для уровня
        for tech_name, pattern in tech_keywords.items():
            tech_count = df_level['Название'].str.contains(pattern, case=False, na=False).sum()
            if tech_count > 0:
                level_data['tech_stats'][tech_name] = {
                    'count': tech_count,
                    'percentage': tech_count / count * 100
                }
        
        # Вывод технологий
        if level_data['tech_stats']:
            print(f"  Технологии для {level}:")
            for tech, stats in sorted(level_data['tech_stats'].items(), key=lambda x: x[1]['count'], reverse=True):
                print(f"    {tech:<10} | {stats['count']:3d} упоминаний ({stats['percentage']:.1f}%)")
            print()  # пустая строка для разделения
    
    # Дополнительный анализ по опыту работы (если есть колонка)
    experience_column = None
    for col in filtered_df.columns:
        if any(word in col.lower() for word in ['опыт', 'experience', 'стаж']):
            experience_column = col
            break
    
    if experience_column:
        print(f"\n=== АНАЛИЗ ОПЫТА ИЗ КОЛОНКИ '{experience_column}' ===")
        experience_counts = filtered_df[experience_column].value_counts()
        print("Распределение по опыту:")
        for exp, count in experience_counts.head(10).items():
            percentage = count / len(filtered_df) * 100
            print(f"{str(exp):<30} | {count:3d} вакансий ({percentage:.1f}%)")
    
    # Анализ технических требований (ключевые слова в названиях)
    print(f"\n=== ПОПУЛЯРНЫЕ ТЕХНОЛОГИИ В НАЗВАНИЯХ ===")
    tech_keywords = {
        'React': 'React',
        'Vue': 'Vue',
        'Angular': 'Angular', 
        'Node.js': 'Node',
        'JavaScript/JS': 'JavaScript|JS',
        'TypeScript': 'TypeScript',
        'Next.js': 'Next',
        'Nuxt': 'Nuxt'
    }
    
    for tech_name, pattern in tech_keywords.items():
        count = filtered_df['Название'].str.contains(pattern, case=False, na=False).sum()
        if count > 0:
            percentage = count / len(filtered_df) * 100
            print(f"{tech_name:<15} | {count:3d} упоминаний ({percentage:.1f}%)")

# %%
# 7. Визуализация данных (совместимо с автотестами)
if 'filtered_df' in locals():
    print("=== СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ ===")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys
    
    # Настройка для headless режима
    plt.style.use('default')
    if not (hasattr(sys, 'ps1') or sys.flags.interactive):
        plt.ioff()  # Отключаем интерактивный режим
    
    # Функция безопасного отображения графиков
    def safe_show_plot():
        if hasattr(sys, 'ps1') or sys.flags.interactive:
            plt.show()
        else:
            plt.close()
    
    try:
        # 1. График распределения зарплат (если есть)
        if 'salary_from_rub' in locals() and salary_from_rub is not None:
            valid_salaries = salary_from_rub.dropna()
            if len(valid_salaries) > 0:
                plt.figure(figsize=(12, 7))
                plt.style.use('seaborn')
                min_salary = valid_salaries.min()
                max_salary = valid_salaries.max()
                
                # Динамические bins с учетом диапазона зарплат
                bins = np.linspace(min_salary, max_salary, 30)
                
                plt.hist(valid_salaries, bins=bins, edgecolor='black', alpha=0.7, color='#4CAF50')
                plt.title('Распределение зарплат "от" (в рублях)', fontsize=15)
                plt.xlabel('Зарплата (руб.)', fontsize=12)
                plt.ylabel('Количество вакансий', fontsize=12)
                
                # Форматирование осей
                plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
                plt.grid(True, alpha=0.3, linestyle='--')
                
                # Добавляем статистические аннотации
                median_salary = valid_salaries.median()
                mean_salary = valid_salaries.mean()
                plt.axvline(median_salary, color='red', linestyle='dashed', linewidth=2, label=f'Медиана: {median_salary:,.0f} руб.')
                plt.axvline(mean_salary, color='blue', linestyle='dashed', linewidth=2, label=f'Среднее: {mean_salary:,.0f} руб.')
                plt.legend()
                
                plt.tight_layout()
                safe_show_plot()
        
        # 2. Топ-10 технологий
        tech_data = {}
        tech_keywords = {
            'React': 'React',
            'Vue': 'Vue', 
            'Angular': 'Angular',
            'Node.js': 'Node',
            'JavaScript': 'JavaScript|JS',
            'TypeScript': 'TypeScript',
            'Next.js': 'Next',
            'Nuxt': 'Nuxt'
        }
        
        for tech_name, pattern in tech_keywords.items():
            count = filtered_df['Название'].str.contains(pattern, case=False, na=False).sum()
            if count > 0:
                tech_data[tech_name] = count
        
        if tech_data:
            plt.figure(figsize=(12, 7))
            plt.style.use('seaborn')
            techs = list(tech_data.keys())
            counts = list(tech_data.values())
            
            # Цветовая палитра с градиентом
            colors = plt.cm.YlGnBu(np.linspace(0.4, 0.8, len(techs)))
            
            plt.bar(techs, counts, color=colors, edgecolor='black', alpha=0.8)
            plt.title('Популярность технологий в названиях вакансий', fontsize=15)
            plt.xlabel('Технологии', fontsize=12)
            plt.ylabel('Количество упоминаний', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            # Добавляем текстовые метки сверху каждого столбца
            for i, v in enumerate(counts):
                plt.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
            
            safe_show_plot()
        
        # 3. Распределение по странам (если определены)
        if 'filtered_df_copy' in locals() and 'Страна' in filtered_df_copy.columns:
            country_counts = filtered_df_copy['Страна'].value_counts()
            if len(country_counts) > 0:
                plt.figure(figsize=(8, 8))
                colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
                plt.pie(country_counts.values, labels=country_counts.index, autopct='%1.1f%%', 
                       colors=colors[:len(country_counts)])
                plt.title('Распределение вакансий по странам')
                safe_show_plot()
        
        # 4. Топ компаний
        if len(filtered_df) > 0:
            top_companies = filtered_df['Компания'].value_counts().head(10)
            if len(top_companies) > 0:
                plt.figure(figsize=(12, 6))
                plt.barh(range(len(top_companies)), top_companies.values, color='lightgreen')
                plt.yticks(range(len(top_companies)), top_companies.index)
                plt.xlabel('Количество вакансий')
                plt.title('Топ-10 компаний по количеству вакансий')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                safe_show_plot()
        
        print("Визуализации созданы успешно")
        
    except Exception as e:
        print(f"Ошибка при создании визуализаций: {e}")
        plt.close('all')  # Закрываем все графики в случае ошибки

# %%
