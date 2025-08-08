# %%
import pandas as pd
import numpy as np

# %%
# Читаем CSV файл с вакансиями
df = pd.read_csv('vacancies_20250808_181820.csv')
print(f"Загружено {len(df)} записей")
print(f"Колонки: {list(df.columns)}")
df.head()

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
# Выводим список компаний для проверки валидности
companies = df_unique['Компания'].value_counts()
print(f"Топ-20 компаний по количеству вакансий:")
print(companies.head(20))

# %%
# Выводим список названий вакансий для проверки
job_titles = df_unique['Название'].value_counts()
print(f"Топ-20 названий вакансий:")
print(job_titles.head(20))

# %%
# Сохраняем уникальные названия вакансий с количеством в txt файл
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
# Анализ отфильтрованных вакансий
# Загружаем отфильтрованные данные для анализа
try:
    # Находим последний созданный файл фильтрованных вакансий
    import glob
    filtered_files = glob.glob('filtered_vacancies_*.csv')
    if filtered_files:
        latest_file = max(filtered_files, key=lambda x: x.split('_')[-1].replace('.csv', ''))
        filtered_df = pd.read_csv(latest_file)
        print(f"Загружен файл: {latest_file}")
        print(f"Количество отфильтрованных вакансий: {len(filtered_df)}")
        
        # Анализ количества вакансий по названиям
        print("\n=== АНАЛИЗ ПО НАЗВАНИЯМ ВАКАНСИЙ ===")
        job_title_counts = filtered_df['Название'].value_counts()
        
        print(f"Уникальных названий: {len(job_title_counts)}")
        print(f"Топ-20 самых популярных названий:")
        for i, (title, count) in enumerate(job_title_counts.head(20).items(), 1):
            print(f"{i:2d}. {title:<60} | {count:3d} ваканси{'я' if count == 1 else 'й' if count < 5 else 'й'}")
        
        # Группировка по ключевым технологиям
        print(f"\n=== ГРУППИРОВКА ПО ТЕХНОЛОГИЯМ ===")
        
        # Анализ по технологиям
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
        
    else:
        print("Файлы с отфильтрованными вакансиями не найдены!")
        
except Exception as e:
    print(f"Ошибка при загрузке отфильтрованных данных: {e}")

# %%
# Анализ по компаниям
if 'filtered_df' in locals():
    print("\n=== АНАЛИЗ ПО КОМПАНИЯМ ===")
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
# Анализ зарплат
if 'filtered_df' in locals():
    print("\n=== АНАЛИЗ ЗАРПЛАТ ===")
    
    # Проверяем наличие колонок с зарплатами
    salary_columns = [col for col in filtered_df.columns if 'зарплат' in col.lower() or 'salary' in col.lower()]
    print(f"Найденные колонки с зарплатами: {salary_columns}")
    
    # Проверим колонки 'Зарплата от' и 'Зарплата до'
    if 'Зарплата от' in filtered_df.columns or 'Зарплата до' in filtered_df.columns:
        # Анализируем зарплаты
        salary_from = filtered_df['Зарплата от'] if 'Зарплата от' in filtered_df.columns else None
        salary_to = filtered_df['Зарплата до'] if 'Зарплата до' in filtered_df.columns else None
        
        if salary_from is not None:
            # Убираем пустые значения и конвертируем в числа
            salary_from_clean = pd.to_numeric(salary_from, errors='coerce')
            valid_salary_from = salary_from_clean.dropna()
            
            if len(valid_salary_from) > 0:
                print(f"Вакансий с указанной зарплатой 'от': {len(valid_salary_from)} ({len(valid_salary_from)/len(filtered_df)*100:.1f}%)")
                print(f"Медианная зарплата 'от': {valid_salary_from.median():,.0f}")
                print(f"Средняя зарплата 'от': {valid_salary_from.mean():,.0f}")
                print(f"Минимальная зарплата 'от': {valid_salary_from.min():,.0f}")
                print(f"Максимальная зарплата 'от': {valid_salary_from.max():,.0f}")
                
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
        
        if salary_to is not None:
            salary_to_clean = pd.to_numeric(salary_to, errors='coerce')
            valid_salary_to = salary_to_clean.dropna()
            
            if len(valid_salary_to) > 0:
                print(f"\nВакансий с указанной зарплатой 'до': {len(valid_salary_to)} ({len(valid_salary_to)/len(filtered_df)*100:.1f}%)")
                print(f"Медианная зарплата 'до': {valid_salary_to.median():,.0f}")
    else:
        print("Колонки с зарплатами не найдены в данных")
    
    # Анализ по валютам (если есть колонка)
    if 'Валюта' in filtered_df.columns:
        currency_counts = filtered_df['Валюта'].value_counts()
        print(f"\nРаспределение по валютам:")
        for currency, count in currency_counts.items():
            print(f"{currency}: {count} вакансий ({count/len(filtered_df)*100:.1f}%)")

# %%
# Анализ географии (города)
if 'filtered_df' in locals():
    print("\n=== АНАЛИЗ ПО ГОРОДАМ ===")
    
    # Ищем колонки с информацией о местоположении
    location_columns = [col for col in filtered_df.columns if any(word in col.lower() for word in ['город', 'регион', 'location', 'адрес'])]
    print(f"Найденные колонки с местоположением: {location_columns}")
    
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
# Анализ опыта/уровня разработчиков
if 'filtered_df' in locals():
    print("\n=== АНАЛИЗ ПО ОПЫТУ/УРОВНЮ ===")
    
    # Извлекаем информацию об уровне из названий вакансий
    level_analysis = {
        'Junior/Стажер': filtered_df[filtered_df['Название'].str.contains(
            'Junior|junior|Стажер|стажер|Trainee|trainee', case=False, na=False)],
        'Middle': filtered_df[filtered_df['Название'].str.contains(
            'Middle|middle|Мидл|мидл', case=False, na=False)],
        'Senior': filtered_df[filtered_df['Название'].str.contains(
            'Senior|senior|Старший|старший|Ведущий|ведущий', case=False, na=False)],
        'Lead/Team Lead': filtered_df[filtered_df['Название'].str.contains(
            'Lead|lead|Лид|лид|Team Lead|team lead|Tech Lead|tech lead|Главный|главный', case=False, na=False)],
        'Без указания уровня': filtered_df[~filtered_df['Название'].str.contains(
            'Junior|junior|Middle|middle|Senior|senior|Lead|lead|Стажер|стажер|Trainee|trainee|Старший|старший|Ведущий|ведущий|Главный|главный|Мидл|мидл|Лид|лид', case=False, na=False)]
    }
    
    print("Распределение по уровням:")
    total_analyzed = 0
    for level, df_level in level_analysis.items():
        count = len(df_level)
        percentage = count / len(filtered_df) * 100
        total_analyzed += count
        print(f"{level:<25} | {count:3d} вакансий ({percentage:.1f}%)")
    
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
