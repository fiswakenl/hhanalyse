# %%
import pandas as pd
import numpy as np

# %%
# Читаем CSV файл с вакансиями
df = pd.read_csv('vacancies_20250808_181211.csv')
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
