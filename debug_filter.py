import pandas as pd
from interactive_dashboard import filter_data, SALARY_MIN, SALARY_MAX

# Загружаем данные
df = pd.read_parquet('data/vacancies.parquet')

print(f"Общее количество записей в файле: {len(df)}")
print(f"SALARY_MIN: {SALARY_MIN}")
print(f"SALARY_MAX: {SALARY_MAX}")

# Тестируем фильтрацию с дефолтными параметрами
filtered_df = filter_data("all", "all", "all", [SALARY_MIN, SALARY_MAX])

print(f"\nПосле фильтрации: {len(filtered_df)} записей")
print(f"Потеряно записей: {len(df) - len(filtered_df)}")

# Анализируем данные о зарплатах
print(f"\nАнализ зарплатных данных:")
print(f"salary_from_rub - не NaN: {df['salary_from_rub'].notna().sum()}")
print(f"salary_from_rub - NaN: {df['salary_from_rub'].isna().sum()}")

# Проверим условие фильтрации зарплат
salary_filter = (
    (df['salary_from_rub'].notna()) &
    (df['salary_from_rub'] >= SALARY_MIN) & 
    (df['salary_from_rub'] <= SALARY_MAX)
)

print(f"\nФильтр зарплат проходят: {salary_filter.sum()} записей")
print(f"Записи без зарплат: {df['salary_from_rub'].isna().sum()}")

# Проверим минимальные и максимальные зарплаты
valid_salaries = df['salary_from_rub'].dropna()
if len(valid_salaries) > 0:
    print(f"\nРеальные зарплаты:")
    print(f"Минимальная: {valid_salaries.min()}")
    print(f"Максимальная: {valid_salaries.max()}")
    print(f"Количество записей с зарплатами: {len(valid_salaries)}")

# Проверим другие столбцы на наличие данных
print(f"\nПроверка других столбцов:")
print(f"company_type - не пустые: {df['company_type'].dropna().sum()}")
print(f"business_domain - не пустые: {df['business_domain'].dropna().sum()}")
print(f"experience_name - не пустые: {df['experience_name'].dropna().sum()}")

# Покажем пример данных без зарплат
no_salary_examples = df[df['salary_from_rub'].isna()].head(5)
print(f"\nПримеры записей без зарплат:")
for i, row in no_salary_examples.iterrows():
    print(f"ID: {row['id']}, company_type: {row['company_type']}, business_domain: {row.get('business_domain', 'N/A')}")