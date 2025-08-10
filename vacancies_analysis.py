# %%
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Configure matplotlib for auto-execution compatibility
# Use non-interactive backend when running as script, interactive when in IDE/Jupyter
if not (hasattr(sys, 'ps1') or sys.flags.interactive or 'ipykernel' in sys.modules):
    plt.switch_backend('Agg')

plt.rcParams['font.size'] = 10

# %%
# Чтение данных из файла vacancies.parquet
df = pd.read_parquet('data/vacancies.parquet')
print(f"Размер датафрейма: {df.shape}")
print(f"Столбцы: {list(df.columns)}")

# %%
# Создание гистограммы по столбцу company_type
plt.figure(figsize=(12, 8))
company_type_counts = df['company_type'].value_counts()

plt.barh(range(len(company_type_counts)), company_type_counts.values)
plt.yticks(range(len(company_type_counts)), company_type_counts.index)
plt.xlabel('Количество вакансий')
plt.ylabel('Тип компании')
plt.title('Распределение вакансий по типам компаний')
plt.grid(axis='x', alpha=0.3)

# Добавление значений на столбцы
for i, v in enumerate(company_type_counts.values):
    plt.text(v + 5, i, str(v), va='center')

plt.tight_layout()

# Safe visualization - only show in interactive mode
if hasattr(sys, 'ps1') or sys.flags.interactive or 'ipykernel' in sys.modules:
    plt.show()
else:
    plt.close()

print(f"Всего уникальных типов компаний: {df['company_type'].nunique()}")
print("\nТоп-10 типов компаний по количеству вакансий:")
print(company_type_counts.head(10))
# %%
