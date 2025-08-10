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
from visualization import plot_company_type_histogram

# %%
# Создание гистограммы по столбцу company_type
company_type_counts = plot_company_type_histogram(df)

print(f"Всего уникальных типов компаний: {df['company_type'].nunique()}")
print("\nТоп-10 типов компаний по количеству вакансий:")
print(company_type_counts.head(10))
# %%
