# %%
import pandas as pd
import dtale
import ast
import numpy as np

# %%
def preprocess_data(df):
    df_clean = df.copy()
    
    # Обработка столбца key_skills (массив numpy)
    if 'key_skills' in df_clean.columns:
        df_clean['key_skills_str'] = df_clean['key_skills'].apply(
            lambda x: ', '.join(x) if isinstance(x, np.ndarray) else str(x)
        )
    
    # Обработка других столбцов со строковыми массивами
    array_columns = ['fe_framework', 'state_mgmt', 'styling', 'testing', 'api_proto']
    
    for col in array_columns:
        if col in df_clean.columns:
            df_clean[f'{col}_str'] = df_clean[col].apply(
                lambda x: ', '.join(ast.literal_eval(x)) if x and x != '[]' else ''
            )
    
    # Удаляем оригинальные столбцы с массивами и raw_json для упрощения
    cols_to_drop = ['key_skills', 'raw_json'] + array_columns
    df_clean = df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns])
    
    return df_clean

# %%
print("Загружаю данные из data/vacancies.parquet...")
df = pd.read_parquet('data/vacancies.parquet')
print(f"Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
df.head()

# %%
print("Обрабатываю массивы для совместимости с D-Tale...")
df_processed = preprocess_data(df)
print(f"Обработано: {df_processed.shape[0]} строк, {df_processed.shape[1]} столбцов")
df_processed.head()

# %%
print("Запускаю D-Tale...")
d = dtale.show(df_processed, host='0.0.0.0', port=40000)
print(f"D-Tale запущен успешно!")
print(f"ID данных: {d._data_id}")
print(f"URL: http://localhost:40000/dtale/main/{d._data_id}")
d