import pandas as pd
import requests
import json
import time
from typing import Dict, Any, Optional, List
import pyarrow as pa
import pyarrow.parquet as pq

def extract_vacancy_data(vacancy_json: Dict[str, Any]) -> Dict[str, Any]:
    """Извлекает нужные поля из JSON ответа API HH.ru"""
    
    def safe_get(obj, *keys):
        """Безопасно извлекает вложенные значения"""
        for key in keys:
            if obj is None or not isinstance(obj, dict):
                return None
            obj = obj.get(key)
        return obj
    
    # Извлекаем salary информацию
    salary = vacancy_json.get('salary')
    salary_from = safe_get(salary, 'from') if salary else None
    salary_to = safe_get(salary, 'to') if salary else None
    salary_currency = safe_get(salary, 'currency') if salary else None
    salary_gross = safe_get(salary, 'gross') if salary else None
    
    # Извлекаем key_skills как список строк
    key_skills = vacancy_json.get('key_skills', [])
    key_skills_list = [skill.get('name', '') for skill in key_skills] if key_skills else []
    
    # Извлекаем work_format
    work_format_list = vacancy_json.get('work_format', [])
    work_format = [wf.get('name', '') for wf in work_format_list] if work_format_list else []
    
    return {
        'employer_id': safe_get(vacancy_json, 'employer', 'id'),
        'employer_name': safe_get(vacancy_json, 'employer', 'name'),
        'id': vacancy_json.get('id'),
        'name': vacancy_json.get('name'),
        'area_name': safe_get(vacancy_json, 'area', 'name'),
        'salary_from': salary_from,
        'salary_to': salary_to,
        'salary_currency': salary_currency,
        'salary_gross': salary_gross,
        'experience_name': safe_get(vacancy_json, 'experience', 'name'),
        'work_format': work_format,
        'raw_json': json.dumps(vacancy_json, ensure_ascii=False),
        'key_skills': key_skills_list
    }

def fetch_vacancy(vacancy_id: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """Получает данные о вакансии по ID с повторными попытками"""
    url = f"https://api.hh.ru/vacancies/{vacancy_id}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                print(f"Вакансия {vacancy_id} не найдена (404)")
                return None
            elif response.status_code == 429:
                # Rate limiting - ждем дольше
                wait_time = 2 ** attempt
                print(f"Rate limit для {vacancy_id}, ждем {wait_time} сек...")
                time.sleep(wait_time)
                continue
            else:
                print(f"Ошибка {response.status_code} для вакансии {vacancy_id}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса для {vacancy_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None
    
    return None

def main():
    # Читаем CSV файл с ID вакансий
    print("Читаем ds_vacancies.csv...")
    df_ids = pd.read_csv('ds_vacancies.csv')
    vacancy_ids = df_ids['id'].astype(str).tolist()
    
    print(f"Найдено {len(vacancy_ids)} вакансий для обработки")
    
    # Список для хранения обработанных данных
    processed_vacancies = []
    
    # Обрабатываем каждую вакансию
    for i, vacancy_id in enumerate(vacancy_ids, 1):
        print(f"Обрабатываем {i}/{len(vacancy_ids)}: {vacancy_id}")
        
        # Получаем данные о вакансии
        vacancy_data = fetch_vacancy(vacancy_id)
        
        if vacancy_data:
            # Извлекаем нужные поля
            extracted_data = extract_vacancy_data(vacancy_data)
            processed_vacancies.append(extracted_data)
            print(f"✓ Успешно обработано: {extracted_data['name'][:50]}...")
        else:
            print(f"✗ Не удалось получить данные для {vacancy_id}")
        
        # Пауза между запросами чтобы не нагружать API
        time.sleep(0.5)
        
        # Промежуточное сохранение каждые 50 вакансий
        if i % 50 == 0:
            print(f"Промежуточное сохранение после {i} вакансий...")
            df_temp = pd.DataFrame(processed_vacancies)
            df_temp.to_parquet(f'vacancies_temp_{i}.parquet', index=False)
    
    # Создаем итоговый DataFrame
    if processed_vacancies:
        df_result = pd.DataFrame(processed_vacancies)
        
        print(f"\nОбработано {len(processed_vacancies)} из {len(vacancy_ids)} вакансий")
        print(f"Столбцы: {list(df_result.columns)}")
        
        # Сохраняем в parquet
        output_file = 'hh_vacancies_data.parquet'
        df_result.to_parquet(output_file, index=False)
        print(f"Данные сохранены в {output_file}")
        
        # Показываем статистику
        print(f"\nСтатистика:")
        print(f"- Вакансий с зарплатой: {df_result['salary_from'].notna().sum()}")
        print(f"- Уникальных работодателей: {df_result['employer_name'].nunique()}")
        print(f"- Уникальных городов: {df_result['area_name'].nunique()}")
        
    else:
        print("Не удалось обработать ни одной вакансии")

if __name__ == "__main__":
    main()