#!/usr/bin/env python3
"""
Класс для извлечения DS-тегов из вакансий с помощью LLM.
"""

import json
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
from openai import OpenAI


class DSTagsExtractor:
    """Извлекает DS-теги и характеристики компаний из вакансий с помощью LLM."""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b:free", batch_size: int = 5):
        """
        Инициализация экстрактора DS-тегов.
        
        Args:
            api_key: API ключ OpenRouter
            model: Модель для извлечения данных
            batch_size: Количество вакансий для обработки за один запрос
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        self.batch_size = batch_size
        
        # Загружаем базовые категории из tags.json
        self.base_categories = self.load_base_categories()
        
        # Колонки которые будут дополняться динамически (массивы)
        self.dynamic_categories = [
            "специализация", "языки_программирования", "ml_библиотеки", 
            "визуализация", "данные_библиотеки", "nlp_библиотеки", 
            "cv_библиотеки", "mlops_инструменты", "облачные_платформы", "базы_данных"
        ]
        
        # Колонки с фиксированными значениями (строки)
        self.fixed_categories = ["уровень", "тип_компании", "индустрия"]
        
        # ДОСТУПНЫЕ МОДЕЛИ
        self.available_models = {
            "free": "openai/gpt-oss-20b:free",
            "haiku": "anthropic/claude-3-5-haiku",
            "sonnet": "anthropic/claude-3-5-sonnet", 
            "llama": "meta-llama/llama-3.1-8b-instruct",
            "gemini": "google/gemini-flash-1.5",
            "qwen": "qwen/qwen-2.5-72b-instruct"
        }
        print(f"Доступные модели: {list(self.available_models.keys())}")
        print(f"Используется: {model}")
    
    def load_base_categories(self) -> Dict[str, List[str]]:
        """Загружает базовые категории из tags.json"""
        try:
            with open('tags.json', 'r', encoding='utf-8') as f:
                categories = json.load(f)
            print(f"Загружены базовые категории из tags.json: {list(categories.keys())}")
            return categories
        except Exception as e:
            print(f"Ошибка загрузки tags.json: {e}")
            return {}
    
    def load_existing_categories(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Загружает существующие категории из ВСЕХ уже обработанных записей для динамического пополнения."""
        categories = self.base_categories.copy()
        
        # Находим ВСЕ уже обработанные записи из всего файла
        if 'ds_extracted_at' in df.columns:
            processed_df = df[df['ds_extracted_at'].notna()]
            
            if not processed_df.empty:
                print(f"Найдено {len(processed_df)} уже обработанных DS записей из всего файла, пополняю категории...")
                
                # Пополняем динамические категории (массивы)
                for category in self.dynamic_categories:
                    if category in processed_df.columns:
                        existing_values = set()
                        for row in processed_df[category]:
                            if pd.notna(row) and row:
                                try:
                                    # Пытаемся распарсить как JSON массив
                                    values = json.loads(row) if isinstance(row, str) else row
                                    if isinstance(values, list):
                                        existing_values.update(values)
                                except (json.JSONDecodeError, TypeError):
                                    pass
                        
                        # Добавляем новые значения к базовым категориям
                        if existing_values:
                            new_count = len(existing_values - set(self.base_categories.get(category, [])))
                            categories[category] = sorted(set(categories.get(category, [])) | existing_values)
                            if new_count > 0:
                                print(f"  {category}: +{new_count} новых значений (всего {len(categories[category])})")
                
                # Пополняем фиксированные категории (строки)
                for category in self.fixed_categories:
                    if category in processed_df.columns:
                        existing_values = set(processed_df[category].dropna().unique())
                        if existing_values:
                            new_count = len(existing_values - set(self.base_categories.get(category, [])))
                            categories[category] = sorted(set(categories.get(category, [])) | existing_values)
                            if new_count > 0:
                                print(f"  {category}: +{new_count} новых значений (всего {len(categories[category])})")
        
        return categories
    
    def clean_html(self, text: str) -> str:
        """Очищает HTML теги и нормализует текст"""
        if not text:
            return ""
        
        # Убираем HTML теги
        text = re.sub(r'<[^>]+>', ' ', text)
        # Убираем HTML entities
        text = re.sub(r'&[^;]+;', ' ', text)
        # Нормализуем пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_description_from_raw_json(self, raw_json_str: str) -> str:
        """Извлекает и очищает описание из raw_json"""
        try:
            raw_data = json.loads(raw_json_str)
            description = raw_data.get('description', '')
            return self.clean_html(description)
        except (json.JSONDecodeError, TypeError):
            return ""
    
    def format_prompt(self, vacancies: List[Dict], categories: Dict[str, List[str]]) -> str:
        """Формирует промпт для LLM с DS вакансиями и текущими категориями."""
        
        # Формируем список существующих категорий
        categories_text = ""
        for category, values in categories.items():
            values_str = '", "'.join(values[:15])  # Ограничиваем до 15 значений для краткости
            more_text = f" (и еще {len(values)-15})" if len(values) > 15 else ""
            categories_text += f"- {category}: [\"{values_str}\"]{more_text}\n"
        
        # Формируем данные вакансий
        vacancies_data = []
        for vacancy in vacancies:
            # Извлекаем описание из raw_json
            description = self.extract_description_from_raw_json(vacancy.get('raw_json', ''))
            
            vacancy_data = {
                "vacancy_id": str(vacancy.get('id', '')),
                "name": vacancy.get('name', ''),
                "employer_name": vacancy.get('employer_name', ''),
                "area_name": vacancy.get('area_name', ''),
                "experience_name": vacancy.get('experience_name', ''),
                "key_skills": list(vacancy.get('key_skills', [])) if vacancy.get('key_skills') is not None else [],
                "description": description[:2000]  # Ограничиваем длину
            }
            vacancies_data.append(vacancy_data)
        
        prompt = f"""Ты эксперт по анализу DS вакансий. Извлеки технологические стеки и характеристики из данных вакансий.

АНАЛИЗИРУЙ ЭТИ ПОЛЯ КАЖДОЙ ВАКАНСИИ:
- "name" - название вакансии (может содержать технологии, уровень, специализацию)
- "description" - основные требования, задачи, технологии
- "key_skills" - явно указанные навыки
- "employer_name" - название работодателя (может указывать на индустрию)
- "area_name" - город (может влиять на тип компании)
- "experience_name" - требуемый опыт (влияет на уровень)

СУЩЕСТВУЮЩИЕ КАТЕГОРИИ (ИСПОЛЬЗУЙ ТОЛЬКО ИХ, НЕ СОЗДАВАЙ ДУБЛИ):
{categories_text}

СТРОГИЕ ПРАВИЛА КОНСИСТЕНТНОСТИ:
1. Для массивов (специализация, языки_программирования, etc.) - выбирай ВСЕ подходящие в массив
2. Для строк (уровень, тип_компании, индустрия) - ОДНУ категорию
3. ИСПОЛЬЗУЙ ТОЛЬКО существующие категории из списка выше
4. НЕ создавай синонимы: "scikit-learn" → используй "sklearn", "TensorFlow" → используй "tensorflow"
5. НЕ создавай дубли на английском если есть русский вариант
6. Всегда заполняй ВСЕ поля
7. КРИТИЧЕСКИ ВАЖНО: Отвечай ТОЛЬКО валидным JSON массивом, БЕЗ объяснений

СПЕЦИАЛИЗАЦИЯ - определяй по задачам:
- "Classification" - классификация, категоризация
- "Regression" - регрессия, предсказание числовых значений
- "Time Series" - временные ряды, прогнозирование, ARIMA, Prophet
- "NLP/LLM" - обработка текста, чат-боты, LLM, BERT, GPT
- "Computer Vision" - изображения, OCR, детекция объектов
- "RecSys" - рекомендательные системы
- "Антифрод/Security" - мошенничество, аномалии, безопасность
- "Finance/Scoring" - кредитный скоринг, риск-модели
- "A/B Testing" - эксперименты, статистика
- "MLOps" - деплой моделей, мониторинг, CI/CD
- "Data Engineering" - ETL, пайплайны, DWH

УРОВЕНЬ - определяй по названию и опыту:
- "Junior" - стажер, без опыта, junior
- "Middle" - 1-3 года опыта, middle
- "Senior" - 3+ года, senior, lead, principal

ПРИМЕРЫ ПРАВИЛЬНОГО ИСПОЛЬЗОВАНИЯ:
- специализация: ["NLP/LLM", "Classification"] 
- ml_библиотеки: ["sklearn", "tensorflow"]
- уровень: "Middle"
- тип_компании: "Корпорация"

ОБЯЗАТЕЛЬНЫЙ ФОРМАТ ОТВЕТА - ТОЛЬКО этот JSON массив:
[
  {{
    "vacancy_id": "123456",
    "специализация": ["Classification", "Time Series"],
    "уровень": "Middle",
    "тип_компании": "Корпорация",
    "индустрия": "Финтех",
    "языки_программирования": ["Python", "SQL"],
    "ml_библиотеки": ["sklearn", "xgboost"],
    "визуализация": ["matplotlib", "plotly"],
    "данные_библиотеки": ["pandas", "numpy"],
    "nlp_библиотеки": [],
    "cv_библиотеки": [],
    "mlops_инструменты": ["docker"],
    "облачные_платформы": ["aws"],
    "базы_данных": ["postgresql"]
  }}
]

НЕ ДОБАВЛЯЙ:
- Никаких объяснений
- Никаких комментариев  
- Никаких markdown блоков (```json)
- Никакого дополнительного текста

НАЧНИ ОТВЕТ СО СИМВОЛА [ И ЗАКОНЧИ СИМВОЛОМ ]

ВАКАНСИИ ДЛЯ АНАЛИЗА:
{json.dumps(vacancies_data, ensure_ascii=False, indent=2)}

ОТВЕТ:"""

        return prompt
    
    def extract_batch(self, batch: List[Dict], categories: Optional[Dict[str, List[str]]] = None) -> List[Dict]:
        """Извлекает DS-теги из батча вакансий через OpenRouter API."""
        if not batch:
            return []
        
        # Используем переданные категории или базовые
        if categories is None:
            categories = self.base_categories
        
        # Формируем промпт
        prompt = self.format_prompt(batch, categories)
        
        # Делаем запрос к OpenRouter API с retry логикой
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Отправляю запрос к {self.model} (попытка {attempt + 1}/{max_retries})...")
                
                response = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://github.com/hhscribe",
                        "X-Title": "HH Scribe DS Tags Extractor",
                    },
                    model=self.model,
                    messages=[
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                # Получаем ответ
                if response.choices and response.choices[0].message:
                    llm_response = response.choices[0].message.content
                    if llm_response:
                        print(f"Получен ответ длиной {len(llm_response)} символов")
                    else:
                        print("Получен пустой ответ от модели")
                        continue
                else:
                    print("Не удалось получить ответ от модели")
                    continue
                
                # Парсим ответ
                extracted_data = self.parse_llm_response(llm_response)
                
                if extracted_data:
                    print(f"Успешно извлечено {len(extracted_data)} записей")
                    return extracted_data
                else:
                    print(f"Не удалось извлечь данные из ответа (попытка {attempt + 1})")
                    
            except Exception as e:
                print(f"Ошибка при запросе к API (попытка {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Ждем {wait_time} секунд перед повтором...")
                    time.sleep(wait_time)
        
        print("Превышено количество попыток, возвращаю пустой результат")
        return []
    
    def parse_llm_response(self, response: str) -> List[Dict]:
        """Парсит ответ LLM и извлекает структурированные данные."""
        if not response or not response.strip():
            print("Получен пустой ответ от LLM")
            return []
        
        # Ищем JSON блок в ответе
        json_text = response.strip()
        
        # Удаляем markdown код блоки если есть
        if "```json" in json_text:
            start = json_text.find("```json") + 7
            end = json_text.find("```", start)
            if end != -1:
                json_text = json_text[start:end].strip()
        elif "```" in json_text:
            start = json_text.find("```") + 3
            end = json_text.find("```", start)
            if end != -1:
                json_text = json_text[start:end].strip()
        
        # Ищем JSON массив в тексте
        start_bracket = json_text.find('[')
        end_bracket = json_text.rfind(']')
        
        if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
            json_text = json_text[start_bracket:end_bracket + 1]
        
        try:
            # Парсим JSON
            data = json.loads(json_text)
            
            if not isinstance(data, list):
                print(f"Ожидался JSON массив, получен {type(data)}")
                return []
            
            # Валидируем структуру данных
            validated_data = []
            for item in data:
                if not isinstance(item, dict):
                    print(f"Пропускаю некорректный элемент: {item}")
                    continue
                
                # Проверяем обязательные поля
                if 'vacancy_id' not in item:
                    print(f"Пропускаю элемент без vacancy_id: {item}")
                    continue
                
                # Валидируем и очищаем данные
                validated_item = {
                    'vacancy_id': str(item['vacancy_id']),
                    'ds_extracted_at': datetime.now().isoformat()
                }
                
                # Добавляем все категории
                for category in self.dynamic_categories + self.fixed_categories:
                    validated_item[category] = item.get(category, [] if category in self.dynamic_categories else "не указано")
                
                # Убеждаемся что массивы действительно массивы
                for array_field in self.dynamic_categories:
                    if not isinstance(validated_item[array_field], list):
                        validated_item[array_field] = []
                
                validated_data.append(validated_item)
            
            print(f"Валидировано {len(validated_data)} из {len(data)} записей")
            return validated_data
            
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}")
            print(f"Проблемный текст: {json_text[:500]}...")
            return []
        except Exception as e:
            print(f"Неожиданная ошибка при парсинге: {e}")
            return []
    
    def process_range(self, start: int, end: int, input_path: str, output_path: str) -> None:
        """Обрабатывает диапазон вакансий с checkpoint системой и динамическим обновлением категорий."""
        print(f"Загружаю данные из {input_path}...")
        
        # Загружаем исходный файл
        try:
            df = pd.read_parquet(input_path)
            print(f"Загружено {len(df)} вакансий")
        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
            return
        
        # Проверяем диапазон
        if start < 0 or end > len(df) or start >= end:
            print(f"Некорректный диапазон: start={start}, end={end}, всего записей={len(df)}")
            return
        
        # Выбираем диапазон для обработки
        process_df = df.iloc[start:end].copy()
        print(f"Обрабатываю вакансии {start}-{end} ({len(process_df)} записей)")
        
        # Показываем статистику уже обработанных записей
        if 'ds_extracted_at' in df.columns:
            already_processed = df['ds_extracted_at'].notna().sum()
            print(f"Уже обработано ранее: {already_processed} записей")
        
        # Разбиваем на батчи
        all_results = []
        total_batches = (len(process_df) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(process_df), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch_df = process_df.iloc[i:i + self.batch_size]
            
            print(f"\nБатч {batch_num}/{total_batches} (записи {start + i}-{start + i + len(batch_df) - 1})")
            
            # КРИТИЧНО: Перезагружаем файл с диска для получения свежих данных
            if i > 0:  # После первого батча
                print("Перезагружаю файл для получения свежих категорий...")
                try:
                    df = pd.read_parquet(input_path)  # Свежие данные с диска
                except Exception as e:
                    print(f"Ошибка перезагрузки файла: {e}")
                    continue
            
            # Загружаем актуальные категории из ВСЕГО файла
            print("Анализирую категории из всего файла...")
            current_categories = self.load_existing_categories(df)
            
            # Конвертируем в список словарей для API
            batch_data = []
            for _, row in batch_df.iterrows():
                row_dict = row.to_dict()
                print(f"Обрабатываю vacancy_id: {row_dict.get('id', 'Unknown')}")
                batch_data.append(row_dict)
            
            # Извлекаем данные через LLM с актуальными категориями
            try:
                batch_results = self.extract_batch(batch_data, current_categories)
            except Exception as e:
                print(f"Ошибка в extract_batch: {e}")
                import traceback
                traceback.print_exc()
                batch_results = []
            
            if batch_results:
                print(f"Обработано {len(batch_results)} записей в батче {batch_num}")
                all_results.extend(batch_results)
                
                # Обновляем DataFrame новыми результатами
                self._update_dataframe_with_results(df, batch_results, start + i)
                
                # CHECKPOINT: Сохраняем после каждого успешного батча
                print("Сохраняю checkpoint...")
                self.save_results(df, output_path)
                
            else:
                print(f"Батч {batch_num} не удалось обработать")
            
            # Небольшая пауза между запросами
            time.sleep(1)
        
        if all_results:
            print(f"\nОбработка завершена! Обработано {len(all_results)} записей")
            
            # Финальная статистика
            if 'ds_extracted_at' in df.columns:
                total_processed = df['ds_extracted_at'].notna().sum()
                print(f"Общий прогресс: {total_processed} из {len(df)} записей обработано")
        else:
            print("\nНе удалось обработать ни одной записи")
    
    def _update_dataframe_with_results(self, df: pd.DataFrame, results: List[Dict], start_idx: int) -> None:
        """Обновляет DataFrame новыми извлеченными данными."""
        # Добавляем новые колонки если их еще нет
        new_columns = self.dynamic_categories + self.fixed_categories + ['ds_extracted_at']
        
        for col in new_columns:
            if col not in df.columns:
                if col in self.dynamic_categories:
                    df[col] = None  # JSON arrays будут строками
                else:
                    df[col] = None  # Строковые значения
        
        # Обновляем данные по vacancy_id
        for result in results:
            vacancy_id = result['vacancy_id']
            # Находим индекс записи в DataFrame
            mask = df['id'].astype(str) == vacancy_id
            
            if mask.any():
                idx = df[mask].index[0]
                
                # Обновляем все поля
                for field, value in result.items():
                    if field != 'vacancy_id' and field in df.columns:
                        if field in self.dynamic_categories:
                            # Сохраняем массивы как JSON строки
                            df.at[idx, field] = json.dumps(value, ensure_ascii=False)
                        else:
                            # Строковые значения
                            df.at[idx, field] = value
    
    def save_results(self, df: pd.DataFrame, output_path: str) -> None:
        """Сохраняет результаты в parquet файл."""
        try:
            # Создаем директорию если не существует
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Сохраняем в parquet
            df.to_parquet(output_path, index=False)
            print(f"Данные сохранены в {output_path}")
            
            # Статистика по обработанным записям
            if 'ds_extracted_at' in df.columns:
                processed_count = df['ds_extracted_at'].notna().sum()
                print(f"Всего обработанных записей: {processed_count} из {len(df)}")
                
        except Exception as e:
            print(f"Ошибка сохранения: {e}")