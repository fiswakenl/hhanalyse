#!/usr/bin/env python3
"""
Класс для извлечения технологических стеков из вакансий с помощью LLM.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
from openai import OpenAI


class TechnologyExtractor:
    """Извлекает технологические стеки и характеристики компаний из вакансий с помощью LLM."""
    
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b:free", batch_size: int = 5):
        """
        Инициализация экстрактора технологий.
        
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
        
        # Базовые категории (будут динамически дополняться)
        self.base_categories = {
            "fe_framework": ["React", "Vue", "Angular", "Svelte", "Next.js"],
            "state_mgmt": ["Redux", "MobX", "Zustand", "RTK Query", "Pinia"],
            "styling": ["Tailwind", "SCSS", "Styled Components", "CSS Modules", "MUI"],
            "testing": ["Jest", "Cypress", "Playwright", "RTL", "Vitest"],
            "api_proto": ["REST", "GraphQL", "WebSocket", "tRPC"],
            "ts_required": ["да", "нет", "не указано"],
            "business_domain": ["финтех", "e-commerce", "образование", "медтех", "геймдев"],
            "company_type": ["продуктовая", "аутсорс", "аутстафф", "веб-студия", "стартап"]
        }
        
        # Колонки которые будут дополняться динамически (массивы)
        self.dynamic_categories = ["fe_framework", "state_mgmt", "styling", "testing", "api_proto"]
        # Колонки с фиксированными значениями (строки)
        self.fixed_categories = ["ts_required", "business_domain", "company_type"]
        
        # ДОСТУПНЫЕ МОДЕЛИ (можно менять здесь)
        self.available_models = {
            "free": "openai/gpt-oss-20b:free",
            "haiku": "anthropic/claude-3-5-haiku",
            "sonnet": "anthropic/claude-3-5-sonnet", 
            "llama": "meta-llama/llama-3.1-8b-instruct",
            "gemini": "google/gemini-flash-1.5",
            "qwen": "qwen/qwen-2.5-72b-instruct"
        }
        print(f"💡 Доступные модели: {list(self.available_models.keys())}")
        print(f"🤖 Используется: {model}")
    
    def load_existing_categories(self, df: pd.DataFrame, start_idx: int) -> Dict[str, List[str]]:
        """Загружает существующие категории из уже обработанных записей для динамического пополнения."""
        categories = self.base_categories.copy()
        
        # Находим уже обработанные записи (те что имеют колонку extracted_at)
        if 'extracted_at' in df.columns:
            processed_df = df[df['extracted_at'].notna()].iloc[:start_idx]
            
            if not processed_df.empty:
                print(f"Найдено {len(processed_df)} уже обработанных записей, пополняю категории...")
                
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
                            categories[category] = sorted(set(categories[category]) | existing_values)
                            print(f"  {category}: +{len(existing_values - set(self.base_categories[category]))} новых значений")
                
                # Пополняем фиксированные категории (строки)
                for category in self.fixed_categories:
                    if category in processed_df.columns:
                        existing_values = set(processed_df[category].dropna().unique())
                        if existing_values:
                            categories[category] = sorted(set(categories[category]) | existing_values)
        
        return categories
    
    def format_prompt(self, vacancies: List[Dict], categories: Dict[str, List[str]]) -> str:
        """Формирует промпт для LLM с вакансиями и текущими категориями."""
        
        # Формируем список существующих категорий
        categories_text = ""
        for category, values in categories.items():
            values_str = '", "'.join(values[:10])  # Ограничиваем до 10 значений для краткости
            more_text = f" (и еще {len(values)-10})" if len(values) > 10 else ""
            categories_text += f"- {category}: [\"{values_str}\"]{more_text}\n"
        
        # Формируем данные вакансий
        vacancies_data = []
        for vacancy in vacancies:
            vacancy_data = {
                "vacancy_id": str(vacancy.get('id', '')),
                "name": vacancy.get('name', ''),
                "employer_name": vacancy.get('employer_name', ''),
                "key_skills": vacancy.get('key_skills', []),
                "description": (vacancy.get('description_markdown', '') or vacancy.get('description', ''))[:1500],
                "branded_description": (vacancy.get('branded_description_markdown', '') or vacancy.get('branded_description', ''))[:800]
            }
            vacancies_data.append(vacancy_data)
        
        prompt = f"""Ты эксперт по анализу вакансий в IT. Извлеки технологические стеки и характеристики компаний из данных вакансий.

АНАЛИЗИРУЙ ЭТИ ПОЛЯ КАЖДОЙ ВАКАНСИИ:
- "name" - название вакансии (может содержать технологии, уровень)
- "description" - основные требования и задачи
- "key_skills" - явно указанные навыки
- "branded_description" - описание компании и её деятельности
- "employer_name" - название работодателя

СУЩЕСТВУЮЩИЕ КАТЕГОРИИ (используй их, если подходят, иначе создавай новые):
{categories_text}

СТРОГИЕ ПРАВИЛА:
1. Для технологий (fe_framework, state_mgmt, styling, testing, api_proto) - выбирай ВСЕ подходящие в массив
2. Для ts_required, business_domain, company_type - ОДНУ категорию
3. Если существующая категория НЕ подходит - создай новую конкретную
4. Всегда заполняй ВСЕ поля
5. КРИТИЧЕСКИ ВАЖНО: Отвечай ТОЛЬКО валидным JSON массивом, БЕЗ объяснений, БЕЗ дополнительного текста, БЕЗ markdown

ОБЯЗАТЕЛЬНЫЙ ФОРМАТ ОТВЕТА - ТОЛЬКО этот JSON массив:
[
  {{
    "vacancy_id": "123822218",
    "fe_framework": ["Vue", "Next.js"],
    "state_mgmt": ["Redux"],
    "styling": ["Bootstrap"],
    "testing": ["Jest"],
    "api_proto": ["REST"],
    "ts_required": "нет",
    "business_domain": "веб-разработка",
    "company_type": "веб-студия"
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
        """Извлекает технологии из батча вакансий через OpenRouter API."""
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
                        "X-Title": "HH Scribe Tech Extractor",
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
                llm_response = response.choices[0].message.content
                print(f"Получен ответ длиной {len(llm_response)} символов")
                
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
        
        # Ищем JSON блок в ответе (может быть обернут в ```json или просто [])
        json_text = response.strip()
        
        # Удаляем markdown код блоки если есть
        if "```json" in json_text:
            start = json_text.find("```json") + 7
            end = json_text.find("```", start)
            if end != -1:
                json_text = json_text[start:end].strip()
        elif "```" in json_text:
            # Попробуем найти любой код блок
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
                    'fe_framework': item.get('fe_framework', []),
                    'state_mgmt': item.get('state_mgmt', []),
                    'styling': item.get('styling', []),
                    'testing': item.get('testing', []),
                    'api_proto': item.get('api_proto', []),
                    'ts_required': item.get('ts_required', 'не указано'),
                    'business_domain': item.get('business_domain', 'не указано'),
                    'company_type': item.get('company_type', 'не указано'),
                    'extracted_at': datetime.now().isoformat()
                }
                
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
        """Обрабатывает диапазон вакансий и сохраняет результаты."""
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
        
        # Загружаем существующие категории для динамического пополнения
        print("Анализирую существующие категории...")
        current_categories = self.load_existing_categories(df, start)
        
        # Разбиваем на батчи
        all_results = []
        total_batches = (len(process_df) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(process_df), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch_df = process_df.iloc[i:i + self.batch_size]
            
            print(f"\n📦 Батч {batch_num}/{total_batches} (записи {start + i}-{start + i + len(batch_df) - 1})")
            
            # Конвертируем в список словарей для API
            batch_data = []
            for _, row in batch_df.iterrows():
                batch_data.append(row.to_dict())
            
            # Обновляем категории перед каждым батчем (для динамического пополнения)
            if i > 0:  # Обновляем после первого батча
                current_categories = self.load_existing_categories(df, start + i)
            
            # Извлекаем данные через LLM с актуальными категориями
            batch_results = self.extract_batch(batch_data, current_categories)
            
            if batch_results:
                print(f"✅ Обработано {len(batch_results)} записей в батче {batch_num}")
                all_results.extend(batch_results)
                
                # Сохраняем промежуточный результат в основной DataFrame
                self._update_dataframe_with_results(df, batch_results, start + i)
                
            else:
                print(f"❌ Батч {batch_num} не удалось обработать")
            
            # Небольшая пауза между запросами
            time.sleep(1)
        
        if all_results:
            print(f"\n🎉 Обработка завершена! Обработано {len(all_results)} записей")
            
            # Сохраняем обновленный файл
            self.save_results(df, output_path)
            print(f"Результаты сохранены в {output_path}")
        else:
            print("\n❌ Не удалось обработать ни одной записи")
    
    def _update_dataframe_with_results(self, df: pd.DataFrame, results: List[Dict], start_idx: int) -> None:
        """Обновляет DataFrame новыми извлеченными данными."""
        # Добавляем новые колонки если их еще нет
        new_columns = ['fe_framework', 'state_mgmt', 'styling', 'testing', 'api_proto', 
                      'ts_required', 'business_domain', 'company_type', 'extracted_at']
        
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
            print(f"📊 Данные сохранены в {output_path}")
            
            # Статистика по обработанным записям
            if 'extracted_at' in df.columns:
                processed_count = df['extracted_at'].notna().sum()
                print(f"📈 Всего обработанных записей: {processed_count} из {len(df)}")
                
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")