import asyncio
import csv
import json
import logging
import time
from pathlib import Path
from typing import List, Optional
import httpx
import pandas as pd
from datetime import datetime

from models import Vacancy, VacancyResponse
from storage import VacancyStorage


class VacancyFetcher:
    def __init__(self, csv_file: str, db_path: str = "vacancies.db", delay: float = 1.0):
        self.csv_file = csv_file
        self.storage = VacancyStorage(db_path)
        self.delay = delay
        self.logger = logging.getLogger(__name__)
        
        # HTTP клиент с настройками
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "HH Vacancy Fetcher 1.0",
                "Accept": "application/json"
            }
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def load_vacancy_ids(self) -> List[str]:
        """Загружает ID вакансий из CSV файла"""
        vacancy_ids = []
        try:
            # Читаем CSV файл
            df = pd.read_csv(self.csv_file)
            # Предполагаем, что ID в первой колонке
            vacancy_ids = df.iloc[:, 0].astype(str).tolist()
            self.logger.info(f"Loaded {len(vacancy_ids)} vacancy IDs from {self.csv_file}")
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {e}")
            raise
        
        return vacancy_ids
    
    def filter_new_vacancies(self, vacancy_ids: List[str]) -> List[str]:
        """Фильтрует только новые вакансии (которых еще нет в БД)"""
        processed_ids = set(self.storage.get_processed_vacancy_ids())
        new_ids = [vid for vid in vacancy_ids if vid not in processed_ids]
        
        self.logger.info(f"Found {len(new_ids)} new vacancies out of {len(vacancy_ids)} total")
        if processed_ids:
            self.logger.info(f"Skipping {len(processed_ids)} already processed vacancies")
        
        return new_ids
    
    async def fetch_vacancy(self, vacancy_id: str) -> Optional[dict]:
        """Получает данные одной вакансии из API HH.ru"""
        url = f"https://api.hh.ru/vacancies/{vacancy_id}"
        
        try:
            self.logger.debug(f"Fetching vacancy {vacancy_id}")
            response = await self.client.get(url)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                self.logger.warning(f"Vacancy {vacancy_id} not found (404)")
                return None
            elif response.status_code == 429:
                self.logger.warning(f"Rate limit exceeded, waiting longer...")
                await asyncio.sleep(5)  # Увеличиваем задержку при rate limit
                return await self.fetch_vacancy(vacancy_id)  # Повторяем запрос
            else:
                self.logger.error(f"HTTP {response.status_code} for vacancy {vacancy_id}: {response.text}")
                return None
                
        except httpx.TimeoutException:
            self.logger.error(f"Timeout fetching vacancy {vacancy_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching vacancy {vacancy_id}: {e}")
            return None
    
    async def process_vacancy(self, vacancy_id: str) -> bool:
        """Обрабатывает одну вакансию: получает данные и сохраняет в БД"""
        try:
            # Получаем данные из API
            api_data = await self.fetch_vacancy(vacancy_id)
            if not api_data:
                return False
            
            # Валидируем и парсим данные через Pydantic
            try:
                vacancy = Vacancy(**api_data)
                vacancy.fetched_at = datetime.now()
            except Exception as e:
                self.logger.error(f"Validation error for vacancy {vacancy_id}: {e}")
                # Сохраняем проблемную запись в отдельный файл для анализа
                self.save_failed_vacancy(vacancy_id, api_data, str(e))
                return False
            
            # Сохраняем в базу данных
            raw_json = json.dumps(api_data, ensure_ascii=False, default=str)
            self.storage.save_vacancy(vacancy, raw_json)
            
            self.logger.info(f"✓ Vacancy {vacancy_id} processed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing vacancy {vacancy_id}: {e}")
            return False
    
    def save_failed_vacancy(self, vacancy_id: str, data: dict, error: str):
        """Сохраняет проблемные вакансии для анализа"""
        failed_dir = Path("failed_vacancies")
        failed_dir.mkdir(exist_ok=True)
        
        failed_data = {
            "vacancy_id": vacancy_id,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        with open(failed_dir / f"{vacancy_id}.json", "w", encoding="utf-8") as f:
            json.dump(failed_data, f, ensure_ascii=False, indent=2, default=str)
    
    async def run(self, resume: bool = True, max_vacancies: Optional[int] = None, start_index: Optional[int] = None, end_index: Optional[int] = None):
        """Основная функция запуска обработки"""
        self.logger.info("Starting vacancy fetcher...")
        
        # Загружаем ID вакансий
        vacancy_ids = self.load_vacancy_ids()
        
        # Фильтрация по индексам ПЕРЕД фильтрацией уже обработанных
        if start_index is not None or end_index is not None:
            original_count = len(vacancy_ids)
            start_idx = start_index or 0
            end_idx = end_index or len(vacancy_ids)
            
            # Валидация индексов
            if start_idx < 0 or start_idx >= len(vacancy_ids):
                raise ValueError(f"start_index {start_idx} вне диапазона [0, {len(vacancy_ids)-1}]")
            if end_idx < start_idx or end_idx > len(vacancy_ids):
                raise ValueError(f"end_index {end_idx} должен быть >= start_index и <= {len(vacancy_ids)}")
            
            vacancy_ids = vacancy_ids[start_idx:end_idx]
            self.logger.info(f"Selected range [{start_idx}:{end_idx}] = {len(vacancy_ids)} vacancies from {original_count} total")
        
        if resume:
            vacancy_ids = self.filter_new_vacancies(vacancy_ids)
        
        # max_vacancies применяется ПОСЛЕ фильтрации по диапазону и resume
        if max_vacancies:
            vacancy_ids = vacancy_ids[:max_vacancies]
            self.logger.info(f"Limited to {max_vacancies} vacancies for processing")
        
        if not vacancy_ids:
            self.logger.info("No vacancies to process")
            return
        
        self.logger.info(f"Processing {len(vacancy_ids)} vacancies with {self.delay}s delay")
        
        # Статистика
        successful = 0
        failed = 0
        start_time = time.time()
        
        # Обрабатываем вакансии по одной
        for i, vacancy_id in enumerate(vacancy_ids, 1):
            self.logger.info(f"[{i}/{len(vacancy_ids)}] Processing vacancy {vacancy_id}")
            
            success = await self.process_vacancy(vacancy_id)
            if success:
                successful += 1
            else:
                failed += 1
            
            # Показываем прогресс каждые 10 вакансий
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed * 60  # вакансий в минуту
                eta = (len(vacancy_ids) - i) / (i / elapsed) if i > 0 else 0
                self.logger.info(f"Progress: {i}/{len(vacancy_ids)} ({i/len(vacancy_ids)*100:.1f}%), "
                               f"Rate: {rate:.1f}/min, ETA: {eta/60:.1f}min")
            
            # Задержка между запросами
            if i < len(vacancy_ids):  # Не ждем после последнего запроса
                await asyncio.sleep(self.delay)
        
        # Финальная статистика
        total_time = time.time() - start_time
        self.logger.info(f"Completed! Processed {len(vacancy_ids)} vacancies in {total_time/60:.1f} minutes")
        self.logger.info(f"Successful: {successful}, Failed: {failed}")
        
        # Статистика базы данных
        stats = self.storage.get_stats()
        self.logger.info(f"Database stats: {stats['total_vacancies']} vacancies, "
                        f"{stats['total_employers']} employers, {stats['unique_skills']} unique skills")


def setup_logging(level: str = "INFO"):
    """Настраивает логирование"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('vacancy_fetcher.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch vacancies from HH.ru API')
    parser.add_argument('csv_file', help='CSV file with vacancy IDs')
    parser.add_argument('--db', default='vacancies.db', help='SQLite database path')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests (seconds)')
    parser.add_argument('--no-resume', action='store_true', help='Process all vacancies (ignore already processed)')
    parser.add_argument('--max', type=int, help='Maximum number of vacancies to process')
    parser.add_argument('--start', type=int, help='Start index in CSV (0-based)')
    parser.add_argument('--end', type=int, help='End index in CSV (exclusive)')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Настраиваем логирование
    setup_logging(args.log_level)
    
    # Запускаем обработку
    async with VacancyFetcher(args.csv_file, args.db, args.delay) as fetcher:
        await fetcher.run(
            resume=not args.no_resume, 
            max_vacancies=args.max,
            start_index=args.start,
            end_index=args.end
        )


if __name__ == "__main__":
    asyncio.run(main())