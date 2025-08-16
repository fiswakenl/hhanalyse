import asyncio
import csv
import logging
import urllib.parse
from typing import List, Dict, Any
import httpx


class VacancyIDExtractor:
    def __init__(self, output_file: str = "vacancy_ids.csv", delay: float = 1.0):
        self.output_file = output_file
        self.delay = delay
        self.logger = logging.getLogger(__name__)
        
        # HTTP клиент с настройками
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "HH Vacancy ID Extractor 1.0",
                "Accept": "application/json"
            }
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def get_frontend_filter(self) -> str:
        """Возвращает фильтр для frontend вакансий"""
        return """(
  ("frontend" OR "front-end" OR "front end" OR фронтенд OR "фронт-енд" OR "фронт энд")
  AND ("developer" OR разработчик)
  AND (javascript OR typescript OR react OR vue OR angular OR "next.js" OR "nuxt.js" OR svelte)
)
AND NOT (
  fullstack OR "full-stack" OR backend OR "node.js" OR "react native" OR mobile OR android OR ios OR flutter
  OR qa OR тестировщик OR devops OR data OR analyst OR analytics OR ml OR "machine learning"
  OR golang OR python OR php OR java OR ".net" OR c# OR "c++" OR 1c OR bitrix OR "bitrix24"
  OR водитель OR курьер OR грузчик OR кладовщик OR комплектовщик OR фасовщик OR прораб
  OR сварщик OR слесарь OR электрик OR токарь OR монтажник OR разнорабоч OR уборщик
  OR кассир OR продавец OR официант OR бармен OR повар OR тракторист
)"""
    
    def get_data_science_ml_filter(self) -> str:
        """Возвращает фильтр для Data Science и Machine Learning вакансий"""
        return """data scientist" OR "data science" OR "machine learning" OR "ml engineer" OR "ai engineer"""
    
    async def fetch_vacancies_page(self, text_filter: str, page: int = 0, per_page: int = 100) -> Dict[str, Any]:
        """Получает страницу вакансий из API HH.ru"""
        base_url = "https://api.hh.ru/vacancies"
        
        params = {
            "per_page": per_page,
            "page": page,
            "text": text_filter
        }
        
        try:
            self.logger.debug(f"Fetching page {page} with {per_page} items")
            response = await self.client.get(base_url, params=params)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                self.logger.error(f"Bad request (400): {response.text}")
                return {}
            elif response.status_code == 429:
                self.logger.warning("Rate limit exceeded, waiting longer...")
                await asyncio.sleep(10)
                return await self.fetch_vacancies_page(text_filter, page, per_page)
            else:
                self.logger.error(f"HTTP {response.status_code}: {response.text}")
                return {}
                
        except httpx.TimeoutException:
            self.logger.error(f"Timeout fetching page {page}")
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching page {page}: {e}")
            return {}
    
    async def extract_all_vacancy_ids(self, text_filter: str) -> List[Dict[str, str]]:
        """Извлекает все ID и названия вакансий по фильтру"""
        vacancies = []
        page = 0
        per_page = 100
        
        self.logger.info("Starting vacancy extraction...")
        
        while True:
            data = await self.fetch_vacancies_page(text_filter, page, per_page)
            
            if not data or 'items' not in data:
                break
            
            items = data['items']
            if not items:
                break
            
            # Извлекаем ID и названия
            for item in items:
                vacancies.append({
                    'id': item['id'],
                    'name': item['name']
                })
            
            self.logger.info(f"Page {page}: extracted {len(items)} vacancies (total: {len(vacancies)})")
            
            # Проверяем, есть ли еще страницы
            total_pages = data.get('pages', 0)
            if page >= total_pages - 1:
                break
            
            page += 1
            
            # Задержка между запросами
            await asyncio.sleep(self.delay)
        
        self.logger.info(f"Extraction completed. Total vacancies: {len(vacancies)}")
        return vacancies
    
    def save_to_csv(self, vacancies: List[Dict[str, str]]):
        """Сохраняет вакансии в CSV файл"""
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['id', 'name']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for vacancy in vacancies:
                    writer.writerow(vacancy)
            
            self.logger.info(f"Saved {len(vacancies)} vacancies to {self.output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")
            raise
    
    async def run(self, filter_type: str = "data_science"):
        """Основная функция запуска извлечения"""
        if filter_type == "frontend":
            text_filter = self.get_frontend_filter()
            self.logger.info("Using frontend developer filter")
        elif filter_type == "data_science":
            text_filter = self.get_data_science_ml_filter()
            self.logger.info("Using data science/ML filter")
        else:
            raise ValueError("filter_type must be 'frontend' or 'data_science'")
        
        # Извлекаем вакансии
        vacancies = await self.extract_all_vacancy_ids(text_filter)
        
        if vacancies:
            # Сохраняем в CSV
            self.save_to_csv(vacancies)
            self.logger.info(f"Successfully extracted and saved {len(vacancies)} vacancies")
        else:
            self.logger.warning("No vacancies found")


def setup_logging(level: str = "INFO"):
    """Настраивает логирование"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('vacancy_extraction.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract vacancy IDs and names from HH.ru API')
    parser.add_argument('--output', default='vacancy_ids.csv', help='Output CSV file path')
    parser.add_argument('--filter', choices=['frontend', 'data_science'], default='data_science', 
                       help='Filter type: frontend or data_science')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests (seconds)')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Настраиваем логирование
    setup_logging(args.log_level)
    
    # Запускаем извлечение
    async with VacancyIDExtractor(args.output, args.delay) as extractor:
        await extractor.run(args.filter)


if __name__ == "__main__":
    asyncio.run(main())