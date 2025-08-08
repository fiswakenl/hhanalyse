from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
import httpx
import pandas as pd
import time


class Vacancy(BaseModel):
    id: str
    name: str
    area_name: str
    employer_name: str
    salary_from: Optional[int] = None
    salary_to: Optional[int] = None
    salary_currency: Optional[str] = None
    experience: Optional[str] = None
    schedule: Optional[str] = None
    employment: Optional[str] = None
    work_format: Optional[str] = None
    requirement: Optional[str] = None
    responsibility: Optional[str] = None
    published_at: str
    alternate_url: str

    @classmethod
    def from_api_response(cls, data: dict):
        return cls(
            id=data["id"],
            name=data["name"],
            area_name=data["area"]["name"],
            employer_name=data["employer"]["name"],
            salary_from=data["salary"]["from"] if data.get("salary") else None,
            salary_to=data["salary"]["to"] if data.get("salary") else None,
            salary_currency=data["salary"]["currency"] if data.get("salary") else None,
            experience=data["experience"]["name"] if data.get("experience") else None,
            schedule=data["schedule"]["name"] if data.get("schedule") else None,
            employment=data["employment"]["name"] if data.get("employment") else None,
            work_format=", ".join([wf.get("name", str(wf)) if isinstance(wf, dict) else str(wf) for wf in data.get("work_format", [])]),
            requirement=data["snippet"]["requirement"] if data.get("snippet") and data["snippet"].get("requirement") else None,
            responsibility=data["snippet"]["responsibility"] if data.get("snippet") and data["snippet"].get("responsibility") else None,
            published_at=data["published_at"],
            alternate_url=data["alternate_url"]
        )


def fetch_vacancies(text: str = "(react or vue) AND ( NOT fullstack NOT full-stack NOT native NOT junior)") -> List[Vacancy]:
    base_url = "https://api.hh.ru/vacancies"
    vacancies = []
    page = 0
    
    print(f"Начинаем сбор вакансий с запросом: {text}")
    
    with httpx.Client() as client:
        while True:
            params = {
                "text": text,
                "per_page": 100,
                "page": page
            }
            
            print(f"Обрабатываем страницу {page}...")
            response = client.get(base_url, params=params)
            
            if response.status_code != 200:
                print(f"Ошибка API: {response.status_code}")
                break
                
            data = response.json()
            items = data.get("items", [])
            
            if not items:
                print("Достигнута последняя страница")
                break
                
            for item in items:
                try:
                    vacancy = Vacancy.from_api_response(item)
                    vacancies.append(vacancy)
                except Exception as e:
                    print(f"Ошибка парсинга вакансии {item.get('id', 'unknown')}: {e}")
                    continue
            
            print(f"Собрано {len(items)} вакансий со страницы {page}")
            
            if page == 0:
                total_pages = data.get("pages", 1)
                total_found = data.get("found", len(items))
                print(f"Всего найдено: {total_found}, страниц: {total_pages}")
            
            page += 1
            time.sleep(1.0)  # Пауза для API чтобы избежать rate limiting
            
            if page >= data.get("pages", 1):
                break
    
    return vacancies


def save_to_csv(vacancies: List[Vacancy]) -> str:
    if not vacancies:
        print("Нет данных для сохранения")
        return ""
    
    df = pd.DataFrame([v.model_dump() for v in vacancies])
    
    # Переименовываем колонки для удобства
    df.columns = [
        "ID", "Название", "Город", "Компания", "Зарплата от", "Зарплата до", 
        "Валюта", "Опыт", "График", "Занятость", "Формат работы", 
        "Требования", "Обязанности", "Дата публикации", "Ссылка"
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vacancies_{timestamp}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"Данные сохранены в {filename}")
    print(f"Всего вакансий: {len(vacancies)}")
    
    return filename


def main():
    vacancies = fetch_vacancies()
    save_to_csv(vacancies)


if __name__ == "__main__":
    main()
