import sqlite3
import json
from typing import Optional, List
from datetime import datetime
from models import Vacancy, Employer, KeySkill
from html_to_markdown import convert_html_to_markdown
import logging

logger = logging.getLogger(__name__)


class VacancyStorage:
    def __init__(self, db_path: str = "vacancies.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Создает таблицы базы данных"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS employers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    url TEXT,
                    alternate_url TEXT,
                    logo_urls TEXT,  -- JSON строка
                    vacancies_url TEXT,
                    accredited_it_employer INTEGER,
                    trusted INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS vacancies (
                    id TEXT PRIMARY KEY,  -- Используем ID вакансии как первичный ключ
                    name TEXT NOT NULL,
                    description TEXT,
                    description_markdown TEXT,  -- Описание в Markdown
                    branded_description TEXT,
                    branded_description_markdown TEXT,  -- Брендированное описание в Markdown
                    
                    -- Area information
                    area_id TEXT,
                    area_name TEXT,
                    area_url TEXT,
                    
                    -- Salary information
                    salary_from INTEGER,
                    salary_to INTEGER,
                    salary_currency TEXT,
                    salary_gross INTEGER,
                    salary_range TEXT,  -- JSON для salary_range если есть
                    
                    -- Job details
                    experience_id TEXT,
                    experience_name TEXT,
                    schedule_id TEXT,
                    schedule_name TEXT,
                    employment_id TEXT,
                    employment_name TEXT,
                    
                    -- Employer reference
                    employer_id TEXT,
                    
                    -- Address information (JSON)
                    address TEXT,
                    
                    -- Type and billing
                    type_id TEXT,
                    type_name TEXT,
                    billing_type_id TEXT,
                    billing_type_name TEXT,
                    
                    -- URLs
                    alternate_url TEXT,
                    apply_alternate_url TEXT,
                    response_url TEXT,
                    
                    -- Work format and conditions (JSON arrays)
                    work_format TEXT,
                    working_days TEXT,
                    working_time_intervals TEXT,
                    working_time_modes TEXT,
                    
                    -- Contact and application settings
                    allow_messages INTEGER,
                    show_contacts INTEGER,
                    contacts TEXT,  -- JSON
                    response_letter_required INTEGER,
                    
                    -- Additional flags and settings
                    premium INTEGER,
                    archived INTEGER,
                    accept_handicapped INTEGER,
                    accept_kids INTEGER,
                    
                    -- Professional data
                    specializations TEXT,  -- JSON array
                    professional_roles TEXT,  -- JSON array
                    
                    -- Timestamps
                    published_at TIMESTAMP,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    fetched_at TIMESTAMP,
                    
                    -- Additional metadata
                    insider_interview TEXT,  -- JSON если есть
                    vacancy_constructor_template TEXT,  -- JSON если есть
                    relations TEXT,  -- JSON array
                    department TEXT,  -- JSON если есть
                    
                    -- Raw data backup
                    raw_json TEXT,
                    
                    FOREIGN KEY (employer_id) REFERENCES employers (id)
                );
                
                CREATE TABLE IF NOT EXISTS vacancy_skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vacancy_id TEXT,
                    skill_name TEXT,
                    FOREIGN KEY (vacancy_id) REFERENCES vacancies (id),
                    UNIQUE(vacancy_id, skill_name)
                );
                
                -- Индексы для быстрого поиска
                CREATE INDEX IF NOT EXISTS idx_vacancies_employer_id ON vacancies (employer_id);
                CREATE INDEX IF NOT EXISTS idx_vacancies_area_id ON vacancies (area_id);
                CREATE INDEX IF NOT EXISTS idx_vacancies_published_at ON vacancies (published_at);
                CREATE INDEX IF NOT EXISTS idx_vacancy_skills_vacancy_id ON vacancy_skills (vacancy_id);
                CREATE INDEX IF NOT EXISTS idx_vacancy_skills_skill_name ON vacancy_skills (skill_name);
            """)
            logger.info("Database initialized successfully")
    
    def vacancy_exists(self, vacancy_id: str) -> bool:
        """Проверяет существование вакансии в базе"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM vacancies WHERE id = ?", (vacancy_id,))
            return cursor.fetchone() is not None
    
    def save_employer(self, employer: Employer) -> None:
        """Сохраняет информацию о работодателе"""
        with sqlite3.connect(self.db_path) as conn:
            # Конвертируем logo_urls в JSON если есть
            logo_urls_json = None
            if employer.logo_urls:
                logo_urls_json = json.dumps(employer.logo_urls.dict())
            
            conn.execute("""
                INSERT OR REPLACE INTO employers 
                (id, name, url, alternate_url, logo_urls, vacancies_url, 
                 accredited_it_employer, trusted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                employer.id,
                employer.name,
                employer.url,
                employer.alternate_url,
                logo_urls_json,
                employer.vacancies_url,
                employer.accredited_it_employer,
                employer.trusted
            ))
    
    def save_vacancy(self, vacancy: Vacancy, raw_json: Optional[str] = None) -> None:
        """Сохраняет вакансию в базу данных"""
        try:
            # Сначала сохраняем работодателя
            self.save_employer(vacancy.employer)
            
            with sqlite3.connect(self.db_path) as conn:
                # Подготавливаем JSON данные
                def to_json(obj):
                    if obj is None:
                        return None
                    if hasattr(obj, 'dict'):
                        return json.dumps(obj.dict(), ensure_ascii=False)
                    if isinstance(obj, list):
                        return json.dumps([item.dict() if hasattr(item, 'dict') else item for item in obj], ensure_ascii=False)
                    return json.dumps(obj, ensure_ascii=False)
                
                address_json = to_json(vacancy.address)
                work_format_json = to_json(vacancy.work_format)
                salary_range_json = to_json(vacancy.salary_range)
                contacts_json = to_json(vacancy.contacts)
                specializations_json = to_json(vacancy.specializations)
                professional_roles_json = to_json(vacancy.professional_roles)
                working_days_json = to_json(vacancy.working_days)
                working_time_intervals_json = to_json(vacancy.working_time_intervals)
                working_time_modes_json = to_json(vacancy.working_time_modes)
                insider_interview_json = to_json(vacancy.insider_interview)
                vacancy_constructor_template_json = to_json(vacancy.vacancy_constructor_template)
                relations_json = to_json(vacancy.relations)
                department_json = to_json(vacancy.department)
                
                # Конвертируем HTML описания в Markdown
                description_md = convert_html_to_markdown(vacancy.description) if vacancy.description else None
                branded_description_md = convert_html_to_markdown(vacancy.branded_description) if vacancy.branded_description else None
                
                # Сохраняем всю информацию о вакансии
                conn.execute("""
                    INSERT OR REPLACE INTO vacancies 
                    (id, name, description, description_markdown, branded_description, branded_description_markdown, 
                     area_id, area_name, area_url,
                     salary_from, salary_to, salary_currency, salary_gross, salary_range,
                     experience_id, experience_name, schedule_id, schedule_name,
                     employment_id, employment_name, employer_id, address,
                     type_id, type_name, billing_type_id, billing_type_name,
                     alternate_url, apply_alternate_url, response_url,
                     work_format, working_days, working_time_intervals, working_time_modes,
                     allow_messages, show_contacts, contacts, response_letter_required,
                     premium, archived, accept_handicapped, accept_kids,
                     specializations, professional_roles,
                     published_at, created_at, expires_at, fetched_at,
                     insider_interview, vacancy_constructor_template, relations, department,
                     raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    vacancy.id,
                    vacancy.name,
                    vacancy.description,
                    description_md,
                    vacancy.branded_description,
                    branded_description_md,
                    vacancy.area.id,
                    vacancy.area.name,
                    vacancy.area.url if hasattr(vacancy.area, 'url') else None,
                    vacancy.salary.from_ if vacancy.salary else None,
                    vacancy.salary.to if vacancy.salary else None,
                    vacancy.salary.currency if vacancy.salary else None,
                    vacancy.salary.gross if vacancy.salary else None,
                    salary_range_json,
                    vacancy.experience.id,
                    vacancy.experience.name,
                    vacancy.schedule.id,
                    vacancy.schedule.name,
                    vacancy.employment.id,
                    vacancy.employment.name,
                    vacancy.employer.id,
                    address_json,
                    vacancy.type.id,
                    vacancy.type.name,
                    vacancy.billing_type.id if vacancy.billing_type else None,
                    vacancy.billing_type.name if vacancy.billing_type else None,
                    vacancy.alternate_url,
                    vacancy.apply_alternate_url,
                    vacancy.response_url,
                    work_format_json,
                    working_days_json,
                    working_time_intervals_json,
                    working_time_modes_json,
                    vacancy.allow_messages,
                    vacancy.show_contacts,
                    contacts_json,
                    vacancy.response_letter_required,
                    vacancy.premium,
                    vacancy.archived,
                    vacancy.accept_handicapped,
                    vacancy.accept_kids,
                    specializations_json,
                    professional_roles_json,
                    vacancy.published_at,
                    vacancy.created_at,
                    vacancy.expires_at,
                    vacancy.fetched_at or datetime.now(),
                    insider_interview_json,
                    vacancy_constructor_template_json,
                    relations_json,
                    department_json,
                    raw_json
                ))
                
                # Удаляем старые навыки для этой вакансии
                conn.execute("DELETE FROM vacancy_skills WHERE vacancy_id = ?", (vacancy.id,))
                
                # Сохраняем навыки
                for skill in vacancy.key_skills:
                    conn.execute("""
                        INSERT INTO vacancy_skills (vacancy_id, skill_name)
                        VALUES (?, ?)
                    """, (vacancy.id, skill.name))
                
                logger.info(f"Vacancy {vacancy.id} saved successfully with {len(vacancy.key_skills)} skills")
                
        except Exception as e:
            logger.error(f"Error saving vacancy {vacancy.id}: {e}")
            raise
    
    def get_processed_vacancy_ids(self) -> List[str]:
        """Возвращает список уже обработанных ID вакансий"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id FROM vacancies")
            return [row[0] for row in cursor.fetchall()]
    
    def get_vacancy_count(self) -> int:
        """Возвращает количество сохраненных вакансий"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM vacancies")
            return cursor.fetchone()[0]
    
    def get_employer_count(self) -> int:
        """Возвращает количество уникальных работодателей"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM employers")
            return cursor.fetchone()[0]
    
    def get_stats(self) -> dict:
        """Возвращает статистику базы данных"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Общее количество вакансий
            cursor = conn.execute("SELECT COUNT(*) FROM vacancies")
            stats['total_vacancies'] = cursor.fetchone()[0]
            
            # Общее количество работодателей
            cursor = conn.execute("SELECT COUNT(*) FROM employers")
            stats['total_employers'] = cursor.fetchone()[0]
            
            # Общее количество навыков
            cursor = conn.execute("SELECT COUNT(DISTINCT skill_name) FROM vacancy_skills")
            stats['unique_skills'] = cursor.fetchone()[0]
            
            # Последняя обработанная вакансия
            cursor = conn.execute("SELECT MAX(fetched_at) FROM vacancies")
            last_fetched = cursor.fetchone()[0]
            stats['last_fetched'] = last_fetched
            
            return stats