from typing import Optional, List, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Area(BaseModel):
    id: str
    name: str
    url: Optional[str] = None


class Salary(BaseModel):
    from_: Optional[int] = Field(None, alias="from")
    to: Optional[int] = None
    currency: Optional[str] = None
    gross: Optional[bool] = None


class Experience(BaseModel):
    id: str
    name: str


class Schedule(BaseModel):
    id: str
    name: str


class Employment(BaseModel):
    id: str
    name: str


class KeySkill(BaseModel):
    name: str


class LogoUrls(BaseModel):
    original: Optional[str] = None
    url_90: Optional[str] = Field(None, alias="90")
    url_240: Optional[str] = Field(None, alias="240")


class Employer(BaseModel):
    id: str
    name: str
    url: Optional[str] = None
    alternate_url: Optional[str] = None
    logo_urls: Optional[LogoUrls] = None
    vacancies_url: Optional[str] = None
    accredited_it_employer: Optional[bool] = None
    trusted: Optional[bool] = None


class Metro(BaseModel):
    station_name: str
    line_name: str
    station_id: str
    line_id: str
    lat: float
    lng: float


class Address(BaseModel):
    city: Optional[str] = None
    street: Optional[str] = None
    building: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    description: Optional[str] = None
    raw: Optional[str] = None
    metro: Optional[Metro] = None
    metro_stations: Optional[List[Metro]] = None
    id: Optional[str] = None


class VacancyType(BaseModel):
    id: str
    name: str


class BillingType(BaseModel):
    id: str
    name: str


class WorkFormat(BaseModel):
    id: str
    name: str


class Vacancy(BaseModel):
    id: str
    name: str
    area: Area
    salary: Optional[Salary] = None
    salary_range: Optional[dict] = None
    type: VacancyType
    address: Optional[Address] = None
    experience: Experience
    schedule: Schedule
    employment: Employment
    description: Optional[str] = None
    branded_description: Optional[str] = None
    key_skills: List[KeySkill] = []
    employer: Employer
    published_at: datetime
    created_at: datetime
    expires_at: Optional[datetime] = None
    premium: Optional[bool] = None
    billing_type: Optional[BillingType] = None
    work_format: Optional[List[WorkFormat]] = None
    
    # URLs
    alternate_url: Optional[str] = None
    apply_alternate_url: Optional[str] = None
    response_url: Optional[str] = None
    
    # Contact and application settings
    allow_messages: Optional[bool] = None
    show_contacts: Optional[bool] = None
    contacts: Optional[dict] = None
    response_letter_required: Optional[bool] = None
    
    # Additional flags
    archived: Optional[bool] = None
    accept_handicapped: Optional[bool] = None
    accept_kids: Optional[bool] = None
    
    # Professional data
    specializations: Optional[List[dict]] = None
    professional_roles: Optional[List[dict]] = None
    
    # Work conditions
    working_days: Optional[List[dict]] = None
    working_time_intervals: Optional[List[dict]] = None
    working_time_modes: Optional[List[dict]] = None
    
    # Additional metadata
    insider_interview: Optional[dict] = None
    vacancy_constructor_template: Optional[dict] = None
    relations: Optional[List[dict]] = None
    department: Optional[dict] = None
    
    # Дополнительные поля для обработки
    fetched_at: Optional[datetime] = None
    raw_json: Optional[str] = None


class VacancyResponse(BaseModel):
    """Модель для полного ответа API"""
    vacancy: Vacancy
    
    @classmethod
    def from_api_response(cls, data: dict) -> "VacancyResponse":
        # Добавляем timestamp получения данных
        data["fetched_at"] = datetime.now()
        vacancy = Vacancy(**data)
        return cls(vacancy=vacancy)