#!/usr/bin/env python3
import sqlite3
import json
import argparse
from datetime import datetime


def view_stats(db_path: str):
    """Показывает общую статистику базы данных"""
    with sqlite3.connect(db_path) as conn:
        print("=== Статистика базы данных ===")
        
        # Общее количество вакансий
        cursor = conn.execute("SELECT COUNT(*) FROM vacancies")
        total_vacancies = cursor.fetchone()[0]
        print(f"Всего вакансий: {total_vacancies}")
        
        # Количество работодателей
        cursor = conn.execute("SELECT COUNT(*) FROM employers")
        total_employers = cursor.fetchone()[0]
        print(f"Всего работодателей: {total_employers}")
        
        # Количество уникальных навыков
        cursor = conn.execute("SELECT COUNT(DISTINCT skill_name) FROM vacancy_skills")
        unique_skills = cursor.fetchone()[0]
        print(f"Уникальных навыков: {unique_skills}")
        
        # Последняя обработанная вакансия
        cursor = conn.execute("SELECT MAX(fetched_at) FROM vacancies")
        last_fetched = cursor.fetchone()[0]
        if last_fetched:
            print(f"Последнее обновление: {last_fetched}")
        
        print()


def view_vacancies(db_path: str, limit: int = 10):
    """Показывает список вакансий"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT v.id, v.name, v.area_name, v.salary_from, v.salary_to, 
                   v.salary_currency, e.name as employer_name, v.published_at
            FROM vacancies v
            LEFT JOIN employers e ON v.employer_id = e.id
            ORDER BY v.published_at DESC
            LIMIT ?
        """, (limit,))
        
        print(f"=== Последние {limit} вакансий ===")
        for row in cursor.fetchall():
            vacancy_id, name, area, sal_from, sal_to, currency, employer, published = row
            
            # Форматируем зарплату
            salary = "не указана"
            if sal_from or sal_to:
                if sal_from and sal_to:
                    salary = f"{sal_from:,}-{sal_to:,} {currency or ''}"
                elif sal_from:
                    salary = f"от {sal_from:,} {currency or ''}"
                elif sal_to:
                    salary = f"до {sal_to:,} {currency or ''}"
            
            print(f"\n{vacancy_id}: {name}")
            print(f"  Работодатель: {employer}")
            print(f"  Город: {area}")
            print(f"  Зарплата: {salary}")
            print(f"  Опубликовано: {published}")


def view_skills(db_path: str, limit: int = 20):
    """Показывает топ навыков"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT skill_name, COUNT(*) as count
            FROM vacancy_skills
            GROUP BY skill_name
            ORDER BY count DESC
            LIMIT ?
        """, (limit,))
        
        print(f"=== Топ {limit} навыков ===")
        for i, (skill, count) in enumerate(cursor.fetchall(), 1):
            print(f"{i:2d}. {skill}: {count} вакансий")


def view_employers(db_path: str, limit: int = 10):
    """Показывает топ работодателей"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT e.name, COUNT(v.id) as vacancy_count
            FROM employers e
            LEFT JOIN vacancies v ON e.id = v.employer_id
            GROUP BY e.id, e.name
            ORDER BY vacancy_count DESC
            LIMIT ?
        """, (limit,))
        
        print(f"=== Топ {limit} работодателей ===")
        for i, (employer, count) in enumerate(cursor.fetchall(), 1):
            print(f"{i:2d}. {employer}: {count} вакансий")


def view_vacancy_detail(db_path: str, vacancy_id: str):
    """Показывает детальную информацию о вакансии"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT v.*, e.name as employer_name
            FROM vacancies v
            LEFT JOIN employers e ON v.employer_id = e.id
            WHERE v.id = ?
        """, (vacancy_id,))
        
        row = cursor.fetchone()
        if not row:
            print(f"Вакансия {vacancy_id} не найдена")
            return
        
        # Получаем названия колонок
        columns = [desc[0] for desc in cursor.description]
        vacancy_data = dict(zip(columns, row))
        
        print(f"=== Детали вакансии {vacancy_id} ===")
        print(f"Название: {vacancy_data['name']}")
        print(f"Работодатель: {vacancy_data['employer_name']}")
        print(f"Город: {vacancy_data['area_name']}")
        
        # Зарплата
        if vacancy_data['salary_from'] or vacancy_data['salary_to']:
            sal_from = vacancy_data['salary_from']
            sal_to = vacancy_data['salary_to']
            currency = vacancy_data['salary_currency'] or ''
            if sal_from and sal_to:
                print(f"Зарплата: {sal_from:,}-{sal_to:,} {currency}")
            elif sal_from:
                print(f"Зарплата: от {sal_from:,} {currency}")
            elif sal_to:
                print(f"Зарплата: до {sal_to:,} {currency}")
        else:
            print("Зарплата: не указана")
        
        print(f"Опыт: {vacancy_data['experience_name']}")
        print(f"График: {vacancy_data['schedule_name']}")
        print(f"Занятость: {vacancy_data['employment_name']}")
        print(f"Опубликовано: {vacancy_data['published_at']}")
        
        # Навыки
        cursor = conn.execute("""
            SELECT skill_name
            FROM vacancy_skills
            WHERE vacancy_id = ?
            ORDER BY skill_name
        """, (vacancy_id,))
        
        skills = [row[0] for row in cursor.fetchall()]
        if skills:
            print(f"Навыки: {', '.join(skills)}")
        
        # Описание - показываем Markdown версию если есть, иначе HTML
        cursor = conn.execute("""
            SELECT description_markdown, description 
            FROM vacancies 
            WHERE id = ?
        """, (vacancy_id,))
        
        desc_row = cursor.fetchone()
        if desc_row:
            md_desc, html_desc = desc_row
            
            if md_desc:
                desc = md_desc[:500]
                desc_type = "Markdown"
            elif html_desc:
                desc = html_desc[:500] 
                desc_type = "HTML"
            else:
                desc = None
                desc_type = None
            
            if desc:
                if len(desc) < (len(md_desc) if md_desc else len(html_desc or "")):
                    desc += "..."
                print(f"\nОписание ({desc_type}):\n{desc}")


def main():
    parser = argparse.ArgumentParser(description='Просмотр данных из базы вакансий')
    parser.add_argument('--db', default='vacancies.db', help='Путь к базе данных')
    parser.add_argument('--stats', action='store_true', help='Показать статистику')
    parser.add_argument('--vacancies', type=int, help='Показать N последних вакансий')
    parser.add_argument('--skills', type=int, help='Показать топ N навыков')
    parser.add_argument('--employers', type=int, help='Показать топ N работодателей')
    parser.add_argument('--detail', help='Показать детали вакансии по ID')
    
    args = parser.parse_args()
    
    # По умолчанию показываем статистику
    if not any([args.vacancies, args.skills, args.employers, args.detail]):
        args.stats = True
    
    if args.stats:
        view_stats(args.db)
    
    if args.vacancies:
        view_vacancies(args.db, args.vacancies)
        print()
    
    if args.skills:
        view_skills(args.db, args.skills)
        print()
    
    if args.employers:
        view_employers(args.db, args.employers)
        print()
    
    if args.detail:
        view_vacancy_detail(args.db, args.detail)


if __name__ == "__main__":
    main()