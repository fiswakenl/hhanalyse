#!/usr/bin/env python3
"""
Экспорт описаний вакансий в Markdown файлы
"""
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime

def export_single_vacancy(vacancy_id: str, output_dir: str = "markdown_export"):
    """Экспортирует одну вакансию в Markdown файл"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with sqlite3.connect("vacancies.db") as conn:
        cursor = conn.execute("""
            SELECT v.id, v.name, v.description_markdown, v.branded_description_markdown,
                   v.area_name, v.salary_from, v.salary_to, v.salary_currency,
                   v.experience_name, v.schedule_name, v.employment_name,
                   v.published_at, e.name as employer_name, v.alternate_url
            FROM vacancies v
            LEFT JOIN employers e ON v.employer_id = e.id
            WHERE v.id = ?
        """, (vacancy_id,))
        
        row = cursor.fetchone()
        if not row:
            print(f"Вакансия {vacancy_id} не найдена")
            return None
        
        (vid, name, description_md, branded_md, area, sal_from, sal_to, currency,
         experience, schedule, employment, published, employer, url) = row
        
        # Формируем зарплату
        salary_text = "Не указана"
        if sal_from or sal_to:
            if sal_from and sal_to:
                salary_text = f"{sal_from:,}-{sal_to:,} {currency or ''}"
            elif sal_from:
                salary_text = f"от {sal_from:,} {currency or ''}"
            elif sal_to:
                salary_text = f"до {sal_to:,} {currency or ''}"
        
        # Получаем навыки
        cursor = conn.execute("""
            SELECT skill_name FROM vacancy_skills 
            WHERE vacancy_id = ? 
            ORDER BY skill_name
        """, (vacancy_id,))
        skills = [row[0] for row in cursor.fetchall()]
        
        # Создаем Markdown файл
        md_content = f"""# {name}

**ID:** {vid}  
**Работодатель:** {employer}  
**Город:** {area}  
**Зарплата:** {salary_text}  
**Опыт:** {experience}  
**График:** {schedule}  
**Занятость:** {employment}  
**Опубликовано:** {published}  
**Ссылка:** {url}  

**Навыки:** {', '.join(skills) if skills else 'Не указаны'}

---

## Описание вакансии

{description_md or 'Описание отсутствует'}
"""
        
        if branded_md:
            md_content += f"""

---

## Брендированное описание

{branded_md}
"""
        
        md_content += f"""

---

*Экспортировано {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Сохраняем файл
        safe_name = name.replace('/', '_').replace('\\', '_').replace(':', '_')[:50]
        filename = f"{vacancy_id}_{safe_name}.md"
        file_path = output_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"Экспортировано: {file_path}")
        return file_path


def export_all_vacancies(output_dir: str = "markdown_export", limit: int = None):
    """Экспортирует все вакансии в отдельные Markdown файлы"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with sqlite3.connect("vacancies.db") as conn:
        query = "SELECT id FROM vacancies ORDER BY published_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = conn.execute(query)
        vacancy_ids = [row[0] for row in cursor.fetchall()]
    
    print(f"Экспортирую {len(vacancy_ids)} вакансий в {output_dir}/")
    
    exported = 0
    for vacancy_id in vacancy_ids:
        try:
            export_single_vacancy(vacancy_id, output_dir)
            exported += 1
        except Exception as e:
            print(f"Ошибка экспорта {vacancy_id}: {e}")
    
    print(f"Успешно экспортировано: {exported}/{len(vacancy_ids)}")


def export_summary(output_dir: str = "markdown_export"):
    """Создает сводный файл со всеми вакансиями"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with sqlite3.connect("vacancies.db") as conn:
        cursor = conn.execute("""
            SELECT v.id, v.name, v.description_markdown,
                   v.area_name, v.salary_from, v.salary_to, v.salary_currency,
                   e.name as employer_name, v.published_at
            FROM vacancies v
            LEFT JOIN employers e ON v.employer_id = e.id
            ORDER BY v.published_at DESC
        """)
        
        vacancies = cursor.fetchall()
    
    # Создаем сводный файл
    md_content = f"""# Сводка по вакансиям

**Всего вакансий:** {len(vacancies)}  
**Дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""
    
    for row in vacancies:
        vid, name, description_md, area, sal_from, sal_to, currency, employer, published = row
        
        # Формируем зарплату
        salary_text = "Не указана"
        if sal_from or sal_to:
            if sal_from and sal_to:
                salary_text = f"{sal_from:,}-{sal_to:,} {currency or ''}"
            elif sal_from:
                salary_text = f"от {sal_from:,} {currency or ''}"
            elif sal_to:
                salary_text = f"до {sal_to:,} {currency or ''}"
        
        # Краткое описание (первые 200 символов)
        short_desc = ""
        if description_md:
            short_desc = description_md.replace('\n', ' ')[:200]
            if len(description_md) > 200:
                short_desc += "..."
        
        md_content += f"""## {name} ({vid})

**Работодатель:** {employer}  
**Город:** {area}  
**Зарплата:** {salary_text}  
**Опубликовано:** {published}

{short_desc}

---

"""
    
    # Сохраняем сводный файл
    summary_path = output_path / "summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Создан сводный файл: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Экспорт описаний вакансий в Markdown')
    parser.add_argument('--output', '-o', default='markdown_export', help='Папка для экспорта')
    parser.add_argument('--vacancy-id', help='Экспортировать одну вакансию по ID')
    parser.add_argument('--all', action='store_true', help='Экспортировать все вакансии')
    parser.add_argument('--summary', action='store_true', help='Создать сводный файл')
    parser.add_argument('--limit', type=int, help='Ограничить количество вакансий')
    
    args = parser.parse_args()
    
    if args.vacancy_id:
        export_single_vacancy(args.vacancy_id, args.output)
    elif args.all:
        export_all_vacancies(args.output, args.limit)
    elif args.summary:
        export_summary(args.output)
    else:
        # По умолчанию создаем сводку
        export_summary(args.output)
        print("\nДля экспорта всех вакансий используйте --all")
        print("Для экспорта одной вакансии используйте --vacancy-id ID")


if __name__ == "__main__":
    main()