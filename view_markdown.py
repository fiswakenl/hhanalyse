#!/usr/bin/env python3
import sqlite3
import argparse

def show_description_comparison(vacancy_id: str):
    """Показывает сравнение HTML и Markdown описаний"""
    with sqlite3.connect("vacancies.db") as conn:
        cursor = conn.execute("""
            SELECT name, description, description_markdown, 
                   branded_description, branded_description_markdown
            FROM vacancies 
            WHERE id = ?
        """, (vacancy_id,))
        
        row = cursor.fetchone()
        if not row:
            print(f"Вакансия {vacancy_id} не найдена")
            return
        
        name, html_desc, md_desc, branded_html, branded_md = row
        
        print(f"=== Вакансия: {name} (ID: {vacancy_id}) ===\n")
        
        if html_desc:
            print("--- HTML описание (первые 300 символов) ---")
            print(html_desc[:300] + "..." if len(html_desc) > 300 else html_desc)
            print()
        
        if md_desc:
            print("--- Markdown описание (первые 500 символов) ---")
            print(md_desc[:500] + "..." if len(md_desc) > 500 else md_desc)
            print()
        
        if branded_html:
            print("--- Брендированное HTML (первые 200 символов) ---")
            print(branded_html[:200] + "..." if len(branded_html) > 200 else branded_html)
            print()
        
        if branded_md:
            print("--- Брендированное Markdown (первые 300 символов) ---")
            print(branded_md[:300] + "..." if len(branded_md) > 300 else branded_md)
            print()


def show_markdown_stats():
    """Показывает статистику по Markdown полям"""
    with sqlite3.connect("vacancies.db") as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM vacancies")
        total = cursor.fetchone()[0]
        
        if total == 0:
            print("База данных пуста")
            return
        
        print(f"=== Статистика Markdown полей (всего {total} вакансий) ===\n")
        
        # Описания
        cursor = conn.execute("""
            SELECT 
                COUNT(CASE WHEN description IS NOT NULL AND description != '' THEN 1 END) as html_desc,
                COUNT(CASE WHEN description_markdown IS NOT NULL AND description_markdown != '' THEN 1 END) as md_desc,
                COUNT(CASE WHEN branded_description IS NOT NULL AND branded_description != '' THEN 1 END) as html_branded,
                COUNT(CASE WHEN branded_description_markdown IS NOT NULL AND branded_description_markdown != '' THEN 1 END) as md_branded
            FROM vacancies
        """)
        
        html_desc, md_desc, html_branded, md_branded = cursor.fetchone()
        
        print(f"Обычные описания:")
        print(f"  HTML: {html_desc}/{total} ({html_desc/total*100:.1f}%)")
        print(f"  Markdown: {md_desc}/{total} ({md_desc/total*100:.1f}%)")
        print()
        
        print(f"Брендированные описания:")
        print(f"  HTML: {html_branded}/{total} ({html_branded/total*100:.1f}%)")
        print(f"  Markdown: {md_branded}/{total} ({md_branded/total*100:.1f}%)")
        print()
        
        # Средние длины
        cursor = conn.execute("""
            SELECT 
                AVG(LENGTH(description)) as avg_html_desc,
                AVG(LENGTH(description_markdown)) as avg_md_desc,
                AVG(LENGTH(branded_description)) as avg_html_branded,
                AVG(LENGTH(branded_description_markdown)) as avg_md_branded
            FROM vacancies
            WHERE description IS NOT NULL OR description_markdown IS NOT NULL
        """)
        
        avg_html_desc, avg_md_desc, avg_html_branded, avg_md_branded = cursor.fetchone()
        
        print(f"Средние длины:")
        if avg_html_desc:
            print(f"  HTML описание: {avg_html_desc:.0f} символов")
        if avg_md_desc:
            print(f"  Markdown описание: {avg_md_desc:.0f} символов")
        if avg_html_branded:
            print(f"  HTML брендированное: {avg_html_branded:.0f} символов")
        if avg_md_branded:
            print(f"  Markdown брендированное: {avg_md_branded:.0f} символов")


def list_vacancies_with_markdown():
    """Показывает список вакансий с информацией о Markdown полях"""
    with sqlite3.connect("vacancies.db") as conn:
        cursor = conn.execute("""
            SELECT id, name,
                   CASE WHEN description_markdown IS NOT NULL AND description_markdown != '' THEN 'MD' ELSE '' END as has_md,
                   CASE WHEN branded_description_markdown IS NOT NULL AND branded_description_markdown != '' THEN 'BRAND' ELSE '' END as has_branded,
                   LENGTH(description_markdown) as md_len
            FROM vacancies
            ORDER BY fetched_at DESC
        """)
        
        print("=== Список вакансий с Markdown полями ===\n")
        print("ID          | Название                           | Поля    | Длина MD")
        print("-" * 80)
        
        for row in cursor.fetchall():
            vacancy_id, name, has_md, has_branded, md_len = row
            name_short = name[:30] + "..." if len(name) > 30 else name
            fields = f"{has_md} {has_branded}".strip()
            md_len_str = str(md_len) if md_len else "0"
            
            print(f"{vacancy_id:<11} | {name_short:<34} | {fields:<7} | {md_len_str}")


def main():
    parser = argparse.ArgumentParser(description='Просмотр Markdown описаний вакансий')
    parser.add_argument('--stats', action='store_true', help='Показать статистику')
    parser.add_argument('--list', action='store_true', help='Показать список вакансий')
    parser.add_argument('--compare', help='Сравнить HTML и Markdown для вакансии (ID)')
    
    args = parser.parse_args()
    
    if args.stats:
        show_markdown_stats()
    elif args.list:
        list_vacancies_with_markdown()
    elif args.compare:
        show_description_comparison(args.compare)
    else:
        # По умолчанию показываем статистику и список
        show_markdown_stats()
        print()
        list_vacancies_with_markdown()


if __name__ == "__main__":
    main()