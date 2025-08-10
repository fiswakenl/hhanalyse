#!/usr/bin/env python3
"""
Тест для проверки исправления фильтрации данных
"""
import pandas as pd
import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем функции из дашборда
from interactive_dashboard import filter_data, SALARY_MIN, SALARY_MAX, df

def test_filter_fix():
    """Тестирует исправление фильтрации данных"""
    
    print("=== Тест исправления фильтрации HR Analytics дашборда ===\n")
    
    # Исходные данные
    print(f"Общее количество записей в датасете: {len(df)}")
    
    # Количество записей с зарплатами
    with_salary = df['salary_from_rub'].notna().sum()
    without_salary = len(df) - with_salary
    print(f"Записи с зарплатой: {with_salary}")
    print(f"Записи без зарплаты: {without_salary}")
    
    print(f"\nДиапазон зарплат: {SALARY_MIN} - {SALARY_MAX}")
    
    # Тест 1: Без фильтров (должно показать все 743)
    print("\n--- Тест 1: Без фильтров ---")
    filtered_no_filters = filter_data("all", "all", "all", [SALARY_MIN, SALARY_MAX])
    print(f"Результат без фильтров: {len(filtered_no_filters)} записей")
    
    if len(filtered_no_filters) == len(df):
        print("✅ ИСПРАВЛЕНО! Показываются все записи без фильтров")
    else:
        print("❌ ПРОБЛЕМА: Все еще теряются записи")
    
    # Тест 2: С фильтром зарплат (должно показать только записи с зарплатой)
    print("\n--- Тест 2: С фильтром зарплат ---")
    filtered_with_salary = filter_data("all", "all", "all", [50000, 200000])
    print(f"Результат с фильтром зарплат 50k-200k: {len(filtered_with_salary)} записей")
    
    # Проверяем что все отфильтрованные записи имеют зарплату в диапазоне
    valid_salary_filter = (
        (filtered_with_salary['salary_from_rub'].notna()) &
        (filtered_with_salary['salary_from_rub'] >= 50000) &
        (filtered_with_salary['salary_from_rub'] <= 200000)
    ).all()
    
    if valid_salary_filter:
        print("✅ Фильтрация по зарплате работает корректно")
    else:
        print("❌ ПРОБЛЕМА: Фильтрация по зарплате работает некорректно")
    
    # Тест 3: С другими фильтрами без фильтрации зарплат
    print("\n--- Тест 3: Фильтр по опыту без зарплат ---")
    experience_options = df['experience_name'].dropna().unique()
    if len(experience_options) > 0:
        test_experience = experience_options[0]
        filtered_experience = filter_data("all", "all", test_experience, [SALARY_MIN, SALARY_MAX])
        expected_count = len(df[df['experience_name'] == test_experience])
        print(f"Фильтр по опыту '{test_experience}': {len(filtered_experience)} из {expected_count} ожидаемых")
        
        if len(filtered_experience) == expected_count:
            print("✅ Фильтрация по опыту работает корректно")
        else:
            print("❌ ПРОБЛЕМА: Фильтрация по опыту работает некорректно")
    
    print("\n=== Результат тестирования ===")
    print(f"Исходная проблема: 225 из 743 записей")
    print(f"После исправления: {len(filtered_no_filters)} из {len(df)} записей")
    
    if len(filtered_no_filters) == len(df):
        print("🎉 ПРОБЛЕМА ИСПРАВЛЕНА! Теперь показываются все вакансии по умолчанию.")
    else:
        print("⚠️  Проблема частично решена, но могут быть другие проблемы фильтрации.")

if __name__ == "__main__":
    test_filter_fix()