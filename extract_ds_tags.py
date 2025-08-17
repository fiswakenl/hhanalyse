#!/usr/bin/env python3
"""
LLM-based DS tags extraction script for job vacancies.
Usage: python extract_ds_tags.py --start 0 --end 100
"""

import argparse
import os
from ds_tags_extractor import DSTagsExtractor


def main():
    """Основная точка входа в скрипт."""
    
    parser = argparse.ArgumentParser(
        description="Извлечение DS-тегов из вакансий с помощью LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python extract_ds_tags.py --start 0 --end 10
  python extract_ds_tags.py --start 100 --end 200 --model meta-llama/llama-3.1-8b-instruct
  python extract_ds_tags.py --start 0 --end 50 --output data/test_results.parquet

Переменные окружения:
  OPENROUTER_API_KEY - API ключ для OpenRouter (обязательно)
        """
    )
    
    parser.add_argument("--start", type=int, required=True, 
                       help="Начальный индекс для обработки")
    parser.add_argument("--end", type=int, required=True, 
                       help="Конечный индекс для обработки")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b:free", 
                       help="Модель для извлечения (по умолчанию: openai/gpt-oss-20b:free)")
    parser.add_argument("--input", type=str, default="data/ds_vacancies.parquet",
                       help="Входной parquet файл (по умолчанию: data/ds_vacancies.parquet)")
    parser.add_argument("--output", type=str, default="data/ds_vacancies.parquet",
                       help="Выходной parquet файл (по умолчанию: data/ds_vacancies.parquet)")
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Размер батча для API запросов (по умолчанию: 5)")
    
    args = parser.parse_args()
    
    # Проверяем корректность аргументов
    if args.start < 0:
        print("Ошибка: --start должен быть >= 0")
        return
    
    if args.end <= args.start:
        print("Ошибка: --end должен быть больше --start")
        return
    
    if args.batch_size < 1 or args.batch_size > 10:
        print("Ошибка: --batch-size должен быть от 1 до 10")
        return
    
    # Используем захардкоженный API ключ
    api_key = "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"
    if not api_key:
        print("Ошибка: API ключ не установлен")
        return
    
    # Проверяем существование входного файла
    if not os.path.exists(args.input):
        print(f"Ошибка: входной файл {args.input} не найден")
        return
    
    # Проверяем существование tags.json
    if not os.path.exists("tags.json"):
        print("Ошибка: файл tags.json не найден в текущей директории")
        print("Убедитесь что файл tags.json с базовыми категориями существует")
        return
    
    # Инициализируем экстрактор
    print(f"Запуск извлечения DS-тегов...")
    print(f"Модель: {args.model}")
    print(f"Диапазон: {args.start}-{args.end} ({args.end - args.start} записей)")
    print(f"Размер батча: {args.batch_size}")
    print(f"Входной файл: {args.input}")
    print(f"Выходной файл: {args.output}")
    print()
    
    try:
        extractor = DSTagsExtractor(
            api_key=api_key, 
            model=args.model, 
            batch_size=args.batch_size
        )
        
        # Обрабатываем вакансии
        extractor.process_range(args.start, args.end, args.input, args.output)
        
        print("\\nОбработка завершена успешно!")
        print(f"Результаты сохранены в {args.output}")
        print("\\nНовые столбцы:")
        print("- специализация (массив)")
        print("- уровень (строка)")
        print("- тип_компании (строка)")
        print("- индустрия (строка)")
        print("- языки_программирования (массив)")
        print("- ml_библиотеки (массив)")
        print("- визуализация (массив)")
        print("- данные_библиотеки (массив)")
        print("- nlp_библиотеки (массив)")
        print("- cv_библиотеки (массив)")
        print("- mlops_инструменты (массив)")
        print("- облачные_платформы (массив)")
        print("- базы_данных (массив)")
        print("- ds_extracted_at (timestamp)")
        
    except KeyboardInterrupt:
        print("\\nОбработка прервана пользователем")
    except Exception as e:
        print(f"\\nОшибка: {e}")
        print("Проверьте правильность параметров и доступность API")


if __name__ == "__main__":
    main()