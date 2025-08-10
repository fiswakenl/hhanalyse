#!/usr/bin/env python3
"""
LLM-based technology extraction script for job vacancies.
Usage: python extract_technologies_llm.py --start 0 --end 100
"""

import argparse
import os
from dotenv import load_dotenv
from technology_extractor import TechnologyExtractor


def main():
    """Основная точка входа в скрипт."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Извлечение технологических стеков из вакансий с помощью LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python extract_technologies_llm.py --start 0 --end 10
  python extract_technologies_llm.py --start 100 --end 200 --model meta-llama/llama-3.1-8b-instruct
  python extract_technologies_llm.py --start 0 --end 50 --output data/test_results.parquet

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
    parser.add_argument("--input", type=str, default="data/vacancies.parquet",
                       help="Входной parquet файл (по умолчанию: data/vacancies.parquet)")
    parser.add_argument("--output", type=str, default="data/vacancies.parquet",
                       help="Выходной parquet файл (по умолчанию: data/vacancies.parquet)")
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
    
    # Получаем API ключ из переменных окружения
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Ошибка: не найден OPENROUTER_API_KEY в переменных окружения")
        print("Создайте файл .env с содержимым:")
        print("OPENROUTER_API_KEY=your_api_key_here")
        return
    
    # Инициализируем экстрактор
    print(f"Запуск извлечения технологий...")
    print(f"Модель: {args.model}")
    print(f"Диапазон: {args.start}-{args.end} ({args.end - args.start} записей)")
    print(f"Размер батча: {args.batch_size}")
    print(f"Входной файл: {args.input}")
    print(f"Выходной файл: {args.output}")
    print()
    
    try:
        extractor = TechnologyExtractor(
            api_key=api_key, 
            model=args.model, 
            batch_size=args.batch_size
        )
        
        # Обрабатываем вакансии
        extractor.process_range(args.start, args.end, args.input, args.output)
        
        print("\\nОбработка завершена успешно!")
        
    except KeyboardInterrupt:
        print("\\nОбработка прервана пользователем")
    except Exception as e:
        print(f"\\nОшибка: {e}")
        print("Проверьте правильность параметров и доступность API")


if __name__ == "__main__":
    main()