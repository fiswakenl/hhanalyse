#!/usr/bin/env python3
"""
Утилита для пакетной обработки вакансий с возможностью выбора диапазонов
"""
import argparse
import subprocess
import sys
import pandas as pd
from pathlib import Path


def get_vacancy_count(csv_file: str) -> int:
    """Получает общее количество вакансий в CSV файле"""
    try:
        df = pd.read_csv(csv_file)
        return len(df)
    except Exception as e:
        print(f"Ошибка чтения CSV: {e}")
        return 0


def show_csv_preview(csv_file: str, start: int = 0, count: int = 10):
    """Показывает превью CSV файла"""
    try:
        df = pd.read_csv(csv_file)
        print(f"=== Превью CSV файла (записи {start}-{start+count-1} из {len(df)}) ===")
        print(f"Колонки: {list(df.columns)}")
        print()
        
        # Показываем выбранные строки
        preview_df = df.iloc[start:start+count]
        for idx, row in preview_df.iterrows():
            vacancy_id = row.iloc[0]  # Первая колонка - ID
            name = row.iloc[1] if len(row) > 1 else "N/A"  # Вторая - название
            print(f"{idx:3d}: {vacancy_id} - {name}")
        
        print(f"\nВсего записей: {len(df)}")
        
    except Exception as e:
        print(f"Ошибка чтения CSV: {e}")


def run_fetcher(csv_file: str, start: int, end: int, **kwargs):
    """Запускает fetch_vacancies.py с указанными параметрами"""
    cmd = [
        "uv", "run", "fetch_vacancies.py",
        csv_file,
        "--start", str(start),
        "--end", str(end)
    ]
    
    # Добавляем дополнительные параметры
    if kwargs.get('db'):
        cmd.extend(["--db", kwargs['db']])
    if kwargs.get('delay'):
        cmd.extend(["--delay", str(kwargs['delay'])])
    if kwargs.get('max'):
        cmd.extend(["--max", str(kwargs['max'])])
    if kwargs.get('no_resume'):
        cmd.append("--no-resume")
    if kwargs.get('log_level'):
        cmd.extend(["--log-level", kwargs['log_level']])
    
    print(f"Выполняю: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка выполнения: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Пакетная обработка вакансий с выбором диапазонов')
    parser.add_argument('csv_file', help='CSV файл с ID вакансий')
    
    # Режимы работы
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Команда preview - показать превью CSV
    preview_parser = subparsers.add_parser('preview', help='Показать превью CSV файла')
    preview_parser.add_argument('--start', type=int, default=0, help='Начальный индекс')
    preview_parser.add_argument('--count', type=int, default=10, help='Количество записей для показа')
    
    # Команда range - обработать диапазон
    range_parser = subparsers.add_parser('range', help='Обработать диапазон записей')
    range_parser.add_argument('start', type=int, help='Начальный индекс (включительно)')
    range_parser.add_argument('end', type=int, help='Конечный индекс (не включительно)')
    range_parser.add_argument('--db', default='vacancies.db', help='Путь к базе данных')
    range_parser.add_argument('--delay', type=float, default=1.0, help='Задержка между запросами')
    range_parser.add_argument('--no-resume', action='store_true', help='Не пропускать уже обработанные')
    range_parser.add_argument('--log-level', default='INFO', help='Уровень логирования')
    
    # Команда batch - разбить на батчи и обработать
    batch_parser = subparsers.add_parser('batch', help='Обработать файл батчами')
    batch_parser.add_argument('--batch-size', type=int, default=50, help='Размер батча')
    batch_parser.add_argument('--start', type=int, default=0, help='Начальный индекс')
    batch_parser.add_argument('--end', type=int, help='Конечный индекс (если не указан - до конца)')
    batch_parser.add_argument('--db', default='vacancies.db', help='Путь к базе данных')
    batch_parser.add_argument('--delay', type=float, default=1.0, help='Задержка между запросами')
    batch_parser.add_argument('--no-resume', action='store_true', help='Не пропускать уже обработанные')
    batch_parser.add_argument('--log-level', default='INFO', help='Уровень логирования')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"Файл {args.csv_file} не найден")
        return 1
    
    total_count = get_vacancy_count(args.csv_file)
    if total_count == 0:
        print("Не удалось прочитать CSV файл")
        return 1
    
    if args.command == 'preview':
        show_csv_preview(args.csv_file, args.start, args.count)
        
    elif args.command == 'range':
        if args.start < 0 or args.start >= total_count:
            print(f"Начальный индекс {args.start} вне диапазона [0, {total_count-1}]")
            return 1
        if args.end <= args.start or args.end > total_count:
            print(f"Конечный индекс {args.end} должен быть > {args.start} и <= {total_count}")
            return 1
        
        print(f"Обрабатываю записи {args.start}-{args.end-1} из {total_count}")
        
        success = run_fetcher(
            args.csv_file, args.start, args.end,
            db=args.db, delay=args.delay, no_resume=args.no_resume, 
            log_level=args.log_level
        )
        
        return 0 if success else 1
        
    elif args.command == 'batch':
        start_idx = args.start
        end_idx = args.end or total_count
        batch_size = args.batch_size
        
        print(f"Обрабатываю записи {start_idx}-{end_idx-1} батчами по {batch_size}")
        
        current_idx = start_idx
        batch_num = 1
        
        while current_idx < end_idx:
            batch_end = min(current_idx + batch_size, end_idx)
            
            print(f"\n=== Батч {batch_num}: записи {current_idx}-{batch_end-1} ===")
            
            success = run_fetcher(
                args.csv_file, current_idx, batch_end,
                db=args.db, delay=args.delay, no_resume=args.no_resume,
                log_level=args.log_level
            )
            
            if not success:
                print(f"Ошибка в батче {batch_num}. Остановка.")
                return 1
            
            current_idx = batch_end
            batch_num += 1
        
        print(f"\nВсе батчи обработаны успешно!")
        
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())