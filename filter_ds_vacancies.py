import pandas as pd
import re
from typing import List


class DataScienceVacancyFilter:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        
        # Ключевые слова для Data Science и ML (включая различные написания)
        self.ds_ml_keywords = [
            # Data Science
            "data scientist", "data science", "дата сайентист", "дата саентист", 
            "специалист по данным", "исследователь данных", "аналитик данных",
            
            # Machine Learning
            "machine learning", "ml engineer", "ml инженер", "машинное обучение",
            "ml-инженер", "ml специалист", "ml разработчик", "мл инженер",
            
            # AI
            "ai engineer", "ai инженер", "искусственный интеллект", "ии инженер",
            "artificial intelligence", "нейросети", "deep learning",
            
            # Специализированные области
            "computer vision", "cv engineer", "nlp engineer", "nlp инженер",
            "llm engineer", "llm инженер", "speech", "nlp", "cv инженер",
            
            # MLOps
            "mlops", "ml ops", "ml platform", "ml инфраструктура",
            
            # Исследования
            "research", "researcher", "исследователь", "r&d", "рнд"
        ]
        
        # Исключающие слова (вакансии, которые НЕ относятся к DS/ML)
        self.exclude_keywords = [
            # Разработка (не ML)
            "frontend", "front-end", "backend", "fullstack", "full-stack",
            "react", "vue", "angular", "javascript", "typescript", "node.js",
            "php", "java developer", ".net", "c#", "c++", "golang", "scala",
            "android", "ios", "mobile", "flutter", "react native",
            
            # QA и тестирование
            "qa engineer", "тестировщик", "qa automation", "тестирование",
            
            # DevOps (не MLOps)
            "devops", "системный администратор", "sre", "network",
            
            # Менеджмент и продажи
            "project manager", "product manager", "менеджер по продажам",
            "sales manager", "account manager", "recruiter", "рекрутер",
            "hr", "бизнес-аналитик", "финансовый аналитик",
            
            # Дизайн
            "ui/ux", "дизайнер", "designer",
            
            # Прочие неrelевантные
            "водитель", "курьер", "грузчик", "офис-менеджер", "бухгалтер",
            "юрист", "кассир", "продавец", "повар", "электрик", "слесарь"
        ]
    
    def is_relevant_vacancy(self, vacancy_name: str) -> bool:
        """Проверяет, относится ли вакансия к Data Science/ML"""
        name_lower = vacancy_name.lower()
        
        # Проверяем исключающие слова
        for exclude_word in self.exclude_keywords:
            if exclude_word.lower() in name_lower:
                # Исключение: если есть исключающее слово, но также есть явные DS/ML слова
                has_ds_keywords = any(keyword.lower() in name_lower for keyword in self.ds_ml_keywords[:15])  # Основные DS/ML слова
                if not has_ds_keywords:
                    return False
        
        # Проверяем наличие ключевых слов DS/ML
        for keyword in self.ds_ml_keywords:
            if keyword.lower() in name_lower:
                return True
        
        return False
    
    def filter_vacancies(self) -> None:
        """Фильтрует вакансии и сохраняет только релевантные"""
        try:
            # Читаем исходный файл
            df = pd.read_csv(self.input_file, encoding='utf-8')
            print(f"Исходное количество вакансий: {len(df)}")
            
            # Фильтруем релевантные вакансии
            filtered_df = df[df['name'].apply(self.is_relevant_vacancy)].copy()
            print(f"Количество релевантных вакансий: {len(filtered_df)}")
            print(f"Удалено вакансий: {len(df) - len(filtered_df)}")
            
            # Сохраняем отфильтрованный файл
            filtered_df.to_csv(self.output_file, index=False, encoding='utf-8')
            print(f"Отфильтрованные вакансии сохранены в: {self.output_file}")
            
            # Показываем примеры удаленных вакансий
            removed_df = df[~df['name'].apply(self.is_relevant_vacancy)]
            if len(removed_df) > 0:
                print("\nПримеры удаленных вакансий:")
                for i, row in removed_df.head(10).iterrows():
                    print(f"- {row['name']}")
            
            # Показываем примеры оставшихся вакансий
            print("\nПримеры оставшихся вакансий:")
            for i, row in filtered_df.head(10).iterrows():
                print(f"+ {row['name']}")
                
        except Exception as e:
            print(f"Ошибка при фильтрации: {e}")
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Фильтрует вакансии, оставляя только Data Science и ML')
    parser.add_argument('--input', default='ds_vacancies.csv', help='Исходный CSV файл')
    parser.add_argument('--output', default='ds_vacancies_filtered.csv', help='Отфильтрованный CSV файл')
    
    args = parser.parse_args()
    
    # Создаем фильтр и запускаем
    filter_tool = DataScienceVacancyFilter(args.input, args.output)
    filter_tool.filter_vacancies()


if __name__ == "__main__":
    main()