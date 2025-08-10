import matplotlib.pyplot as plt
import sys


def plot_company_type_histogram(df):
    """Создает гистограмму распределения вакансий по типам компаний"""
    plt.figure(figsize=(12, 8))
    company_type_counts = df['company_type'].value_counts()
    
    plt.barh(range(len(company_type_counts)), company_type_counts.values)
    plt.yticks(range(len(company_type_counts)), company_type_counts.index)
    plt.xlabel('Количество вакансий')
    plt.ylabel('Тип компании')
    plt.title('Распределение вакансий по типам компаний')
    plt.grid(axis='x', alpha=0.3)
    
    # Добавление значений на столбцы
    for i, v in enumerate(company_type_counts.values):
        plt.text(v + 5, i, str(v), va='center')
    
    plt.tight_layout()
    
    # Safe visualization - only show in interactive mode
    if hasattr(sys, 'ps1') or sys.flags.interactive or 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close()
    
    return company_type_counts