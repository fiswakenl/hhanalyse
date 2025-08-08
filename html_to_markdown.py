#!/usr/bin/env python3
"""
Утилиты для конвертации HTML в Markdown для описаний вакансий
"""
import re
from typing import Optional
try:
    import html2text
except ImportError:
    html2text = None

try:
    from markdownify import markdownify
except ImportError:
    markdownify = None

from bs4 import BeautifulSoup


def clean_html(html_text: str) -> str:
    """Очищает и нормализует HTML перед конвертацией"""
    if not html_text:
        return ""
    
    # Убираем лишние пробелы и переносы
    html_text = re.sub(r'\s+', ' ', html_text)
    html_text = html_text.strip()
    
    # Заменяем некоторые специфичные для HH.ru конструкции
    html_text = html_text.replace('<highlighttext>', '**')
    html_text = html_text.replace('</highlighttext>', '**')
    
    return html_text


def html_to_markdown_html2text(html_text: str) -> str:
    """Конвертирует HTML в Markdown используя html2text"""
    if not html_text or not html2text:
        return html_text or ""
    
    # Настраиваем конвертер
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.body_width = 0  # Отключаем перенос строк
    h.ul_item_mark = '-'  # Используем дефис для списков
    h.emphasis_mark = '*'  # Используем * для выделения
    h.strong_mark = '**'  # Используем ** для жирного
    
    # Конвертируем
    markdown_text = h.handle(clean_html(html_text))
    
    # Постобработка
    # Убираем лишние переносы строк
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    # Убираем пробелы в конце строк
    markdown_text = re.sub(r' +\n', '\n', markdown_text)
    
    return markdown_text.strip()


def html_to_markdown_markdownify(html_text: str) -> str:
    """Конвертирует HTML в Markdown используя markdownify"""
    if not html_text or not markdownify:
        return html_text or ""
    
    # Конвертируем
    markdown_text = markdownify(
        clean_html(html_text),
        heading_style="ATX",  # Используем # для заголовков
        bullets="-",  # Используем дефис для списков
        strong_em="**",  # Используем ** для жирного
        strip=['span'],  # Удаляем span теги
    )
    
    # Постобработка
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    markdown_text = re.sub(r' +\n', '\n', markdown_text)
    
    return markdown_text.strip()


def html_to_markdown_simple(html_text: str) -> str:
    """Простая конвертация HTML в Markdown используя BeautifulSoup"""
    if not html_text:
        return ""
    
    soup = BeautifulSoup(clean_html(html_text), 'html.parser')
    
    # Заменяем теги на Markdown эквиваленты
    for strong in soup.find_all(['strong', 'b']):
        strong.replace_with(f"**{strong.get_text()}**")
    
    for em in soup.find_all(['em', 'i']):
        em.replace_with(f"*{em.get_text()}*")
    
    for code in soup.find_all('code'):
        code.replace_with(f"`{code.get_text()}`")
    
    # Заголовки
    for i, tag in enumerate(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], 1):
        for header in soup.find_all(tag):
            header.replace_with(f"{'#' * i} {header.get_text()}\n\n")
    
    # Списки
    for ul in soup.find_all('ul'):
        items = []
        for li in ul.find_all('li'):
            items.append(f"- {li.get_text().strip()}")
        ul.replace_with('\n'.join(items) + '\n\n')
    
    for ol in soup.find_all('ol'):
        items = []
        for i, li in enumerate(ol.find_all('li'), 1):
            items.append(f"{i}. {li.get_text().strip()}")
        ol.replace_with('\n'.join(items) + '\n\n')
    
    # Параграфы
    for p in soup.find_all('p'):
        p.replace_with(f"{p.get_text().strip()}\n\n")
    
    # Переносы строк
    for br in soup.find_all('br'):
        br.replace_with('\n')
    
    # Получаем текст
    markdown_text = soup.get_text()
    
    # Постобработка
    markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
    markdown_text = re.sub(r' +\n', '\n', markdown_text)
    
    return markdown_text.strip()


def convert_html_to_markdown(html_text: str, method: str = "auto") -> str:
    """
    Конвертирует HTML в Markdown
    
    Args:
        html_text: HTML текст для конвертации
        method: Метод конвертации ("html2text", "markdownify", "simple", "auto")
    
    Returns:
        Текст в формате Markdown
    """
    if not html_text:
        return ""
    
    if method == "auto":
        # Пробуем методы в порядке предпочтения
        if html2text:
            return html_to_markdown_html2text(html_text)
        elif markdownify:
            return html_to_markdown_markdownify(html_text)
        else:
            return html_to_markdown_simple(html_text)
    elif method == "html2text":
        return html_to_markdown_html2text(html_text)
    elif method == "markdownify":
        return html_to_markdown_markdownify(html_text)
    elif method == "simple":
        return html_to_markdown_simple(html_text)
    else:
        raise ValueError(f"Неизвестный метод: {method}")


def test_conversion():
    """Тестирует конвертацию на примере"""
    sample_html = """
    <p><strong>Компания ООО "ЦифроТех"</strong></p>
    <p><strong>Стек:</strong></p>
    <ul>
        <li>Angular 12</li>
        <li>TypeScript 4</li>
        <li>GraphQL, REST, WebSockets</li>
    </ul>
    <p><strong>Что ждем от кандидата:</strong></p>
    <ul>
        <li>Опыт разработки Front-End от 3-х лет</li>
        <li>Знание <em>JavaScript ES6+</em> и <strong>TypeScript 4+</strong></li>
    </ul>
    """
    
    methods = ["simple", "html2text", "markdownify"]
    
    for method in methods:
        try:
            result = convert_html_to_markdown(sample_html, method)
            print(f"=== Метод: {method} ===")
            print(result)
            print()
        except Exception as e:
            print(f"Ошибка с методом {method}: {e}")


if __name__ == "__main__":
    test_conversion()