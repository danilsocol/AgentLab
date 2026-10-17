"""Инструменты ассистента с реальными публичными API, без ключей."""

from datetime import datetime
from decimal import Decimal, InvalidOperation
from html import unescape
import re
from urllib.parse import quote
import xml.etree.ElementTree as ET

import requests


USER_AGENT = "AgentLabEducationalAssistant/1.0 (local educational project; Python requests)"


class PublicAPIError(RuntimeError):
    """Ошибка внешнего сервиса, которую можно показать пользователю."""


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Укажите {label}.")
    value = value.strip()
    if len(value) > 200 or any(char in value for char in "\r\n\x00"):
        raise ValueError(f"{label}: допустим текст длиной до 200 символов без перевода строки.")
    return value


def _get(service: str, url: str, params=None):
    try:
        response = requests.get(
            url, params=params, headers={"User-Agent": USER_AGENT}, timeout=(8, 20),
        )
    except requests.Timeout:
        raise PublicAPIError(f"{service}: превышено время ожидания. Попробуйте позже.") from None
    except requests.RequestException:
        raise PublicAPIError(f"{service}: не удалось получить данные. Проверьте соединение.") from None
    if response.status_code != 200:
        raise PublicAPIError(f"{service}: ошибка HTTP {response.status_code}. Попробуйте позже.")
    return response


def _json(service: str, url: str, params: dict) -> dict:
    try:
        data = _get(service, url, params).json()
    except ValueError:
        raise PublicAPIError(f"{service}: получен некорректный JSON.") from None
    if not isinstance(data, dict) or data.get("error"):
        raise PublicAPIError(f"{service}: неожиданный ответ сервиса.")
    return data


def get_weather(city: str) -> str:
    """Текущая погода wttr.in в проверенном коротком текстовом формате."""
    city = _text(city, "город")
    response = _get("wttr.in", "https://wttr.in/" + quote(city, safe=""),
                    {"format": "3", "lang": "ru", "m": ""})
    weather = " ".join(response.text.split())
    if len(weather) > 500 or not re.search(r"[+−-]?\d+(?:\.\d+)?\s*°C", weather):
        raise PublicAPIError("wttr.in: погода для этого города не получена. Уточните город.")
    return f"Погода для {city} (wttr.in, получено сейчас): {weather}"


def get_currency(currency: str = "USD") -> str:
    """Официальный курс ЦБ РФ с датой и правильным учётом номинала валюты."""
    currency = _text(currency, "трёхбуквенный код валюты").upper()
    if not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError("Используйте трёхбуквенный код валюты: USD, EUR, CNY, JPY и т. п.")
    if currency == "RUB":
        return "1 RUB = 1 RUB: рубль является базовой валютой расчёта."
    response = _get("ЦБ РФ", "https://www.cbr.ru/scripts/XML_daily.asp")
    try:
        root = ET.fromstring(response.content)
        rates_date = root.attrib["Date"]
        datetime.strptime(rates_date, "%d.%m.%Y")
        if root.tag != "ValCurs":
            raise ValueError("Неожиданный корневой элемент")
        for item in root.findall("Valute"):
            if item.findtext("CharCode") != currency:
                continue
            nominal = int(item.findtext("Nominal"))
            value = Decimal(item.findtext("Value").replace(",", "."))
            if nominal <= 0 or not value.is_finite() or value <= 0:
                raise ValueError("Некорректный курс")
            name = item.findtext("Name") or currency
            return (f"ЦБ РФ, официальный курс на {rates_date}: {nominal} {currency} "
                    f"({name}) = {value} RUB. За 1 {currency}: {value / nominal:.6f} RUB. "
                    "Это официальный курс, а не курс покупки или продажи в банке. "
                    "Источник: https://www.cbr.ru/scripts/XML_daily.asp")
    except (ET.ParseError, KeyError, TypeError, ValueError, AttributeError, InvalidOperation):
        raise PublicAPIError("ЦБ РФ: не удалось прочитать официальный курс из ответа.") from None
    return f"ЦБ РФ не публикует курс {currency} в полученной таблице на {rates_date}."


def search_wikipedia(query: str) -> str:
    """Найти до трёх статей русской Wikipedia и вернуть аннотации со ссылками."""
    query = _text(query, "тему статьи")
    data = _json("Wikipedia", "https://ru.wikipedia.org/w/api.php", {
        "action": "query", "list": "search", "srsearch": query,
        "srnamespace": 0, "srlimit": 3, "format": "json", "utf8": 1,
    })
    try:
        rows = data["query"]["search"]
        if not isinstance(rows, list):
            raise ValueError("Ожидался список")
        lines = []
        for row in rows:
            snippet = unescape(re.sub(r"<[^>]+>", "", row.get("snippet", "")))
            snippet = " ".join(snippet.split())[:320]
            title, pageid = row["title"], int(row["pageid"])
            if not isinstance(title, str) or pageid <= 0:
                raise ValueError("Некорректная статья")
            lines.append(f"{len(lines) + 1}. {title}: {snippet}\n"
                         f"https://ru.wikipedia.org/?curid={pageid}")
    except (KeyError, TypeError, ValueError, AttributeError):
        raise PublicAPIError("Wikipedia: не удалось прочитать результаты поиска.") from None
    return "Wikipedia, результаты поиска:\n" + "\n".join(lines) if lines else "Wikipedia: статьи не найдены."


def search_books(query: str) -> str:
    """Найти книги Open Library по названию, автору или общей поисковой фразе."""
    query = _text(query, "название книги или автора")
    data = _json("Open Library", "https://openlibrary.org/search.json", {
        "q": query, "limit": 3, "fields": "key,title,author_name,first_publish_year",
    })
    try:
        rows = data["docs"]
        if not isinstance(rows, list):
            raise ValueError("Ожидался список")
        lines = []
        for row in rows:
            title, key = row["title"], row["key"]
            authors = row.get("author_name") or []
            if (not isinstance(title, str) or not isinstance(authors, list)
                    or not isinstance(key, str) or not re.fullmatch(r"/works/OL\d+W", key)):
                raise ValueError("Некорректная книга")
            author_text = ", ".join(authors[:3]) if authors else "автор не указан"
            year = row.get("first_publish_year") or "не указан"
            lines.append(f"{len(lines) + 1}. {title[:160]} — {author_text[:180]}; "
                         f"первое издание: {year}.\nhttps://openlibrary.org{key}")
    except (KeyError, TypeError, ValueError, AttributeError):
        raise PublicAPIError("Open Library: не удалось прочитать результаты поиска.") from None
    return "Open Library, найденные книги:\n" + "\n".join(lines) if lines else "Open Library: книги не найдены."
