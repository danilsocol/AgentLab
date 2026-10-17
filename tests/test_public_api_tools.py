"""Проверки формата API и ошибок без сетевых запросов."""

from types import SimpleNamespace

import pytest
import requests

import public_api_tools as api


def response(monkeypatch, *, data=None, text="", status=200):
    fake = SimpleNamespace(status_code=status, text=text, content=text.encode("utf-8"),
                           json=lambda: data)
    monkeypatch.setattr(api.requests, "get", lambda *args, **kwargs: fake)


def test_weather_uses_encoded_city_and_real_response(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, text="Москва: ☁️ +13°C\n")

    monkeypatch.setattr(api.requests, "get", get)
    assert "+13°C" in api.get_weather("Москва")
    assert "%D0%9C" in calls[0][0]
    assert calls[0][1]["params"]["format"] == "3"
    assert calls[0][1]["timeout"] == (8, 20)


@pytest.mark.parametrize("text", ["<html>Service unavailable</html>", "Unknown location", ""])
def test_invalid_weather_is_not_replaced_with_fake_data(monkeypatch, text):
    response(monkeypatch, text=text)
    with pytest.raises(api.PublicAPIError):
        api.get_weather("Москва")


@pytest.mark.parametrize("error", [requests.Timeout(), requests.ConnectionError()])
def test_network_failure_is_readable(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(api.requests, "get", fail)
    with pytest.raises(api.PublicAPIError):
        api.get_weather("Москва")


def test_http_error(monkeypatch):
    response(monkeypatch, status=429)
    with pytest.raises(api.PublicAPIError, match="HTTP 429"):
        api.search_books("Hobbit")


def test_cbr_nominal_and_date(monkeypatch):
    response(monkeypatch, text='<ValCurs Date="17.10.2026"><Valute>'
             '<CharCode>JPY</CharCode><Nominal>100</Nominal><Value>54,3210</Value>'
             '<Name>Иена</Name></Valute></ValCurs>')
    result = api.get_currency("jpy")
    assert "100 JPY" in result
    assert "За 1 JPY: 0.543210 RUB" in result
    assert "17.10.2026" in result


@pytest.mark.parametrize("xml", ["not XML", '<ValCurs/>',
    '<ValCurs Date="17.10.2026"><Valute><CharCode>USD</CharCode>'
    '<Nominal>0</Nominal><Value>83,6</Value></Valute></ValCurs>'])
def test_invalid_cbr_response(monkeypatch, xml):
    response(monkeypatch, text=xml)
    with pytest.raises(api.PublicAPIError):
        api.get_currency("USD")


def test_unknown_currency_has_no_invented_rate(monkeypatch):
    response(monkeypatch, text='<ValCurs Date="17.10.2026"/>')
    assert "не публикует курс XYZ" in api.get_currency("XYZ")


def test_wikipedia_removes_markup_and_keeps_link(monkeypatch):
    response(monkeypatch, data={"query": {"search": [
        {"title": "Python", "pageid": 2705, "snippet": '<span>Язык</span> &amp; код'}]}})
    result = api.search_wikipedia("Python")
    assert "Язык & код" in result
    assert "<span>" not in result
    assert "https://ru.wikipedia.org/?curid=2705" in result


def test_books_missing_optional_metadata(monkeypatch):
    response(monkeypatch, data={"docs": [{"title": "Book", "key": "/works/OL123W"}]})
    result = api.search_books("Book")
    assert "автор не указан" in result
    assert "https://openlibrary.org/works/OL123W" in result


@pytest.mark.parametrize("function,data,expected", [
    (api.search_wikipedia, {"query": {"search": []}}, "статьи не найдены"),
    (api.search_books, {"docs": []}, "книги не найдены"),
])
def test_empty_search(monkeypatch, function, data, expected):
    response(monkeypatch, data=data)
    assert expected in function("unknown")


@pytest.mark.parametrize("function", [api.search_wikipedia, api.search_books])
def test_unexpected_schema(monkeypatch, function):
    response(monkeypatch, data={"unexpected": []})
    with pytest.raises(api.PublicAPIError):
        function("Python")


@pytest.mark.parametrize("value", ["", "x\ny", "a" * 201, None])
def test_invalid_input_is_rejected_before_request(value):
    with pytest.raises(ValueError):
        api.search_books(value)
