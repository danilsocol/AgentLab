"""Учебный цикл вызова функций: подставляется только модель и ответ API."""

import logging
import sys
from types import SimpleNamespace

import pytest

import assistant


class ScriptedModel:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.messages = []

    def tokenize(self, content, **kwargs):
        return content.split()

    def n_ctx(self):
        return 8192

    def create_chat_completion(self, *, messages, **kwargs):
        self.messages.append([dict(item) for item in messages])
        return {"choices": [{"message": {"content": next(self.answers)}, "finish_reason": "stop"}]}


@pytest.mark.parametrize("text,name,args", [
    ('$function$get_weather("Москва")$function$', "get_weather", ["Москва"]),
    ('$function$search_books("I, Robot (Asimov)")$function$', "search_books", ["I, Robot (Asimov)"]),
    ("get_currency(USD)", "get_currency", ["USD"]),
    ("get_currency()", "get_currency", []),
    ("get_time()", "get_time", []),
])
def test_parse_valid_call(text, name, args):
    assert assistant.parse_tool_call(text) == (name, args)


@pytest.mark.parametrize("expression", [
    '__import__("os").system("echo unwanted")',
    'get_weather(__import__("os"))', 'get_weather(city="Москва")',
    'get_weather("Москва", "Казань")', 'get_weather()', 'get_weather(42)',
    'get_time("Москва")', 'unknown("x")',
])
def test_disallowed_call(expression):
    assert assistant.parse_tool_call(f"$function${expression}$function$") == (None, None)


def history():
    return [{"role": "system", "content": assistant.SYSTEM_PROMPT}]


def test_tool_result_reaches_model_and_final_history(monkeypatch):
    monkeypatch.setitem(assistant.TOOLS, "get_currency", lambda code: "17.10.2026: 1 USD = 83.6177 RUB")
    model = ScriptedModel('$function$get_currency("USD")$function$', "Курс: 83.6177 рубля.")
    chat = history()
    answer = assistant.answer_turn(model, chat, "Курс доллара?", logging.getLogger("test"))
    assert "83.6177" in answer
    assert "17.10.2026: 1 USD = 83.6177 RUB" in model.messages[1][-1]["content"]
    assert len(chat) == 3
    assert chat[-1]["content"] == answer


def test_tool_failure_returns_without_model_hallucination(monkeypatch):
    def fail(city):
        raise RuntimeError("Сервис недоступен")

    monkeypatch.setitem(assistant.TOOLS, "get_weather", fail)
    model = ScriptedModel('$function$get_weather("Москва")$function$')
    answer = assistant.answer_turn(model, history(), "Погода?", logging.getLogger("test"))
    assert "Сервис недоступен" in answer
    assert len(model.messages) == 1


def test_duplicate_call_is_not_executed_twice(monkeypatch):
    calls = []
    monkeypatch.setitem(assistant.TOOLS, "get_time", lambda: calls.append(1) or "12:00")
    model = ScriptedModel("get_time()", "get_time()")
    answer = assistant.answer_turn(model, history(), "Время?", logging.getLogger("test"))
    assert len(calls) == 1
    assert answer == "12:00"


def test_multiple_tools_in_one_turn(monkeypatch):
    monkeypatch.setitem(assistant.TOOLS, "get_time", lambda: "12:00")
    monkeypatch.setitem(assistant.TOOLS, "get_currency", lambda code: "1 USD = 83 RUB")
    model = ScriptedModel("get_time()", 'get_currency("USD")', "12:00; доллар 83 рубля")
    assert "83" in assistant.answer_turn(model, history(), "Время и курс?", logging.getLogger("test"))
    assert len(model.messages) == 3


def test_history_keeps_last_three_turns():
    model = ScriptedModel(*["Привет!"] * 10)
    chat = history()
    for i in range(10):
        assistant.answer_turn(model, chat, str(i), logging.getLogger("test"))
    assert len(chat) == 7
    assert chat[1]["content"] == "7"


def test_invented_link_triggers_real_tool_call(monkeypatch):
    result = "The Hobbit, 1937. https://openlibrary.org/works/OL27482W"
    monkeypatch.setitem(assistant.TOOLS, "search_books", lambda query: result)
    model = ScriptedModel("The Hobbit https://example.org/invented",
                          'search_books("The Hobbit")', "The Hobbit, 1937.")
    answer = assistant.answer_turn(model, history(), "Найди The Hobbit", logging.getLogger("test"))
    assert "https://openlibrary.org/works/OL27482W" in answer
    assert "example.org" not in answer


def test_invented_final_link_returns_original_api_data(monkeypatch):
    result = "Python https://ru.wikipedia.org/?curid=2705"
    monkeypatch.setitem(assistant.TOOLS, "search_wikipedia", lambda query: result)
    model = ScriptedModel('search_wikipedia("Python")', "Python https://example.org/invented")
    assert assistant.answer_turn(model, history(), "Найди Python", logging.getLogger("test")) == result


def test_cbr_date_is_preserved(monkeypatch):
    monkeypatch.setitem(assistant.TOOLS, "get_currency", lambda code:
                        "ЦБ РФ, официальный курс на 17.10.2026: 1 USD = 83.6177 RUB.")
    model = ScriptedModel('get_currency("USD")', "1 USD = 83.6177 RUB.")
    answer = assistant.answer_turn(model, history(), "Курс доллара?", logging.getLogger("test"))
    assert "17.10.2026" in answer


def test_currency_answer_without_api_is_retried(monkeypatch):
    calls = []
    monkeypatch.setitem(assistant.TOOLS, "get_currency", lambda code:
                        calls.append(code) or "1 USD = 83.6177 RUB")
    model = ScriptedModel("USD: 68.48", 'get_currency("USD")', "1 USD = 83.6177 RUB")
    answer = assistant.answer_turn(model, history(), "Какой курс доллара?", logging.getLogger("test"))
    assert calls == ["USD"]
    assert "68.48" not in answer


def test_previous_topic_tool_is_not_executed(monkeypatch):
    monkeypatch.setitem(assistant.TOOLS, "get_weather", lambda city: pytest.fail("Wrong tool executed"))
    monkeypatch.setitem(assistant.TOOLS, "search_books", lambda query: "The Hobbit, 1937")
    model = ScriptedModel('get_weather("Москва")', 'search_books("The Hobbit")', "The Hobbit, 1937")
    answer = assistant.answer_turn(model, history(), "Найди книгу The Hobbit", logging.getLogger("test"))
    assert "1937" in answer


def test_missing_city_can_be_clarified():
    model = ScriptedModel("В каком городе нужна погода?")
    answer = assistant.answer_turn(model, history(), "Какая погода?", logging.getLogger("test"))
    assert answer == "В каком городе нужна погода?"


@pytest.mark.parametrize("message,expected", [
    ("Спасибо", "Пожалуйста!"), ("«Спасибо!»", "Пожалуйста!"),
    ("Большое спасибо!", "Пожалуйста!"), ("Пока", "До свидания!"),
    ("До свидания!", "До свидания!"), ("Хорошо", "Хорошо."),
    ("Привет!", "Привет! Чем могу помочь?"),
])
def test_social_turn_after_books_does_not_call_model_or_repeat_data(message, expected):
    chat = history() + [{"role": "user", "content": "Найди книгу The Hobbit"},
                        {"role": "assistant", "content": "The Hobbit — Tolkien, 1937."}]
    model = ScriptedModel()
    answer = assistant.answer_turn(model, chat, message, logging.getLogger("test"))
    assert answer == expected
    assert not model.messages
    assert chat[-1]["content"] == expected


def test_thanks_with_new_request_still_uses_model_and_tool(monkeypatch):
    calls = []
    monkeypatch.setitem(assistant.TOOLS, "search_books", lambda query:
                        calls.append(query) or "Dune — Frank Herbert, 1965.")
    model = ScriptedModel('search_books("Dune")', "Dune — Frank Herbert, 1965.")
    answer = assistant.answer_turn(model, history(), "Спасибо! Найди книгу Dune.", logging.getLogger("test"))
    assert calls == ["Dune"]
    assert "Dune" in answer


@pytest.mark.parametrize("debug", [False, True])
def test_api_trace_is_only_printed_in_debug_mode(monkeypatch, capsys, debug):
    result = "The Hobbit — Tolkien, 1937."
    monkeypatch.setitem(assistant.TOOLS, "search_books", lambda query: result)
    model = ScriptedModel('search_books("The Hobbit")', result)
    assistant.answer_turn(model, history(), "Найди книгу The Hobbit", logging.getLogger("test"), debug=debug)
    output = capsys.readouterr().out
    if debug:
        assert "[Инструмент]" in output
        assert "[Результат]" in output
        assert result in output
    else:
        assert output == ""


def test_placeholder_link_uses_verified_book_result(monkeypatch):
    result = "The Hobbit — J.R.R. Tolkien, 1937. https://openlibrary.org/works/OL27482W"
    monkeypatch.setitem(assistant.TOOLS, "search_books", lambda query: result)
    model = ScriptedModel('search_books("The Hobbit")', "The Hobbit — J.R. Tolkien, 1937, [ссылка]")
    answer = assistant.answer_turn(model, history(), "Найди книгу The Hobbit", logging.getLogger("test"))
    assert answer == result


def test_internal_tool_label_is_not_returned_as_chat_reply():
    model = ScriptedModel("ДАННЫЕ ИНСТРУМЕНТА\nВластелин колец, Герберт Уэллс, 1903")
    answer = assistant.answer_turn(model, history(), "Что дальше?", logging.getLogger("test"))
    assert "ДАННЫЕ ИНСТРУМЕНТА" not in answer
    assert "Герберт Уэллс" not in answer


def test_interactive_books_thanks_goodbye(monkeypatch, capsys):
    result = "The Hobbit — J.R.R. Tolkien, 1937. https://openlibrary.org/works/OL27482W"
    calls = []
    monkeypatch.setitem(assistant.TOOLS, "search_books", lambda query: calls.append(query) or result)
    model = ScriptedModel('search_books("The Hobbit")', result)
    closed = []
    model.close = lambda: closed.append(True)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=lambda **kwargs: model))
    monkeypatch.setattr(assistant, "setup_logger", lambda name:
                        (logging.getLogger("interactive_test"), assistant.PROJECT_DIR / "logs/test-unused.log"))
    inputs = iter(["Найди книгу The Hobbit", "Спасибо", "Пока"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    assistant.run_assistant("unused.gguf", "test")

    dialogue = capsys.readouterr().out.split("Команды:", 1)[1]
    assert dialogue.count("Ассистент: ") == 3
    assert dialogue.count(result) == 1
    assert "Ассистент: Пожалуйста!" in dialogue
    assert "Ассистент: До свидания!" in dialogue
    assert "[Результат]" not in dialogue
    assert calls == ["The Hobbit"]
    assert len(model.messages) == 2
    assert closed == [True]
