"""Продолжение ДЗ 1: четыре API и отдельная память фактов в LCEL."""

import json

from langchain_core.runnables import RunnableLambda
import pytest

import assistant
from assistant_with_memory import AssistantWithMemory, FileFactMemory, Profile
from test_assistant import ScriptedModel


def make_chat(path, answers, extracted=(), events=None):
    updates = iter(extracted)
    model = ScriptedModel(*answers)
    chat = AssistantWithMemory(model, RunnableLambda(lambda prompt: next(updates)),
                               FileFactMemory(path), on_event=None if events is None else events.append)
    return chat, model


def test_facts_are_saved_and_loaded_by_new_agent(tmp_path):
    path = tmp_path / "memory.json"
    events = []
    first, _ = make_chat(path, ["Запомнил, Анна."], ['{"name":"Анна","stack":"Python","city":"Казань"}'], events)
    first.chain.invoke("Меня зовут Анна. Я пишу на Python и живу в Казани.")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["profiles"]["student"]["name"] == "Анна"
    assert "history" not in document and "messages" not in document
    before = path.read_bytes()
    second, model = make_chat(path, ["Вас зовут Анна, вы пишете на Python."])
    answer = second.chain.invoke("Как меня зовут и на чём я пишу?")
    assert "Анна" in answer and "Python" in answer
    assert '"city":"Казань"' in model.messages[0][0]["content"]
    assert path.read_bytes() == before
    assert [event["event"] for event in events] == ["loaded", "saved"]


@pytest.mark.parametrize("name,query,arg,result", [
    ("get_weather", "Погода в Москве?", "Москва", "Москва: +13°C"),
    ("get_currency", "Какой курс доллара?", "USD", "1 USD = 83 RUB"),
    ("search_wikipedia", "Найди Python в Википедии", "Python", "Python https://ru.wikipedia.org/?curid=2705"),
    ("search_books", "Найди книгу The Hobbit", "The Hobbit", "The Hobbit https://openlibrary.org/works/OL27482W"),
])
def test_all_homework_one_tools_remain_available(monkeypatch, tmp_path, name, query, arg, result):
    calls = []
    monkeypatch.setitem(assistant.TOOLS, name, lambda value: calls.append(value) or result)
    chat, _ = make_chat(tmp_path / "memory.json", [f'$function${name}("{arg}")$function$', result])
    assert chat.chain.invoke(query) == result
    assert calls == [arg]
    assert not chat.memory.path.exists()  # API-ответ не превращается в профиль пользователя.


def test_saved_city_reaches_tool_prompt(monkeypatch, tmp_path):
    path = tmp_path / "memory.json"
    FileFactMemory(path).save(Profile(city="Казань"))
    calls = []
    monkeypatch.setitem(assistant.TOOLS, "get_weather", lambda city: calls.append(city) or "Казань: +10°C")
    chat, model = make_chat(path, ['get_weather("Казань")', "Казань: +10°C"])
    assert "+10°C" in chat.chain.invoke("Какая погода в моём городе?")
    assert calls == ["Казань"]
    assert '"city":"Казань"' in model.messages[0][0]["content"]


def test_courtesy_and_clear_do_not_erase_facts(tmp_path):
    path = tmp_path / "memory.json"
    FileFactMemory(path).save(Profile(name="Анна"))
    chat, model = make_chat(path, [])
    assert chat.chain.invoke("Спасибо") == "Пожалуйста!"
    chat.clear_dialogue()
    assert len(chat.history) == 1
    assert FileFactMemory(path).load().name == "Анна"
    assert not model.messages


def test_mixed_message_saves_only_statement(tmp_path):
    received = []
    json_llm = RunnableLambda(lambda prompt: received.append(prompt.to_messages()[-1].content) or '{"name":"Анна"}')
    chat = AssistantWithMemory(ScriptedModel("Вас зовут Анна."), json_llm,
                               FileFactMemory(tmp_path / "memory.json"))
    chat.chain.invoke("Меня зовут Анна. Как меня зовут?")
    assert "Меня зовут Анна." in received[0]
    assert "Как меня зовут?" not in received[0]


def test_bad_extraction_keeps_existing_file(tmp_path):
    path = tmp_path / "memory.json"
    FileFactMemory(path).save(Profile(name="Анна"))
    before = path.read_bytes()
    chat, _ = make_chat(path, [], ['{"name":123}'])
    with pytest.raises(ValueError):
        chat.chain.invoke("Меня зовут Борис")
    assert path.read_bytes() == before
