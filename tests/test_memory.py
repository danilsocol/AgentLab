"""Общая файловая память: перезапуск, исправления и защита сохранённых данных."""

import json
import subprocess
import sys

import pytest

from homework_2.chat_memory import FileFactMemory, Profile


def test_restart_in_separate_process(tmp_path):
    path = tmp_path / "memory.json"
    FileFactMemory(path, "demo").save(Profile(name="Анна", stack="Python"))
    code = ("import sys; from homework_2.chat_memory import FileFactMemory; "
            "print(FileFactMemory(sys.argv[1], 'demo').load().model_dump_json(exclude_none=True))")
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", code, str(path)],
                            capture_output=True, encoding="utf-8", check=True)
    assert json.loads(result.stdout) == {"name": "Анна", "stack": "Python"}


def test_corrections_no_duplicates_and_separate_profiles(tmp_path):
    path = tmp_path / "memory.json"
    anna = FileFactMemory(path, "anna")
    anna.save(Profile(name="Анна", city="Казань"))
    FileFactMemory(path, "boris").save(Profile(name="Борис"))
    anna.save(Profile(city="Екатеринбург"))
    before = path.read_bytes()
    assert anna.save(Profile(city="Екатеринбург")) == {}
    assert path.read_bytes() == before
    assert anna.load().city == "Екатеринбург"
    assert anna.load().name == "Анна"
    assert FileFactMemory(path, "boris").load().city is None


def test_none_does_not_erase_existing_fact(tmp_path):
    memory = FileFactMemory(tmp_path / "memory.json")
    memory.save(Profile(name="Анна"))
    memory.save(Profile(name=None, stack="Python"))
    assert memory.load().name == "Анна"
    assert memory.load().stack == "Python"


@pytest.mark.parametrize("content", ["not json", '{"version":2,"profiles":{}}',
                                     '{"version":1,"profiles":{"demo":{"name":12}}}'])
def test_corrupt_file_is_not_overwritten(tmp_path, content):
    path = tmp_path / "memory.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="файл не перезаписан"):
        FileFactMemory(path)
    assert path.read_text(encoding="utf-8") == content


def test_missing_file_starts_with_empty_profile(tmp_path):
    path = tmp_path / "nested/memory.json"
    memory = FileFactMemory(path)
    assert memory.load() == Profile()
    assert not path.exists()
    memory.save(Profile(name="Анна"))
    assert path.is_file()
