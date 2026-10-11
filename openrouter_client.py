import os
import requests
from dotenv import load_dotenv

# Загружаем ключ из .env
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Актуальные бесплатные модели (на текущий момент)
FREE_MODELS = [
    "apodex/apodex-1.1-mini:free",
]

def test_openrouter():
    if not API_KEY or API_KEY == "ваш_ключ_здесь":
        print("❌ Ошибка: Ключ не найден в файле .env")
        return False

    print("🚀 Запуск теста OpenRouter...\n")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",  # Обязательно для бесплатных моделей!
        "X-Title": "AgentLab Homework"
    }

    for model in FREE_MODELS:
        print(f"🔄 Пробуем модель: {model}")
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Привет! Напиши короткое приветствие на русском языке."}
            ]
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )

            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                print(f"✅ УСПЕХ! Ответ модели:\n{text}\n")
                print("=" * 50)
                print("🎉 OpenRouter полностью настроен и работает!")
                return True
            else:
                print(f"❌ Ошибка {response.status_code}: {response.text}\n")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Сетевая ошибка: {e}\n")

    print("⚠️ Не удалось подключиться ни к одной бесплатной модели.")
    print("💡 Совет: Проверьте баланс аккаунта на openrouter.ai/credits")
    return False

if __name__ == "__main__":
    test_openrouter()