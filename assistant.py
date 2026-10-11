import re
import os
import sys
import logging
from datetime import datetime
from llama_cpp import Llama

# ==========================================
# 1. НАСТРОЙКА ЛОГИРОВАНИЯ (Задание 4.1)
# ==========================================
def setup_logger(model_name: str) -> logging.Logger:
    """Создает логгер для каждой модели"""
    os.makedirs("logs", exist_ok=True)
    
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    
    # Файловый обработчик
    log_file = f"logs/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    
    return logger, log_file

# ==========================================
# 2. ОПРЕДЕЛЕНИЕ ИНСТРУМЕНТОВ (ФУНКЦИЙ)
# ==========================================
def get_weather(city: str) -> str:
    """Получает погоду в указанном городе."""
    # Захардкоженные данные (как в задании)
    weather_data = {
        "москва": "В Москве сейчас солнечно, +22°C, ветер слабый.",
        "санкт-петербург": "В Санкт-Петербурге облачно, +18°C, возможен дождь.",
        "челябинск": "В Челябинске ясно, +20°C, ветер умеренный.",
        "новосибирск": "В Новосибирске пасмурно, +15°C, легкий дождь."
    }
    city_lower = city.lower().strip()
    return weather_data.get(city_lower, f"В городе {city} сейчас ясно, +20°C.")

def get_time() -> str:
    """Получает текущее время."""
    return f"Текущее время: {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}"

def get_currency(currency: str = "USD") -> str:
    """Получает курс валют."""
    rates = {
        "USD": "Курс USD: 95.50 рублей за доллар.",
        "EUR": "Курс EUR: 105.20 рублей за евро.",
        "CNY": "Курс CNY: 13.80 рублей за юань."
    }
    return rates.get(currency.upper(), f"Курс {currency.upper()}: 90.00 рублей.")

# Словарь доступных инструментов
TOOLS = {
    "get_weather": get_weather,
    "get_time": get_time,
    "get_currency": get_currency
}

# ==========================================
# 3. СИСТЕМНЫЙ ПРОМПТ (Задание 4.2 - улучшенный для слабых моделей)
# ==========================================
SYSTEM_PROMPT = """Ты — полезный ИИ-ассистент с доступом к инструментам.

ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
1. get_weather(город) — узнать погоду в городе
   Пример: get_weather(Москва)
2. get_time() — узнать текущее время
   Пример: get_time()
3. get_currency(валюта) — узнать курс валюты (USD, EUR, CNY)
   Пример: get_currency(USD)

ПРАВИЛА ВЫЗОВА ИНСТРУМЕНТОВ:
- Если пользователь спрашивает о погоде, времени или курсе валют — ОБЯЗАТЕЛЬНО вызови инструмент
- Формат вызова: $function$имя_функции(аргументы)$function$
- НЕ придумывай ответы сам! Сначала вызови функцию, потом ответь на основе результата
- Если инструмент не нужен — отвечай обычным текстом

ПРИМЕРЫ:
Пользователь: Какая погода в Москве?
Ассистент: $function$get_weather(Москва)$function$

Пользователь: Сколько сейчас времени?
Ассистент: $function$get_time()$function$

Пользователь: Какой курс доллара?
Ассистент: $function$get_currency(USD)$function$

Пользователь: Привет!
Ассистент: Привет! Чем могу помочь?"""

# ==========================================
# 4. ПАРСЕР ВЫЗОВА ФУНКЦИЙ
# ==========================================
def parse_tool_call(text: str):
    """Ищет в тексте вызов функции, прощая мелкие ошибки в формате."""
    # Гибкий regex: ловит $function$get_weather(Москва)$function$ или $get_weather(Москва)$
    pattern = r'(?:\$function\$)?\s*(\w+)\s*\(\s*(.*?)\s*\)\s*(?:\$function\$)?'
    match = re.search(pattern, text)
    
    if match:
        func_name = match.group(1)
        args_str = match.group(2)
        
        if func_name in TOOLS:
            args = [arg.strip().strip('"').strip("'") for arg in args_str.split(',')] if args_str.strip() else []
            return func_name, args
            
    return None, None

# ==========================================
# 5. ОСНОВНОЙ ЦИКЛ АССИСТЕНТА
# ==========================================
def run_assistant(model_path: str, model_name: str):
    print(f"\n{'='*60}")
    print(f"🚀 Запуск ассистента: {model_name}")
    print(f"{'='*60}\n")
    
    # Настройка логгера
    logger, log_file = setup_logger(model_name)
    print(f"📝 Логирование: {log_file}")
    
    print("⏳ Загрузка модели...")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,  # 0 если нет GPU
        verbose=False
    )
    print("✅ Модель загружена!\n")
    
    # История диалога
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    print("💡 Попробуйте спросить:")
    print("   - Какая погода в Москве?")
    print("   - Сколько времени?")
    print("   - Какой курс доллара?")
    print("   - Напиши стих (обычный вопрос)")
    print("Введите 'exit' для выхода.\n")
    
    logger.info(f"=== НАЧАЛО ДИАЛОГА ===")
    logger.info(f"Модель: {model_name}")
    
    while True:
        try:
            user_input = input("👤 Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Выход из чата...")
            logger.info("=== КОНЕЦ ДИАЛОГА (пользователь вышел) ===")
            break
        
        if user_input.lower() in ("exit", "quit", "выход"):
            print("👋 До свидания!")
            logger.info("=== КОНЕЦ ДИАЛОГА (команда выхода) ===")
            break
        
        if not user_input:
            continue
        
        logger.info(f"ПОЛЬЗОВАТЕЛЬ: {user_input}")
        history.append({"role": "user", "content": user_input})
        
        print("🤖 Ассистент думает...", end="\r")
        
        # Первый вызов: поиск инструмента (низкая температура для точности)
        response = llm.create_chat_completion(
            messages=history,
            max_tokens=256,
            temperature=0.1,
            stream=False
        )
        assistant_text = response["choices"][0]["message"]["content"].strip()
        print(" " * 30, end="\r")
        
        logger.info(f"АССИСТЕНТ (первый ответ): {assistant_text}")
        
        func_name, args = parse_tool_call(assistant_text)
        
        if func_name and func_name in TOOLS:
            print(f"\n🔧 [Вызов инструмента]: {func_name}({', '.join(args)})")
            logger.info(f"ИНСТРУМЕНТ: {func_name}({', '.join(args)})")
            
            try:
                # Выполняем функцию
                if func_name == "get_time":
                    tool_result = TOOLS[func_name]()
                else:
                    tool_result = TOOLS[func_name](*args)
                    
                print(f"📥 [Результат]: {tool_result}")
                logger.info(f"РЕЗУЛЬТАТ: {tool_result}")
                
                # Добавляем в историю
                history.append({"role": "assistant", "content": assistant_text})
                history.append({
                    "role": "user", 
                    "content": f"Результат выполнения {func_name}: {tool_result}. Теперь ответь пользователю на основе этих данных."
                })
                
                # Второй вызов: финальный ответ (выше температура для естественности)
                final_response = llm.create_chat_completion(
                    messages=history,
                    max_tokens=256,
                    temperature=0.7,
                    stream=False
                )
                final_text = final_response["choices"][0]["message"]["content"].strip()
                print(f"\n🤖 Ассистент: {final_text}\n")
                logger.info(f"ФИНАЛЬНЫЙ ОТВЕТ: {final_text}")
                
                history.append({"role": "assistant", "content": final_text})
                
            except Exception as e:
                print(f"\n❌ [Ошибка инструмента]: {e}\n")
                logger.info(f"ОШИБКА ИНСТРУМЕНТА: {e}")
                history.append({"role": "assistant", "content": assistant_text})
        else:
            # Обычный ответ без инструмента
            print(f"\n🤖 Ассистент: {assistant_text}\n")
            logger.info(f"ОБЫЧНЫЙ ОТВЕТ: {assistant_text}")
            history.append({"role": "assistant", "content": assistant_text})

# ==========================================
# 6. ГЛАВНАЯ ФУНКЦИЯ
# ==========================================
def main():
    print("="*60)
    print(" АССИСТЕНТ С ВЫЗОВОМ ФУНКЦИЙ (Задание №4)")
    print("="*60)
    
    # Проверяем наличие моделей
    models = {
        "qwen2.5-3b": "./models/qwen2.5-3b-instruct-q4_k_m.gguf",
        "llama-3.2-3b": "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    }
    
    available_models = {k: v for k, v in models.items() if os.path.exists(v)}
    
    if not available_models:
        print("❌ Ошибка: Модели не найдены в папке ./models/")
        sys.exit(1)
    
    print(f"\n📦 Доступные модели:")
    for i, (name, path) in enumerate(available_models.items(), 1):
        print(f"   {i}. {name}")
    
    print(f"\nВыберите модель (1-{len(available_models)}) или 'all' для теста всех:")
    choice = input("> ").strip()
    
    if choice.lower() == "all":
        # Тест всех моделей (Задание 4.1)
        for model_name, model_path in available_models.items():
            run_assistant(model_path, model_name)
    elif choice.isdigit() and 1 <= int(choice) <= len(available_models):
        model_name = list(available_models.keys())[int(choice) - 1]
        model_path = available_models[model_name]
        run_assistant(model_path, model_name)
    else:
        print(" Неверный выбор")
        sys.exit(1)

if __name__ == "__main__":
    main()