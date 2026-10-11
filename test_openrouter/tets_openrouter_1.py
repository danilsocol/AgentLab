import requests, json

PROXIES = {"http": "http://192.168.1.10:8080", "https": "http://192.168.1.10:8080"}

r = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
      "Authorization": "Bearer sk-or-v1-a2cef299fce0b7f48d2a5a80",
      "Content-Type": "application/json",
    },
    data=json.dumps({
        #"model": "qwen/qwen3.8-27b:free",
        "model": "inclusionai/ling-3.0-flash-vl:free",
        "messages": [{"role": "user", "content": "hi"}]
    }),
    proxies=PROXIES,
    timeout=(10, 60),
)
print("status:", r.status_code)
print("body:", r.text[:500])