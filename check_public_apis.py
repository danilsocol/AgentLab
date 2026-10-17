"""Прямой запрос четырёх публичных API: без LLM, ключей и регистрации."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from public_api_tools import get_currency, get_weather, search_books, search_wikipedia


def main():
    checks = [("wttr.in", get_weather, "Москва"), ("ЦБ РФ", get_currency, "USD"),
              ("Wikipedia", search_wikipedia, "Python"), ("Open Library", search_books, "The Hobbit")]

    def run(service, function, argument):
        start = time.monotonic()
        try:
            result = function(argument)
            return {"service": service, "ok": True, "result": result,
                    "seconds": round(time.monotonic() - start, 2)}
        except (RuntimeError, ValueError) as exc:
            return {"service": service, "ok": False, "error": str(exc),
                    "seconds": round(time.monotonic() - start, 2)}

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(run, *check) for check in checks]):
            item = future.result()
            results.append(item)
            print(f"\n{'OK' if item['ok'] else 'FAIL'}: {item['service']} ({item['seconds']} с)\n"
                  f"{item.get('result', item.get('error'))}", flush=True)
    path = Path(__file__).resolve().parent / "logs" / "public_apis_check.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"checked_at_utc": datetime.now(timezone.utc).isoformat(),
                                "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nОтчёт: {path}")
    return 0 if all(item["ok"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
