from llama_cpp import Llama

# Тест Qwen2.5-3B
print("=" * 50)
print("Тест модели Qwen2.5-3B")
print("=" * 50)

llm_qwen = Llama(
    model_path="./qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,  # Поставьте 0, если нет NVIDIA GPU
    verbose=False
)

output = llm_qwen("Привет! Как дела?", max_tokens=100, echo=False)
print(output["choices"][0]["text"])

# Тест Llama-3.2-3B
print("\n" + "=" * 50)
print("Тест модели Llama-3.2-3B")
print("=" * 50)

llm_llama = Llama(
    model_path="./Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False
)

output = llm_llama("Привет! Как дела?", max_tokens=100, echo=False)
print(output["choices"][0]["text"])

print("\n✓ Обе модели работают!")