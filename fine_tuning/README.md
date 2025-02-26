# Fine-Tuning Qwen2-0.5B for Function Calling and Reasoning with LoRA 🚀

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hqmTaQZJhKSy-lQLxvBXh3o536Nky4K0)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/Patrik1352/Agent_Qven)

🔧 **Проект**: Тонкая настройка модели **Qwen2-0.5B-Instruct** для выполнения вызовов функций и генерации рассуждений с использованием **LoRA**.  
🎯 **Цель**: Создание интеллектуального ассистента, способного анализировать запросы, генерировать промежуточные рассуждения (`<think>...</think>`) и вызывать внешние инструменты (`<tool_call>...</tool_call>`).

---

## ✨ Ключевые особенности
- **LoRA-оптимизация**: Адаптация модели с минимальным количеством параметров (ранг 16, alpha 64).
- **Функциональный вызов**: Модель умеет генерировать структурированные JSON-запросы для вызова инструментов.
- **Рассуждения в реальном времени**: Добавление тегов `<think>` для демонстрации внутреннего мыслительного процесса.
- **Эффективность**: Обучение заняло **1.5 часа** на Google Colab (T4 GPU).
- **Интеграция с Hugging Face**: Модель и токенизатор загружены в Hugging Face Hub.

---

## 🛠️ Технологии
- **Модель**: [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- **Оптимизация**: [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation)
- **Библиотеки**: `transformers`, `peft`, `trl`, `bitsandbytes`
- **Данные**: [Hermes-Function-Calling Dataset](https://huggingface.co/datasets/Jofthomas/hermes-function-calling-thinking-V1)
- **Инфраструктура**: Google Colab, Hugging Face Hub

---

## 📊 Гиперпараметры обучения
```python
rank_dimension = 16     # Ранг матриц LoRA
lora_alpha = 64         # Коэффициент масштабирования
lora_dropout = 0.05     # Регуляризация
learning_rate = 1e-4    # Скорость обучения
max_seq_length = 1500   # Макс. длина контекста
epochs = 1              # Кол-во эпох
```
## 🚀 Как использовать модель?
1. Загрузка модели и токенизатора
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = PeftModel.from_pretrained(model, "Patrik1352/Agent_Qven")
tokenizer = AutoTokenizer.from_pretrained("Patrik1352/Agent_Qven")
```
2. Пример генерации
python
```python
prompt = """<|im_start|>system
You are a function calling AI model. Here are the tools: <tools>[...]</tools>
<|im_end|>
<|im_start|>user
Draw a cat<|im_end|>
<|im_start|>assistant
<think>"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0]))
```
## 📈 Результаты
Модель успешно генерирует:

Мой запрос:

```
Draw a cat
```

Рассуждения внутри <think>:

```
<think>Okay, so the user asked to draw a cat.
I need to figure out how to respond.
Looking at the available tools, there's a function called draw_it which does exactly that.
It takes a title as input, which is the description of the image. The user provided the title as "cat."
So, I should call the draw_it function with that title.
That makes sense because it directly addresses the user's request.</think>
```

Структурированные вызовы функций:

```
<tool_call>
{"name": "draw_it", "arguments": {"title": "cat"}}
</tool_call>
```
