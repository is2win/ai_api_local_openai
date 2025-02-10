# Local LLM API с использованием llama.cpp

Локальный API сервер для работы с LLM моделями в формате GGUF через llama.cpp. API совместим с форматом OpenAI Chat Completions API.

## Требования

- Python 3.8+
- GGUF модель (можно скачать с [HuggingFace](https://huggingface.co/models?search=gguf))
- CMake (для сборки llama-cpp-python)

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Создайте виртуальное окружение и активируйте его:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
# или
.\venv\Scripts\activate  # для Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Создайте директорию для моделей и скачайте GGUF модель:
```bash
mkdir models
# Скачайте модель в формате GGUF и поместите её в директорию models/
```

5. Настройте конфигурацию в файле .env:
```bash
MODEL_PATH=models/your_model.gguf  # Путь к вашей модели
N_GPU_LAYERS=0  # Количество слоев для GPU (0 для CPU)
N_CTX=2048  # Размер контекстного окна
```

## Запуск сервера

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Использование API

API совместим с форматом OpenAI Chat Completions API. Пример запроса:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "local-model",
       "messages": [
         {"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Hello!"}
       ],
       "temperature": 0.7,
       "max_tokens": 150
     }'
```

## Логирование

Логи сохраняются в директории `applogs/`:
- `main.log` - логи основного приложения
- `model.log` - логи работы модели

## GPU Ускорение

Для использования GPU установите значение `N_GPU_LAYERS` в файле `.env`. Значение больше 0 активирует GPU ускорение для указанного количества слоев модели.

## Примечания

- Убедитесь, что у вас достаточно оперативной памяти для загрузки модели
- Размер контекстного окна (`N_CTX`) влияет на потребление памяти
- При использовании GPU убедитесь, что установлены необходимые драйверы CUDA 