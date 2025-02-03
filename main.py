from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import os
import logging

from model import GGUFModel  # Импортируем наш модуль для работы с моделью

app = FastAPI()

# Настройка логгирования для основного приложения
log_dir = "applogs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler(os.path.join(log_dir, "main.log"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# Схема входящего запроса, аналогичная структуре запроса OpenAI Chat API
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 150

# Схема ответа, аналогичная OpenAI API
class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    choices: List[dict]
    usage: dict

# Загружаем модель при запуске сервера
model_path = "path/to/your_model.gguf"  # Укажите путь к вашей модели в формате GGUF
model_instance = GGUFModel(model_path)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completion(request: ChatCompletionRequest):
    logger.info("Получен запрос /v1/chat/completions")
    logger.debug(f"Данные запроса: {request.dict()}")
    
    # Формируем prompt из переданных сообщений
    prompt = ""
    for msg in request.messages:
        prompt += f"{msg.role.capitalize()}: {msg.content}\n"
    
    # Генерируем ответ с помощью локальной модели
    try:
        generated_text = model_instance.generate(
            prompt=prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        logger.info("Сгенерирован ответ модели")
        logger.debug(f"Сгенерированный текст: {generated_text}")
    except Exception as e:
        logger.error(f"Ошибка генерации: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # Формируем ответ API, аналогичный формату OpenAI (с ключом "message")
    response = ChatCompletionResponse(
        id="chatcmpl-123456",
        object="chat.completion",
        created=int(time.time()),
        choices=[{
            "message": {"role": "assistant", "content": generated_text},
            "index": 0,
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(generated_text.split()),
            "total_tokens": len(prompt.split()) + len(generated_text.split())
        }
    )
    logger.debug(f"Ответ API: {response.dict()}")
    return response 