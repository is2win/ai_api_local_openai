from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import os
import logging
from dotenv import load_dotenv

from model import GGUFModel

# Загружаем переменные окружения
load_dotenv()

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
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.gguf")  # Путь к GGUF файлу
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "0"))  # Количество GPU слоев
N_CTX = int(os.getenv("N_CTX", "2048"))  # Размер контекстного окна

model_instance = GGUFModel(
    model_path=MODEL_PATH,
    n_gpu_layers=N_GPU_LAYERS,
    n_ctx=N_CTX
)

def format_prompt(messages: List[Message]) -> str:
    """Форматирование сообщений в промпт для модели"""
    formatted_prompt = ""
    for msg in messages:
        if msg.role == "system":
            formatted_prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            formatted_prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            formatted_prompt += f"Assistant: {msg.content}\n"
    formatted_prompt += "Assistant: "
    return formatted_prompt

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completion(request: ChatCompletionRequest):
    logger.info("Получен запрос /v1/chat/completions")
    logger.debug(f"Данные запроса: {request.dict()}")
    
    # Форматируем промпт из сообщений
    prompt = format_prompt(request.messages)
    
    # Генерируем ответ
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
    
    # Формируем ответ API
    response = ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
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