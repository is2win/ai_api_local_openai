import time
import os
import logging
from typing import Optional
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Настройка логгирования для модели
log_dir = "applogs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler(os.path.join(log_dir, "model.log"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

class GGUFModel:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0):
        """
        Инициализация модели LlamaCpp
        
        Args:
            model_path: Путь к GGUF файлу модели
            n_ctx: Размер контекстного окна
            n_gpu_layers: Количество слоев для выполнения на GPU (0 для CPU)
        """
        self.model_path = model_path
        logger.info(f"Инициализация модели из файла: {model_path}")
        
        try:
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            self.model = LlamaCpp(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                callback_manager=callback_manager,
                verbose=True
            )
            logger.info("Модель успешно загружена")
            
        except Exception as e:
            logger.exception(f"Ошибка загрузки модели: {str(e)}")
            raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")

    def generate(self, prompt: str, temperature: float = 1.0, max_tokens: int = 150, **kwargs) -> str:
        """
        Генерация текста с помощью модели
        
        Args:
            prompt: Входной текст
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов для генерации
            
        Returns:
            str: Сгенерированный текст
        """
        logger.debug(f"Начало генерации текста. Prompt: {prompt}")
        try:
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            logger.debug(f"Генерация завершена. Результат: {response}")
            return response
            
        except Exception as e:
            logger.exception(f"Ошибка генерации текста: {str(e)}")
            raise RuntimeError(f"Ошибка генерации текста: {str(e)}") 