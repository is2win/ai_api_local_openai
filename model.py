import time
import os
import logging

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
    def __init__(self, model_path: str):
        # Инициализируем модель из файла в формате GGUF
        self.model_path = model_path
        self.model = self.load_model(model_path)
    
    def load_model(self, path: str):
        if not os.path.exists(path):
            logger.error(f"Модель не найдена по пути: {path}")
            raise FileNotFoundError(f"Модель не найдена по пути: {path}")

        logger.info(f"Загрузка модели из: {path}")
        # Импортируем библиотеку ctransformers
        try:
            from ctransformers import AutoModelForCausalLM
        except ImportError:
            logger.exception("Пакет ctransformers не установлен")
            raise ImportError("Пакет ctransformers не установлен. Установите его командой 'pip install ctransformers'.")

        # Создаем экземпляр модели с заданным путём через ctransformers и загружаем модель на GPU
        model = AutoModelForCausalLM(model=path, device="cuda")
        logger.info("Модель успешно загружена на GPU")
        return model

    def generate(self, prompt: str, temperature: float, max_tokens: int, context_window: int = 2048, repetition_penalty: float = 1.0, stop_tokens: list = None) -> str:
        logger.debug(f"Начало генерации текста. Prompt: {prompt}")
        logger.debug(f"Параметры генерации: max_tokens={max_tokens}, temperature={temperature}, context_window={context_window}, repetition_penalty={repetition_penalty}, stop_tokens={stop_tokens}")
        try:
            generated_text = self.model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                context_window=context_window,
                repetition_penalty=repetition_penalty,
                stop=stop_tokens
            )
            logger.debug(f"Генерация завершена. Результат: {generated_text}")
            return generated_text
        except Exception as e:
            logger.exception(f"Ошибка генерации текста: {str(e)}")
            raise RuntimeError(f"Ошибка генерации текста: {str(e)}") 