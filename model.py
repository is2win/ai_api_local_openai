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
    def __init__(self, model_id: str, gguf_file: str):
        # Инициализируем модель из файла в формате GGUF с использованием Transformers Hugging Face
        self.model_id = model_id
        self.gguf_file = gguf_file
        self.model = self.load_model(model_id, gguf_file)
        self.tokenizer = self.load_tokenizer(model_id, gguf_file)
    
    def load_model(self, model_id: str, gguf_file: str):
        logger.info(f"Загрузка модели {model_id} с использованием gguf файла: {gguf_file}")
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=gguf_file)
        except Exception as e:
            logger.exception(f"Ошибка загрузки модели: {str(e)}")
            raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")
        logger.info("Модель успешно загружена с использованием Transformers и gguf")
        return model

    def load_tokenizer(self, model_id: str, gguf_file: str):
        logger.info(f"Загрузка токенизатора для модели {model_id} с использованием gguf файла: {gguf_file}")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=gguf_file)
        except Exception as e:
            logger.exception(f"Ошибка загрузки токенизатора: {str(e)}")
            raise RuntimeError(f"Ошибка загрузки токенизатора: {str(e)}")
        logger.info("Токенизатор успешно загружен")
        return tokenizer

    def generate(self, prompt: str, temperature: float = 1.0, max_tokens: int = 150, **kwargs) -> str:
        logger.debug(f"Начало генерации текста. Prompt: {prompt}")
        try:
            inputs = self.tokenizer(prompt, return_tensors='pt')
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, **kwargs)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"Генерация завершена. Результат: {generated_text}")
            return generated_text
        except Exception as e:
            logger.exception(f"Ошибка генерации текста: {str(e)}")
            raise RuntimeError(f"Ошибка генерации текста: {str(e)}") 