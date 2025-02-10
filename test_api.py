import requests
import json
import time

def test_chat_completion():
    """
    Тестирование эндпоинта chat/completions
    """
    url = "http://localhost:8000/v1/chat/completions"
    
    # Тестовые сообщения
    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "system",
                "content": "Ты - полезный ассистент, который отвечает кратко и по делу."
            },
            {
                "role": "user",
                "content": "Привет! Расскажи о себе в двух предложениях."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("Отправка запроса к API...")
        start_time = time.time()
        
        response = requests.post(url, headers=headers, json=payload)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\nВремя обработки запроса: {processing_time:.2f} секунд")
        
        if response.status_code == 200:
            result = response.json()
            print("\nУспешный ответ от API:")
            print("------------------------")
            print(f"ID: {result['id']}")
            print(f"Созданный: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['created']))}")
            print("\nСгенерированный ответ:")
            print("---------------------")
            print(result['choices'][0]['message']['content'])
            print("\nСтатистика использования токенов:")
            print("------------------------------")
            print(json.dumps(result['usage'], indent=2))
        else:
            print(f"\nОшибка при запросе: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\nПроизошла ошибка: {str(e)}")

if __name__ == "__main__":
    print("Начало тестирования API...")
    test_chat_completion()
    print("\nТестирование завершено.") 