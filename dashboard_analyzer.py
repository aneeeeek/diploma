import json
import base64
import re
from config import client, logger

class DashboardAnalyzer:
    def encode_image(self, image_path: str) -> str:
        """Кодирует изображение в формат base64."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                logger.info(f"Изображение {image_path} успешно закодировано в base64")
                return encoded_string
        except Exception as e:
            logger.error(f"Ошибка при кодировании изображения {image_path}: {str(e)}")
            raise

    def extract_json_from_markdown(self, content: str) -> str:
        """Извлекает JSON из markdown-блока, удаляя ```json и ```."""
        json_pattern = r"```json\s*([\s\S]*?)\s*```"
        match = re.search(json_pattern, content)
        if match:
            return match.group(1).strip()
        return content.strip()

    def analyze_dashboard(self, image_path: str) -> dict:
        """Анализирует изображение дашборда, извлекая только основную метрику."""
        # Кодируем изображение
        try:
            base64_image = self.encode_image(image_path)
        except Exception as e:
            return {"error": f"Не удалось закодировать изображение: {str(e)}"}

        # Формируем промпт для извлечения только метрики
        prompt = """Ты успешный аналитик данных. Проанализируй изображение дашборда, содержащее временной ряд. 
        Извлеки из графика основную метрику/показатель у временного ряда (например, "Продажи золота", "Выручка", "Объем производства"). 
        Сформулируй понятно для человека, на русском языке. Будь внимателен, может быть такое, что метрика указана в названии графика или в легенде.
        Можешь указать дополнительную информацию, если видишь: страну, крайние даты на оси Х, важные подписи на графике. 
        Верни результат в формате JSON с полем "main_metric". Если метрику определить невозможно, укажи "неизвестно".
        Пример: {"main_metric": "Детская смертность в Бразилии с 1934 по 2023 год"}
        """

        try:
            # Отправляем запрос с текстом и изображением
            response = client.chat.completions.create(
                model="openai.gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100,  # Уменьшаем, так как задача проще
                temperature=0.5,
                stream=False
            )
            content = response.choices[0].message.content
            # Извлекаем JSON из markdown, если он есть
            json_content = self.extract_json_from_markdown(content)

            try:
                # Пробуем распарсить JSON
                result = json.loads(json_content)
                if "main_metric" not in result:
                    logger.error(f"Поле main_metric отсутствует в ответе: {json_content}")
                    return {"main_metric": "неизвестно"}
                logger.info(f"Основная метрика дашборда: {result}")
                return result
            except json.JSONDecodeError:
                logger.error(f"Некорректный JSON от LLM: {json_content}")
                return {"main_metric": "неизвестно"}
        except Exception as e:
            logger.error(f"Ошибка анализа изображения дашборда: {str(e)}")
            return {"main_metric": "неизвестно"}