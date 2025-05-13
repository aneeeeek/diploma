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
        """Анализирует изображение дашборда, извлекая характеристики временного ряда."""
        # Кодируем изображение
        try:
            base64_image = self.encode_image(image_path)
        except Exception as e:
            return {"error": f"Не удалось закодировать изображение: {str(e)}"}

        # Формируем промпт для анализа изображения
        prompt = """Ты успешный аналитик временных рядов. 
        Проанализируй изображение дашборда, содержащее временной ряд. Извлеки следующие характеристики и верни их в формате JSON:
        - main_metric: основной показатель для временного ряда (например, "Продажи золота")
        - trend: визуальный тренд (восходящий, нисходящий, стабильный)
        - seasonality: наличие сезонности (присутствует, отсутствует)
        - anomalies: наличие аномалий (описание или количество, если есть)
        - min_value: минимальное значение и предполагаемая дата (например, "$50,000 на 2023-01-15")
        - max_value: максимальное значение и предполагаемая дата (например, "$150,000 на 2023-06-30")
        Если какие-то данные не удается определить, укажи "неизвестно".
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
                max_tokens=300,
                temperature=0.5,
                stream=False
            )
            content = response.choices[0].message.content
            # Извлекаем JSON из markdown, если он есть
            json_content = self.extract_json_from_markdown(content)

            try:
                # Пробуем распарсить JSON
                result = json.loads(json_content)
                logger.info(f"Характеристики дашборда: {result}")
                return result
            except json.JSONDecodeError:
                logger.error(f"Некорректный JSON от LLM: {json_content}")
                return {"error": f"Некорректные данные дашборда от LLM: {json_content}"}
        except Exception as e:
            logger.error(f"Ошибка анализа изображения дашборда: {str(e)}")
            return {"error": f"Ошибка анализа изображения: {str(e)}"}