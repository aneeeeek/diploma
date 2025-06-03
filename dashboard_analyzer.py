import json
import base64
import re
from config import client, logger, llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, Dict


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
        try:
            base64_image = self.encode_image(image_path)
        except Exception as e:
            logger.error(f"Не удалось закодировать изображение: {str(e)}")
            return {"main_metric": "неизвестно"}

        prompt = """Ты успешный аналитик данных. Проанализируй изображение дашборда, содержащее временной ряд. 
        Извлеки из графика основную метрику/показатель у временного ряда (например, "Продажи золота", "Выручка", "Объем производства"). 
        Сформулируй понятно для человека, на русском языке. Будь внимателен, может быть такое, что метрика указана в названии графика или в легенде.
        Верни результат в формате JSON с полем "main_metric". Если метрику определить невозможно, укажи "неизвестно".
        Пример: {"main_metric": "Детская смертность в Бразилии с 1934 по 2023 год"}
        Ответ должен быть заключен в ```json ```.
        """

        try:
            response = client.chat.completions.create(
                model="aimediator.gpt-4.1-mini",
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
                max_tokens=100,
                temperature=0.5,
                stream=False
            )
            content = response.choices[0].message.content
            json_content = self.extract_json_from_markdown(content)

            try:
                result = json.loads(json_content)
                if "main_metric" not in result:
                    logger.error(f"Поле main_metric отсутствует в ответе: {json_content}")
                    return {"main_metric": "неизвестно"}
                logger.info(f"Основная метрика дашборда: {result['main_metric']}")
                return result
            except json.JSONDecodeError:
                logger.error(f"Некорректный JSON от LLM: {json_content}")
                return {"main_metric": "неизвестно"}
        except Exception as e:
            logger.error(f"Ошибка анализа изображения дашборда: {str(e)}")
            return {"main_metric": "неизвестно"}

    def query_dashboard(self, query: str, image_path: Optional[str], context: str,
                        dash_features: Optional[Dict] = None) -> str:
        """Обрабатывает запрос пользователя, связанный с визуальными элементами дашборда."""
        if not image_path:
            return "неизвестно"

        try:
            base64_image = self.encode_image(image_path)
        except Exception as e:
            logger.error(f"Ошибка при кодировании изображения: {str(e)}")
            return "неизвестно"

        main_metric = dash_features.get("main_metric", "неизвестно") if dash_features else "неизвестно"
        prompt = ChatPromptTemplate.from_template(
            """Ты аналитик дашбордов. Твоя роль — анализировать визуальные элементы дашборда (графики, метрики, подписи).
            Запрос пользователя: {query}
            Контекст: {context}
            Основная метрика дашборда: {main_metric}
            Изображение дашборда в base64: {base64_image}

            Ответь на вопрос, если он связан с визуальными элементами дашборда (например, метрика, название графика, легенда).
            Используй переданную метрику и изображение для ответа.
            Используй термины, специфичные для финансовой области, если применимо.
            Если вопрос не относится к твоей роли, верни "неизвестно".
            Верни ответ кратко, одним-двумя предложениями."""
        )

        chain = prompt | llm | StrOutputParser()

        try:
            response = chain.invoke({
                "query": query,
                "context": context,
                "main_metric": main_metric,
                "base64_image": base64_image
            })
            logger.info(f"Ответ Dashboard Agent: {response}")
            return response
        except Exception as e:
            logger.error(f"Ошибка Dashboard Agent: {str(e)}")
            return "неизвестно"