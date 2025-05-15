import base64
import json
import re
import pandas as pd
from pathlib import Path
from config import client, logger
from typing import Optional

class DomainSpecificAnalyzer:
    def __init__(self, default_domain: str = "finance"):
        self.default_domain = default_domain

    def encode_image(self, image_path: str) -> str:
        """Кодирует изображение в формат base64."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                logger.info(f"Изображение {image_path} успешно закодировано в base64")
                return encoded_string
        except Exception as e:
            logger.error(f"Ошибка при кодировании изображения {image_path}: {str(e)}")
            return ""

    def encode_data(self, data_path: Path) -> str:
        """Читает данные и кодирует их в CSV в формате base64."""
        try:
            if data_path.suffix == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(data_path, engine='openpyxl')
            else:
                logger.error(f"Неподдерживаемый формат файла: {data_path.suffix}")
                return ""

            # Преобразуем в CSV и кодируем в base64
            csv_buffer = df.to_csv(index=False)
            encoded_csv = base64.b64encode(csv_buffer.encode("utf-8")).decode("utf-8")
            logger.info(f"Данные {data_path} успешно закодированы в base64")
            return encoded_csv
        except Exception as e:
            logger.error(f"Ошибка при кодировании данных {data_path}: {str(e)}")
            return ""

    def extract_text_from_response(self, content: str) -> str:
        """Извлекает текст из ответа LLM, удаляя markdown, если он есть."""
        json_pattern = r"```json\s*([\s\S]*?)\s*```"
        match = re.search(json_pattern, content)
        if match:
            try:
                json_content = json.loads(match.group(1).strip())
                return json_content.get("domain", self.default_domain)
            except json.JSONDecodeError:
                logger.error(f"Некорректный JSON в ответе: {content}")
        return content.strip() or self.default_domain

    def suggest_domain(self, image_path: Optional[str], data_path: Optional[str]) -> str:
        """Обращается к LLM для предположения области дашборда на основе изображения и данных."""
        # Подготовка данных
        base64_image = self.encode_image(image_path) if image_path else ""
        base64_data = self.encode_data(Path(data_path)) if data_path else ""

        # Формируем промпт
        prompt = f"""Ты аналитик данных. На основе изображения дашборда и данных временного ряда определи область применения дашборда (например, финансы, ритейл, производство, здравоохранение и т.д.).
        Изображение в base64: {base64_image if base64_image else 'отсутствует'}
        Данные в CSV (base64): {base64_data if base64_data else 'отсутствуют'}
        Верни результат в формате JSON, указав предполагаемую область в поле "domain" (например, {{"domain": "финансы"}}).
        Если область определить невозможно, укажи "{self.default_domain}".
        Ответ должен быть кратким и содержать только JSON в блоке ```json ```.
        Инструкция: Декодируй base64, проанализируй данные и изображение, определи область.
        """

        try:
            # Отправляем запрос к LLM
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
                            } if base64_image else {"type": "text", "text": "Изображение отсутствует"}
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0.5,
                stream=False
            )
            content = response.choices[0].message.content
            logger.info(f"Ответ LLM для определения области: {content}")

            # Извлекаем область из ответа
            domain = self.extract_text_from_response(content)
            logger.info(f"Предполагаемая область дашборда: {domain}")
            return f"Предполагаемая область дашборда: {domain}."
        except Exception as e:
            logger.error(f"Ошибка при обращении к LLM: {str(e)}")
            return f"Предполагаемая область дашборда: {self.default_domain}."

    def adapt_to_domain(self, annotation: str, image_path: Optional[str] = None, data_path: Optional[str] = None) -> str:
        """Адаптирует аннотацию к домену, предполагая область через LLM."""
        domain_suggestion = self.suggest_domain(image_path, data_path)
        logger.info(f"Адаптация аннотации с учетом области: {domain_suggestion}")
        return f"{domain_suggestion} {annotation}"