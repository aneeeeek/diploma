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
        self.max_rows = 100  # Ограничение на количество строк для CSV
        self.max_base64_length = 50000  # Максимальная длина base64-строки (примерно)

    def encode_image(self, image_path: str) -> str:
        """Кодирует изображение в формат base64."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                logger.info(f"Изображение {image_path} закодировано, длина: {len(encoded_string)}")
                if len(encoded_string) > self.max_base64_length:
                    logger.warning(f"Размер base64 изображения превышает {self.max_base64_length}, пропускается")
                    return ""
                return encoded_string
        except Exception as e:
            logger.error(f"Ошибка при кодировании изображения {image_path}: {str(e)}")
            return ""

    def encode_data(self, data_path: Path) -> str:
        """Читает данные и кодирует их в CSV в формате base64, ограничивая количество строк."""
        try:
            if data_path.suffix == '.csv':
                df = pd.read_csv(data_path, nrows=self.max_rows)
            elif data_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(data_path, engine='openpyxl', nrows=self.max_rows)
            else:
                logger.error(f"Неподдерживаемый формат файла: {data_path.suffix}")
                return ""
            # Преобразуем в CSV и кодируем в base64
            csv_buffer = df.to_csv(index=False)
            encoded_csv = base64.b64encode(csv_buffer.encode("utf-8")).decode("utf-8")
            logger.info(f"Данные {data_path} закодированы, длина: {len(encoded_csv)}")
            if len(encoded_csv) > self.max_base64_length:
                logger.warning(f"Размер base64 данных превышает {self.max_base64_length}, пропускается")
                return ""
            return encoded_csv
        except Exception as e:
            logger.error(f"Ошибка при кодировании данных {data_path}: {str(e)}")
            return ""

    def extract_text_from_response(self, content: str) -> str:
        """Извлекает домен из ответа LLM, удаляя markdown, если он есть."""
        if not content:
            logger.error("Пустой ответ от LLM")
            return self.default_domain
        json_pattern = r"```json\s*([\s\S]*?)\s*```"
        match = re.search(json_pattern, content)
        if match:
            try:
                json_content = json.loads(match.group(1).strip())
                domain = json_content.get("domain", self.default_domain)
                return domain
            except json.JSONDecodeError:
                logger.error(f"Некорректный JSON в ответе: {content}")
        else:
            logger.error(f"JSON не найден в ответе LLM: {content}")
        return self.default_domain

    def suggest_domain(self, image_path: Optional[str], data_path: Optional[str]) -> str:
        """Определяет область дашборда на основе изображения и данных, возвращая строку."""
        # Подготовка данных
        base64_image = self.encode_image(image_path) if image_path else ""
        base64_data = self.encode_data(Path(data_path)) if data_path else ""
        # Проверка, есть ли хотя бы один источник данных
        if not base64_image and not base64_data:
            logger.warning("Отсутствуют данные и изображение, возвращается default_domain")
            return self.default_domain
        # Формируем промпт
        prompt = f"""Ты аналитик данных. На основе изображения дашборда и данных временного ряда определи область применения дашборда.
Извлеки контекст из полученных данных, а именно область применения дашборда (например, финансы, экономика, криптовалюта, медицина, политика, компьютерные вычисления и прочее, что можешь распознать).
Изображение в base64: {base64_image if base64_image else 'отсутствует'}
Данные в CSV (base64): {base64_data if base64_data else 'отсутствуют'}
Верни результат в формате JSON с полем "domain" (например, {{"domain": "Заболевания"}}).
Ответ должен быть заключен в ```json ```.
Инструкция: Декодируй base64, проанализируй данные и изображение, выбери наиболее подходящую область.
"""
        # Проверка длины промпта
        prompt_length = len(prompt)
        logger.info(f"Длина промпта: {prompt_length} символов")
        if prompt_length > 100000:  # Примерный лимит для предотвращения превышения токенов
            logger.error(f"Промпт слишком длинный ({prompt_length} символов), возвращается default_domain")
            return self.default_domain
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
            # Проверяем, что ответ существует и содержит choices
            if not response or not hasattr(response, 'choices') or not response.choices:
                logger.error("Ответ от LLM пустой или не содержит choices")
                return self.default_domain
            content = response.choices[0].message.content
            if not content:
                logger.error("Пустое содержимое ответа от LLM")
                return self.default_domain
            logger.info(f"Ответ LLM для определения области: {content}")
            # Извлекаем домен из ответа
            domain = self.extract_text_from_response(content)
            logger.info(f"Предполагаемая область дашборда: {domain}")
            return domain
        except Exception as e:
            logger.error(f"Ошибка при обращении к LLM: {str(e)}")
            return self.default_domain

    def adapt_to_domain(self, annotation: str, image_path: Optional[str] = None, data_path: Optional[str] = None) -> str:
        """Адаптирует аннотацию, добавляя название домена."""
        domain = self.suggest_domain(image_path, data_path)
        # Формируем итоговую аннотацию с указанием домена
        result = f"Область дашборда: {domain}. {annotation}"
        logger.info(f"Адаптированная аннотация: {result}")
        return result