import base64
import json
import re
import pandas as pd
from pathlib import Path
from config import client, logger, llm
from typing import Optional, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from PIL import Image
import io


class DomainSpecificAnalyzer:
    def __init__(self, default_domain: str = ""):
        self.default_domain = default_domain

    def encode_image(self, image_path: str) -> str:
        """Кодирует изображение в формат base64 с предварительным сжатием."""
        try:
            # Открываем изображение
            img = Image.open(image_path)
            # Уменьшаем размер до 800x600
            img.thumbnail((800, 600))
            # Сохраняем в буфер с сжатием JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=50)  # Сжатие с качеством 50
            encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
            logger.info(f"Изображение {image_path} закодировано, длина: {len(encoded_string)}")
            return encoded_string
        except Exception as e:
            logger.error(f"Ошибка при кодировании изображения {image_path}: {str(e)}")
            return f"Ошибка: Не удалось закодировать изображение: {str(e)}"

    def encode_data(self, data_path: str) -> str:
        """Кодирует данные в CSV в формате base64, ограничивая до 50 строк."""
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path, nrows=50)  # Ограничиваем до 50 строк
                logger.info(f"Прочитано {len(df)} строк из CSV файла {data_path}")
            elif data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path, nrows=50)  # Ограничиваем до 50 строк
                logger.info(f"Прочитано {len(df)} строк из Excel файла {data_path}")
            else:
                logger.error(f"Неподдерживаемый формат файла: {data_path}")
                return ""
            # Ограничиваем до двух столбцов (дата и первое числовое значение)
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                df = df[[df.columns[0], numeric_cols[0]]]  # Дата и первый числовой столбец
            else:
                df = df.iloc[:, :2]  # Первые два столбца, если числовых нет
            csv_buffer = df.to_csv(index=False)
            encoded_csv = base64.b64encode(csv_buffer.encode("utf-8")).decode("utf-8")
            logger.info(f"Данные {data_path} закодированы, длина: {len(encoded_csv)}")
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

    def suggest_domain(self, image_path: Optional[str], data_path: Optional[str]) -> Dict:
        """Определяет область дашборда на основе изображения и данных, возвращая JSON."""
        base64_image = self.encode_image(image_path) if image_path else ""
        base64_data = self.encode_data(data_path) if data_path else ""
        if not base64_image and not base64_data:
            logger.warning("Отсутствуют данные и изображение, возвращается default_domain")
            return {"domain": self.default_domain}

        logger.info(f"Размер base64_image: {len(base64_image)} байт, base64_data: {len(base64_data)} байт")
        prompt = f"""Ты аналитик данных. На основе изображения дашборда и данных временного ряда определи область применения дашборда.
        Извлеки контекст из полученных данных, а именно область применения дашборда (например, финансы, экономика, криптовалюта, медицина, политика, компьютерные вычисления и прочее, что можешь распознать).
        Изображение в base64: {base64_image if base64_image else 'отсутствует'}
        Данные в CSV (base64): {base64_data if base64_data else 'отсутствуют'}
        Верни результат в формате JSON с полем "domain" (например, {{"domain": "Заболевания"}}).
        Ответ должен быть заключен в ```json ```.
        Инструкция: Декодируй base64, проанализируй данные и изображение, выбери наиболее подходящую область.
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
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            } if base64_image and not base64_image.startswith("Ошибка") else {"type": "text",
                                                                                              "text": "Изображение отсутствует"}
                        ]
                    }
                ],
                max_tokens=500,  # Уменьшено для оптимизации
                temperature=0.5,
                stream=False
            )
            if not response or not hasattr(response, 'choices') or not response.choices:
                logger.error("Ответ от LLM пустой или не содержит choices")
                return {"domain": self.default_domain}
            content = response.choices[0].message.content
            if not content:
                logger.error("Пустое содержимое ответа от LLM")
                return {"domain": self.default_domain}
            logger.info(f"Ответ LLM для определения области: {content}")
            json_content = self.extract_text_from_response(content)
            return {"domain": json_content}
        except Exception as e:
            logger.error(f"Ошибка при обращении к LLM: {str(e)}")
            if "413" in str(e) or "request too large" in str(e).lower():
                return {"domain": self.default_domain,
                        "error": "Слишком большой объем данных или изображения. Пожалуйста, уменьшите размер файла."}
            return {"domain": self.default_domain, "error": f"Ошибка анализа: {str(e)}"}

    def query_domain(self, query: str, context: str, domain_features: Optional[Dict] = None) -> str:
        """Обрабатывает запрос пользователя, связанный с областью применения дашборда."""
        domain = domain_features.get("domain", self.default_domain) if domain_features else self.default_domain
        prompt = ChatPromptTemplate.from_template(
            """Ты аналитик данных, специализирующийся на определении области применения дашборда.
            Запрос пользователя: {query}
            Контекст: {context}
            Область дашборда: {domain}

            Ответь на вопрос, если он связан с областью применения дашборда (например, финансы, медицина, экономика).
            Используй указанную область для ответа.
            Используй термины, специфичные для финансовой области, если применимо.
            Если вопрос не относится к твоей роли, верни "неизвестно".
            Верни ответ кратко, одним-двумя предложениями."""
        )

        chain = prompt | llm | StrOutputParser()

        try:
            response = chain.invoke({
                "query": query,
                "context": context,
                "domain": domain
            })
            logger.info(f"Ответ Domain Agent: {response}")
            return response
        except Exception as e:
            logger.error(f"Ошибка Domain Agent: {str(e)}")
            return "неизвестно"