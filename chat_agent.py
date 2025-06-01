import json
from typing import Dict, Optional, List
from config import llm, logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ChatAgent:
    def generate_general_annotation(self, ts_features: Dict) -> str:
        """Создает аннотацию на основе характеристик временного ряда через LLM."""
        # Проверяем наличие обязательных ключей
        if "error" in ts_features or not all(
                key in ts_features for key in ["metric", "domain", "trend", "seasonality"]):
            error_msg = f"Ошибка: Некорректные данные временного ряда: {ts_features.get('error', 'Недостаточно данных')}"
            logger.error(error_msg)
            return error_msg

        # Формируем промпт для LLM
        prompt = ChatPromptTemplate.from_template(
            """Ты аналитик данных. На основе характеристик временного ряда составь аннотацию по следующему плану:
            1. Опиши область и метрику дашборда одним предложением.
            2. Опиши тренды и сезонность, указав их характер и особенности.
            3. Укажи максимальное и минимальное значения с датами.
            4. Опиши обнаруженные аномалии, если они есть, или укажи их отсутствие.
            5. Предложи гипотезы, объясняющие тренды, сезонность или аномалии.

            Характеристики временного ряда: {ts_features}

            Верни аннотацию одним абзацем, кратко и естественно, как для человека, используя термины в нужной области.
            Не используй подзаголовки или списки, только связный текст.
            Если данные отсутствуют или некорректны, укажи это в аннотации."""
        )

        # Создаем цепочку обработки
        chain = prompt | llm | StrOutputParser()

        try:
            # Отправляем запрос к LLM с JSON-строкой характеристик
            response = chain.invoke({
                "ts_features": json.dumps(ts_features, ensure_ascii=False)
            })
            logger.info(f"Сгенерирована аннотация: {response}")
            return response
        except Exception as e:
            logger.error(f"Ошибка при генерации аннотации: {str(e)}")
            return f"Ошибка генерации аннотации: {str(e)}"

    def process_user_query(self, query: str, image_path: Optional[str], data_path: Optional[str],
                           chat_history: List[Dict]) -> str:
        """Обрабатывает запрос пользователя с учетом контекста и финансовых терминов."""
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in
               ["тренд", "закономерность", "сезонность", "аномалия", "минимум", "максимум"]):
            agent = "dashboard"
        elif any(keyword in query_lower for keyword in ["показатель", "kpi"]):
            agent = "dashboard"
        else:
            agent = "general"

        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
        prompt = ChatPromptTemplate.from_template(
            """Запрос пользователя: {query}
            Контекст: {context}
            Агент: {agent}
            Дайте краткий ответ на основе данных дашборда и временного ряда. Используйте термины, специфичные для финансовой области."""
        )
        chain = prompt | llm | StrOutputParser()
        try:
            response = chain.invoke({
                "query": query,
                "context": context,
                "agent": agent
            })
            logger.info(f"Ответ на запрос пользователя: {response}")
            return response
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {str(e)}")
            return f"Ошибка обработки запроса: {str(e)}"