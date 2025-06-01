import json
from typing import Dict, Optional, List
from config import llm, logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dashboard_analyzer import DashboardAnalyzer
from domain_specific_analyzer import DomainSpecificAnalyzer
from timeseries_analyzer import TimeSeriesAnalyzer


class ChatAgent:
    def __init__(self):
        self.dashboard_analyzer = DashboardAnalyzer()
        self.domain_analyzer = DomainSpecificAnalyzer(default_domain="finance")
        self.timeseries_analyzer = TimeSeriesAnalyzer()

    def generate_general_annotation(self, ts_features: Dict) -> str:
        """Создает аннотацию на основе характеристик временного ряда через LLM."""
        if "error" in ts_features or not all(
                key in ts_features for key in ["metric", "domain", "trend", "seasonality"]):
            error_msg = f"Ошибка: Некорректные данные временного ряда: {ts_features.get('error', 'Недостаточно данных')}"
            logger.error(error_msg)
            return error_msg

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

        chain = prompt | llm | StrOutputParser()

        try:
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
        """Пересылает запрос пользователя агентам и объединяет их ответы."""
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])

        # Пересылаем запрос каждому агенту
        dashboard_response = self.dashboard_analyzer.query_dashboard(query, image_path, context)
        domain_response = self.domain_analyzer.query_domain(query, image_path, data_path, context)
        timeseries_response = self.timeseries_analyzer.query_timeseries(query, data_path, context)

        # Собираем ответы
        responses = {
            "dashboard": dashboard_response,
            "domain": domain_response,
            "timeseries": timeseries_response
        }
        logger.info(f"Ответы агентов: {responses}")

        # Проверяем, есть ли содержательные ответы
        meaningful_responses = [resp for resp in responses.values() if resp != "неизвестно"]

        if not meaningful_responses:
            logger.info("Все агенты вернули 'неизвестно', запрашиваем переформулировку")
            return "Пожалуйста, переформулируйте ваш вопрос, чтобы он был связан с дашбордом, областью или временным рядом."

        # Объединяем ответы с помощью LLM
        prompt = ChatPromptTemplate.from_template(
            """Ты аналитик данных. Объедини ответы от трех агентов в один связный ответ на запрос пользователя.
            Запрос пользователя: {query}
            Контекст: {context}
            Ответы агентов:
            - Dashboard Agent: {dashboard_response}
            - Domain Agent: {domain_response}
            - Timeseries Agent: {timeseries_response}

            Правила:
            1. Если агент ответил "неизвестно", игнорируй его ответ.
            2. Сформируй связный ответ, объединяя информацию из всех содержательных ответов.
            3. Используй термины, специфичные для финансовой области, если применимо.
            4. Ответ должен быть кратким, естественным и адресованным человеку.
            5. Если есть противоречия между ответами, выбери наиболее логичный или укажи неопределенность.

            Верни объединенный ответ одним абзацем."""
        )

        chain = prompt | llm | StrOutputParser()

        try:
            combined_response = chain.invoke({
                "query": query,
                "context": context,
                "dashboard_response": dashboard_response,
                "domain_response": domain_response,
                "timeseries_response": timeseries_response
            })
            logger.info(f"Объединенный ответ: {combined_response}")
            return combined_response
        except Exception as e:
            logger.error(f"Ошибка при объединении ответов: {str(e)}")
            return f"Ошибка обработки запроса: {str(e)}"