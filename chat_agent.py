import json
from typing import List, Dict, Optional
from config import llm, logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ChatAgent:
    def generate_general_annotation(self, ts_features: Dict) -> str:
        """Создает краткую аннотацию на естественном языке из JSON временного ряда."""
        if "error" in ts_features or not all(key in ts_features for key in ["metric", "domain", "trend", "seasonality"]):
            error_msg = f"Ошибка: Некорректные данные временного ряда: {ts_features.get('error', 'Недостаточно данных')}"
            logger.error(error_msg)
            return error_msg

        # Извлекаем данные
        metric = ts_features.get("metric", "неизвестно")
        domain = ts_features.get("domain", "неизвестно")
        trend = ts_features.get("trend", "неизвестно")
        seasonality = ts_features.get("seasonality", "неизвестно")
        min_value = ts_features.get("min_value", "неизвестно")
        max_value = ts_features.get("max_value", "неизвестно")
        anomalies = ts_features.get("anomalies", [])
        hypotheses = ts_features.get("hypotheses", "Гипотезы отсутствуют")

        # Формируем базовую аннотацию
        annotation = f"Дашборд отображает {metric} в области {domain}. {trend}. {seasonality}. "
        if max_value != "неизвестно" or min_value != "неизвестно":
            annotation += f"Максимум: {max_value}, минимум: {min_value}. "
        if anomalies:
            anomalies_text = ", ".join([f"{a['value']} на {a['date']}" for a in anomalies])
            annotation += f"Обнаружены аномалии: {anomalies_text}. "
        annotation += f"{hypotheses}"

        # Удаляем лишние пробелы
        annotation = " ".join(annotation.split())
        logger.info(f"Сгенерирована аннотация: {annotation}")

        # Проверяем и улучшаем аннотацию
        return self.review_annotation(annotation, ts_features)

    def review_annotation(self, annotation: str, ts_features: Dict) -> str:
        """Проверяет и улучшает аннотацию на естественность и согласованность."""
        prompt = ChatPromptTemplate.from_template(
            """Проверь аннотацию на естественность, краткость и согласованность с данными:
            Аннотация: {annotation}
            Характеристики временного ряда: {ts_features}
            Убедись, что аннотация звучит естественно, как текст для человека, и не содержит противоречий.
            Верни улучшенную версию аннотации, сохраняя краткость, БЕЗ ПОДПУНКТОВ одним абзацем!"""
        )
        chain = prompt | llm | StrOutputParser()
        try:
            response = chain.invoke({
                "annotation": annotation,
                "ts_features": json.dumps(ts_features)
            })
            logger.info(f"Улучшенная аннотация: {response}")
            return response
        except Exception as e:
            logger.error(f"Ошибка при проверке аннотации: {str(e)}")
            return annotation

    def process_user_query(self, query: str, image_path: Optional[str], data_path: Optional[str], chat_history: List[Dict]) -> str:
        """Обрабатывает запрос пользователя с учетом контекста и финансовых терминов."""
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["тренд", "закономерность", "сезонность", "аномалия", "минимум", "максимум"]):
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