import json
from typing import List, Dict, Optional
from config import llm, logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ChatAgent:
    def generate_general_annotation(self, ts_features: Dict, dash_features: Dict) -> str:
        """Создает краткую аннотацию на естественном языке с приоритизацией информации."""
        if "error" in dash_features or ("mathematical" not in ts_features and "llm" not in ts_features):
            error_msg = f"Ошибка: Невозможно создать аннотацию из-за некорректных данных. Временной ряд: {ts_features.get('error', '')}, Дашборд: {dash_features.get('error', '')}"
            logger.error(error_msg)
            return error_msg

        # Извлекаем данные с учетом приоритетов
        ts_math = ts_features.get("mathematical", {})
        ts_llm = ts_features.get("llm", {})

        # Приоритет: математический анализ для тренда, сезонности, максимума и минимума
        trend = ts_math.get("trend", "неизвестный")
        seasonality = ts_math.get("seasonality", "неизвестно")
        math_min = ts_math.get("min_value", "неизвестно")
        math_max = ts_math.get("max_value", "неизвестно")

        # Визуальные максимум и минимум из дашборда
        visual_min = dash_features.get("min_value", "неизвестно")
        visual_max = dash_features.get("max_value", "неизвестно")
        main_metric = dash_features.get("main_metric", "неизвестный")

        # Аномалии из LLM (текстовое описание)
        anomalies_description = ts_llm.get("anomalies_description", "Аномалии не обнаружены")

        # Формируем базовую аннотацию
        annotation = f"Дашборд отображает {main_metric} с {trend} трендом и {seasonality} сезонностью. "
        if visual_max != "неизвестно" or visual_min != "неизвестно":
            annotation += f"Визуально максимальное значение {visual_max}, минимальное {visual_min}. "
        if math_max != "неизвестно" or math_min != "неизвестно":
            annotation += f"По данным, максимум {math_max}, минимум {math_min}. "
        if anomalies_description != "Аномалии не обнаружены":
            annotation += f"{anomalies_description} "

        # Удаляем лишние пробелы и возвращаем аннотацию
        annotation = " ".join(annotation.split())
        logger.info(f"Сгенерирована аннотация: {annotation}")
        return annotation

    def review_annotation(self, annotation: str, ts_features: Dict, dash_features: Dict) -> str:
        """Проверяет и улучшает аннотацию на есте complements and consistency."""
        prompt = ChatPromptTemplate.from_template(
            """Проверь аннотацию на естественность, краткость и согласованность с данными:
            Аннотация: {annotation}
            Характеристики временного ряда: {ts_features}
            Характеристики дашборда: {dash_features}
            Убедись, что аннотация звучит естественно, как текст для человека, и не содержит противоречий.
            Верни улучшенную версию аннотации, сохраняя краткость, БЕЗ ПОДПУНКТОВ одним абзацем!"""
        )
        chain = prompt | llm | StrOutputParser()
        try:
            response = chain.invoke({
                "annotation": annotation,
                "ts_features": json.dumps(ts_features),
                "dash_features": json.dumps(dash_features)
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