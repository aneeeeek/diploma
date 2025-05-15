import json
from typing import List, Dict, Optional
from config import llm, logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ChatAgent:
    def generate_general_annotation(self, ts_features: Dict, dash_features: Dict) -> str:
        """Создает общую аннотацию на основе характеристик данных."""
        if "error" in dash_features or ("mathematical" not in ts_features and "llm" not in ts_features):
            error_msg = f"Ошибка: Невозможно создать аннотацию из-за некорректных данных. Временной ряд: {ts_features.get('error', '')}, Дашборд: {dash_features.get('error', '')}"
            logger.error(error_msg)
            return error_msg

        annotation = []

        # Аннотация для дашборда
        annotation.append(f"Дашборд показывает тренд {dash_features.get('trend', 'неизвестный')}.")
        annotation.append(f"Основной показатель: {dash_features.get('main_metric', 'неизвестный')}.")
        annotation.append(f"Сезонность: {dash_features.get('seasonality', 'неизвестно')}.")
        if dash_features.get('anomalies', 'неизвестно') != 'неизвестно':
            anomalies_str = ", ".join([f"{a['value']} на {a['date']}" for a in dash_features.get('anomalies', [])])
            annotation.append(f"Аномалии: {anomalies_str if anomalies_str else 'не обнаружены'}.")
        annotation.append(f"Минимальное значение: {dash_features.get('min_value', 'неизвестно')}.")
        annotation.append(f"Максимальное значение: {dash_features.get('max_value', 'неизвестно')}.")

        # Аннотация для математического анализа временного ряда
        ts_math = ts_features.get("mathematical", {})
        annotation.append(f"Математический анализ временного ряда показывает тренд {ts_math.get('trend', 'неизвестный')}.")
        annotation.append(f"Сезонность: {ts_math.get('seasonality', 'неизвестно')}.")
        if ts_math.get('anomalies', []):
            anomalies_str = ", ".join([f"{a['value']} на {a['date']}" for a in ts_math.get('anomalies', [])])
            annotation.append(f"Аномалии: {anomalies_str}.")
        else:
            annotation.append("Аномалии: не обнаружены.")
        annotation.append(f"Сравнение с дашбордом: тренд {'совпадает' if dash_features.get('trend') == ts_math.get('trend') else 'различается'}.")

        # Аннотация для LLM анализа временного ряда
        ts_llm = ts_features.get("llm", {})
        annotation.append(f"Анализ временного ряда через LLM показывает тренд {ts_llm.get('trend', 'неизвестный')}.")
        annotation.append(f"Сезонность: {ts_llm.get('seasonality', 'неизвестно')}.")
        if ts_llm.get('anomalies', []):
            anomalies_str = ", ".join([f"{a['value']} на {a['date']}" for a in ts_llm.get('anomalies', [])])
            annotation.append(f"Аномалии: {anomalies_str}.")
        else:
            annotation.append("Аномалии: не обнаружены.")
        annotation.append(f"Минимальное значение: {ts_llm.get('min_value', 'неизвестно')}.")
        annotation.append(f"Максимальное значение: {ts_llm.get('max_value', 'неизвестно')}.")
        annotation.append(f"Сравнение с математическим анализом: тренд {'совпадает' if ts_llm.get('trend') == ts_math.get('trend') else 'различается'}.")

        return " ".join(annotation)

    def review_annotation(self, annotation: str, ts_features: Dict, dash_features: Dict) -> str:
        """Проверяет аннотацию на соответствие данным."""
        prompt = ChatPromptTemplate.from_template(
            """Проверьте аннотацию на согласованность с данными:
            Аннотация: {annotation}
            Характеристики временного ряда (математический и LLM): {ts_features}
            Характеристики дашборда: {dash_features}
            Убедитесь, что аннотация ясна, полна и соответствует данным. Верните исправленную аннотацию."""
        )
        chain = prompt | llm | StrOutputParser()
        try:
            response = chain.invoke({
                "annotation": annotation,
                "ts_features": json.dumps(ts_features),
                "dash_features": json.dumps(dash_features)
            })
            return response
        except Exception as e:
            logger.error(f"Ошибка при проверке аннотации: {str(e)}")
            return annotation

    def process_user_query(self, query: str, image_path: Optional[str], data_path: Optional[str], chat_history: List[Dict]) -> str:
        """Обрабатывает запрос пользователя."""
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
            return response
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {str(e)}")
            return f"Ошибка обработки запроса: {str(e)}"