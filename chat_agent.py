import json
from typing import List, Dict, Optional
from config import client, logger

class ChatAgent:
    def generate_general_annotation(self, ts_features: Dict, dash_features: Dict) -> str:
        """Создает общую аннотацию на основе характеристик данных."""
        if "error" in ts_features or "error" in dash_features:
            error_msg = f"Ошибка: Невозможно создать аннотацию из-за некорректных данных. Временной ряд: {ts_features.get('error', '')}, Дашборд: {dash_features.get('error', '')}"
            logger.error(error_msg)
            return error_msg

        annotation = []
        annotation.append(f"Дашборд отображает график типа {dash_features.get('graph_type', 'неизвестный')} с трендом {dash_features.get('trend', 'неизвестный')}.")
        annotation.append(f"Основной показатель: {dash_features.get('main_metric', 'неизвестный')}.")
        annotation.append(f"Анализ временного ряда показывает тренд {ts_features.get('trend', 'неизвестный')}.")
        if ts_features.get('seasonality') == "present":
            annotation.append("Обнаружены сезонные закономерности.")
        if ts_features.get('anomalies', 0) > 0:
            annotation.append(f"Обнаружено {ts_features.get('anomalies')} аномалий.")
        annotation.append(f"Уровень поддержки: {ts_features.get('support', 'неизвестный')}, Уровень сопротивления: {ts_features.get('resistance', 'неизвестный')}.")

        return " ".join(annotation)

    def review_annotation(self, annotation: str, ts_features: Dict, dash_features: Dict) -> str:
        """Проверяет аннотацию на соответствие данным."""
        prompt = f"""Проверьте аннотацию на согласованность с данными:
        Аннотация: {annotation}
        Характеристики временного ряда: {json.dumps(ts_features)}
        Характеристики дашборда: {json.dumps(dash_features)}
        Убедитесь, что аннотация ясна, полна и соответствует данным. Верните исправленную аннотацию."""
        try:
            response = client.chat.completions.create(
                model="openai.gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.5,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Ошибка при проверке аннотации: {str(e)}")
            return annotation

    def process_user_query(self, query: str, image_path: Optional[str], data_path: Optional[str], chat_history: List[Dict]) -> str:
        """Обрабатывает запрос пользователя."""
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["тренд", "закономерность", "сезонность", "аномалия"]):
            agent = "time_series"
        elif any(keyword in query_lower for keyword in ["показатель", "kpi", "график", "дашборд"]):
            agent = "dashboard"
        else:
            agent = "general"

        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
        prompt = f"""Запрос пользователя: {query}
        Контекст: {context}
        Агент: {agent}
        Дайте краткий ответ на основе данных дашборда и временного ряда. Используйте термины, специфичные для финансовой области."""

        try:
            response = client.chat.completions.create(
                model="openai.gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.5,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {str(e)}")
            return f"Ошибка обработки запроса: {str(e)}"