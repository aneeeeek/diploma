import json
from typing import List, Dict, Optional
from config import client, logger

class ChatAgent:
    def generate_general_annotation(self, ts_features: Dict, dash_features: Dict) -> str:
        """Генерирует общую аннотацию."""
        if "error" in ts_features or "error" in dash_features:
            error_msg = f"Error: Unable to generate annotation due to invalid data. TS: {ts_features.get('error', '')}, Dash: {dash_features.get('error', '')}"
            logger.error(error_msg)
            return error_msg

        annotation = []
        annotation.append(f"The dashboard shows a {dash_features.get('graph_type', 'unknown')} graph with a {dash_features.get('trend', 'unknown')} trend.")
        annotation.append(f"Main metric: {dash_features.get('main_metric', 'unknown')}.")
        annotation.append(f"Time series analysis indicates a {ts_features.get('trend', 'unknown')} trend.")
        if ts_features.get('seasonality') == "present":
            annotation.append("Seasonal patterns are detected.")
        if ts_features.get('anomalies', 0) > 0:
            annotation.append(f"Found {ts_features.get('anomalies')} anomalies.")
        annotation.append(f"Support level: {ts_features.get('support', 'unknown')}, Resistance level: {ts_features.get('resistance', 'unknown')}.")

        return " ".join(annotation)

    def review_annotation(self, annotation: str, ts_features: Dict, dash_features: Dict) -> str:
        """Проверяет аннотацию на согласованность."""
        prompt = f"""Review the annotation for consistency with the data:
Annotation: {annotation}
Time series features: {json.dumps(ts_features)}
Dashboard features: {json.dumps(dash_features)}
Ensure the annotation is clear, complete, and matches the data. Return the revised annotation."""
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
            logger.error(f"Error in annotation review: {str(e)}")
            return annotation

    def process_user_query(self, query: str, image_path: Optional[str], data_path: Optional[str], chat_history: List[Dict]) -> str:
        """Обрабатывает вопрос пользователя."""
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["trend", "pattern", "seasonality", "anomaly"]):
            agent = "time_series"
        elif any(keyword in query_lower for keyword in ["metric", "kpi", "graph", "dashboard"]):
            agent = "dashboard"
        else:
            agent = "general"

        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
        prompt = f"""User query: {query}
Context: {context}
Agent: {agent}
Provide a concise answer based on the dashboard and time series data. Use domain-specific terms for finance."""

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
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"