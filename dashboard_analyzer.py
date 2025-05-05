import json
from config import client, logger

class DashboardAnalyzer:
    def analyze_dashboard(self, image_path: str) -> dict:
        """Эмулирует анализ изображения дашборда через LLM."""
        prompt = f"""Analyze the dashboard image at {image_path}. Extract key metrics and visual patterns (e.g., graphs, KPIs). Provide a JSON with:
- main_metric: primary KPI or value
- graph_type: type of graph (line, bar, etc.)
- trend: visual trend (upward, downward, stable)
"""
        try:
            response = client.chat.completions.create(
                model="openai.gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.5,
                stream=False
            )
            content = response.choices[0].message.content
            try:
                result = json.loads(content) if content.startswith("{") else {
                    "main_metric": "unknown",
                    "graph_type": "unknown",
                    "trend": "unknown"
                }
                logger.info(f"Dashboard features: {result}")
                return result
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from LLM: {content}")
                return {"error": "Invalid dashboard data from LLM"}
        except Exception as e:
            logger.error(f"Error analyzing dashboard image: {str(e)}")
            return {"error": f"Image analysis failed: {str(e)}"}