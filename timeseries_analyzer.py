import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from config import logger, client
import json
import re
import base64
import tempfile
import os

class TimeSeriesAnalyzer:
    def read_data(self, file_path: Path) -> Tuple[Optional[pd.DataFrame], str]:
        """Читает данные временного ряда с подробной проверкой."""
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
                logger.info(f"CSV файл прочитан: {file_path}, размер: {df.shape}")
            elif file_path.suffix in ['.xlsx', '.xls']:
                xls = pd.ExcelFile(file_path, engine='openpyxl')
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    if not df.empty and len(df.columns) >= 2:
                        if pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                            logger.info(f"Данные Excel найдены на листе '{sheet_name}', размер: {df.shape}")
                            return df, f"Данные найдены на листе '{sheet_name}'"
                return None, "Подходящие данные в файле Excel не найдены"
            else:
                return None, f"Неподдерживаемый формат файла: {file_path.suffix}"
            if df.empty:
                return None, "Таблица данных пуста"
            return df, "Данные успешно прочитаны"
        except Exception as e:
            logger.error(f"Ошибка чтения {file_path}: {str(e)}")
            return None, f"Ошибка чтения файла: {str(e)}"

    def analyze_time_series_mathematical(self, df: pd.DataFrame) -> Dict:
        """Анализирует временной ряд математически и извлекает ключевые характеристики."""
        features = {}
        numeric_cols = df.select_dtypes(include=['number']).columns

        if len(numeric_cols) == 0:
            logger.error("Числовые столбцы в таблице данных не найдены")
            return {"error": "Числовые столбцы не найдены"}

        col = numeric_cols[0]
        data = df[col]

        # Определение тренда
        trend = np.polyfit(range(len(data)), data, 1)[0]
        features["trend"] = "восходящий" if trend > 0 else "нисходящий" if trend < 0 else "стабильный"

        # Определение сезонности
        autocorr = data.autocorr(lag=1)
        features["seasonality"] = "присутствует" if abs(autocorr) > 0.3 else "не обнаружена"

        # Определение аномалий с датами
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        anomalies = data[(data < lower_bound) | (data > upper_bound)]

        # Проверяем наличие столбца с датами (предполагаем, что это первый столбец)
        date_col = df.columns[0] if len(df.columns) > 1 and pd.api.types.is_datetime64_any_dtype(df[df.columns[0]]) else None
        anomalies_list = []
        if len(anomalies) > 0:
            for idx in anomalies.index:
                value = anomalies.loc[idx]
                date = df.loc[idx, date_col].strftime('%Y-%m-%d') if date_col else "неизвестно"
                anomalies_list.append({"value": round(float(value), 2), "date": date})
        features["anomalies"] = anomalies_list

        logger.info(f"Характеристики временного ряда (математический): {features}")
        return features

    def analyze_time_series_llm(self, df: pd.DataFrame) -> Dict:
        """Анализирует временной ряд с помощью LLM, передавая данные как CSV-файл в base64."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            logger.error("Числовые столбцы в таблице данных не найдены")
            return {"error": "Числовые столбцы не найдены"}

        # Подготовка данных для сохранения в CSV
        col = numeric_cols[0]
        date_col = df.columns[0] if len(df.columns) > 1 and pd.api.types.is_datetime64_any_dtype(df[df.columns[0]]) else None
        temp_df = pd.DataFrame({
            "Дата": [row[date_col].strftime('%Y-%m-%d') if date_col else f"Запись {idx}" for idx, row in df.iterrows()],
            "Значение": [round(float(row[col]), 2) for _, row in df.iterrows()]
        })

        # Логируем первые несколько строк данных для отладки
        logger.info(f"Первые 5 строк данных для LLM:\n{temp_df.head().to_string()}")

        # Создаем временный CSV-файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
            temp_df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name

        try:
            # Кодируем CSV-файл в base64
            with open(temp_file_path, "rb") as file:
                encoded_csv = base64.b64encode(file.read()).decode("utf-8")
            logger.info(f"CSV-файл закодирован в base64: {temp_file_path}")

            # Формируем промпт для LLM
            prompt = f"""Ты успешный аналитик временных рядов.
Проанализируй временной ряд, представленный в CSV-файле (формат: Дата,Значение), закодированном в base64:
```
{encoded_csv}
```
Извлеки следующие характеристики и верни их в формате JSON:
- trend: тренд временного ряда (восходящий, нисходящий, стабильный)
- seasonality: наличие сезонности (присутствует, отсутствует)
- anomalies_description: текстовое описание аномалий, включая их даты, значения и предполагаемые причины (например, "Аномалия 5000 на 2023-06-15, возможно, из-за рыночного сбоя")
- min_value: минимальное значение и предполагаемая дата (например, "<число> на ГГГГ-ММ-ДД")
- max_value: максимальное значение и предполагаемая дата (например, "<число> на ГГГГ-ММ-ДД")
Если аномалии отсутствуют, укажи в anomalies_description: "Аномалии не обнаружены".
Если какие-то данные не удается определить, укажи "неизвестно".
Верни результат в формате JSON, заключенном в ```json ```.
Инструкция: Декодируй base64 в CSV, проанализируй данные, определи аномалии и предположи возможные причины, основываясь на контексте данных. Используй только значения из CSV.
"""

            logger.info(f"Промпт для LLM:\n{prompt[:500]}...")  # Логируем начало промпта для отладки

            try:
                # Отправляем запрос к LLM
                response = client.chat.completions.create(
                    model="openai.gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.5,
                    stream=False
                )
                content = response.choices[0].message.content
                logger.info(f"Сырой ответ от LLM:\n{content}")

                # Извлекаем JSON из markdown
                json_pattern = r"```json\s*([\s\S]*?)\s*```"
                match = re.search(json_pattern, content)
                if match:
                    json_content = match.group(1).strip()
                    try:
                        result = json.loads(json_content)
                        logger.info(f"Характеристики временного ряда (LLM): {result}")
                        return result
                    except json.JSONDecodeError:
                        logger.error(f"Некорректный JSON от LLM: {json_content}")
                        return {"error": f"Некорректные данные от LLM: {json_content}"}
                else:
                    logger.error(f"JSON не найден в ответе LLM: {content}")
                    return {"error": f"JSON не найден в ответе LLM: {content}"}
            except Exception as e:
                logger.error(f"Ошибка анализа временного ряда через LLM: {str(e)}")
                return {"error": f"Ошибка анализа: {str(e)}"}
        finally:
            # Удаляем временный файл
            try:
                os.unlink(temp_file_path)
                logger.info(f"Временный файл удален: {temp_file_path}")
            except Exception as e:
                logger.error(f"Ошибка при удалении временного файла {temp_file_path}: {str(e)}")

    def analyze_time_series(self, df: pd.DataFrame) -> Dict:
        """Анализирует временной ряд, возвращая результаты обоих методов."""
        mathematical_features = self.analyze_time_series_mathematical(df)
        llm_features = self.analyze_time_series_llm(df)
        return {
            "mathematical": mathematical_features,
            "llm": llm_features
        }