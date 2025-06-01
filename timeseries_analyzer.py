import pandas as pd
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
                logger.info(f"CSV файл прочитан: {file_path}, размер: {df.shape}, колонки: {df.columns.tolist()}")
                logger.info(f"Типы данных колонок:\n{df.dtypes}")
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

    def encode_image(self, image_path: str) -> str:
        """Кодирует изображение в формат base64."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                logger.info(f"Изображение {image_path} закодировано, длина: {len(encoded_string)}")
                return encoded_string
        except Exception as e:
            logger.error(f"Ошибка при кодировании изображения {image_path}: {str(e)}")
            return f"Ошибка: Не удалось закодировать изображение: {str(e)}"

    def encode_data(self, df: pd.DataFrame) -> str:
        """Кодирует данные DataFrame в CSV в формате base64."""
        try:
            csv_buffer = df.to_csv(index=False)
            encoded_csv = base64.b64encode(csv_buffer.encode("utf-8")).decode("utf-8")
            logger.info(f"Данные закодированы, длина: {len(encoded_csv)}")
            return encoded_csv
        except Exception as e:
            logger.error(f"Ошибка при кодировании данных: {str(e)}")
            return f"Ошибка: Не удалось закодировать данные: {str(e)}"

    def analyze_time_series(self, df: pd.DataFrame, image_path: Optional[str], main_metric: str, domain: str) -> Dict:
        """Анализирует временной ряд с учетом изображения, данных, метрики и домена."""
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            logger.error("Числовые столбцы в таблице данных не найдены")
            return {
                "metric": main_metric,
                "domain": domain,
                "trend": "неизвестно",
                "seasonality": "неизвестно",
                "min_value": "неизвестно",
                "max_value": "неизвестно",
                "anomalies": [],
                "hypotheses": "Данные отсутствуют"
            }

        date_col_name = df.columns[0]
        numeric_cols = [col for col in numeric_cols if col != date_col_name]
        if not numeric_cols:
            logger.error("Числовые столбцы для значений не найдены (исключая колонку с датами)")
            return {
                "metric": main_metric,
                "domain": domain,
                "trend": "неизвестно",
                "seasonality": "неизвестно",
                "min_value": "неизвестно",
                "max_value": "неизвестно",
                "anomalies": [],
                "hypotheses": "Числовые данные отсутствуют"
            }
        col = numeric_cols[0]
        logger.info(f"Выбрана числовая колонка для значений: {col}")

        try:
            if pd.api.types.is_numeric_dtype(df[date_col_name]):
                df[date_col_name] = pd.to_datetime(df[date_col_name].astype(int).astype(str) + '-01-01', errors='coerce')
                logger.info(f"Колонка {date_col_name} преобразована в datetime")
            elif df[date_col_name].dtype == 'object':
                if df[date_col_name].str.match(r'Занлись \d+').any():
                    df[date_col_name] = df[date_col_name].str.extract(r'Занлись (\d+)').astype(float).astype(int) + 1945
                    df[date_col_name] = pd.to_datetime(df[date_col_name].astype(str) + '-01-01')
                    logger.info(f"Колонка {date_col_name} преобразована из формата 'Занлись \d+' в datetime")
                else:
                    df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
                    logger.info(f"Колонка {date_col_name} преобразована в datetime (попытка автопреобразования)")
            if pd.api.types.is_datetime64_any_dtype(df[date_col_name]):
                date_col = df[date_col_name]
                logger.info(f"Колонка {date_col_name} успешно установлена как date_col с типом datetime")
            else:
                date_col = None
                logger.warning(f"Колонка {date_col_name} не является datetime после преобразования")
        except Exception as e:
            logger.error(f"Ошибка при преобразовании даты {date_col_name}: {str(e)}")
            date_col = None

        def format_date_for_human(date):
            if pd.isna(date):
                return "неизвестно"
            if isinstance(date, str):
                return date
            year = date.year
            month = date.month
            day = date.day
            if month == 1 and day == 1:
                return f"в {year} году" if "min" in context or "max" in context else f"на {year} год"
            return f"на {day} {'января' if month == 1 else 'февраля' if month == 2 else 'марта' if month == 3 else 'апреля' if month == 4 else 'мая' if month == 5 else 'июня' if month == 6 else 'июля' if month == 7 else 'августа' if month == 8 else 'сентября' if month == 9 else 'октября' if month == 10 else 'ноября' if month == 11 else 'декабря'} {year} года"

        context = {}
        temp_df = pd.DataFrame({
            "Дата": [format_date_for_human(row[date_col_name]) if date_col is not None and pd.notna(row[date_col_name]) else f"Запись {idx}" for idx, row in df.iterrows()],
            "Значение": [round(float(row[col]), 2) if pd.notna(row[col]) else 0.0 for _, row in df.iterrows()]
        })

        min_value = temp_df["Значение"].min()
        max_value = temp_df["Значение"].max()
        min_date = temp_df.loc[temp_df["Значение"].idxmin(), "Дата"]
        max_date = temp_df.loc[temp_df["Значение"].idxmax(), "Дата"]
        min_max_hint = f"Минимальное значение: {min_value} {min_date}, Максимальное значение: {max_value} {max_date}"

        logger.info(f"Первые 5 строк данных для LLM:\n{temp_df.head().to_string()}")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_file:
            temp_df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name

        try:
            encoded_csv = self.encode_data(temp_df)
            logger.info(f"CSV-файл закодирован в base64: {temp_file_path}")
            base64_image = self.encode_image(image_path) if image_path else ""

            prompt_parts = [
                f"Дашборд в области {domain} показывает метрику {main_metric}.",
                "Ты успешный аналитик временных рядов. Проанализируй временной ряд, представленный в CSV-файле (формат: Дата,Значение), закодированном в base64:",
                f"```\n{encoded_csv}\n```",
                f"Изображение дашборда в base64: {'присутствует' if base64_image else 'отсутствует'}",
                f"Подсказка по данным: {min_max_hint}",
                "Извлеки следующие характеристики и верни их в формате JSON:",
                "- metric: метрика временного ряда (используй переданную метрику)",
                "- domain: область дашборда (используй переданную область)",
                "- trend: подробное описание трендов временного ряда (например, 'С начала периода до 1990 года наблюдается восходящий тренд, затем с 1990 по 2000 год тренд стабилизируется', если применимо; если один тренд, опиши его детально)",
                "- seasonality: подробное описание наличия и характера сезонности (например, 'Сезонность присутствует с годовым циклом, с пиками в летние месяцы', если применимо; если отсутствует, укажи это с объяснением)",
                "- min_value: минимальное значение и дата (проверь и скорректируй на основе данных и изображения, например, '500 на 1 мая 1999 года')",
                "- max_value: максимальное значение и дата (проверь и скорректируй на основе данных и изображения, например, '5000 на 15 декабря 2000 года')",
                "- anomalies: список аномалий с подробным описанием (например, [{'value': 5000, 'date': '1 июня 1999 года', 'description': 'Резкий скачок, возможно, связанный с событием X'}])",
                "- hypotheses: подробные гипотезы, объясняющие характеристики временного ряда (например, 'Восходящий тренд с 1980 по 1990 год может быть связан с экономическим ростом, а скачок в 1995 году — с технологическим прорывом'), с объяснением каждого наблюдаемого тренда, пика или скачка",
                "Если аномалии отсутствуют, верни пустой список в anomalies.",
                "Если какие-то данные не удается определить, укажи 'неизвестно'.",
                "Верни результат в формате JSON, заключенном в ```json ```.",
                "Инструкция: Декодируй base64 в CSV, проанализируй данные и изображение, учти метрику, область и подсказку. Скорректируй даты для min_value и max_value в человеко-читаемом формате (например, 'в 1999 году' для года или 'на 1 мая 1999 года' для полной даты). Опиши тренды, сезонность и аномалии максимально детально, включая несколько этапов или пиков, если они есть. Сформируй гипотезы для каждого наблюдаемого явления."
            ]
            prompt = "\n".join(prompt_parts)
            logger.info(f"Полный промпт для LLM (первые 500 символов):\n{prompt[:500]}...")

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
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                } if base64_image and not base64_image.startswith("Ошибка") else {"type": "text", "text": "Изображение отсутствует"}
                            ]
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.5,
                    stream=False
                )
                content = response.choices[0].message.content
                logger.info(f"Сырой ответ от LLM:\n{content}")

                json_pattern = r"```json\s*([\s\S]*?)\s*```"
                match = re.search(json_pattern, content)
                if match:
                    json_content = match.group(1).strip()
                    try:
                        result = json.loads(json_content)
                        result["metric"] = main_metric
                        result["domain"] = domain
                        if "min_value" in result and result["min_value"] != f"{min_value} {min_date}":
                            logger.warning(f"LLM изменил min_value: {result['min_value']} vs {min_value} {min_date}")
                        if "max_value" in result and result["max_value"] != f"{max_value} {max_date}":
                            logger.warning(f"LLM изменил max_value: {result['max_value']} vs {max_value} {max_date}")
                        logger.info(f"Характеристики временного ряда: {result}")
                        return result
                    except json.JSONDecodeError:
                        logger.error(f"Некорректный JSON от LLM: {json_content}")
                        return {
                            "metric": main_metric,
                            "domain": domain,
                            "trend": "неизвестно",
                            "seasonality": "неизвестно",
                            "min_value": f"{min_value} {min_date}",
                            "max_value": f"{max_value} {max_date}",
                            "anomalies": [],
                            "hypotheses": "Некорректные данные от LLM"
                        }
                else:
                    logger.error(f"JSON не найден в ответе LLM: {content}")
                    return {
                        "metric": main_metric,
                        "domain": domain,
                        "trend": "неизвестно",
                        "seasonality": "неизвестно",
                        "min_value": f"{min_value} {min_date}",
                        "max_value": f"{max_value} {max_date}",
                        "anomalies": [],
                        "hypotheses": "JSON не найден в ответе LLM"
                    }
            except Exception as e:
                logger.error(f"Ошибка анализа временного ряда: {str(e)}")
                if "413" in str(e) or "request too large" in str(e).lower():
                    return {
                        "metric": main_metric,
                        "domain": domain,
                        "trend": "неизвестно",
                        "seasonality": "неизвестно",
                        "min_value": f"{min_value} {min_date}",
                        "max_value": f"{max_value} {max_date}",
                        "anomalies": [],
                        "hypotheses": "Слишком большой объем данных или изображения. Пожалуйста, уменьшите размер файла."
                    }
                return {
                    "metric": main_metric,
                    "domain": domain,
                    "trend": "неизвестно",
                    "seasonality": "неизвестно",
                    "min_value": f"{min_value} {min_date}",
                    "max_value": f"{max_value} {max_date}",
                    "anomalies": [],
                    "hypotheses": f"Ошибка анализа: {str(e)}"
                }
        finally:
            try:
                os.unlink(temp_file_path)
                logger.info(f"Временный файл удален: {temp_file_path}")
            except Exception as e:
                logger.error(f"Ошибка при удалении временного файла {temp_file_path}: {str(e)}")