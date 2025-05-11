import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from config import logger

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

    def analyze_time_series(self, df: pd.DataFrame) -> Dict:
        """Анализирует временной ряд и извлекает ключевые характеристики."""
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

        # Определение аномалий
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        anomalies = data[(data < lower_bound) | (data > upper_bound)]
        features["anomalies"] = len(anomalies)

        # Уровни поддержки и сопротивления
        features["support"] = round(q1, 2)
        features["resistance"] = round(q3, 2)

        logger.info(f"Характеристики временного ряда: {features}")
        return features