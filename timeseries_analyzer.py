import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from config import logger

class TimeSeriesAnalyzer:
    def read_data(self, file_path: Path) -> Tuple[Optional[pd.DataFrame], str]:
        """Читает данные временного ряда с детальной проверкой."""
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
                logger.info(f"CSV file read: {file_path}, shape: {df.shape}")
            elif file_path.suffix in ['.xlsx', '.xls']:
                xls = pd.ExcelFile(file_path, engine='openpyxl')
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    if not df.empty and len(df.columns) >= 2:
                        if pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                            logger.info(f"Excel data found on sheet '{sheet_name}', shape: {df.shape}")
                            return df, f"Data found on sheet '{sheet_name}'"
                return None, "No suitable data found in Excel file"
            else:
                return None, f"Unsupported file format: {file_path.suffix}"
            if df.empty:
                return None, "Data frame is empty"
            return df, "Data read successfully"
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return None, f"Error reading file: {str(e)}"

    def analyze_time_series(self, df: pd.DataFrame) -> Dict:
        """Анализирует временной ряд."""
        features = {}
        numeric_cols = df.select_dtypes(include=['number']).columns

        if len(numeric_cols) == 0:
            logger.error("No numeric columns found in data frame")
            return {"error": "No numeric columns found"}

        col = numeric_cols[0]
        data = df[col]

        # Тренд
        trend = np.polyfit(range(len(data)), data, 1)[0]
        features["trend"] = "upward" if trend > 0 else "downward" if trend < 0 else "stable"

        # Сезонность
        autocorr = data.autocorr(lag=1)
        features["seasonality"] = "present" if abs(autocorr) > 0.3 else "not detected"

        # Аномалии
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        anomalies = data[(data < lower_bound) | (data > upper_bound)]
        features["anomalies"] = len(anomalies)

        # Уровни поддержки/сопротивления
        features["support"] = round(q1, 2)
        features["resistance"] = round(q3, 2)

        logger.info(f"Time series features: {features}")
        return features