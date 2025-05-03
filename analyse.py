# from openai import OpenAI
# import os
#
# client = OpenAI(
#     api_key=os.getenv("API_KEY","sk-eae1582d53c2402b9d7be1f1a882c79f"),
#     base_url="https://llm.glowbyteconsulting.com/api"
# )
# response = client.chat.completions.create(
#     model="openai.gpt-4o-mini",
#     messages=[
#         {"role": "user", "content": "Why is the sky blue?"}
#     ],
#     stream=False
# )
# print("API_KEY:", os.getenv("API_KEY"))
# print(response)

import os
from pathlib import Path
from openai import OpenAI
import pandas as pd
import openpyxl
from typing import Optional, Tuple

# Инициализация клиента API
client = OpenAI(
    api_key=os.getenv("API_KEY", "sk-eae1582d53c2402b9d7be1f1a882c79f"),
    base_url="https://llm.glowbyteconsulting.com/api"
)


def read_financial_data(file_path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """Читает финансовые данные с обработкой ошибок."""
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            return df, "CSV файл успешно прочитан"
        elif file_path.suffix in ['.xlsx', '.xls']:
            # Читаем Excel файл
            xls = pd.ExcelFile(file_path, engine='openpyxl')

            # Проверяем все листы
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                if not df.empty and len(df.columns) >= 2:
                    # Проверяем, есть ли числовые данные во второй колонке
                    if pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                        return df, f"Данные найдены на листе '{sheet_name}'"

            return None, "Не найдены подходящие данные в файле"
        else:
            return None, f"Неподдерживаемый формат файла: {file_path.suffix}"
    except Exception as e:
        return None, f"Ошибка чтения файла: {str(e)}"


def prepare_financial_report(df: pd.DataFrame) -> str:
    """Готовит отчет по финансовым данным."""
    report = []

    # Основные статистики
    report.append("=== ОСНОВНЫЕ СТАТИСТИКИ ===")
    report.append(df.describe().to_string())

    # Анализ аномалий (только для числовых колонок)
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        report.append("\n=== ПОТЕНЦИАЛЬНЫЕ АНОМАЛИИ ===")
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            anomalies = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            report.append(f"\nКолонка '{col}':")
            report.append(f"Границы: [{lower_bound:.2f}, {upper_bound:.2f}]")
            report.append(f"Найдено аномалий: {len(anomalies)}")

    return "\n".join(report)


def analyze_gold_prices(data_path: Path) -> Tuple[Optional[str], str]:
    """Анализирует данные о ценах на золото."""
    df, message = read_financial_data(data_path)
    if df is None:
        return None, message

    try:
        # Подготовка данных для анализа
        data_preview = df.head().to_string()
        report = prepare_financial_report(df)

        prompt = f"""Анализ данных о ценах на золото:

Превью данных:
{data_preview}

{report}

Проанализируй данные и подготовь отчет:
1. Основные характеристики ценового ряда
2. Волатильность цен
3. Ключевые уровни поддержки/сопротивления
4. Рекомендации для трейдеров

Формат: профессиональный отчет на 1-2 абзаца"""

        response = client.chat.completions.create(
            model="openai.gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.5,
            stream=False
        )

        return response.choices[0].message.content, "Анализ выполнен успешно"
    except Exception as e:
        return None, f"Ошибка анализа: {str(e)}"


def main():
    """Основная функция выполнения."""
    data_dir = Path("data")
    data_files = list(data_dir.glob("*"))

    if not data_files:
        print("Ошибка: не найден файл с данными в папке 'data'")
        print("Поместите файл с данными (CSV/XLSX) в папку 'data'")
        return

    data_file = data_files[0]
    print(f"\nАнализируем файл: {data_file.name}")

    # Диагностика файла
    print("\n=== ДИАГНОСТИКА ФАЙЛА ===")
    print(f"Размер: {data_file.stat().st_size} байт")
    print(f"Расширение: {data_file.suffix}")

    # Анализ данных
    print("\n=== ЗАПУСК АНАЛИЗА ===")
    analysis_result, status_message = analyze_gold_prices(data_file)

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТ АНАЛИЗА:")
    print("=" * 50)
    print(f"\nСтатус: {status_message}")

    if analysis_result:
        print("\nАНАЛИТИЧЕСКИЙ ОТЧЕТ:\n")
        print(analysis_result)
    else:
        print("\nНе удалось сгенерировать отчет.")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()