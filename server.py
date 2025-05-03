import streamlit as st
import os
import pandas as pd
from pathlib import Path
import shutil

# Конфигурация
UPLOAD_DIR = "uploads"
DATA_DIR = "data"
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_DATA_EXTENSIONS = {'csv', 'txt', 'xlsx'}

# Создаем директории, если их нет
Path(UPLOAD_DIR).mkdir(exist_ok=True)
Path(DATA_DIR).mkdir(exist_ok=True)


def get_current_file(directory):
    """Получаем текущий загруженный файл (если есть)"""
    files = os.listdir(directory)
    return files[0] if files else None


def read_data_preview(file_path):
    """Читаем первые 5 строк из файла данных"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, nrows=5)
        return df
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, nrows=5)
        return df
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            lines = [next(f) for _ in range(5)]
        return "\n".join(lines)
    return ""


def check_uploaded_files():
    """Проверяем загруженные файлы и возвращаем их пути"""
    image_file = get_current_file(UPLOAD_DIR)
    data_file = get_current_file(DATA_DIR)

    if image_file and data_file:
        return f"Я вижу пути...\nИзображение: {os.path.abspath(os.path.join(UPLOAD_DIR, image_file))}\nДанные: {os.path.abspath(os.path.join(DATA_DIR, data_file))}"
    else:
        missing = []
        if not image_file:
            missing.append("изображение")
        if not data_file:
            missing.append("данные")
        return f"Не хватает: {', '.join(missing)}"


def clear_directory(directory):
    """Очищает указанную директорию"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            st.error(f'Ошибка при удалении {file_path}: {e}')


# Настройка страницы
st.set_page_config(layout="wide")
st.title("Загрузка изображений и данных")

# Создаем две колонки
col1, col2 = st.columns(2)

# Панель для изображения
with col1:
    st.header("Изображение")
    current_image = get_current_file(UPLOAD_DIR)

    if current_image:
        st.image(os.path.join(UPLOAD_DIR, current_image), use_column_width=True)
        if st.button("Удалить изображение", key="remove_image"):
            clear_directory(UPLOAD_DIR)
            st.experimental_rerun()
    else:
        st.info("Изображение не загружено")

    uploaded_image = st.file_uploader(
        "Выберите изображение",
        type=ALLOWED_IMAGE_EXTENSIONS,
        key="image_uploader",
        accept_multiple_files=False
    )

    if uploaded_image is not None:
        clear_directory(UPLOAD_DIR)
        with open(os.path.join(UPLOAD_DIR, uploaded_image.name), "wb") as f:
            f.write(uploaded_image.getbuffer())
        st.experimental_rerun()

# Панель для данных
with col2:
    st.header("Данные")
    current_data = get_current_file(DATA_DIR)

    if current_data:
        data_path = os.path.join(DATA_DIR, current_data)
        preview = read_data_preview(data_path)

        if isinstance(preview, pd.DataFrame):
            st.dataframe(preview)
        else:
            st.text(preview)

        if st.button("Удалить данные", key="remove_data"):
            clear_directory(DATA_DIR)
            st.experimental_rerun()
    else:
        st.info("Данные не загружены")

    uploaded_data = st.file_uploader(
        "Выберите файл с данными",
        type=ALLOWED_DATA_EXTENSIONS,
        key="data_uploader",
        accept_multiple_files=False
    )

    if uploaded_data is not None:
        clear_directory(DATA_DIR)
        with open(os.path.join(DATA_DIR, uploaded_data.name), "wb") as f:
            f.write(uploaded_data.getbuffer())
        st.experimental_rerun()

# Кнопка проверки путей
if st.button("Проверить пути к файлам", key="check_paths"):
    result = check_uploaded_files()
    st.info(result)