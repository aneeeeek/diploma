import streamlit as st
from templates.page_config import set_page_config
import os
import pandas as pd
from pathlib import Path
import shutil
from templates.interface import setup_interface
from config import UPLOAD_DIR, DATA_DIR, ALLOWED_IMAGE_EXTENSIONS, ALLOWED_DATA_EXTENSIONS, logger
from dashboard_analyzer import DashboardAnalyzer
from timeseries_analyzer import TimeSeriesAnalyzer
from domain_specific_analyzer import DomainSpecificAnalyzer
from chat_agent import ChatAgent

# Вызываем set_page_config как первую команду
set_page_config()

# Функция для создания директорий (вызывается позже)
def initialize_directories():
    Path(UPLOAD_DIR).mkdir(exist_ok=True)
    Path(DATA_DIR).mkdir(exist_ok=True)

# Инициализация агентов и директорий
initialize_directories()
dashboard_analyzer = DashboardAnalyzer()
timeseries_analyzer = TimeSeriesAnalyzer()
domain_specific_analyzer = DomainSpecificAnalyzer(domain="finance")
chat_agent = ChatAgent()

def get_current_file(directory):
    """Получаем текущий загруженный файл"""
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

def clear_directory(directory):
    """Очищает директорию"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            st.error(f'Ошибка при удалении {file_path}: {e}')
            logger.error(f'Ошибка при удалении {file_path}: {e}')

# Callback-функции для интерфейса
def upload_image_callback(uploaded_image):
    clear_directory(UPLOAD_DIR)
    with open(os.path.join(UPLOAD_DIR, uploaded_image.name), "wb") as f:
        f.write(uploaded_image.getbuffer())
    st.session_state.chat_history = []
    st.rerun()

def upload_data_callback(uploaded_data):
    clear_directory(DATA_DIR)
    with open(os.path.join(DATA_DIR, uploaded_data.name), "wb") as f:
        f.write(uploaded_data.getbuffer())
    st.session_state.chat_history = []
    st.rerun()

def display_image_callback(current_image):
    if current_image:
        st.image(os.path.join(UPLOAD_DIR, current_image), use_column_width=True)
        if st.button("Remove Image", key="remove_image"):
            clear_directory(UPLOAD_DIR)
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No image uploaded")

def display_data_callback(current_data):
    if current_data:
        data_path = os.path.join(DATA_DIR, current_data)
        preview = read_data_preview(data_path)
        if isinstance(preview, pd.DataFrame):
            st.dataframe(preview)
        else:
            st.text(preview)
        if st.button("Remove Data", key="remove_data"):
            clear_directory(DATA_DIR)
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No data uploaded")

def chat_callback(chat_container):
    # Инициализация состояния чата
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Автоматический анализ при наличии обоих файлов
    if get_current_file(UPLOAD_DIR) and get_current_file(DATA_DIR):
        image_path = os.path.join(UPLOAD_DIR, get_current_file(UPLOAD_DIR))
        data_path = os.path.join(DATA_DIR, get_current_file(DATA_DIR))
        if not st.session_state.chat_history:
            df, message = timeseries_analyzer.read_data(Path(data_path))
            if df is None:
                st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {message}"})
            else:
                ts_features = timeseries_analyzer.analyze_time_series(df)
                dash_features = dashboard_analyzer.analyze_dashboard(image_path)
                general_annotation = chat_agent.generate_general_annotation(ts_features, dash_features)
                domain_annotation = domain_specific_analyzer.adapt_to_domain(general_annotation)
                final_annotation = chat_agent.review_annotation(domain_annotation, ts_features, dash_features)
                st.session_state.chat_history.append({"role": "assistant", "content": final_annotation})
                logger.info(f"Initial annotation generated: {final_annotation}")

    # Отображение истории чата
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Ввод пользователя
    user_input = st.chat_input("Ask a question about the dashboard...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        image_path = os.path.join(UPLOAD_DIR, get_current_file(UPLOAD_DIR)) if get_current_file(UPLOAD_DIR) else None
        data_path = os.path.join(DATA_DIR, get_current_file(DATA_DIR)) if get_current_file(DATA_DIR) else None
        response = chat_agent.process_user_query(user_input, image_path, data_path, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

# Настройка интерфейса
setup_interface(
    upload_image_callback=upload_image_callback,
    upload_data_callback=upload_data_callback,
    display_image_callback=display_image_callback,
    display_data_callback=display_data_callback,
    chat_callback=chat_callback,
    get_current_image=lambda: get_current_file(UPLOAD_DIR),
    get_current_data=lambda: get_current_file(DATA_DIR),
    clear_directory_callback=clear_directory
)