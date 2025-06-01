import streamlit as st
from templates.page_config import set_page_config
import os
import pandas as pd
from pathlib import Path
import shutil
from templates.interface import setup_interface
from config import UPLOAD_DIR, DATA_DIR, logger
import asyncio
from graph_workflow import AgentState, create_graph

# Вызов функции настройки страницы как первой команды
set_page_config()

# Функция для создания директорий
def initialize_directories():
    Path(UPLOAD_DIR).mkdir(exist_ok=True)
    Path(DATA_DIR).mkdir(exist_ok=True)

# Инициализация директорий
initialize_directories()

# Инициализация графа
graph = create_graph()

# Callback-функции для интерфейса
def get_current_file(directory):
    """Получает текущий загруженный файл в указанной директории."""
    files = os.listdir(directory)
    return files[0] if files else None

def read_data_preview(file_path):
    """Читает первые 5 строк из файла данных для предварительного просмотра."""
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
    """Очищает указанную директорию от всех файлов."""
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
        st.image(os.path.join(UPLOAD_DIR, current_image), use_container_width=True)
        if st.button("Удалить изображение", key="remove_image"):
            clear_directory(UPLOAD_DIR)
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Изображение не загружено")

def display_data_callback(current_data):
    if current_data:
        data_path = os.path.join(DATA_DIR, current_data)
        preview = read_data_preview(data_path)
        if isinstance(preview, pd.DataFrame):
            st.dataframe(preview)
        else:
            st.text(preview)
        if st.button("Удалить данные", key="remove_data"):
            clear_directory(DATA_DIR)
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Данные не загружены")

async def run_graph(state):
    """Запускает граф асинхронно."""
    return await graph.ainvoke(state)

def chat_callback(chat_container):
    # Инициализация состояния чата
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Автоматический анализ при наличии обоих файлов
    if get_current_file(UPLOAD_DIR) and get_current_file(DATA_DIR):
        image_path = os.path.join(UPLOAD_DIR, get_current_file(UPLOAD_DIR))
        data_path = os.path.join(DATA_DIR, get_current_file(DATA_DIR))
        if not st.session_state.chat_history:
            state = AgentState(
                image_path=image_path,
                data_path=data_path,
                chat_history=st.session_state.chat_history,
                dash_features=None,
                domain_features=None,
                ts_features=None,
                final_annotation=None,
                user_query=None,
                response=None
            )
            result = asyncio.run(run_graph(state))
            if result["final_annotation"]:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": result["final_annotation"]}
                )
                logger.info(f"Создана начальная аннотация: {result['final_annotation']}")

    # Отображение истории чата
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Ввод пользователя
    user_input = st.chat_input("Задайте вопрос о дашборде...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        image_path = os.path.join(UPLOAD_DIR, get_current_file(UPLOAD_DIR)) if get_current_file(UPLOAD_DIR) else None
        data_path = os.path.join(DATA_DIR, get_current_file(DATA_DIR)) if get_current_file(DATA_DIR) else None
        state = AgentState(
            image_path=image_path,
            data_path=data_path,
            chat_history=st.session_state.chat_history,
            user_query=user_input,
            dash_features=None,
            domain_features=None,
            ts_features=None,
            final_annotation=None,
            response=None
        )
        result = asyncio.run(run_graph(state))
        if result["response"]:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["response"]}
            )
        st.rerun()

# Настройка пользовательского интерфейса
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