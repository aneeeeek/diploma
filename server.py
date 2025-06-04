import streamlit as st
from templates.page_config import set_page_config
import os
import pandas as pd
from pathlib import Path
import shutil
from templates.interface import setup_interface
from config import UPLOAD_DIR, DATA_DIR, logger, ALLOWED_IMAGE_EXTENSIONS
import asyncio
from graph_workflow import AgentState, create_graph
from PIL import Image
import io

logger.info("Начало инициализации приложения")

try:
    set_page_config()
    logger.info("set_page_config выполнен")
except Exception as e:
    logger.error(f"Ошибка в set_page_config: {str(e)}")
    raise

def initialize_directories():
    try:
        Path(UPLOAD_DIR).mkdir(exist_ok=True)
        Path(DATA_DIR).mkdir(exist_ok=True)
        logger.info("Директории инициализированы")
    except Exception as e:
        logger.error(f"Ошибка при инициализации директорий: {str(e)}")
        raise

initialize_directories()

try:
    graph = create_graph()
    logger.info("Граф создан")
except Exception as e:
    logger.error(f"Ошибка при создании графа: {str(e)}")
    raise

def get_current_file(directory):
    try:
        files = os.listdir(directory)
        logger.info(f"Получены файлы в {directory}: {files}")
        return files[0] if files else None
    except Exception as e:
        logger.error(f"Ошибка в get_current_file для {directory}: {str(e)}")
        return None

def read_data_preview(file_path):
    try:
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
    except Exception as e:
        logger.error(f"Ошибка в read_data_preview для {file_path}: {str(e)}")
        return ""

def clear_directory(directory):
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        logger.info(f"Директория {directory} очищена")
    except Exception as e:
        st.error(f'Ошибка при очистке {directory}: {e}')
        logger.error(f'Ошибка при очистке {directory}: {e}')

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def upload_image_callback(uploaded_image):
    try:
        if uploaded_image.size > MAX_FILE_SIZE:
            st.error(f"Ошибка: Размер файла {uploaded_image.name} превышает допустимый лимит (10 МБ).")
            logger.error(f"Слишком большой файл: {uploaded_image.name}, размер: {uploaded_image.size} байт")
            return
        if Path(uploaded_image.name).suffix[1:].lower() not in ALLOWED_IMAGE_EXTENSIONS:
            st.error(f"Ошибка: Формат файла {uploaded_image.name} не поддерживается. Разрешены: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}")
            logger.error(f"Неподдерживаемый формат файла: {uploaded_image.name}")
            return
        clear_directory(UPLOAD_DIR)
        image_data = uploaded_image.read()
        img = Image.open(io.BytesIO(image_data))
        img.verify()
        img.close()
        file_path = os.path.join(UPLOAD_DIR, uploaded_image.name)
        with open(file_path, "wb") as f:
            f.write(image_data)
        if not os.path.exists(file_path):
            st.error(f"Ошибка: Не удалось сохранить изображение {uploaded_image.name}")
            logger.error(f"Не удалось сохранить файл: {file_path}")
            return
        st.session_state.chat_history = []
        st.session_state.last_image = uploaded_image.name
        st.session_state.annotation_triggered = True
        st.session_state.rerun_count = 0
        st.session_state.image_uploaded = True
        logger.info(f"Изображение загружено: {uploaded_image.name}")
    except Exception as e:
        st.error(f"Ошибка: Загруженный файл не является изображением: {str(e)}")
        logger.error(f"Ошибка валидации изображения {uploaded_image.name}: {str(e)}")

def upload_data_callback(uploaded_data):
    try:
        clear_directory(DATA_DIR)
        with open(os.path.join(DATA_DIR, uploaded_data.name), "wb") as f:
            f.write(uploaded_data.getbuffer())
        st.session_state.chat_history = []
        st.session_state.last_data = uploaded_data.name
        st.session_state.annotation_triggered = True
        st.session_state.rerun_count = 0
        st.session_state.data_uploaded = True
        logger.info(f"Данные загружены: {uploaded_data.name}")
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        logger.error(f"Ошибка при загрузке данных {uploaded_data.name}: {str(e)}")

def display_image_callback(current_image):
    try:
        if current_image:
            file_path = os.path.join(UPLOAD_DIR, current_image)
            if os.path.exists(file_path):
                st.image(file_path, use_container_width=True)
                logger.info(f"Отображено изображение: {file_path}")
            else:
                st.error(f"Ошибка: Файл {current_image} не найден в {UPLOAD_DIR}")
                logger.error(f"Файл не найден: {file_path}")
            if st.button("Удалить изображение", key="remove_image"):
                clear_directory(UPLOAD_DIR)
                st.session_state.chat_history = []
                st.session_state.last_image = None
                st.session_state.annotation_triggered = True
                st.session_state.rerun_count = 0
                st.session_state.image_uploaded = False
                logger.info("Изображение удалено пользователем")
                st.rerun()
        else:
            st.info("Изображение не загружено")
    except Exception as e:
        logger.error(f"Ошибка в display_image_callback: {str(e)}")

def display_data_callback(current_data):
    try:
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
                st.session_state.last_data = None
                st.session_state.annotation_triggered = True
                st.session_state.rerun_count = 0
                st.session_state.data_uploaded = False
                logger.info("Данные удалены пользователем")
                st.rerun()
        else:
            st.info("Данные не загружены")
    except Exception as e:
        logger.error(f"Ошибка в display_data_callback: {str(e)}")

async def run_graph(state):
    try:
        result = await graph.ainvoke(state)
        logger.info("Граф успешно выполнен")
        return result
    except Exception as e:
        logger.error(f"Ошибка в run_graph: {str(e)}")
        return {"final_annotation": f"Ошибка выполнения графа: {str(e)}"}

def chat_callback(chat_container):
    try:
        logger.info("Начало chat_callback")
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'last_image' not in st.session_state:
            st.session_state.last_image = None
        if 'last_data' not in st.session_state:
            st.session_state.last_data = None
        if 'annotation_triggered' not in st.session_state:
            st.session_state.annotation_triggered = False
        if 'rerun_count' not in st.session_state:
            st.session_state.rerun_count = 0
        if 'image_uploaded' not in st.session_state:
            st.session_state.image_uploaded = False
        if 'data_uploaded' not in st.session_state:
            st.session_state.data_uploaded = False
        if 'reset_uploaders' not in st.session_state:
            st.session_state.reset_uploaders = False
        if 'needs_rerun' not in st.session_state:
            st.session_state.needs_rerun = False

        current_image = get_current_file(UPLOAD_DIR)
        current_data = get_current_file(DATA_DIR)
        logger.info(f"chat_callback: current_image={current_image}, current_data={current_data}, annotation_triggered={st.session_state.annotation_triggered}, rerun_count={st.session_state.rerun_count}, image_uploaded={st.session_state.image_uploaded}, data_uploaded={st.session_state.data_uploaded}, reset_uploaders={st.session_state.reset_uploaders}, needs_rerun={st.session_state.needs_rerun}")

        # Проверяем пользовательский ввод
        user_input = st.chat_input("Задайте вопрос о дашборде...", key="chat_input")
        if user_input:
            logger.info(f"Получен пользовательский ввод: {user_input}")
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            image_path = os.path.join(UPLOAD_DIR, current_image) if current_image else None
            data_path = os.path.join(DATA_DIR, current_data) if current_data else None
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
                if "Слишком большой объем" in result["response"]:
                    st.error(result["response"])
                    logger.error(f"Ошибка в ответе: {result['response']}")
                else:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": result["response"]}
                    )
                    logger.info(f"Сгенерирован ответ: {result['response']}")
            st.session_state.rerun_count = 0
            st.session_state.reset_uploaders = True
            st.session_state.needs_rerun = True

        # Синхронизируем last_image
        if current_image and current_image != st.session_state.last_image:
            st.session_state.last_image = current_image
            logger.info(f"Синхронизировано last_image: {current_image}")

        # Сбрасываем file_uploader'ы, если установлен флаг
        if st.session_state.reset_uploaders:
            if "image_uploader" in st.session_state:
                del st.session_state["image_uploader"]
                logger.info("Сброшен image_uploader")
            if "data_uploader" in st.session_state:
                del st.session_state["data_uploader"]
                logger.info("Сброшен data_uploader")
            st.session_state.reset_uploaders = False
            st.session_state.needs_rerun = True

        # Устанавливаем annotation_triggered для начальной генерации
        if current_image and current_data and not st.session_state.chat_history and not st.session_state.annotation_triggered:
            st.session_state.annotation_triggered = True
            logger.info("Установлен annotation_triggered для начальной генерации аннотации")

        # Обрабатываем аннотацию
        if current_image and current_data and st.session_state.annotation_triggered:
            if st.session_state.rerun_count >= 3:
                st.error("Слишком много перезапусков. Пожалуйста, обновите страницу.")
                logger.error("Превышен лимит перезапусков")
                st.session_state.annotation_triggered = False
                st.session_state.image_uploaded = False
                st.session_state.data_uploaded = False
                st.session_state.reset_uploaders = True
                st.session_state.needs_rerun = True
            else:
                st.session_state.rerun_count += 1
                image_path = os.path.join(UPLOAD_DIR, current_image) if current_image else None
                data_path = os.path.join(DATA_DIR, current_data) if current_data else None
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
                    if "Слишком большой объем" in result["final_annotation"]:
                        st.error(result["final_annotation"])
                        logger.error(f"Ошибка в аннотации: {result['final_annotation']}")
                    else:
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": result["final_annotation"]}
                        )
                        logger.info(f"Создана начальная аннотация: {result['final_annotation']}")
                st.session_state.annotation_triggered = False
                st.session_state.image_uploaded = False
                st.session_state.data_uploaded = False
                st.session_state.last_image = current_image
                st.session_state.last_data = current_data
                st.session_state.rerun_count = 0
                st.session_state.reset_uploaders = True
                st.session_state.needs_rerun = True

        # Обрабатываем загрузку
        if st.session_state.image_uploaded or st.session_state.data_uploaded:
            st.session_state.image_uploaded = False
            st.session_state.data_uploaded = False
            st.session_state.reset_uploaders = True
            st.session_state.needs_rerun = True

        # Отображаем чат
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            logger.info(f"Чат отображен, chat_history: {len(st.session_state.chat_history)} сообщений")

        # Выполняем rerun, если нужно
        if st.session_state.needs_rerun:
            st.session_state.needs_rerun = False
            logger.info("Выполняется st.rerun()")
            st.rerun()

        # Отладочный вывод
        # st.write(f"Debug: {st.session_state}")

    except Exception as e:
        logger.error(f"Ошибка в chat_callback: {str(e)}")
        # Отображаем чат даже при ошибке
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            logger.info(f"Чат отображен при ошибке, chat_history: {len(st.session_state.chat_history)} сообщений")

try:
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
    logger.info("Интерфейс успешно настроен")
except Exception as e:
    logger.error(f"Ошибка при настройке интерфейса: {str(e)}")
    raise