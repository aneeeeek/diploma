import tempfile

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

from timeseries_analyzer import TimeSeriesAnalyzer

logger.info("Начало инициализации приложения")

# установка параметров страницы интерфейса
try:
    set_page_config()
except Exception as e:
    logger.error(f"Ошибка в set_page_config: {str(e)}")
    raise

# Инициализация директорий проекта
def initialize_directories():
    try:
        Path(UPLOAD_DIR).mkdir(exist_ok=True)
        Path(DATA_DIR).mkdir(exist_ok=True)
    except Exception as e:
        logger.error(f"Ошибка при инициализации директорий: {str(e)}")
        raise

initialize_directories()

# создание и компиляция графа
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
        st.session_state.last_image = uploaded_image.name
        st.session_state.run_triggered = False
        st.session_state.rerun_count = 0
        st.session_state.image_uploaded = True
        st.session_state.processing = False
        logger.info(f"Изображение загружено: {uploaded_image.name}")
    except Exception as e:
        st.error(f"Ошибка: Загруженный файл не является изображением: {str(e)}")
        logger.error(f"Ошибка валидации изображения {uploaded_image.name}: {str(e)}")

def upload_data_callback(uploaded_data):
    try:
        # Проверяем данные перед сохранением
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_data.name, mode='wb') as temp_file:
            temp_file.write(uploaded_data.getbuffer())
            temp_file_path = temp_file.name

        # Проверяем данные с помощью TimeSeriesAnalyzer
        timeseries_analyzer = TimeSeriesAnalyzer()
        df, message = timeseries_analyzer.read_data(Path(temp_file_path))

        # Удаляем временный файл
        try:
            os.unlink(temp_file_path)
            logger.info(f"Временный файл удален: {temp_file_path}")
        except Exception as e:
            logger.error(f"Ошибка при удалении временного файла {temp_file_path}: {str(e)}")

        if df is None:
            st.error(message)  # Отображаем сообщение об ошибке
            logger.error(f"Ошибка валидации данных {uploaded_data.name}: {message}")
            return

        clear_directory(DATA_DIR)
        with open(os.path.join(DATA_DIR, uploaded_data.name), "wb") as f:
            f.write(uploaded_data.getbuffer())
        st.session_state.last_data = uploaded_data.name
        st.session_state.run_triggered = False
        st.session_state.rerun_count = 0
        st.session_state.data_uploaded = True
        st.session_state.processing = False
        st.success("Данные успешно загружены!")
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
                st.session_state.chat_history = []  # Очищаем историю чата
                st.session_state.has_initial_annotation = False  # Сбрасываем состояние аннотации
                st.session_state.last_image = None
                st.session_state.run_triggered = False
                st.session_state.rerun_count = 0
                st.session_state.image_uploaded = False
                st.session_state.processing = False
                logger.info("Изображение удалено пользователем, история чата очищена")
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
                st.session_state.chat_history = []  # Очищаем историю чата
                st.session_state.has_initial_annotation = False  # Сбрасываем состояние аннотации
                st.session_state.last_data = None
                st.session_state.run_triggered = False
                st.session_state.rerun_count = 0
                st.session_state.data_uploaded = False
                st.session_state.processing = False
                logger.info("Данные удалены пользователем, история чата очищена")
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
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'last_image' not in st.session_state:
            st.session_state.last_image = None
        if 'last_data' not in st.session_state:
            st.session_state.last_data = None
        if 'run_triggered' not in st.session_state:
            st.session_state.run_triggered = False
        if 'rerun_count' not in st.session_state:
            st.session_state.rerun_count = 0
        if 'image_uploaded' not in st.session_state:
            st.session_state.image_uploaded = False
        if 'data_uploaded' not in st.session_state:
            st.session_state.data_uploaded = False
        if 'reset_uploaders' not in st.session_state:
            st.session_state.reset_uploaders = False
        if 'has_initial_annotation' not in st.session_state:
            st.session_state.has_initial_annotation = False
        if 'error_message' not in st.session_state:
            st.session_state.error_message = None
        if 'pending_user_input' not in st.session_state:
            st.session_state.pending_user_input = None
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'pending_processing' not in st.session_state:
            st.session_state.pending_processing = False

        current_image = get_current_file(UPLOAD_DIR)
        current_data = get_current_file(DATA_DIR)
        logger.info(f"chat_callback: current_image={current_image}, current_data={current_data}, run_triggered={st.session_state.run_triggered}, rerun_count={st.session_state.rerun_count}, image_uploaded={st.session_state.image_uploaded}, data_uploaded={st.session_state.data_uploaded}, processing={st.session_state.processing}, pending_processing={st.session_state.pending_processing}")

        # Определяем, нужно ли скрывать элементы
        chat_empty = len(st.session_state.chat_history) == 0
        should_hide = st.session_state.processing or st.session_state.pending_processing
        logger.info(f"Состояние скрытия: should_hide={should_hide}, processing={st.session_state.processing}, pending_processing={st.session_state.pending_processing}, chat_empty={chat_empty}")

        # Индикатор загрузки
        loading_container = st.empty()
        if should_hide:
            with loading_container:
                st.spinner("Обработка...")

        # Контейнер для кнопки "Запустить"
        run_button_container = st.empty()
        # Контейнер для поля ввода чата
        chat_input_container = st.empty()

        # Отображаем кнопку "Запустить", если не нужно скрывать и файлы загружены
        if not should_hide and current_image and current_data and not st.session_state.run_triggered and chat_empty:
            if run_button_container.button("Запустить", key="run_button"):
                st.session_state.run_triggered = True
                st.session_state.processing = True
                logger.info("Нажата кнопка 'Запустить'")
                st.rerun()
        else:
            run_button_container.empty()

        # Отображаем поле ввода чата, если не нужно скрывать и есть аннотация или файлы
        if not should_hide and (len(st.session_state.chat_history) > 0 or (current_image and current_data)):
            user_input = chat_input_container.chat_input("Задайте вопрос о дашборде...", key="chat_input_main")
            if user_input and not st.session_state.pending_processing and not st.session_state.processing:
                logger.info(f"Обработка пользовательского ввода: {user_input}")
                st.session_state.processing = True
                st.session_state.pending_processing = True
                st.session_state.pending_user_input = user_input
                logger.info(f"Установлено processing=True и pending_processing=True для запроса: {user_input}")
                st.rerun()
        else:
            chat_input_container.empty()

        # Обрабатываем загрузку файлов
        if st.session_state.image_uploaded or st.session_state.data_uploaded:
            if st.session_state.pending_user_input:
                user_input = st.session_state.pending_user_input
                logger.info(f"Сохранён пользовательский ввод перед перезапуском: {user_input}")
            st.session_state.image_uploaded = False
            st.session_state.data_uploaded = False
            st.session_state.reset_uploaders = True
            st.rerun()

        # Проверяем и обрабатываем отложенный запрос
        if st.session_state.pending_processing and st.session_state.pending_user_input and not st.session_state.image_uploaded and not st.session_state.data_uploaded:
            user_input = st.session_state.pending_user_input
            logger.info(f"Обработка отложенного запроса: {user_input}")
            if len(st.session_state.chat_history) > 0 or st.session_state.run_triggered:
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
                st.session_state.processing = False
                logger.info(f"Сброшено processing=False после обработки запроса: {user_input}")
                st.session_state.pending_processing = False
                st.session_state.pending_user_input = None
                if result["response"]:
                    if "Слишком большой объем" in result["response"]:
                        st.session_state.error_message = result["response"]
                        logger.error(f"Ошибка в ответе: {result['response']}")
                    else:
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": result["response"]}
                        )
                        logger.info(f"Сгенерирован ответ: {result['response']}")
                st.session_state.rerun_count = 0
                st.session_state.reset_uploaders = True
                st.rerun()

        # Синхронизируем last_image и last_data
        if current_image and current_image != st.session_state.last_image:
            st.session_state.last_image = current_image
            logger.info(f"Синхронизировано last_image: {current_image}")
        if current_data and current_data != st.session_state.last_data:
            st.session_state.last_data = current_data
            logger.info(f"Синхронизировано last_data: {current_data}")

        # Сбрасываем file_uploader'ы
        if st.session_state.reset_uploaders:
            if "image_uploader" in st.session_state:
                del st.session_state["image_uploader"]
                logger.info("Сброшен image_uploader")
            if "data_uploader" in st.session_state:
                del st.session_state["data_uploader"]
                logger.info("Сброшен data_uploader")
            st.session_state.reset_uploaders = False

        # Обрабатываем аннотацию при нажатии кнопки "Запустить"
        if st.session_state.run_triggered:
            if not current_image or not current_data:
                st.session_state.error_message = "Пожалуйста, загрузите изображение и данные перед запуском."
                logger.info("Ошибка: отсутствует изображение или данные при попытке запуска")
                st.session_state.run_triggered = False
                st.session_state.processing = False
                st.rerun()
            elif st.session_state.rerun_count >= 3:
                st.session_state.error_message = "Слишком много перезапусков. Пожалуйста, обновите страницу."
                logger.error("Превышен лимит перезапусков")
                st.session_state.run_triggered = False
                st.session_state.image_uploaded = False
                st.session_state.data_uploaded = False
                st.session_state.has_initial_annotation = False
                st.session_state.reset_uploaders = True
                st.session_state.processing = False
                st.rerun()
            else:
                st.session_state.error_message = None
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
                st.session_state.processing = False
                if result["final_annotation"]:
                    if "Слишком большой объем" in result["final_annotation"]:
                        st.session_state.error_message = result["final_annotation"]
                        logger.error(f"Ошибка в аннотации: {result['final_annotation']}")
                    else:
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": result["final_annotation"]}
                        )
                        logger.info(f"Создана начальная аннотация: {result['final_annotation']}")
                        st.session_state.has_initial_annotation = True
                st.session_state.run_triggered = False
                st.session_state.image_uploaded = False
                st.session_state.data_uploaded = False
                st.session_state.last_image = current_image
                st.session_state.last_data = current_data
                st.session_state.rerun_count = 0
                st.session_state.reset_uploaders = True
                st.rerun()

        # Убираем индикатор загрузки после завершения обработки
        if not should_hide:
            loading_container.empty()

        # Отображаем чат
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            logger.info(f"Чат отображен, chat_history: {len(st.session_state.chat_history)} сообщений")

    except Exception as e:
        logger.error(f"Ошибка в chat_callback: {str(e)}")
        st.session_state.processing = False
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