import streamlit as st
from config import logger


def setup_interface(
        upload_image_callback,
        upload_data_callback,
        display_image_callback,
        display_data_callback,
        chat_callback,
        get_current_image,
        get_current_data,
        clear_directory_callback
):
    """Настраивает интерфейс с двумя колонками: дашборд и данные в одной, чат и кнопка запуска в другой."""
    logger.info("Начало настройки интерфейса")
    # Создаем две колонки
    col1, col2 = st.columns([3, 3])

    # Первая колонка: изображение дашборда и данные
    with col1:
        # Панель для изображения
        st.header("Изображение дашборда")
        uploaded_image = st.file_uploader(
            "Загрузите изображение дашборда",
            type=["png", "jpg", "jpeg"],
            key="image_uploader",
            accept_multiple_files=False
        )
        if uploaded_image is not None:
            try:
                upload_image_callback(uploaded_image)
                st.session_state["image_uploaded"] = True
                st.session_state["error_message"] = None  # Clear error message on successful upload
                st.success("Изображение успешно загружено!")
                logger.info("Обработана загрузка изображения")
                # Сбрасываем file_uploader, чтобы избежать повторной загрузки
                if "image_uploader" in st.session_state:
                    del st.session_state["image_uploader"]
            except Exception as e:
                st.error(f"Ошибка при загрузке изображения: {str(e)}")
                logger.error(f"Ошибка при загрузке изображения: {str(e)}")
        display_image_callback(get_current_image())
        logger.info("Вызван display_image_callback")

        # Панель для данных
        st.header("Данные временного ряда")
        uploaded_data = st.file_uploader(
            "Загрузить файл данных",
            type=["csv", "txt", "xlsx"],
            key="data_uploader",
            accept_multiple_files=False
        )
        if uploaded_data is not None:
            try:
                upload_data_callback(uploaded_data)
                st.session_state["data_uploaded"] = True
                st.session_state["error_message"] = None  # Clear error message on successful upload
                st.success("Данные успешно загружены!")
                logger.info("Обработана загрузка данных")
                # Сбрасываем file_uploader, чтобы избежать повторной загрузки
                if "data_uploader" in st.session_state:
                    del st.session_state["data_uploader"]
            except Exception as e:
                st.error(f"Ошибка при загрузке данных: {str(e)}")
                logger.error(f"Ошибка при загрузке данных: {str(e)}")
        display_data_callback(get_current_data())
        logger.info("Вызван display_data_callback")

    # Вторая колонка: кнопка запуска и чат
    with col2:
        st.header("Аннотация к временному ряду и чат")
        if st.button("Запустить", key="run_button"):
            st.session_state["run_triggered"] = True
            st.session_state.chat_history = []  # Clear chat history on Run button click
            st.session_state["error_message"] = None  # Clear error message on Run button click
            logger.info("Кнопка 'Запустить' нажата, история чата очищена")

        # Display error message if present, before chat_callback
        if "error_message" in st.session_state and st.session_state["error_message"]:
            st.error(st.session_state["error_message"])

        chat_container = st.container()
        chat_callback(chat_container)
        logger.info("Вызван chat_callback")

    logger.info("Интерфейс настроен")
    return col1, col2