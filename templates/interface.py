import streamlit as st

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
    """Настраивает интерфейс с двумя колонками: дашборд и данные в одной, чат в другой."""
    # Создаем две колонки
    col1, col2 = st.columns([3, 3])

    # Первая колонка: изображение дашборда и данные
    with col1:
        # Панель для изображения
        st.header("Изображение дашборда")
        display_image_callback(get_current_image())
        uploaded_image = st.file_uploader(
            "Загрузите изображение дашборда",
            type=["png", "jpg", "jpeg", "gif"],
            key="image_uploader",
            accept_multiple_files=False
        )
        if uploaded_image is not None:
            upload_image_callback(uploaded_image)

        # Панель для данных
        st.header("Данные временного ряда")
        display_data_callback(get_current_data())
        uploaded_data = st.file_uploader(
            "Загрузить файл данных",
            type=["csv", "txt", "xlsx"],
            key="data_uploader",
            accept_multiple_files=False
        )
        if uploaded_data is not None:
            upload_data_callback(uploaded_data)

    # Вторая колонка: чат
    with col2:
        st.header("Аннотация к временному ряду и чат")
        chat_container = st.container()
        chat_callback(chat_container)

    return col1, col2