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
    """Настраивает интерфейс с тремя колонками для изображения, данных и чата."""
    # Создаем три колонки
    col1, col2, col3 = st.columns([2, 2, 3])

    # Панель для изображения
    with col1:
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
    with col2:
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

    # Панель для чата
    with col3:
        st.header("Аннотация к временному ряду и чат")
        chat_container = st.container()
        chat_callback(chat_container)

    return col1, col2, col3