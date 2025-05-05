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
        st.header("Dashboard Image")
        display_image_callback(get_current_image())
        uploaded_image = st.file_uploader(
            "Upload Dashboard Image",
            type=["png", "jpg", "jpeg", "gif"],
            key="image_uploader",
            accept_multiple_files=False
        )
        if uploaded_image is not None:
            upload_image_callback(uploaded_image)

    # Панель для данных
    with col2:
        st.header("Time Series Data")
        display_data_callback(get_current_data())
        uploaded_data = st.file_uploader(
            "Upload Data File",
            type=["csv", "txt", "xlsx"],
            key="data_uploader",
            accept_multiple_files=False
        )
        if uploaded_data is not None:
            upload_data_callback(uploaded_data)

    # Панель для чата
    with col3:
        st.header("Annotations & Chat")
        chat_container = st.container()
        chat_callback(chat_container)

    return col1, col2, col3