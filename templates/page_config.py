import streamlit as st

def set_page_config():
    """Настраивает конфигурацию страницы Streamlit."""
    st.set_page_config(
        page_title="TimeSeries Annotator",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )