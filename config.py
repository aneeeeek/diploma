import os
from openai import OpenAI
import logging
from langchain_openai import ChatOpenAI

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Инициализация клиента API
client = OpenAI(
    api_key=os.getenv("API_KEY", "sk-eae1582d53c2402b9d7be1f1a882c79f"),
    base_url="https://llm.glowbyteconsulting.com/api"
)

# Инициализация LangChain LLM
llm = ChatOpenAI(
    openai_api_key=os.getenv("API_KEY", "sk-eae1582d53c2402b9d7be1f1a882c79f"),
    openai_api_base="https://llm.glowbyteconsulting.com/api",
    model_name="openai.gpt-4o-mini",
    temperature=0.5,
    max_tokens=500
)

# Конфигурация директорий
UPLOAD_DIR = "uploads"
DATA_DIR = "data"
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_DATA_EXTENSIONS = {'csv', 'txt', 'xlsx'}