import os
from openai import OpenAI
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Инициализация клиента API
client = OpenAI(
    api_key=os.getenv("API_KEY", "sk-eae1582d53c2402b9d7be1f1a882c79f"),
    base_url="https://llm.glowbyteconsulting.com/api"
)

# Пример домен-специфичных терминов
DOMAIN_TERMS = {
    "finance": {
        "trend": "ценовой тренд",
        "anomaly": "аномальное движение цены",
        "support": "уровень поддержки",
        "resistance": "уровень сопротивления"
    },
    "retail": {
        "trend": "тренд продаж",
        "anomaly": "необычный всплеск спроса",
        "support": "базовый уровень продаж",
        "resistance": "потолок продаж"
    }
}

# Конфигурация директорий
UPLOAD_DIR = "uploads"
DATA_DIR = "data"
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_DATA_EXTENSIONS = {'csv', 'txt', 'xlsx'}