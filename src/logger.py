import logging
import os

LOG_DIR = "artifacts/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "training.log"),
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)