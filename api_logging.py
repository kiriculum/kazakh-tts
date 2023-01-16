from logging import Formatter, getLogger
from logging.handlers import RotatingFileHandler
from pathlib import Path

Path('logs').mkdir(parents=True, exist_ok=True)

logger = getLogger('tts_api')
verbose_fmt = Formatter('{asctime}-{levelname}-{name}: {message}', style='{', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = RotatingFileHandler('logs/tts_api.log', maxBytes=5 * 1024 * 1024, backupCount=3)
file_handler.setFormatter(verbose_fmt)
logger.addHandler(file_handler)
logger.setLevel('INFO')
