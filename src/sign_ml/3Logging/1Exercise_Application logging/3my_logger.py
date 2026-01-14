"""Configure logging.

- Show only WARNING+ in the terminal (user-facing)
- Save DEBUG+ (incl. INFO) to a rotating log file
"""
import sys
from pathlib import Path

from loguru import logger

log_path = Path(__file__).resolve().parent / "my_log.log"
# 3. Configure the logging level such that only messages of level warning and higher are shown
logger.remove()  # Remove the default logger
logger.add(sys.stdout, level="WARNING")  # Only show WARNING and above

# 4 send all logs to a file
logger.add(str(log_path), level="DEBUG", rotation="100 MB")  # Save everything for debugging


#2. Create a script called my_logger.py
logger.debug("Used for debugging your code.")
logger.info("Informative messages from your code.")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong and process may terminate.")
