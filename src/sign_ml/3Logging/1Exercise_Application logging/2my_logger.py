""" Configure the logging level such that only messages of level 
warning and higher are shown."""
import sys

from loguru import logger
# 3. Configure the logging level such that only messages of level warning and higher are show
logger.remove()  # Remove the default logger
logger.add(sys.stdout, level="WARNING")  # Only show WARNING and above

#2. Create a script called my_logger.py
logger.debug("Used for debugging your code.")
logger.info("Informative messages from your code.")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong and process may terminate.")
