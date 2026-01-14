from pathlib import Path

import hydra # hydra is a framework for managing configurations
from loguru import logger # loguru is a library for logging


@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg):
    """Function that shows how to use loguru with hydra."""
    # Write logs next to this script (independent of Hydra's output directory / chdir)
    log_path = Path(__file__).resolve().parent / "my_logger_hydra.log"
    
     # Add a log file to the logger
    logger.add(str(log_path))
    logger.info(cfg)

    logger.debug("Used for debugging your code.")
    logger.info("Informative messages from your code.")
    logger.warning("Everything works but there is something to be aware of.")
    logger.error("There's been a mistake with the process.")
    logger.critical("There is something terribly wrong and process may terminate.")


if __name__ == "__main__":
    main()
