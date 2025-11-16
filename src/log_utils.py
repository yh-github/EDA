import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: Path, file_prefix: str=""):
    """
    Configures the root logger to log to both stdout and a timestamped file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"{file_prefix}_{timestamp}.log"

    # Create handlers locally. They will be "owned" by the root logger.
    stdout_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[stdout_handler, file_handler],
        force=True
    )

    logging.info(f"Logging to {log_filename}")


def flush_logger():
    """
    Finds and flushes any root logger handlers that stream to sys.stdout.
    This is a "stateless" function and does not depend on setup_logging.
    """
    for handler in logging.root.handlers:
        if (isinstance(handler, logging.StreamHandler) and
                handler.stream == sys.stdout):
            handler.flush()