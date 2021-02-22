import logging
from colorlog import ColoredFormatter
import os

def setup_logger(log_dir=None):

    


    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(yellow)s%(name)s: %(white)s%(message)s",
        "%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'green',
            'INFO': 'green',
            'WARNING': 'red',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
    )

    logger = logging.getLogger('saic-dc')
    if log_dir:
        log_file_save_path = os.path.join(log_dir, "exp_output.log")
        logging.basicConfig(filename=log_file_save_path)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger

