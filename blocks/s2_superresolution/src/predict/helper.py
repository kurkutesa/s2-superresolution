import logging

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def get_logger(name, level=logging.DEBUG):
    """
    This method creates logger object and sets the default log level to DEBUG
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
