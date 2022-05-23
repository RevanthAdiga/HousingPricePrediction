import os

from .utils import set_logger

log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs")
_LOGGER = set_logger(__name__, verbose=2, log_dir=log_dir)
