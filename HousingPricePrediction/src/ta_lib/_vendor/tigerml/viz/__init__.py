import os

from tigerml.core.utils import set_logger

from .dashboard import Dashboard
from .data_exploration import DataExplorer

# Configure logger for the module
log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs")
_LOGGER = set_logger(__name__, verbose=2, log_dir=log_dir)
_LOGGER.propagate = False
