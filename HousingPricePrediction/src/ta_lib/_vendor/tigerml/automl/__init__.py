"""AutoML module automates the process of selecting best models."""

# from ._lib import (
# 	load_data,
# 	perform_model_optimization,
# 	prepare_output_file
# )
import os

from tigerml.core.utils import set_logger

from .core import DATA_TYPES, TASKS, AutoML

# Configure logger for the module
log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs")
_LOGGER = set_logger(__name__, verbose=2, log_dir=log_dir)
_LOGGER.propagate = False

auto_ml = AutoML()
data_types = DATA_TYPES
tasks = TASKS
