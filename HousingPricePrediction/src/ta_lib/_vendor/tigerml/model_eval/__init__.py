"""Model evaluation and explanation utility.

All plots in this utility are built using holoviews library. Interactive plots will be rendered in jupyter and others refer `holoviews <https://hvplot.holoviz.org/user_guide/Viewing.html>`_
"""

import os

from tigerml.core.utils import set_logger

from .base import (
    ClassificationEvaluation,
    ClassificationReport,
    ModelInterpretation,
    RegressionEvaluation,
    RegressionReport,
)
from .comparison import ClassificationComparison, RegressionComparison
from .multi_model import MultiModelComparisonRegression
from .plotters import *
from .segmented import ClassificationSegmentedReport, RegressionSegmentedReport

# Configure logger for the module
LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs")
_LOGGER = set_logger(__name__, verbose=2, log_dir=LOG_DIR)
_LOGGER.propagate = False
