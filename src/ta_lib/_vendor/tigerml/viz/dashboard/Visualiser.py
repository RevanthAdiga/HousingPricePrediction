import panel as pn

from ..data_exploration import DataExplorer


class Visualizer:
    """A class for the Visualizer sub-module in viz module."""

    def __init__(self, data):
        self.data = data

    def lag_analysis(self):
        """A method to perform lag analysis on the data."""
        pass

    def residual_analysis(self):
        """A method to perform residual analysis on the data."""
        pass

    def explore_data(self):
        """A method that launches DataExplorer as a standalone widget."""
        explorer = DataExplorer(self.data)
        explorer.open(5006)

    def show(self):
        """A method that launches Visualizer as a standalone widget."""
        pane = pn.Column(self.lag_analysis, self.residual_analysis, self.explore_data)
        pane.show(5006)
