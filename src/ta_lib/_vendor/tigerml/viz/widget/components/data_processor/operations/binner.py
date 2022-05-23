import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from tigerml.viz.widget.states import StatefulUI


class Binner(StatefulUI):
    """A class for binning Operation in DataProcessor."""

    def __init__(self, data, parent=None):
        super().__init__()
        self.name = "Bin"
        self.input_data_cols = data.columns.tolist()
        self.parent = parent
        self.new_col_name = self.TextInput(name="New column name", placeholder="Enter the binned column name...",  # noqa
                                           width=200)
        self.bin_col = self.Select(name="Select the cols to bin", options=self.input_data_cols, width=200)
        self.bin_nos = self.Select(name="No of bins", options=list(range(3, 11)), width=50)
        self.current_state.update({"bin_col.value": self.bin_col.value, "bin_nos.value": self.bin_nos.value})
        self.pane = self.Column()
        self.create_ui()

    def create_ui(self):
        """A method to create the UI components."""
        self.pane.clear()
        self.bin_col.options = self.input_data_cols
        self.pane.extend([self.new_col_name, self.Row(self.bin_col, self.bin_nos)])
        return self.pane

    def exec_func(self, data):
        """A method to execute the operation."""
        if self.bin_col.value:
            processed_data = data.copy()
            new_col_name = (self.new_col_name.value if self.new_col_name.value else "binned_" + self.bin_col.value)
            processed_data[new_col_name] = pd.cut(processed_data[self.bin_col.value], bins=self.bin_nos.value)
            cat_type = CategoricalDtype(categories=[str(item) for
                                                    item in np.sort(processed_data[new_col_name].unique().tolist())],
                                        ordered=True)
            processed_data[new_col_name] = processed_data[new_col_name].astype(str).astype(cat_type)
            return processed_data
        else:
            return "No column selected for binning."
