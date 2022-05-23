from tigerml.viz.backends.panel import PanelBackend
from tigerml.viz.widget.states import StatefulUI


class DropCols(StatefulUI):
    """A class for droping a column Operation in DataProcessor."""

    def __init__(self, data, parent=None):
        super().__init__()
        self.name = "Drop"
        self.input_data_cols = data.columns.tolist()
        self.parent = parent
        self.delete_cols = self.MultiChoice(name="Select the cols to delete", options=self.input_data_cols, width=200)  # noqa
        self.current_state.update({"delete_cols.value": self.delete_cols.value})
        self.pane = self.Row()
        self.create_ui()

    def create_ui(self):
        """A method to create the UI components."""
        self.pane.clear()
        self.delete_cols.options = self.input_data_cols
        self.pane.extend([self.delete_cols])
        return self.pane

    def exec_func(self, data):
        """A method to execute the operation."""
        if self.delete_cols.value != []:
            processed_data = data.copy()
            processed_data = processed_data.drop(self.delete_cols.value, axis=1)
            return processed_data
        else:
            return "No column selected for deletion."
