import gc

from tigerml.core.plots import hvPlot
from tigerml.viz.backends.panel import PanelBackend

from .operations import (
    Binner,
    CustomTransformer,
    DropCols,
    Sorters,
    Transformer,
)


class DataProcessing(PanelBackend):
    """A class for handling preprocessed and processed data in DataProcessor sub-module."""

    DP_OPTIONS = {
        "Sort": "Sort data",
        "Create": "Add new column",
        "Drop": "Delete columns",
        "Bin": "Bin a column",
        "Custom": "Custom transformation",
    }

    def __init__(self, data, operation=None, parent=None):
        self.data = data.copy()
        self.parent = parent
        self.original_data = data.copy()
        self.operation = operation
        self.operations_wrapper = self.Column("", width=600)
        self.operations = self.Select(name="Select operation", options=[""] + list(self.DP_OPTIONS.values()), width=200)  # noqa
        self.operations.param.watch(self.opt_func, "value")
        # self.reset_button = self.Button(name='Reset df', width=130, css_classes=['button', 'tertiary'])
        # self.reset_button.on_click(self.reset_df)
        self.save_button = self.Button(name="Save Operation", width=130, css_classes=["button", "primary"])
        self.save_button.on_click(self.save_operation)
        self.save_button.js_on_click(args={}, code=self.remove_glass)
        self.update_button = self.Button(name="View Output", width=130, css_classes=["button", "secondary"])
        self.update_button.on_click(self.update_data)
        # self.cancel_button = self.Button(name='Cancel', width=130, css_classes=['button', 'tertiary'])
        # self.cancel_button.on_click(self.cancel_dp)
        self.delete_operation = ""
        self.processed_data = ""
        self.buttons = self.Row(self.save_button, self.update_button, self.delete_operation)
        self.output_data_preview = self.Column(self.display_df(self.processed_data))
        if self.operation is not None:
            self.operations.value = self.DP_OPTIONS[self.operation.name]
            self.operations_wrapper[0] = self.operation.create_ui()
        self.pane = self.Row(self.Column(self.display_df(self.data)),
                             self.Column("Operations Panel", self.operations, self.operations_wrapper, self.buttons),
                             self.output_data_preview)
        self.main_pane = self.Column(self.display_df(self.data), css_classes=["widget_header"])

    def display_df(self, data, table_width=600):
        """A method to customize and configure the data for display on UI."""
        table_height = 400
        if isinstance(data, str):
            return self.Column(data, height=table_height + 110, width=600, background="#ffffff")
        else:
            PREVIEW_LIMIT = 1000
            if str(table_width) == "auto":
                TABLE_WIDTH = None
            else:
                TABLE_WIDTH = table_width
            message = ""
            if len(data) > PREVIEW_LIMIT:
                message = f"Data is too big. Limiting preview to {PREVIEW_LIMIT} rows"
                data = data[:PREVIEW_LIMIT]
            for col in data.columns:
                try:
                    non_na_element = data[col].dropna().unique()[0]
                except:
                    non_na_element = None
                if "Interval" in str(type(non_na_element)):
                    data[col] = data[col].astype(str)
            return self.Column(self.Row(hvPlot(data).table(width=(len(data.columns) + 1) * 100, height=table_height),
                                        height=table_height + 50,
                                        width=TABLE_WIDTH,
                                        css_classes=["overflow-auto"]),
                               self.Row(message, height=30),
                               self.Row("shape of data = {}".format(data.shape), height=30), background="#ffffff")

    # def reset_df(self, event=None):
    #     self.data = self.original_data.copy()
    #     self.pane[0][1] = self.display_df(self.data)

    def update_data(self, event=None):
        """A method to update the display data after executing the operation."""
        self.operation.save_current_state()
        self.exec_func()

    def save_operation(self, event=None):
        """A method to update the actual data after executing the operation."""
        self.exec_func()
        if "DataFrame" in str(type(self.processed_data)):
            self.data = self.processed_data
            if self.parent is not None:
                self.parent.complete_active_operation(self.operation, self.processed_data)
        else:
            self.output_data_preview[0] = self.display_df("Error in executing operation!")

    def cancel_dp(self, event=None):
        """A callback function to delete a particular operation from the WorkFlow."""
        branch_workflow = self.parent
        parent_workflow = self.parent.parent
        self.parent.overlays.clear()
        if not branch_workflow.children and branch_workflow.parent:
            del parent_workflow.branches[self.parent]
            del parent_workflow.branches_ui
            gc.collect()
            parent_workflow.complete_workflow[1] = parent_workflow.get_branches_ui()

    def opt_func(self, event=None):
        """A callback function to change the operation based on the selection."""
        if self.operation is not None:
            if not event.new:  # If the user did not seletc any option
                return
            else:
                selection_option = list(self.DP_OPTIONS.keys())[list(self.DP_OPTIONS.values()).index(event.new)]
                if selection_option == self.operation.name:  # If the option user selected is same as the operation already present  # noqa
                    return
        self.processed_data = ""
        self.output_data_preview[0] = self.display_df(self.processed_data)
        if event.new == "":
            self.operation = None
            self.operations_wrapper[0] = ""
        elif event.new == "Sort data":
            self.operation = Sorters(self.data)
            self.operations_wrapper[0] = self.operation.pane
        elif event.new == "Add new column":
            self.operation = Transformer(self.data)
            self.operations_wrapper[0] = self.operation.pane
        elif event.new == "Delete columns":
            self.operation = DropCols(data=self.data, parent=self)
            self.operations_wrapper[0] = self.operation.pane
        elif event.new == "Bin a column":
            self.operation = Binner(data=self.data, parent=self)
            self.operations_wrapper[0] = self.operation.pane
        elif event.new == "Custom transformation":
            self.operation = CustomTransformer(data=self.data)
            self.operations_wrapper[0] = self.operation.pane

    def exec_func(self, event=None):
        """A method to trigger the particular Operation's exec_func."""
        if self.operations.value != "":
            try:
                self.processed_data = self.operation.exec_func(self.data)
            except Exception as e:
                import traceback

                self.processed_data = str(type(e).__name__) + " : " + str(e)
                self.err_traceback = traceback.format_exc()
            self.output_data_preview[0] = self.display_df(self.processed_data)
