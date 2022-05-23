import gc
import logging

import tigerml.core.dataframe as td
from tigerml.core.utils import measure_time

from .ui import WidgetUI

_LOGGER = logging.getLogger(__name__)


class VizWidget(WidgetUI):
    """Base class for viz module."""

    def __init__(self, data=td.DataFrame(), initial_state={}, children=None, data_access=True,  # noqa
                 show_summary=False, debugger=False, datashade_beyond=None):
        """
        Initializes the DataExplorer.

            - infers Data types from data set, creates a UI for filters, y expressions and
            story board captures the current values in the widgets.

        :parameters data : pd.DataFrame (input data set)
        """
        self.data = data
        self.processed_data = self.data
        self._compute_dtypes()
        super().__init__(initial_state=initial_state, children=children)
        self.debugger = debugger
        self.data_access = data_access
        self.show_summary = show_summary
        self.notebook = False
        if not data_access:
            self.data_loader = self.null_component
        self.DATASHADE_BEYOND = (datashade_beyond or 200000)  # Beyond these number of points, use datashader
        # if data is not None:
        #     self._initiate()

    # @measure_time(_LOGGER)
    def _compute_dtypes(self):
        """Infers data type from the dataset.

        :returns dictionary : containing the columns and the corresponding data types
        """
        dtype_mapping = {}
        if self.processed_data is not None:
            from tigerml.core.utils.pandas import get_num_cols, get_bool_cols, get_dt_cols

            num_cols = get_num_cols(self.processed_data)
            dtype_mapping.update({col: "numeric" for col in num_cols})
            bool_cols = get_bool_cols(self.processed_data)
            dtype_mapping.update({col: "bool" for col in bool_cols})
            dt_cols = get_dt_cols(self.processed_data)
            dtype_mapping.update({col: "datetime" for col in dt_cols})
            dtypes = self.processed_data.dtypes.astype(str)
            cat_cols = dtypes[dtypes == "category"].index.tolist()
            dtype_mapping.update({col: "category" for col in cat_cols})
            remaining_cols = [x for x in self.processed_data.columns if x not in num_cols + bool_cols + dt_cols + cat_cols]  # noqa
            dtype_mapping.update({col: "string" for col in remaining_cols})
            dtype_mapping = dict(sorted(dtype_mapping.items()))
        self.dtypes = dtype_mapping

    def save_current_state(self):
        """
        Saves the values in the filter, Y and X blocks.

          - Used to capture the changes in the values, after creating the plot.
        """
        super().save_current_state()
        if "splitter.value" in self.current_state:
            if self.current_state["splitter.value"]:
                self.split_plots.disabled = False
            else:
                self.split_plots.disabled = True

    def update_viz_config_for_data(self):
        """A method that updates the Widget configuration based on input data."""
        self._compute_dtypes()
        # self.data_changed = True
        self._update_data_columns(self.processed_data, self.dtypes)

    def load_data(self, data):
        """A method that loads the data on Widget from DataProcessor module."""
        del self.data
        gc.collect()
        self.data = data
        self.data_processor.initial_data = self.data
        self.preprocess_data(recompute=True)
        self.update_plot()
        # self.data_changed = False
        return

    def load_state(self, passed_state):
        """A method that loads the give state on the Widget."""
        state = passed_state.copy()
        # dp_changed = state['children'][1] != self.data_processor.get_state()
        children_states = state.pop("children")
        self.set_state(state)
        upstream_changed = False
        for ind, child_state in enumerate(children_states):
            child = self.children[ind]
            if child_state != child.get_state():
                upstream_changed = True
                child.set_state(child_state)
                child.compute()
                # child.save_current_state()
            elif upstream_changed and hasattr(child, "compute"):
                child.compute()
        # if dp_changed:
        # import pdb
        # pdb.set_trace()
        # self.data_processor.set_state(state['children'][1])
        # self.data_processor.refresh_ui()
        # self.preprocess_data()
        # self.data_processor.save_current_state()
        self.refresh_ui()
        self.save_current_state()
        self.refresh_plot()

    def load_config(self, event=None):
        """A method that derives the Widget configuration from json file."""
        import json

        config_dict = json.loads(self.config_selector.value)
        self.load_state(config_dict)
        self.update_states_list()

    def save_data(self, event=None):
        """A method save the data generated through current Widget configuration."""
        x_col = self.current_state["x_col.value"] if "x_col.value" in self.current_state else self.initial_state["x_col.value"]  # noqa
        group_by_cols = self.current_state["splitter.value"] if "splitter.value" in self.current_state else self.initial_state["splitter.value"]  # noqa
        self.y_exprs.save_data(x_col, group_by_cols=group_by_cols)

    def preprocess_data(self, recompute=False):
        """A method to preprocess the input data."""
        if not self.data_processor.has_state_change() and not self.data_loader.has_state_change() and not recompute:
            return
        if hasattr(self, "processed_data"):
            del self.processed_data
            gc.collect()
        self.processed_data = self.data_processor.get_processed_data(recompute=recompute)
        self.update_viz_config_for_data()

    # @measure_time(_LOGGER)
    def filter_data(self, recompute=False):
        """Filters the data sets as per the filter conditions selected.

        :returns filtered_data : pd.DataFrame(resultant data frame after performing filters)
        """
        # import pdb
        # pdb.set_trace()
        if not self.filters.has_changes and not self.data_processor.has_state_change() \
                and not self.data_loader.has_state_change() and not recompute:
            return
        del self.filtered_data
        gc.collect()
        self.filtered_data = self.processed_data
        filter_expr = self.filters.get_filters()
        if filter_expr is not None:
            self.filtered_data = self.filtered_data[filter_expr]
        else:
            self.filtered_data = self.filtered_data

    @property
    def has_x_data_changes(self):
        """Monitors any changes in x value and the split values in the UI.

        :returns True if changes and False if not
        """
        return self.has_state_change("x_col.value", "splitter.value")

    @property
    def has_x_sort_changes(self):
        """Monitors any changes in sort value of X.

        :returns True if changes and False if not
        """
        return self.has_state_change("x_sort.value")

    @property
    def has_y_data_changes(self):
        """Monitors the changes in the y Values(segments, sorts, aggregation, y col) in the UI.

        :returns True if changes and False if not
        """
        return self.y_exprs.has_data_changes

    @property
    def has_data_changes(self):
        """Monitors the filter input values, y input values, X input values, splitter input values from the UI.

        :returns True if changes and False if not
        """
        return (
            self.data_loader.has_state_change()
            or self.data_processor.has_state_change()
            or self.filters.has_changes
            or self.has_y_data_changes
            or self.has_x_data_changes
            or self.has_state_change("splitter.value")
        )

    @property
    def has_sort_changes(self):
        """Monitors the  y sort input and X  sort input values from the UI.

        :returns True if changes are present and False if not
        """
        return self.y_exprs.has_sort_changes or self.has_x_sort_changes

    @property
    def has_plot_changes(self):
        """Monitors the  PlotType input and Axis input values from the UI.

        :returns True if changes are present and False if not
        """
        return (
            self.has_data_changes
            or self.has_sort_changes
            or self.y_exprs.has_plot_changes
            or self.has_state_change("split_plots.value", "color_axis.value")
        )

    @measure_time(_LOGGER)
    def sort_data(self, recompute=False):
        """Sorts the data as per the sort rule selected in UI.

        :returns value : sorting index for the plot data
        """
        if self.has_data_changes or self.has_sort_changes or recompute:
            self.y_exprs.sort_data(self.x_col.value, self.x_sort.value, group_by_cols=self.splitter.value)

    @measure_time(_LOGGER)
    def create_plot_data(self, data=None, recompute=False):
        """Creates plot data for the filtered data set.

        :returns plot_data: pd.DataFrame (plot data for the selected x and Y's, having X col as index)
        """
        if not self.has_data_changes and data is None and not recompute:
            # print('Not computing plot data')
            return
        # print('Computing plot data')
        data = self.filtered_data if data is None else data
        self.y_exprs.compute_plot_data(data=data, x_col=self.x_col.value, split_by_cols=self.splitter.value,
                                       recompute=self.has_data_changes or self.filters.has_changes)

    @measure_time(_LOGGER)
    def create_plot(self, recompute=False):
        """A method that creates the plot from plot_data."""
        if not self.has_plot_changes and not recompute:
            return
        plot = self.y_exprs.create_plot(
            self.x_col.value,
            self.splitter.value,
            self.split_plots.value,
            self.show_summary,
        )
        # if self.notebook:
        #     self.plot.opts(width=950, height=500)
        # else:
        #     self.plot.opts(width=1000, height=500)
        if self.plot.__class__.__name__ == "DynamicMap":
            plot = self.panel(self.plot, widget_location="top_right")
        return plot

    @measure_time(_LOGGER)
    def get_plot(self, recompute=False, event=None):
        """Performs 3 operations : creates the plot data, sorts it and creates the plot."""
        # self.preprocess_data(recompute=recompute)
        self.filter_data(recompute=recompute)
        if self.filtered_data is None:
            return "Please load data to start exploration"
        if self.filtered_data.empty:
            return "No matching data in given filters."
        self.create_plot_data(recompute=recompute)
        # if self.plot_data.empty:
        # 	return 'No matching data for given y expressions.'
        self.sort_data(recompute=recompute)
        self.plot = self.create_plot(recompute=recompute)

    def refresh_plot(self):
        """A method that initiates plot creation from plot_data."""
        self.sort_data(recompute=True)
        self.plot = self.create_plot(recompute=True)
        self.plot_wrapper[0] = self.plot

    @measure_time(_LOGGER)
    def update_plot(self, event=None, bokeh=False, check=False, recompute=False):
        """A callback function that initiates plot creation."""
        if self.debugger and self.debug.value == "ON":
            import pdb

            pdb.set_trace()
        if self.has_plot_changes or recompute:
            self.plot_wrapper[0] = self.loading_message
            del self.plot
            gc.collect()
            self.plot = None
            try:
                self.preprocess_data(recompute=recompute)
                self.get_plot(recompute=recompute, event=event)
                if self.plot:
                    if bokeh:
                        from tigerml.core.plots import get_bokeh_plot

                        self.plot = get_bokeh_plot(self.plot)
                    self.plot_wrapper[0] = self.plot
                    if self.show_summary:
                        self.summary_stats[1] = self.y_exprs.summary_table
                    self.save_current_state()
            except Exception as e:
                if not check:
                    self.plot_wrapper[0] = "Error occured. {}".format(e)
                raise e

    @measure_time(_LOGGER)
    def _create_ui(self):
        self.processed_data = self.data
        self.filtered_data = self.data
        self.plot_data = td.DataFrame()
        self.plot = None
        self.y_vals = {}
        self._create_ui_elements()

    @measure_time(_LOGGER)
    def _initiate(self):
        self._create_ui()
        self._create_child_panes()
        if self.notebook:
            self.widget = self._create_notebook_widget()
        else:
            self.widget = self._create_standalone_widget()
        self.widget.append(self.message_box.show())
        self.save_current_state()

    @measure_time(_LOGGER)
    def create_pane(self):
        """A method that creates the Widget UI component."""
        self._initiate()
        try:
            self.update_plot(recompute=True)
        except Exception as e:
            self.plot_wrapper[0] = "Could not create plot. Error - {}".format(e)
        classes = ["data_explorer"]
        if self.notebook:
            classes.append("gray_bg")
        self.pane = self.Column("", css_classes=classes)
        self.pane[0] = self.widget
        # self.pane[1] = self.story_board.pane

    def show(self):
        """A method that launches the Widget as a reactive object on Notebook."""
        self.notebook = True
        self.create_pane()
        # self.pane[1] = self.story_board.pane
        display(self.pane)  # noqa: F821

    def open(self, port=None, threaded=True):
        """A method that launches the Widget as a standalone application."""
        self.notebook = False
        self.create_pane(threaded=threaded)
        if port:
            self.server = self.pane.show(title="TigerML Explorer", port=port, threaded=threaded)
        else:
            self.server = self.pane.show(title="TigerML Explorer", threaded=threaded)

    def close(self, event=None):
        """A method that terminates the standalone Widget application."""
        self.pane.clear()
        self.pane.append("Data Explorer terminated.")
        try:
            self.server.stop()
        except:
            self.server._stop_event.set()
        import sys

        sys.exit()

    def refresh_x(self):
        """A method that reloads the X-Pane UI object."""
        self.x_pane[1] = self.null_component
        self.x_pane[1] = self.x_options

    def refresh_splitter(self):
        """A method that reloads the SplitterPane UI object."""
        self.splitter_pane[1] = self.null_component
        self.splitter_pane[1] = self.splitter_widget

    def refresh_ui(self, enforce=False, only_children=True):
        """A method that reloads the entire Widget UI."""
        if not only_children:
            self.refresh_x()
            self.refresh_splitter()
        if enforce or self.data_processor.has_state_change():
            self.data_processor.refresh_ui()
        if enforce or self.filters.has_changes:
            self.filters.refresh_ui()
        if enforce or self.y_exprs.has_state_change():
            self.y_exprs.refresh_ui()

    @property
    def widget_title(self):
        """A property that describes the Widget based on its configuration."""
        return ", ".join([y.description for y in self.y_exprs.children]) + (" vs {}".format(self.x_col.value))
