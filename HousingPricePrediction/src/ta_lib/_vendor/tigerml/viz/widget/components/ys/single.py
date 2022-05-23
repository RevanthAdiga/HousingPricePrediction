import logging
from functools import partial

import numpy as np

import tigerml.core.dataframe as td
from tigerml.core.dataframe.helpers import get_formatted_values
from tigerml.core.plots import hvPlot
from tigerml.core.plots.bokeh import (
    add_to_primary,
    add_to_secondary,
    colorize_barplot,
    colorize_boxplot,
    dodge_barplot,
    finalize_axes_left,
    finalize_axes_right,
    left_y_as_datetimeaxis,
    right_y_as_datetimeaxis,
    x_as_datetimeaxis,
)
from tigerml.core.utils import (
    compute_if_dask,
    get_cat_cols,
    get_dt_cols,
    get_num_cols,
    is_numeric,
    measure_time,
)

from .ui import AGGS, YExprUI

_LOGGER = logging.getLogger(__name__)

PLOT_TYPES = {
    "numeric": {
        "numeric": {
            "single_y": {
                "has_ColorAxis": [
                    ("line", {}),
                    ("scatter", {}),
                    ("area", {"stacked": True, "alpha": 0.7}),
                    ("step", {}),
                ],
                "no_ColorAxis": [
                    ("line", {}),
                    ("scatter", {}),
                    ("area", {"stacked": True, "alpha": 0.7}),
                    ("step", {}),
                ],
            },
            "multiple_y": {
                "has_ColorAxis": [("scatter", {}), ("hexbin", {})],
                "no_ColorAxis": [("scatter", {}), ("hexbin", {})],
            },
        },
        "non_numeric": {
            "single_y": {
                "has_ColorAxis": [
                    ("bar", {"invert": True}),
                    ("grouped_bar", {"invert": True}),
                ],
                "no_ColorAxis": [
                    ("bar", {"invert": True}),
                    ("grouped_bar", {"invert": True}),
                ],
            },
            "multiple_y": {
                "has_ColorAxis": [
                    ("box", {"invert": True}),
                    ("violin", {"invert": True}),
                ],
                "no_ColorAxis": [
                    ("box", {"invert": True}),
                    ("violin", {"invert": True}),
                ],
            },
        },
    },
    "non_numeric": {
        "numeric": {
            "single_y": {
                "has_ColorAxis": [
                    ("line", {}),
                    ("scatter", {}),
                    ("bar", {}),
                    ("grouped_bar", {}),
                ],
                "no_ColorAxis": [
                    ("line", {}),
                    ("scatter", {}),
                    ("bar", {}),
                    ("grouped_bar", {}),
                ],
            },
            "multiple_y": {
                "has_ColorAxis": [("scatter", {}), ("box", {}), ("violin", {})],
                "no_ColorAxis": [("scatter", {}), ("box", {}), ("violin", {})],
            },
        },
        "non_numeric": {
            "single_y": {
                "has_ColorAxis": [],
                "no_ColorAxis": [
                    ("heatmap", {}),
                    ("bar", {"stacked": True}),
                    ("grouped_bar", {"stacked": False}),
                ],
            },
            "multiple_y": {
                "has_ColorAxis": [],
                "no_ColorAxis": [
                    ("heatmap", {}),
                    ("bar", {"stacked": True}),
                    ("grouped_bar", {"stacked": False}),
                ],
            },
        },
    },
    "no_x": {
        "numeric": {
            "has_ColorAxis": [("kde", {}), ("hist", {}), ("box", {}), ("violin", {})],
            "no_ColorAxis": [("kde", {}), ("hist", {}), ("box", {}), ("violin", {})],
        },
        "non_numeric": {
            "has_ColorAxis": [
                ("bar", {"invert": True}),
                ("grouped_bar", {"invert": True}),
            ],
            "no_ColorAxis": [
                ("bar", {"invert": True}),
                ("grouped_bar", {"invert": True}),
            ],
        },
    },
}


class YExpr(YExprUI):
    """A class for all the individual Y-Expression object."""

    def __init__(self, parent, data, dtypes, initial_state={}):
        self.data = data
        self.parent = parent
        self.dtypes = dtypes
        self.id = None
        self.plot_data = None
        self.plot = None
        super().__init__(data, dtypes, initial_state=initial_state)
        self._create_ui()
        if not initial_state:
            self._set_defaults()

    @property
    def y_col(self):
        """A property that returns the selected Y-Column."""
        return self.col_name.value

    @property
    def segment_by(self):
        """A property that returns the selected ColorAxis."""
        return self.parent.parent.color_axis.value if self.have_color_axis.value else ""

    @property
    def display_name(self):
        """A property that returns the name of Y-Column along with the aggregation applied on it."""
        return "{}({})".format(self.agg_func.value, self.y_col) if self.agg_func.value else self.y_col  # noqa

    @property
    def description(self):
        """A property that returns the description of Y-Column along with the aggregation applied on it."""
        return self.display_name + " segmented by {}".format(self.segment_by) if self.segment_by else ""  # noqa

    @property
    def has_color_axis(self):
        """A property that tracks if ColorAxis is activated."""
        return "has_ColorAxis" if self.segment_by else "no_ColorAxis"

    def has_multiple_y_values(self, ignore_segment=True):
        """A method to determine the no. of y values for each x value. Used to determine appropriate plot-type."""
        if self.parent.parent.splitter.value:
            # index_name is used to avoid `KeyError: 0.0` which occurs in group_by apply()
            # for dataframes with unsorted float type as index
            groupby_cols = self.parent.parent.splitter.value.copy()
            if not ignore_segment and self.segment_by:
                groupby_cols += [self.segment_by]
            index_name = self.plot_data.index.name
            if index_name:
                self.plot_data.reset_index(inplace=True)
                value = any(self.plot_data.groupby(groupby_cols).apply(lambda df: df[index_name].nunique() < len(df)).values)  # noqa
                self.plot_data.set_index(index_name, inplace=True)
            else:
                value = any(self.plot_data.groupby(groupby_cols).apply(lambda df: df.index.nunique() < len(df)).values)
        else:
            if not ignore_segment and self.segment_by:
                index_name = self.plot_data.index.name
                if index_name:
                    self.plot_data.reset_index(inplace=True)
                    value = any(self.plot_data.groupby(self.segment_by).apply(lambda df: df[index_name].nunique() < len(df)).values)  # noqa
                    self.plot_data.set_index(index_name, inplace=True)
                else:
                    value = any(self.plot_data.groupby(self.segment_by).apply(lambda df: df.index.nunique() < len(df)).values)  # noqa
            else:
                value = self.plot_data.index.nunique() < len(self.plot_data)
        return bool(compute_if_dask(value))

    def _is_valid_state(self, event):
        if event and event.obj.name in ["Select Columns(Splitter)", "Select Column(X)", "Select Column(Y)",  # noqa
                                        "Color Axis", "Have Color Axis", "UPDATE PLOT", "Aggregation"]:
            x_col = self.parent.parent.x_col.value
            y_col = self.col_name.value
            seg_by = self.segment_by
            has_color = self.have_color_axis.value
            splitter = self.parent.parent.splitter.value
            color_by = self.parent.parent.color_axis.value
            # if x_col == y_col:
            #     self.parent.parent.plot_wrapper[0] = 'Invalid state. Please select again'
            #     return False
            # else:
            #     return True

            condition_dict = {
                "x_col and x_col == y_col": "Invalid state. X-axis and Y-axis cannot be same.",
                "x_col and x_col == seg_by": "Invalid state. X-axis and Color-axis cannot be same.",
                "x_col and x_col in splitter": "Invalid state. X-axis and Splitter value cannot be same.",
                "seg_by and seg_by in splitter": "Invalid state. Color-axis and Splitter value cannot be same.",
                "has_color and not color_by": "Invalid state. Select a Color-axis variable from dropdown.",
                "seg_by and seg_by == y_col": "Invalid state. Color-axis and Y-axis cannot be same.",
                "y_col in splitter": "Invalid state. Y-axis and Splitter value cannot be same.",
            }
            for key in condition_dict.keys():
                if eval(key):
                    self.parent.parent.plot_wrapper[0] = condition_dict[key]
                    return False
            if event and event.obj.name != "UPDATE PLOT":  # In order to show the loading message
                self.parent.parent.plot_wrapper[0] = ""
            return True
        else:
            if event and event.obj.name != "UPDATE PLOT":  # In order to show the loading message
                self.parent.parent.plot_wrapper[0] = ""
            return True

    @measure_time(_LOGGER)
    def refresh_plot_types(self, event=None):
        """A method to set the PlotType options based on the selections on Y-Expression."""
        if hasattr(self.parent.parent, "x_col") and self._is_valid_state(event):
            x_col = self.parent.parent.x_col.value
            self.x_type = "no_x" if not x_col else "numeric" if self.dtypes[x_col] in ["numeric", "datetime"] else "non_numeric"  # noqa
            if self.agg_func.value:
                self.y_type = "numeric"
                if self.x_type == "no_x":
                    self.x_type = "numeric" if self.dtypes[self.y_col] in ["numeric", "datetime"] else "non_numeric"
            elif self.y_col is None:
                self.y_type = "numeric"
            else:
                self.y_type = "numeric" if self.dtypes[self.y_col] in ["numeric", "datetime"] else "non_numeric"
            if len(self.data.columns) != len(self.parent.parent.filtered_data.columns) and \
                    set(list(self.data.columns)) != set(list(self.parent.parent.filtered_data.columns)):
                self.parent.parent.filter_data()
            self.compute_plot_data(self.parent.parent.filtered_data, x_col, self.parent.parent.splitter.value.copy())
            self.no_of_y_values = "multiple_y" if self.has_multiple_y_values(ignore_segment=False) else "single_y"
            logging.info(f"X col is {x_col} - {self.x_type}")
            logging.info(f"y col is {self.y_col} - {self.y_type} - {self.no_of_y_values}")
            if self.x_type == "no_x":
                self.plot_types = PLOT_TYPES[self.x_type][self.y_type][self.has_color_axis].copy()
            else:
                self.plot_types = PLOT_TYPES[self.x_type][self.y_type][self.no_of_y_values][self.has_color_axis].copy()    # noqa
            self.plot_types.append(("table", {}))
            new_options = [ptype[0] for ptype in self.plot_types]
            if self.plot_type.options != new_options:
                self.plot_type.options = []
                self.plot_type.options = new_options
                self.plot_type.value = self.plot_type.options[0]  # noqa  # to avoid robustness issue (the option seen in UI and
                # the actual value will be different once the plot_type options list is changed from backend)

    def describe(self):
        """A method that provides the description of the Y-Expression object in widget_builder."""
        description = self.Row()
        if "agg_func.value" in self.initial_state:
            if self.initial_state["agg_func.value"]:
                description = self.Row(f'{self.initial_state["agg_func.value"]} of ')
        else:
            description = self.Row(self.agg_func, " of ")
        description.append(self.col_name if "col_name.value" not in self.initial_state else f"{self.col_name.value}")  # noqa
        if "have_color_axis.value" in self.initial_state:
            if self.initial_state["have_color_axis.value"]:
                description.append(f' segmented by {self.initial_state["have_color_axis.value"]}')    # noqa
            else:
                if "sort_rule.value" not in self.initial_state:
                    description.extend([f" sort in ", self.sort_rule])
        else:
            description.extend([f" segmented by ", self.segment_by])
            if "sort_rule.value" not in self.initial_state:
                description.extend([f" sort in ", self.sort_rule])
        if "plot_type.value" not in self.initial_state:
            description.extend([f" plot as ", self.plot_type])
        if "axis.value" not in self.initial_state:
            description.extend([f" on ", self.axis])
        return description

    def set_sortable(self):
        """A method to set the Sorting options based on the initial state of Y-Expression."""
        if "sort_rule.value" not in self.initial_state:
            self.sort_rule.disabled = self.has_multiple_y_values()

    def remove_other_sorts(self, event=None):
        """A method to disable other the Sort options if the current Y-Expression's sort is active."""
        if event.new:
            self.parent.remove_y_sorts(self)

    def delete_y(self, event):
        """A method to delete the current Y-Expression."""
        self.parent.delete_y(self, event)

    def toggle_advanced(self, event):
        """A method to enable and disable advanced Y-Expression options."""
        if event.new:
            self.pane[2] = self.advanced_options
        else:
            self.pane[2] = self.null_component

    @property
    def has_data_changes(self):
        """
        Monitors the changes in y input values.

        - this check is done before getting the plot data

        :return: True if changes and False if not
        """
        return self.has_state_change("col_name.value", "agg_func.value", "have_color_axis.value",
                                     "normalize_by.value", "normalize_across.value")    # noqa

    @property
    def has_sort_changes(self):
        """A method to track the changes in sort rules."""
        return self.has_state_change("sort_rule.value")

    @property
    def has_plot_changes(self):
        """A method to track the changes in PlotType and Axis."""
        return self.has_state_change("axis.value", "plot_type.value")

    @property
    def normalize_data(self):
        """A method to track if Normalizer is active."""
        return not self.normalize_by.disabled

    def _get_plot_type_options(self):
        if not hasattr(self, "plot_types") and self.plot_type.value:
            self.refresh_plot_types()
        ptype_index = [ptype[0] for ptype in self.plot_types].index(self.plot_type.value)    # noqa
        ptype_kwargs = self.plot_types[ptype_index][1]
        return ptype_kwargs

    def preprocess_data_for_plot(self, series_data, x_col, group_by_cols=[], segment_col=""):    # noqa
        """A method to preprocess the computed data for plotting."""
        if x_col == "" and self.x_type != "no_x":
            x_col = self.y_col
        series_data = series_data.dropna()
        if segment_col:
            series_data[segment_col] = series_data[segment_col].astype(str)
        if (self.plot_type.value in ["heatmap"] or self.plot_type.value in ["bar", "grouped_bar"] and "stacked" in self._get_plot_type_options()):    # noqa
            series_data.reset_index(inplace=True)
            plot_data = series_data.groupby([x_col, self.display_name] + group_by_cols).agg(len)
            # plot_data.name = 'count'
            series_data = plot_data.reset_index().rename(columns={0: "count"})
        elif (self.plot_type.value in ["bar", "grouped_bar"] and self.x_type == "no_x" and self.y_type == "non_numeric"):    # noqa
            group_by = group_by_cols + [self.display_name]
            if segment_col:
                group_by += [segment_col]
            plot_data = series_data.reset_index().groupby(group_by).agg(len)
            series_data = plot_data.reset_index().rename(columns={"index": "count"}).sort_values(by="count", ascending=False)    # noqa
        elif self.plot_type.value in ["hexbin"]:
            series_data.reset_index(inplace=True)
        x_is_date = False
        y_is_date = False
        index_name = series_data.index.name
        if index_name:
            series_data.reset_index(inplace=True)
        if x_col and get_dt_cols(series_data[[x_col]]):
            series_data[x_col] = series_data[x_col].astype(np.int64) / 10 ** 6
            x_is_date = True
        if self.display_name in self.dtypes.keys() and get_dt_cols(series_data[[self.display_name]]) and x_col:
            series_data[self.display_name] = series_data[self.display_name].astype(np.int64) / 10 ** 6
            y_is_date = True
        if not x_col and self.display_name in self.dtypes.keys() and get_dt_cols(series_data[[self.display_name]]):
            series_data[self.display_name] = series_data[self.display_name].astype(np.int64) / 10 ** 6
            if self.plot_type.value in ["box", "violin"]:
                x_is_date = False
                y_is_date = True
            else:
                x_is_date = True
                y_is_date = False
        if index_name:
            series_data.set_index(index_name, inplace=True)
        return series_data, x_is_date, y_is_date

    def set_kwargs_for_bars(self, kwargs, series_data, x_col, y_col, segment_col, group_by_cols, multiple_series,
                            only_bars):
        """A method to set the plotter_kwargs for Bar plots."""
        if x_col == "" and self.x_type != "no_x":
            x_col = self.y_col
        kwargs["kind"] = self.plot_type.value
        if kwargs["kind"] == "grouped_bar":
            kwargs["kind"] = "bar"
        ptype_kwargs = self._get_plot_type_options()
        kwargs.update(ptype_kwargs)
        if self.plot_type.value in ["bar", "grouped_bar"] and self.x_type == "no_x" and self.y_type == "non_numeric":
            kwargs["x"] = y_col
            kwargs["y"] = "count"
            if segment_col:
                kwargs["by"] = segment_col
            return
        if ptype_kwargs:
            if "invert" in ptype_kwargs:
                series_data.reset_index(inplace=True)
                kwargs["by"] = y_col
                kwargs["y"] = x_col
            elif self.plot_type.value in ["bar", "grouped_bar"] and "stacked" in ptype_kwargs:
                kwargs["by"] = y_col
                kwargs["x"] = x_col
                kwargs["y"] = "count"
            return
        elif kwargs["kind"] == "bar":
            series_data["metric"] = self.display_name
            series_data.rename(columns={self.display_name: "value"}, inplace=True)
            if segment_col:
                kwargs["by"] = segment_col
        if not only_bars and segment_col and segment_col not in group_by_cols:
            if multiple_series:
                prefix = self.display_name + " - "
                series_data[segment_col + "_levels"] = prefix + series_data[segment_col].astype(str)
                # kwargs["by"] = segment_col + "_levels"
                kwargs["by"] = segment_col
            else:
                if self.dtypes[segment_col] not in ["numeric", "datetime"]:
                    series_data[segment_col + "_levels"] = series_data[segment_col].astype(str)
                    # kwargs["by"] = segment_col + "_levels"
                    kwargs["by"] = segment_col
                else:
                    kwargs["by"] = segment_col
        if self.plot_type.value in ["heatmap"]:
            kwargs["x"] = x_col

    def post_process_plot(self, current_plot, kwargs, x_col, last_y, mapper, mapper_df, x_name, x_is_date=False,
                          y_is_date=False, color_hook_kwargs={}):
        """A method to post-process the generated plot."""
        if x_col == "" and self.x_type != "no_x":
            x_col = self.y_col
        hooks = []
        if "box_with_color" in color_hook_kwargs.keys():
            hooks.append(partial(colorize_boxplot, color_hook_kwargs["box_with_color"]))
            # colorize_boxplot should be the first hook to be applied
            # because glyph type changes to 'patch' after applying other hooks (related to axes)
        if "bar_with_color" in color_hook_kwargs.keys():
            hooks.append(partial(dodge_barplot, color_hook_kwargs["bar_with_color"]))
        if current_plot.__class__.__name__ not in ["DynamicMap"]:
            if self.axis.value == "right":
                hooks.append(add_to_secondary)
                if y_is_date:
                    hooks.append(right_y_as_datetimeaxis)
            else:
                hooks.append(add_to_primary)
                if y_is_date:
                    hooks.append(left_y_as_datetimeaxis)
            if last_y:
                if x_is_date:
                    hooks.append(x_as_datetimeaxis)
                if self.axis.value == "right":
                    hooks.append(finalize_axes_right)
                else:
                    hooks.append(finalize_axes_left)
        if hooks:
            current_plot = current_plot.options(hooks=hooks)
        if "datashade" in kwargs and kwargs["datashade"]:
            if mapper is not None:
                if len(mapper_df) > 30:
                    sample_length = round(len(mapper) / 20)
                    indices = [sample_length * i for i in range(0, 20)]
                else:
                    sample_length = len(mapper_df)
                    indices = list(range(0, sample_length))
                mapper_df = mapper_df.loc[indices]
                ticks = list(zip(mapper_df["tigerml_mapper"].values.tolist(), get_formatted_values(mapper_df[x_name])))
                current_plot = current_plot.opts(xlabel=x_col, xticks=ticks)
            from holoviews.operation.datashader import dynspread

            current_plot = dynspread(current_plot, threshold=0.75)
        # if ('x' in kwargs and 'datetime' in str(series_data[kwargs['x']].dtype)) \
        # 		or 'datetime' in str(series_data.index.dtype):
        # 	from bokeh.models import DatetimeTickFormatter
        # 	formatter = DatetimeTickFormatter(days="%m/%d",
        # 		hours="%H",
        # 		minutes="%H:%M")
        # 	current_plot = current_plot.opts(xformatter=formatter)
        return current_plot

    def set_kwargs_for_plotter(self, plotter, kwargs, series_data, x_col, y_col, segment_col, group_by_cols):
        """A method to set the plotter_kwargs for all plots."""
        if x_col == "" and self.x_type != "no_x":
            x_col = self.y_col
        max_data_length = True
        x_name = None
        color_hook_kwargs = {}
        if group_by_cols:
            kwargs["groupby"] = group_by_cols.copy()
            kwargs["widget_location"] = "top_right"
            index_name = series_data.index.name
            if index_name:
                series_data.reset_index(inplace=True)
                max_data_length = any(series_data.groupby(group_by_cols).apply(lambda df: len(df) > self.parent.parent.DATASHADE_BEYOND))  # noqa
                series_data.set_index(index_name, inplace=True)
            else:
                max_data_length = any(series_data.groupby(group_by_cols).apply(lambda df: len(df) > self.parent.parent.DATASHADE_BEYOND))  # noqa
        if x_col and "tigerml_xcol_{}".format(x_col) in series_data.columns:
            kwargs["x"] = "tigerml_xcol_{}".format(x_col)
            kwargs["xlabel"] = x_col
        elif "tigerml_sort_ranking" in series_data and not x_col:
            kwargs["x"] = "tigerml_sort_ranking"
            kwargs["xlabel"] = x_col or "index"
        mapper = None
        mapper_df = None
        if len(series_data) > self.parent.parent.DATASHADE_BEYOND and max_data_length:
            if kwargs["kind"] in ["scatter", "line"]:
                kwargs["datashade"] = True
                x_name = None
                new_data = None
                if "x" in kwargs and kwargs["x"] not in get_num_cols(series_data):
                    x_name = kwargs["x"]
                    new_data = series_data
                elif not is_numeric(series_data.index.dtype):
                    x_name = series_data.index.name or "index"
                    new_data = series_data.reset_index()
                if x_name is not None:
                    unique_vals = new_data[x_name].unique().categories.tolist() if str(new_data.dtypes[x_name]) == "category" else list(new_data[x_name].unique())  # noqa
                    mapper = dict(zip(unique_vals, range(0, new_data[x_name].nunique())))
                    mapper_df = td.DataFrame(mapper, index=[0]).T.reset_index().rename(columns={0: "tigerml_mapper", "index": x_name})  # noqa
                    new_data = new_data.merge(mapper_df, on=x_name)
                    plotter = hvPlot(new_data.set_index("tigerml_mapper"))
            elif kwargs["kind"] == "bar":
                raise Exception("Dataset too big for a bar plot.")
        if "by" in kwargs:
            if kwargs["kind"] == "bar":
                if self.plot_type.value == "bar":  # should not be set to True for 'grouped_bar' case
                    kwargs["stacked"] = True
            elif kwargs["kind"] == "scatter" and kwargs["by"] not in get_cat_cols(series_data) and ("datashade" not in kwargs or not kwargs["datashade"]):  # noqa
                kwargs["c"] = kwargs.pop("by")
        if kwargs["kind"] == "table":
            kwargs["columns"] = ([x_col] if x_col else []) + list(series_data.columns)
        elif kwargs["kind"] in ["violin", "box"]:
            if "by" in kwargs and segment_col:
                color_hook_kwargs["box_with_color"] = kwargs["by"]
                kwargs["c"] = "white"
                if x_col:
                    kwargs["by"] = [x_col, kwargs["by"]]
            if "by" not in kwargs and (x_col or segment_col):
                kwargs["by"] = []
                if x_col:
                    kwargs["by"] += [x_col]
                if segment_col:
                    kwargs["by"] += [segment_col]
                    color_hook_kwargs["box_with_color"] = segment_col
                    kwargs["c"] = "white"
            if "x" in kwargs:
                kwargs.pop("x")
        elif kwargs["kind"] == "bar":
            # kwargs['alpha'] = 0.8  # alpha < 1 disables point_selection
            kwargs["muted_fill_alpha"] = 0.1
            y_col = "value"
        elif kwargs["kind"] == "heatmap":
            kwargs["C"] = "count"
        if "y" not in kwargs:
            kwargs["y"] = y_col
        if self.plot_type.value == "grouped_bar" and "by" in kwargs:
            if "invert" in kwargs and kwargs["invert"]:
                color_hook_kwargs["bar_with_color"] = (kwargs["by"], series_data.index.name, "inverted")
            else:
                color_hook_kwargs["bar_with_color"] = (kwargs["by"], series_data.index.name, "upright")
            kwargs["groupby"] = kwargs["groupby"] if "groupby" in kwargs else []
            kwargs["groupby"] += [kwargs["by"]]
            kwargs.pop("by")
        return plotter, mapper, mapper_df, x_name, color_hook_kwargs

    def get_plot_by_type(self, series_data, group_by_cols, x_col, last_y=False, multiple_series=False, only_bars=False):
        """
        Generates the Plot based on the type of plot selected.

        :param series_data: pd.DataFrame (data set after filtering and sorting)
        :param group_by_cols: list (columns selected in split by cols)
        :param x_col: str (selected x column)
        :param last_y: bool (To check for last Y object)
        :param multiple_series: bool (To check if many Y-Expressions are present)
        :param only_bars: bool (To check if only Bar plots are present in other Y-Expressions)
        :return: plot of the selected type
        """
        if x_col == "" and self.x_type != "no_x":
            x_col = self.y_col
        width = 950 if self.parent.parent.notebook else 1000
        kwargs = {"legend": "top", "width": width, "height": 500}
        if self.display_name in series_data.columns:
            y_col = self.display_name
        else:
            cols_with_name = [col for col in series_data.columns if self.display_name in col]
            if cols_with_name:
                y_col = cols_with_name[0]
            else:
                raise Exception(f"{self.display_name} not found in series_data")
        # import pdb
        # pdb.set_trace()
        segment_col = self.segment_by
        series_data, x_is_date, y_is_date = self.preprocess_data_for_plot(series_data, x_col, group_by_cols, segment_col)    # noqa
        self.set_kwargs_for_bars(kwargs, series_data, x_col, y_col, segment_col, group_by_cols, multiple_series, only_bars)   # noqa

        plotter = hvPlot(series_data)
        plotter, mapper, self.mapper_df, x_name, color_hook_kwargs = \
            self.set_kwargs_for_plotter(plotter, kwargs, series_data, x_col, y_col, segment_col, group_by_cols)
        self.plotter_kwargs = kwargs
        current_plot = plotter(**kwargs)
        entire_plot = current_plot
        if current_plot.__module__ == "panel.layout":  # If the plot is split, it will be a panel layout on which the
            # further processing won't work. Need to extract the plot inside the layout to post process.
            current_plot = entire_plot[1].object
        if "bar_with_color" in color_hook_kwargs.keys():
            current_plot = current_plot.overlay(color_hook_kwargs["bar_with_color"][0])
            # if entire_plot.__module__ == 'panel.layout':
            #     for select in entire_plot[0][1]:
            #         if select.name == color_hook_kwargs['bar_with_color'][0]:
            #             entire_plot[0][1].remove(select)
            color_hook_kwargs["bar_with_color"] = (color_hook_kwargs["bar_with_color"][1], color_hook_kwargs["bar_with_color"][2])    # noqa
        if "box_with_color" in color_hook_kwargs.keys():  # Factors of Level_1 are to be passed to colorize_boxplot hook
            level_1_factors = [str(x) for x in series_data[color_hook_kwargs["box_with_color"]].dropna().unique().tolist()]    # noqa
            color_hook_kwargs["box_with_color"] = (color_hook_kwargs["box_with_color"], level_1_factors)
        current_plot = self.post_process_plot(current_plot, kwargs, x_col, last_y, mapper, self.mapper_df, x_name,
                                              x_is_date, y_is_date, color_hook_kwargs)
        self.set_sortable()
        # if "PointSelector" in [ch.__class__.__name__ for ch in self.parent.parent.children]:
        #     # Add selection feature only if widget has PointSelector class as child and one y_expr is being plotted
        #     selector = [ch for ch in self.parent.parent.children if ch.__class__.__name__ == "PointSelector"][0]
        #     if len(self.parent.children) == 1:
        #         current_plot = selector.set_selection_tools(self, current_plot, kwargs, x_col, plotter)
        #     else:
        #         selector.reset_selection()
        if entire_plot.__module__ == "panel.layout":
            entire_plot[1].object = current_plot
            current_plot = entire_plot
        self.plot = current_plot
        return current_plot

    def get_sorter(self, x_col, group_by_cols=[]):
        """A method generate the sorted data for plot."""
        if x_col == "" and self.x_type != "no_x":
            x_col = self.y_col
        col_name = self.display_name
        order = self.sort_rule.value
        y_vals = self.plot_data
        y_vals.sort_values(col_name, ascending=(order == "ASC"), inplace=True)
        if x_col:
            y_vals["tigerml_xcol_{}".format(x_col)] = y_vals.index.astype(str)
        index = y_vals.index
        index_name = index.name or "tigerml_index"
        sorter = td.DataFrame(index.values).rename(columns={0: index_name}).rename_axis("index")
        if group_by_cols:
            sorter = td.concat([sorter, y_vals[group_by_cols].reset_index(drop=True)], axis=1)
        sorter = sorter.reset_index().set_index([index_name] + group_by_cols).rename(columns={"index": "tigerml_sort_ranking"})    # noqa
        y_vals["tigerml_sort_ranking"] = sorter["tigerml_sort_ranking"].values
        return sorter

    def compute_plot_data(self, data, x_col, group_by_cols):
        """
        Computes the plot data after filtering.

          - computes plot data for a single Y expression

        :param data: pd.DataFrame (input data set after filtering)
        :param x_col: str (selected x col)
        :param group_by_cols: list (list of split by cols selected)
        :return: pd.DataFrame (plot data after aggregation(if any) with x col as index)
        """
        y = self.y_col
        series_name = self.display_name
        self.source_data = None  # Source data is used to map selected points to original indices in data
        if self.segment_by and self.segment_by not in group_by_cols:
            group_by_cols = group_by_cols.copy() + [self.segment_by]
        for col in data.columns:
            try:
                non_na_element = data[col].dropna().unique()[0]
            except:
                non_na_element = None
            if "Interval" in str(type(non_na_element)):
                data[col] = data[col].astype(str)
        if data.empty:
            self.plot_data = data
            self.source_data = data
        else:
            if x_col or self.x_type != "no_x":
                if x_col not in group_by_cols and x_col != "":
                    group_by_cols.append(x_col)
                agg_func = AGGS[self.agg_func.value]
                if agg_func:
                    if x_col:
                        self.plot_data = data[[y] + group_by_cols].rename(columns={y: series_name}).groupby(group_by_cols).agg(agg_func)    # noqa
                        if self.normalize_data:
                            norm_by = AGGS[self.normalize_by.value]
                            norm_across_cols = [x for x in group_by_cols if x not in self.normalize_across.value]
                            norm_data = data[[y] + norm_across_cols].rename(columns={y: "norm_by_value"}).groupby(norm_across_cols).agg(norm_by)    # noqa
                            self.plot_data = self.plot_data.merge(norm_data, how="left", left_on=norm_across_cols, right_index=True)    # noqa
                            self.plot_data[series_name] = self.plot_data[[series_name, "norm_by_value"]]\
                                .apply(lambda x: x[series_name] / x["norm_by_value"], axis=1)
                            self.plot_data.drop(["norm_by_value"], axis=1, inplace=True)
                        self.source_data = data[group_by_cols].merge(self.plot_data, how="left", left_on=group_by_cols, right_index=True)    # noqa
                        self.plot_data = self.plot_data.reset_index().set_index(x_col)
                    else:
                        self.plot_data = data[[y] + group_by_cols].groupby([y] + group_by_cols).agg({y: agg_func}).rename(columns={y: series_name})    # noqa
                        if self.normalize_data:
                            norm_by = AGGS[self.normalize_by.value]
                            norm_across_cols = [x for x in group_by_cols if x not in self.normalize_across.value]
                            norm_data = data[[y] + norm_across_cols].groupby([y] + norm_across_cols).agg({y: norm_by}).rename(columns={y: "norm_by_value"})    # noqa
                            self.plot_data = self.plot_data.merge(norm_data, how="left",
                                                                  left_on=[y] + norm_across_cols, right_index=True)    # noqa
                            self.plot_data[series_name] = self.plot_data[[series_name, "norm_by_value"]]\
                                .apply(lambda x: x[series_name] / x["norm_by_value"], axis=1)
                            self.plot_data.drop(["norm_by_value"], axis=1, inplace=True)
                        self.source_data = data[[y] + group_by_cols].merge(self.plot_data, how="left",
                                                                           left_on=[y] + group_by_cols, right_index=True)    # noqa
                        self.plot_data = self.plot_data.reset_index().set_index(y)
                else:
                    self.plot_data = data[[y] + group_by_cols].set_index(x_col)
                    self.source_data = self.plot_data.reset_index()
            else:
                self.plot_data = data[[y] + group_by_cols].rename(columns={y: series_name})
                self.source_data = self.plot_data

    def get_summary_stats(self, group_by_cols=None):
        """
        Generates a table containing the summary stats for the plot data.

        - describes the plot data using describe method for individual groups

        :return:
        summary_stats : pd.DataFrame (Data frame containing the summary for each group)
        """
        series_data = self.plot_data
        y_col = self.display_name
        segment_col = self.segment_by
        if segment_col and group_by_cols:
            summary_df = series_data.groupby(group_by_cols)
            summary_stats = summary_df.apply(lambda x: x.groupby(segment_col).apply(lambda df: df[y_col].describe().T).T)    # noqa
        elif group_by_cols:
            summary_stats = series_data.groupby(group_by_cols).apply(lambda df: df.describe().T).T
            summary_stats.columns = summary_stats.columns.swaplevel()
        elif segment_col:
            summary_stats = series_data.groupby(segment_col).apply(lambda df: df.describe().T).T
            summary_stats.columns = summary_stats.columns.swaplevel()
        else:
            summary_stats = series_data.describe()
        return summary_stats
