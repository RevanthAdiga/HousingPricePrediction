import gc
import logging
from functools import partial

import pandas as pd

from tigerml.core.plots import hvPlot
from tigerml.core.plots.bokeh import (
    add_to_primary_dynamicmap,
    add_to_secondary_dynamicmap,
    colorize_barplot,
    finalize_axes_left,
    finalize_axes_right,
    fix_overlay_legends,
    legend_policy,
)
from tigerml.core.utils import time_now_readable

from .single import YExpr
from .ui import YExprsUI


class YExprs(YExprsUI):
    """A class for multiple Y-Expressions object."""

    def __init__(self, parent, data, dtypes, initial_state={}):
        self.add_yexpr = self.Button(name="Add New Series", css_classes=["add_y", "inline_button"])  # noqa
        self.data = data
        self.parent = parent
        self.dtypes = dtypes
        # self.children = [YExpr(self, data, self.dtypes)]
        self.add_yexpr.on_click(self.add_new_y)
        self.plot_data = None
        self.plot = None
        self.summary_table = None
        super().__init__(data, dtypes, children=[], initial_state=initial_state)
        if initial_state:
            [self.create_child(state, initial=True) for state in initial_state["children"]]
        else:
            self._create_child()

    @property
    def has_data_changes(self):
        """A property that tracks the data changes in YExpr."""
        return any([y.has_data_changes for y in self.children]) or self.current_state["no_of_children"] != len(self.children)  # noqa

    @property
    def has_sort_changes(self):
        """A property that tracks the sort option changes in YExpr."""
        return any([y.has_sort_changes for y in self.children])

    @property
    def has_plot_changes(self):
        """A property that tracks the plot_type and axis selection changes in YExpr."""
        return any([y.has_plot_changes for y in self.children])

    @property
    def sort_col(self):
        """A property that tracks the column by which the YExpr has to be sorted."""
        sort_cols = [y for y in self.children if y.sort_rule.value and not y.segment_by]
        if sort_cols:
            return sort_cols[0]
        else:
            return ""

    def refresh_plot_types(self, event=None):
        """A method to invoke the refresh_plot_type function in all individual YExpr."""
        for y in self.children:
            y.refresh_plot_types(event=event)

    def update_normalizer(self, event=None):
        """A method to invoke the update_normalizer function in all individual YExpr."""
        for y in self.children:
            y.update_normalizer(event=event)

    def _is_valid_state(self, event=None):
        valid_state_list = []
        for y in self.children:
            valid_state_list += [y._is_valid_state(event=event)]
        return all(valid_state_list)

    def describe(self):
        """A method that provides the description of the entire Y-Expressions object in widget_builder."""
        ys = self.Column(*[y.describe() for y in self.children], css_classes=["description"])
        if len(ys) > 1:
            for ind, y in enumerate(ys):
                if ind != 0:
                    y.insert(0, "<b> and </b>")
        return ys

    def _create_child(self, initial_state={}):
        new_y_expr = YExpr(self, self.data, self.dtypes, initial_state=initial_state)
        new_y_expr.refresh_plot_types()  # to add plot_type options for the new yexpr
        self.children.append(new_y_expr)
        return new_y_expr

    def create_child(self, child_state, initial=False):
        """A method to set the state of Y-Expressions if any initial state is passed."""
        initial_state = {}
        if initial:
            initial_state = child_state
        child = self._create_child(initial_state=initial_state)
        child.set_state(child_state)

    def add_new_y(self, event=None):
        """A method to add new Y-Expression child to Y-Expressions."""
        new_y_expr = self._create_child()
        self.refresh_ui()
        # self.children_ui.append(new_y_expr.show())

    def delete_y(self, y_expr, event=None):
        """A method to delete a particular Y-Expression child of Y-Expressions."""
        # index = self.children.index(y_expr)
        # self.children_ui.remove(self.children_ui[index])
        self.children.remove(y_expr)
        self.refresh_ui()

    def remove_y_sorts(self, y_expr=None):
        """A method to deactivate sorting by Y-column if X-column sort is active."""
        if y_expr:
            self.parent.remove_x_sort()
            index = self.children.index(y_expr)
            ys = [y for ind, y in enumerate(self.children) if index != ind]
        else:
            ys = self.children
        for y_expr in ys:
            y_expr.sort_rule.value = ""

    def compute_plot_data(self, data, x_col, split_by_cols, recompute):
        """
        Computes the plot data after filtering.

             - computes plot data for a multiple Y expressions

        :param data: pd.DataFrame (input data set after filtering)
        :param x_col: str (selected x col)
        :param split_by_cols: list (list of split by cols selected)
        :param recompute: bool (monitors for data changes in UI)
        :return: pd.DataFrame (plot data after aggregation(if any) with x col as index)
        """
        if self.parent.data_processor.has_state_change():
            recompute = True
        if recompute:
            for y in self.children:
                del y.plot_data
                gc.collect()
                y.plot_data = None
            changed_y_cols = [y for y in self.children if y.y_col]
        else:
            unchanged_ys = [y for y in self.children if not y.has_data_changes]
            for y in [y for y in self.children if y not in unchanged_ys]:
                del y.plot_data
                gc.collect()
                y.plot_data = None
            changed_y_cols = [
                y for y in self.children if y.y_col and y.has_data_changes
            ]
        group_by_cols = split_by_cols.copy()
        for y_expr in changed_y_cols:
            y_expr.compute_plot_data(data, x_col, group_by_cols)

    def sort_data(self, x_col, x_sort, group_by_cols=[]):
        """A method to sort data based on selected sort options."""
        y_sort_col = self.sort_col
        if y_sort_col:
            sorter = y_sort_col.get_sorter(x_col, group_by_cols=group_by_cols)
        # elif x_col:
        # 	col_name = x_col
        elif not x_col:
            return
        for y in self.children:
            if y_sort_col and y == y_sort_col:
                continue
            if "tigerml_sort_ranking" in y.plot_data:
                y.plot_data.drop("tigerml_sort_ranking", axis=1, inplace=True)
            if x_col and "tigerml_xcol_{}".format(x_col) in y.plot_data:
                y.plot_data.drop("tigerml_xcol_{}".format(x_col), axis=1, inplace=True)
            if y_sort_col:
                if x_col:
                    y.plot_data["tigerml_xcol_{}".format(x_col)] = y.plot_data.index.astype(str)
                y.plot_data["tigerml_x_copy"] = y.plot_data.index.values
                y.plot_data = y.plot_data.merge(sorter, right_index=True, left_on=["tigerml_x_copy"] + group_by_cols, copy=False)  # noqa
                y.plot_data.drop("tigerml_x_copy", axis=1, inplace=True)
                if y.segment_by:
                    y.plot_data.sort_values("tigerml_sort_ranking", ascending=True, inplace=True)
                else:
                    y.plot_data.sort_values("tigerml_sort_ranking", ascending=True, inplace=True)
            else:
                order = x_sort or "ASC"
                y.plot_data.sort_index(ascending=(order == "ASC"), inplace=True)

    def _add_to_plot(self, plot, y_expr, x_col, group_by_cols=None, parent_plot="", last_y=False, only_bars=False):
        # plot_type = y_expr.plot_type.value
        segment_col = y_expr.segment_by
        y_axis = y_expr.axis.value
        if parent_plot == "":
            parent_plot = self.plot
        if plot.__class__.__name__ == "DynamicMap" or group_by_cols:
            hooks = []
            if y_axis == "right":
                hooks.append(add_to_secondary_dynamicmap)
                if last_y:
                    hooks.append(finalize_axes_right)
            else:
                hooks.append(add_to_primary_dynamicmap)
                if last_y:
                    hooks.append(finalize_axes_left)
            hooks.append(fix_overlay_legends)
            callback = None
            if "_callback_param_value" in plot.__dict__:
                callback = plot.__dict__["_callback_param_value"]
            if callback and hasattr(callback, "operation") and hasattr(callback.operation, "__name__") and \
                    callback.operation.__name__ == "dynspread":
                pass
            else:
                if group_by_cols:
                    existing_hooks = []
                    # in case of Dynamicmap the existing hooks can be accessed only through individual plot objects
                    # also When splitter is used, the plot object becomes panel column with widget on top right
                    DMap = plot[1].object
                    eval_str = "DMap" + str([dims.values[0] for dims in DMap.kdims])
                    first_plot = eval(eval_str)  # getting first_plot in case of multi-level split by
                    if "hooks" in first_plot.opts.get().kwargs:
                        existing_hooks = first_plot.opts.get().kwargs["hooks"]
                    plot[1].object.opts(hooks=existing_hooks + hooks)
                else:
                    existing_hooks = []
                    if "hooks" in plot.opts.get().kwargs:
                        existing_hooks = plot.opts.get().kwargs["hooks"]
                    plot.opts(hooks=existing_hooks + hooks)
        if only_bars and plot.__class__.__name__ == "Bars" and not (y_expr.x_type in ["non_numeric", "no_x"] and y_expr.y_type == "non_numeric"):  # noqa
            plot_data = plot.data
            if segment_col:
                plot_data["metric"] = plot_data["metric"] + " - " + plot_data[segment_col].astype(str)
            if parent_plot is not None:
                complete_data = pd.concat([parent_plot.data, plot_data])
            else:
                complete_data = plot_data
            # by_cols = ['metric']
            # parent_plot = hvPlot(complete_data).bar(f'tigerml_xcol_{x_col}'
            #                                         if f'tigerml_xcol_{x_col}' in complete_data else x_col,
            #                                         'value', by=by_cols)
            kwargs = y_expr.plotter_kwargs.copy()
            kwargs["x"] = f"tigerml_xcol_{x_col}" if f"tigerml_xcol_{x_col}" in complete_data else x_col
            kwargs["y"] = "value"
            kwargs["by"] = "metric"
            kwargs.pop("legend")
            plotter = hvPlot(complete_data)
            parent_plot = plotter(**kwargs)

            existing_hooks = []
            if "hooks" in plot.opts.get().kwargs:
                existing_hooks = plot.opts.get().kwargs["hooks"]
                for i in range(len(existing_hooks)):
                    if existing_hooks[i].__class__.__name__ == "partial" and existing_hooks[i].func.__name__ == "colorize_barplot":  # noqa
                        existing_hooks[i] = partial(colorize_barplot, ("metric", existing_hooks[i].args[0][1]))
            parent_plot.opts(xlabel=x_col, multi_level=False, hooks=existing_hooks)
        elif parent_plot is not None:
            if group_by_cols:
                parent = parent_plot[1].object
                current = plot[1].object
            else:
                parent = parent_plot
                current = plot
            if current.__class__.__name__ == "DynamicMap":
                # getting the type of first plot in case of DynamicMap
                current_class_name = current.values()[0].__class__.__name__
            else:
                current_class_name = current.__class__.__name__
            if parent.__class__.__name__ == "DynamicMap":
                # getting the type of first plot in case of DynamicMap
                parent_class_name = parent.values()[0].__class__.__name__
            else:
                parent_class_name = parent.__class__.__name__
            separate_plots = ["Table", "Layout", "HeatMap"]
            if current_class_name in separate_plots or parent_class_name in separate_plots:
                parent = parent + current
                parent.cols(1)
            else:
                parent = parent * current
            if group_by_cols:
                parent_plot[1].object = parent
            else:
                parent_plot = parent
        else:
            parent_plot = plot
        existing_hooks = []
        if parent_plot.__class__.__name__ == "Layout":
            for plot in parent_plot:
                if "hooks" in plot.opts.get().kwargs:
                    existing_hooks = plot.opts.get().kwargs["hooks"]
                hooks = existing_hooks + [fix_overlay_legends]
                plot = plot.options(hooks=hooks)
        elif parent_plot.__class__.__name__ == "Column":
            plot_obj = parent_plot[1].object
            if plot_obj.values() and "hooks" in plot_obj.values()[0].opts.get().kwargs:
                # plot_obj.values()[0].opts.get().kwargs["hooks"] += [fix_overlay_legends]
                plot_obj.values()[0].opts.get().kwargs["hooks"] += [fix_overlay_legends]
            #     existing_hooks = plot_obj.values()[0].opts.get().kwargs["hooks"]
            # hooks = existing_hooks + [fix_overlay_legends]
            # plot_obj = plot_obj.options(hooks=hooks)
            # parent_plot[1].object = plot_obj
        else:
            if "hooks" in parent_plot.opts.get().kwargs:
                existing_hooks = parent_plot.opts.get().kwargs["hooks"]
            hooks = existing_hooks + [fix_overlay_legends]
            parent_plot = parent_plot.options(hooks=hooks)
        return parent_plot

    def create_plot(self, x_col, split_by_cols, grid_split, show_summary):
        """A method to invoke the get_plot_by_type function in all individual YExpr."""
        self.plot = None
        # self.selections = None
        group_by_cols = split_by_cols.copy()
        last_y = False
        y_exprs = self.children
        summary_df = None
        bar_plots = [y for y in y_exprs if y.plot_type.value in ["bar", "grouped_bar"]]
        non_bar_plots = [y for y in y_exprs if y.plot_type.value not in ["bar", "grouped_bar"]]
        only_bars = False if non_bar_plots else True
        y_exprs = bar_plots + non_bar_plots
        for ind, y_expr in enumerate(y_exprs):
            if ind == len(y_exprs) - 1:
                last_y = True
            plot_series = y_expr.plot_data
            current_plot = y_expr.get_plot_by_type(plot_series, group_by_cols, x_col, last_y=last_y,
                                                   multiple_series=len(y_exprs) > 1, only_bars=only_bars)
            if show_summary:
                summary_stats = y_expr.get_summary_stats(group_by_cols)
                if summary_df is None:
                    summary_df = summary_stats
                else:
                    summary_df.merge(summary_stats, how="outer", left_index=True, right_index=True)
            self.plot = self._add_to_plot(current_plot, y_expr=y_expr, x_col=x_col, group_by_cols=group_by_cols,
                                          last_y=last_y, only_bars=only_bars)
        if "table" not in [y_expr.plot_type.value for y_expr in y_exprs]:
            try:
                existing_hooks = []
                if "hooks" in self.plot.opts.get().kwargs:
                    existing_hooks = self.plot.opts.get().kwargs["hooks"]
                self.plot.opts(active_tools=["wheel_zoom"], xrotation=45, hooks=existing_hooks + [legend_policy])
            except Exception:
                logging.info("legend_policy hook not applied to the plot")
        if grid_split:
            self.plot = self.plot[1].object.layout().cols(1)
        # elif self.plot.__class__.__name__ == 'DynamicMap' and len(self.plot.kdims) > 0:
        #     self.plot.opts(widget_location='top')
        # IMPORTANT: For summary table - commented for now
        # elif self.plot.__class__.__name__ == 'Layout':
        #     if split_by_cols:
        #         self.plot = self.plot.cols(1)
        #     else:
        #         self.plot = self.plot.cols(2)
        if show_summary:
            kwargs = {}
            if summary_df.columns.nlevels > 1:
                summary_df.columns = [f"{i}|{j}" if j != "" else f"{i}" for i, j in summary_df.columns]
            summary_df.index.name = "metrics"
            summary_data = summary_df.T.reset_index().rename(columns={"index": "Y Series"})
            plotter = hvPlot(summary_data)
            # if group_by_cols:
            #     kwargs['groupby'] = group_by_cols
            kwargs["kind"] = "table"
            kwargs["width"] = 1000
            kwargs["columns"] = list(summary_data.columns)
            self.summary_table = plotter(**kwargs)
            # self.summary_table = summary_data
        # if len(y_exprs) == 1:
        #     return (self.plot + y_exprs[0].selection_summary).cols(2)
        # else:
        return self.plot

    def save_data(self, x_col, group_by_cols=[]):
        """A method to concatenate and save the data from all individual YExpr."""
        # import pdb
        # pdb.set_trace()
        all_data = None
        for y in self.children:
            drop_cols = []
            if "tigerml_sort_ranking" in y.plot_data:
                drop_cols.append("tigerml_sort_ranking")
            if x_col and "tigerml_xcol_{}".format(x_col) in y.plot_data:
                drop_cols.append("tigerml_xcol_{}".format(x_col))
            data = y.plot_data.drop(drop_cols, axis=1)
            merge_cols = group_by_cols + ([x_col] if x_col else [])
            if y.segment_by:
                wide_data = None
                for level in data[y.segment_by].unique():
                    filtered_data = data[data[y.segment_by] == level].drop(y.segment_by, axis=1)
                    if wide_data is not None:
                        if merge_cols:
                            wide_data = wide_data.merge(filtered_data, how="outer", left_on=merge_cols,
                                                        right_on=merge_cols)
                            wide_data = wide_data.rename(columns={y.display_name: "{}_{}".format(y.display_name, level)})  # noqa
                        else:
                            wide_data["{}_{}".format(y.display_name, level)] = filtered_data[y.display_name]
                    else:
                        wide_data = filtered_data
                        wide_data = wide_data.rename(columns={y.display_name: "{}_{}".format(y.display_name, level)})
                data = wide_data
            if all_data is not None:
                all_data = all_data.merge(data, how="outer", left_on=merge_cols, right_on=merge_cols)
            else:
                all_data = data
        if all_data is not None:
            all_data.to_csv("data_saved_at_{}.csv".format(time_now_readable()))

    def get_children_ui(self, row=False):
        """A method that returns the UI objects of all the children of Y-Expressions."""
        if row:
            return self.Row(*[y.show() for y in self.children])
        else:
            return self.Column(*[y.show() for y in self.children])

    def show(self, row=False):
        """A method that returns the Y-Expressions UI object."""
        self.children_ui = self.get_children_ui()
        self.pane = self.Column(self.children_ui)
        if not self.initial_state:
            self.pane.append(self.add_yexpr)
        return self.pane

    def refresh_ui(self):
        """A method that reloads the Y-Expressions UI object."""
        self.children_ui = self.get_children_ui()
        self.pane[0] = self.children_ui

    def compute(self):
        """A method that creates the source data for plot generation."""
        return self.parent.create_plot_data(recompute=True)
