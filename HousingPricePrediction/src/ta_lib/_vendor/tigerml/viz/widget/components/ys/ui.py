from tigerml.core.utils import get_num_cols

from ...states import StatefulUI


def distinct_count(series):
    return series.nunique()


AGGS = {
    "": "",
    "sum": sum,
    "mean": "mean",
    "max": max,
    "min": min,
    "count": "count",
    "distinct count": distinct_count,
}


class YExprUI(StatefulUI):
    """A class for all the single Y-Expression's UI components."""

    def __init__(self, data, dtypes, initial_state={}):
        self.data = data
        self.dtypes = dtypes
        self.ui_created = False
        self.pane = None
        super().__init__(initial_state=initial_state)

    def _compute_y_cols(self, data, dtypes):
        self.data = data
        self.dtypes = dtypes
        return get_num_cols(data) + [x for x in data.columns if x not in get_num_cols(data)]  # noqa

    def _create_ui(self):
        col_options_func = lambda data, dtypes: self._compute_y_cols(data, dtypes)
        col_name = self.select(width=210, name="Select Column(Y)", options=col_options_func)
        from functools import partial

        agg_func = self.select(value="", name="Aggregation", width=110, options=partial(self.agg_options, None))
        have_color_axis = self.checkbox(name="Have Color Axis", width=110, css_classes=["color_axis_checkbox"])
        # self._update_data_columns()
        # self.col_name.options = [col for col in self.dtypes if self.dtypes[col] == 'numeric']
        sort_rule = self.select(options=["", "ASC", "DESC"], name="Sort", width=80)
        normalize_by = self.select(name="Normalize by", options=["count", "sum"], width=80)
        normalize_across = self.multichoice(name="Normalize across", options=[], width=225)
        axis = self.togglegroup(options=["left", "right"], name="Y Axis", width=100, behavior="radio", css_classes=["tabs"])  # noqa
        plot_type = self.select(name="Plot Type", width=170)
        ui_config = {
            "col_name": col_name,
            "agg_func": agg_func,
            "have_color_axis": have_color_axis,
            "sort_rule": sort_rule,
            "normalize_by": normalize_by,
            "normalize_across": normalize_across,
            "axis": axis,
            "plot_type": plot_type,
        }
        self._create_ui_from_config(ui_config, self.data, self.dtypes)
        self.normalize_by.disabled = True
        if not self.initial_state:
            if self.data is not None:
                self.agg_func.value = "count" if self.col_name.value not in get_num_cols(self.data) else ""
            self.delete_button = self.Button(width=30, height=30,
                                             css_classes=["icon-button", "icon-delete", "is_hidden", "y_delete"])
            self.toggle = self.Toggle(name="", width=30, height=30,
                                      css_classes=["icon-button", "icon-expand", "y_expand"])
            # bindings
            self.col_name.param.watch(self.set_agg_and_segmentations, "value")
            self.sort_rule.param.watch(self.remove_other_sorts, "value")
            self.delete_button.on_click(self.delete_y)
            self.toggle.param.watch(self.toggle_advanced, "value")
            self.col_name.param.watch(self.refresh_plot_types, "value")
            self.have_color_axis.param.watch(self.refresh_plot_types, "value")
            self.agg_func.param.watch(self.refresh_plot_types, "value")
            self.normalize_by.param.watch(self.refresh_plot_types, "value")
            self.normalize_across.param.watch(self.refresh_plot_types, "value")
            self.col_name.param.watch(self.update_normalizer, "value")
            self.have_color_axis.param.watch(self.update_normalizer, "value")
            self.agg_func.param.watch(self.update_normalizer, "value")
            self.normalize_across.param.watch(self.activate_normalizer, "value")
        else:
            if "col_name.value" not in self.initial_state:
                if "agg_func.value" not in self.initial_state:
                    self.col_name.param.watch(self.agg_options, "value")
            if ("sort_rule.value" not in self.initial_state and  # noqa
                    len([y for y in self.parent.children if "sort_rule.value" not in y.initial_state]) > 1):
                self.sort_rule.param.watch(self.remove_other_sorts, "value")
            if "plot_type.value" not in self.initial_state:
                self.col_name.param.watch(self.refresh_plot_types, "value")
                self.have_color_axis.param.watch(self.refresh_plot_types, "value")
                self.agg_func.param.watch(self.refresh_plot_types, "value")

        self.ui_created = True

    def _set_defaults(self):
        for key in self.current_state:
            self.current_state[key] = ""
        self.current_state["plot_type.value"] = self.plot_type.options[0] if self.plot_type.options else ""

    def set_state(self, passed_state):
        """A method to set the state of Y-Expression if any initial state is passed."""
        super().set_state(passed_state)
        self.refresh_plot_types()
        self.plot_type.value = passed_state["plot_type.value"]

    def agg_options(self, event=None, data=None, dtypes=None):
        """A method to compute the Aggregation options available for the selected Y-column."""
        col = event.new if event else self.col_name.value
        dtypes = dtypes if dtypes is not None else self.dtypes
        # data = data if data is not None else self.data
        if not col:
            return []
        elif dtypes[col] != "numeric":
            return [""] + [func for func in AGGS.keys() if "count" in func]
        else:
            return list(AGGS.keys())

    def set_agg_and_segmentations(self, event=None):
        """A method to set the Aggregation options based on the selected Y-column."""
        self.agg_func.options = self.agg_options(event=event)

    def update_normalizer(self, event=None):
        """A method to update the Normalizer options based on the selected Y-column."""
        if event.obj.name == "Select Column(Y)":
            if self.dtypes[event.new] == "numeric":
                self.normalize_by.options = ["count", "sum"]
            else:
                self.normalize_by.options = ["count"]
        elif event.obj.name == "Aggregation":
            if event.new == "sum":
                self.normalize_by.options = ["count", "sum"]
            else:
                self.normalize_by.options = ["count"]
        else:
            option_list = self.parent.parent.splitter.value.copy()
            if self.have_color_axis.value:
                option_list += [self.segment_by]
            self.normalize_across.options = option_list
        self.activate_normalizer()

    def activate_normalizer(self, event=None):
        """Determines if it is possible to normalize plot_data for chosen state and enables/disables normalize_by."""
        if self.agg_func.value in ["sum", "count"] and self.normalize_across.value != []:
            self.normalize_by.disabled = False
        else:
            self.normalize_by.disabled = True

    def create_pane(self):
        """A method that creates UI components of Y-Expression object."""
        if not self.ui_created:
            self._create_ui()
        self.y_def = self.Row(self.agg_func, self.col_name, width=460, css_classes=["y_def"])
        self.layout_options = self.Row(self.plot_type,
                                       self.Column(self.Markdown("Y Axis", css_classes=["no_spacing"]), self.axis),
                                       self.have_color_axis,
                                       width=460,
                                       css_classes=["y_def"])
        if not self.initial_state:
            self.y_def.extend([self.toggle, self.delete_button])
        self.normalize = self.Row(self.normalize_by, self.normalize_across, width=360)
        self.advanced_options = self.Row(self.normalize, self.sort_rule, width=460, css_classes=["y_advanced_options"])
        self.pane = self.Column(self.y_def, self.layout_options, width=460, css_classes=["y_expr"])
        if self.initial_state:
            self.pane.append(self.advanced_options)
        else:
            self.pane.append(self.null_component)
        return self.pane

    def show(self):
        """A method that returns the full Y-Expression's UI component."""
        if not self.pane:
            return self.create_pane()
        else:
            return self.pane


class YExprsUI(StatefulUI):
    """A class for multiple Y-Expressions UI component."""

    def __init__(self, data, dtypes, children=[], initial_state={}):
        self.data = data
        self.dtypes = dtypes
        super().__init__(children=children, dynamic_ui=True, initial_state=initial_state)
