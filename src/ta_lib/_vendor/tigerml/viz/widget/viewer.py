from .states import StatefulUI
from .widget import VizWidget


class WidgetView(VizWidget):
    """A class for multiple WidgetViewer sub-module in viz."""

    def __init__(self, initial_state, free_states, *args, delete_callback=None, **kwargs):  # noqa
        self.initial_state = initial_state
        self.free_states = free_states
        super().__init__(*args, **kwargs, initial_state=initial_state)
        if delete_callback:
            self.del_callback = delete_callback
            self.close_btn = self.Button(css_clsses=["icon-button", "icon-cancel", "right"], width=50, height=50)
            self.close_btn.on_click(self.delete_self)

    def set_watchers(self, element, element_state):
        """A method to set the watcher for UI components of WidgetViewer."""
        if "children" in element_state:
            for ind, child_state in enumerate(element_state["children"]):
                try:
                    self.set_watchers(element.children[ind], child_state)
                except Exception as e:
                    import pdb

                    pdb.set_trace()
                    print(e)
        for key in [k for k in element_state.keys() if ".value" in k]:
            el_name = key.replace(".value", "")
            if hasattr(element, el_name):
                el = eval(f"element.{el_name}")
                el.param.watch(self.update_plot, "value")
                el.value = element_state[key]
                el.name = ""

    def describe(self, options, row=False):
        """A method that returns the description of the WidgetViewer."""
        if isinstance(options, list):
            if row:
                return self.Row(*[self.describe(opt) for opt in options])
            else:
                return self.Column(*[self.describe(opt) for opt in options])
        elif options == "x":
            x_description = ""
            if "x_col.value" in self.initial_state and self.initial_state["x_col.value"]:
                x_description = self.Row(f'<b> vs </b>{self.initial_state["x_col.value"]}', css_classes=["description"])  # noqa
            elif "x_col.value" in self.free_states:
                x_description = self.Row(" vs ", self.x_col, css_classes=["description"])
            if x_description and "x_sort.value" in self.free_states:
                x_description.append(self.x_sort)
            return x_description
        elif options == "y":
            y_description = self.y_exprs.describe()
            return y_description
        elif options == "filters":
            filter_decription = self.filters.describe()
            return filter_decription
        elif options == "splitter":
            splitter_description = ""
            if "splitter.value" in self.initial_state and self.initial_state["splitter.value"]:
                splitter_description = self.Row(f' split by {self.initial_state["splitter.value"]}',css_classes=["description"])  # noqa
            elif "splitter.value" in self.free_states:
                splitter_description = self.Row(" split by ", self.splitter, css_classes=["description"])
            if splitter_description and "split_plots.value" in self.free_states:
                self.split_plots.name = "Show in a Grid"
                self.split_plots.css_classes = self.split_plots.css_classes + ["show_grid_option"]
                splitter_description.extend([self.split_plots])
            return splitter_description
        else:
            raise Exception("options should be one or multiple of x, y, filters, splitter")

    def create_widget_view(self):
        """A method that creates the UI of WidgetViewer."""
        description = self.Column(
            self.close_btn,
            self.describe("filters"),
            self.Row(
                "<b>Plot: </b>",
                self.Column(
                    self.describe("y"), self.describe(["x", "splitter"], row=True)
                ),
                css_classes=["description"],
            ),
            "",
            css_classes=["full_width", "gray_bg", "description_pane"],
        )
        self.widget = self.Column(description, self.plot_wrapper)

    def delete_self(self, event=None):
        """A method that deletes the UI of WidgetViewer."""
        self.del_callback(self.pane)

    def _initiate(self):
        self._create_ui()
        # self._create_child_panes()
        self.create_widget_view()
        self.save_current_state()
        # super()._initiate()
        self.set_watchers(self, self.free_states)
