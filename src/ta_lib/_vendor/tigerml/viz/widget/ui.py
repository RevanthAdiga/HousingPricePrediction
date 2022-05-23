import logging
import threading
import time
from functools import partial

from tigerml.core.utils import measure_time, params_to_dict

from ..backends.panel import PanelBackend, zoom_style_path
from .components import DataLoader, DPWorkFlow, FilterPanel, YExprs
from .states import StatefulUI

_LOGGER = logging.getLogger(__name__)


class MessageBox(PanelBackend):
    """A class for custom panel MessageBox UI element."""

    def __init__(self):
        self.messages_ui = self.Column(css_classes=["message_box"])

    def show_timer(self, msg_ui, sleep_time):
        """A method to add timer to the MessageBox."""
        time.sleep(sleep_time)
        if msg_ui in self.messages_ui:
            self.messages_ui.remove(msg_ui)
        return

    def show_message(self, message, time=5):
        """A method to display the MessageBox."""
        close_btn = self.Button(css_classes=["icon-button", "icon-cancel"], width=50, height=40)  # noqa
        message_ui = self.Row(message, close_btn, css_classes=["message"])
        close_btn.on_click(partial(self.remove_message, message_ui))
        self.messages_ui.append(message_ui)
        thread = threading.Thread(target=partial(self.show_timer, message_ui, time))
        thread.start()

    def remove_message(self, mui, event=None):
        """A method to remove the MessageBox."""
        self.messages_ui.remove(mui)

    def show(self):
        """A method that returns the MessageBox UI component."""
        return self.messages_ui


class WidgetUI(StatefulUI):
    """Base UI class for viz module."""

    def __init__(self, initial_state={}, children=None):
        self.loading_message = self.Markdown("Computing the changes. Please wait.", height=490)
        if initial_state:
            dl_state, filter_state, y_state = initial_state["children"]
        else:
            dl_state, filter_state, y_state = [{}] * 3
        self.filters = FilterPanel(self, self.data, self.dtypes, initial_state=filter_state)
        self.y_exprs = YExprs(self, self.data, self.dtypes, initial_state=y_state)
        self.data_loader = DataLoader(self, self.load_data, initial_state)
        self.data_processor = DPWorkFlow(self.data, parent_widget=self)
        self.message_box = MessageBox()
        self.data_dependent = []
        if children:
            assert isinstance(children, list)
            children = [self.data_loader, self.data_processor, self.filters, self.y_exprs] + children
        super().__init__(children=children, initial_state=initial_state)

    @measure_time(_LOGGER)
    def _create_ui_config(self):
        x_col_options = lambda data, dtypes: [""] + sorted(list(data.columns))
        x_col = self.get_ui_dict(type="select", name="Select Column(X)", options=x_col_options,
                                 callback=self.y_exprs.refresh_plot_types)
        if not self.initial_state or all(["sort_rule.value" not in child_state for child_state in self.initial_state["children"][2]["children"]]):  # noqa
            x_sort_callback = self.remove_y_sorts
        else:
            x_sort_callback = None
        x_sort = self.get_ui_dict(type="select", options=["", "ASC", "DESC"], value="ASC",
                                  name="Sort", width=70, callback=x_sort_callback)
        x_options = self.get_ui_dict(type="row", children=params_to_dict(x_col=x_col, x_sort=x_sort),
                                     css_classes=["x_options"])
        split_plots = self.get_ui_dict(type="checkbox", name="Show Plots in Grid", css_classes=["grid_option"])
        splitter_options = lambda data, dtypes: sorted([col for col in self.dtypes if self.dtypes[col]
                                                        in ["bool", "category", "string"]])
        if not self.initial_state or ("splitter.value" in self.initial_state and "split_plots.value" in self.initial_state):  # noqa
            splitter_callback = self.set_grid_option
        else:
            splitter_callback = None
        splitter = self.get_ui_dict(type="multichoice", name="Select Columns(Splitter)", width=270,
                                    options=splitter_options, css_classes=["filter_multiselect"],
                                    callback=[splitter_callback, self.y_exprs.refresh_plot_types,
                                              self.y_exprs.update_normalizer])
        splitter_widget = self.get_ui_dict(type="column",
                                           children=params_to_dict(
                                               dummy=self.get_ui_dict(type="row",
                                                                      children=params_to_dict(splitter=splitter)),
                                               split_plots=None), css_classes=["splitter_widget"])
        color_options = lambda data, dtypes: [""] + sorted([col for col in self.dtypes if self.dtypes[col] in ["bool", "category"]])  # noqa
        color_axis = self.get_ui_dict(type="select", name="Color Axis", options=color_options,
                                      callback=[self.y_exprs.refresh_plot_types, self.y_exprs.update_normalizer])
        if not self.initial_state:
            update_button = self.get_ui_dict(type="button", name="UPDATE PLOT", width=200,
                                             css_classes=["primary", "button", "icon-button-prefix",
                                                          "icon-refresh-white"],
                                             callback=self.update_plot)
        else:
            update_button = self.null_component
        ctas = self.get_ui_dict(type="row", children={"update_button": update_button, "save_data_button": None},
                                css_classes=["ctas"])
        if self.data_access:
            save_data_button = self.get_ui_dict(type="button", name="DATA", width=50,
                                                css_classes=["button", "icon-button-prefix", "icon-download",
                                                             "download_button"],
                                                callback=self.save_data)
            ctas["kwargs"]["children"].update({"save_data_button": save_data_button})

        plot_wrapper = self.get_ui_dict(type="column", children={"plot": self.plot}, css_classes=["plot_wrapper"])
        if self.show_summary:
            summary_stats = self.get_ui_dict(type="column",
                                             children=params_to_dict(
                                                 dummy=self.get_ui_dict("### Summary Stats", type="markdown",
                                                                        css_classes=["section_header"]),
                                                 summary_table=None))
            plot_wrapper.update({"summary_stats": summary_stats})
        elements = {"x_options": x_options, "splitter_widget": splitter_widget, "ctas": ctas,
                    "plot_wrapper": plot_wrapper, "split_plots": split_plots, "color_axis": color_axis}
        if self.debugger:
            debug = self.togglegroup(options=["OFF", "ON"], name="Debugger", behavior="radio",
                                     css_classes=["tabs"], width=100)
            elements["debug"] = debug
        return elements

    @measure_time(_LOGGER)
    def _create_ui_elements(self):
        ui_config = self._create_ui_config()
        self._create_ui_from_config(ui_config, self.data, self.dtypes)
        self.body_row = self.Row(self.plot_wrapper, "")
        self.body_main = self.Column(self.body_row, css_classes=["body_right"])
        self.body = self.Row(self.body_main, css_classes=["widget_body", "full_width"])
        # self.plot_wrapper[0] = self.state_navigator()

    @measure_time(_LOGGER)
    def _create_child_panes(self):
        self.x_pane = self.Column(self.Row(self.SectionHeader('<h3 class="section_header">X Axis: </h3>'),
                                           self.Spacer(width=20), self.x_options),
                                  self.Row(self.SectionHeader('<h3 class="section_header">Color Axis: </h3>'),
                                           self.color_axis))
        self.splitter_pane = self.Row(self.SectionHeader('<h3 class="section_header">Split Plot by : </h3>'),
                                      self.splitter_widget)
        self.y_pane = self.Row(self.SectionHeader('<h3 class="section_header">Y Axes: </h3>'),
                               self.y_exprs.show(), css_classes=["y_exprs"])

    @measure_time(_LOGGER)
    def _create_standalone_widget(self):
        style_file = open(zoom_style_path, "r")
        zoom_style = style_file.read()
        # self.data_changer = self.Column(self.data_loader.show(), )
        self.plot_config = self.Row(self.y_pane, self.x_pane, self.splitter_pane, css_classes=["plot_data_controls"])
        self.viz_config = self.Column(self.data_processor.show(), self.filters.show(width=1350),
                                      self.plot_config, css_classes=["gray_bg", "full_width", "viz_config"])
        # self.top_row = self.Row(self.data_changer)
        self.header = self.Row(self.data_loader.show(), "",  # For additional space at the end
                               css_classes=["full_width", "widget_header", "widget-box"])

        if self.debugger:
            self.viz_config.insert(0, self.Row("Debugger", self.debug))
        self.body_top_row = self.Row(self.ctas, css_classes=["full_width"])
        self.body_main.insert(0, self.body_top_row)
        widget = self.Column(self.HTML(f"<style>{zoom_style}</style>", height=0, css_classes=["is_hidden"]),
                             self.header, self.viz_config, self.body,
                             css_classes=["iwidget", "standalone", "add_glass"])
        return widget

    def _create_notebook_widget(self):
        self.controls = self.Column(self.data_loader.show(), self.data_processor.show(), self.filters.show(width=950),
                                    self.Row(self.y_pane, self.Column(self.x_pane, self.splitter_pane),
                                             css_classes=["plot_data_controls"]), self.ctas, css_classes=["controls"])
        widget = self.Column(self.controls, self.body, css_classes=["iwidget", "notebook", "add_glass"])
        return widget

    def remove_x_sort(self):
        """Removes the x sort value when y sort is enabled."""
        self.x_sort.value = ""
        self.x_options[1].value = ""

    def set_grid_option(self, event):
        """A callback fuction that sets the split_plots option."""
        # self.splitter_widget.clear()
        # self.splitter_widget[0][1] = ''
        # self.splitter_widget[0][1] = self.splitter
        if event.new:
            self.splitter_widget[1] = self.split_plots
            self.split_plots.disabled = False
        else:
            self.splitter_widget[1] = self.null_component
            self.split_plots.value = False

    def remove_y_sorts(self, event):
        """Removes y sorts if any of the other Y or X has sort enabled."""
        if event.new:
            self.y_exprs.remove_y_sorts()
