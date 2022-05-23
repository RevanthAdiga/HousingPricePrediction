from tigerml.core.utils import flatten_list, set_logger_config

from .backends.panel import tigerml_logo_path
from .point_selector import PointSelector
from .state_tracker import StateTracker
from .widget import VizWidget
from .widget.components.story_board import StoryBoard


class DataExplorer(VizWidget, StateTracker):
    """Base class for DataExplorer in viz module."""

    def __init__(self, *args, **kwargs):
        """
        Initializes the DataExplorer.

            - Infers Data types from data set, creates a UI for filters, y expressions and
            story board captures the current values in the widgets

        :parameter data : pd.DataFrame (input data set)
        """
        # if 'verbose' in kwargs.keys():
        #     set_logger_config(verbose=kwargs['verbose'])
        #     kwargs.pop('verbose')
        # else:
        #     set_logger_config(verbose=0)
        self.story_board = StoryBoard()
        StateTracker.__init__(self)
        self.selector = PointSelector(self)
        children = [self.selector]
        # PointSelector.__init__(self)
        super().__init__(*args, **kwargs, children=children)

    def _create_ui_elements(self):
        super()._create_ui_elements()
        self.pin_plot = self.Button(name="PLOT", css_classes=["tertiary", "button", "icon-button-prefix", "icon-pin"],  # noqa
                                    width=50)
        self.pin_plot.on_click(self.save_to_board)
        self.ctas.insert(1, self.pin_plot)
        self.shutdown = self.Button(css_classes=["icon-button", "icon-shutdown", "right"], width=50, height=35)
        self.shutdown.on_click(self.close)

    def _create_standalone_widget(self):
        widget = super()._create_standalone_widget()
        self.ctas.css_classes.append("right")
        logo_file = open(tigerml_logo_path, "r")
        logo_str = logo_file.read()
        self.logo = self.HTML(
            '<div class="logo">'
            '<img src="' + logo_str + '" width="200">'
            '<span class="logo_text">DATA EXPLORER</span>'
            "</div>",
            width=500,
        )
        self.right_bar = self.Column(self.get_bookmarks_ui(), css_classes=["right_side_bar"], width=440)
        self.header.insert(0, self.logo)
        self.body.append(self.right_bar)
        self.body_top_row.insert(0, self.state_navigator())
        self.body_row[1] = self.selector.get_ui()
        widget.append(self.story_board.pane)
        return widget

    def _create_notebook_widget(self):
        widget = super()._create_notebook_widget()
        # widget.append(self.shutdown)
        bookmark_pane = self.Column(self.get_bookmarks_ui(), css_classes=["right"])
        self.controls.append(self.Row(self.state_navigator(), bookmark_pane, css_classes=["full_width"]))
        self.controls.append(self.selector.get_ui())
        # self.body_main.insert(0, )
        # self.body_main.insert(1, )
        widget.append(self.story_board.pane)
        return widget

    def save_to_board(self, event=None):
        """Adds the plot to the story board along with the summary table."""
        self.story_board.add_item(self.plot, title=self.widget_title)

    @property
    def has_data_changes(self):
        """Tracks if there are any data or selection changes."""
        return super().has_data_changes or self.selector.has_state_change()

    def get_plot(self, **kwargs):
        """A method that generates the plot after processing the data."""
        if "event" in kwargs.keys():
            if not self.y_exprs._is_valid_state(event=kwargs["event"]):
                self.plot = None
                return
            kwargs.pop("event")
        self.filter_data(**kwargs)
        data = self.filtered_data
        limit_to_sels_data = list(set(flatten_list([sel.data for sel in self.selector.children if sel.actions.value == "limit to"])))  # noqa
        exclude_sels_data = list(set(flatten_list([sel.data for sel in self.selector.children if sel.actions.value == "exclude"])))  # noqa
        if limit_to_sels_data:
            data = data[data.index.isin(limit_to_sels_data)]
        if exclude_sels_data:
            data = data[~data.index.isin(exclude_sels_data)]
        # super().get_plot()
        compare_sels = [sel for sel in self.selector.children if sel.actions.value == "compare"]
        comparison_plots = None
        if compare_sels:
            for sel in compare_sels:
                sel_data = data[data.index.isin(sel.data)]
                self.create_plot_data(data=sel_data)
                for y_expr in self.y_exprs.children:
                    y_expr.plot_data.rename(columns={y_expr.display_name: y_expr.display_name + f" - {sel.name}"},
                                            inplace=True)
                self.sort_data()
                if comparison_plots:
                    comparison_plots *= self.create_plot()
                else:
                    comparison_plots = self.create_plot()
            compare_sels_data = list(set(flatten_list([sel.data for sel in compare_sels])))
            data = data[~data.index.isin(compare_sels_data)]
        self.create_plot_data(data=data, **kwargs)
        self.sort_data(**kwargs)
        plot = self.create_plot(**kwargs)
        if comparison_plots:
            plot *= comparison_plots
        self.plot = plot

    def update_plot(self, event=None, **kwargs):
        """A callback function that triggers the plot generation process once UPDATE PLOT button is clicked."""
        super().update_plot(event=event, **kwargs)
        self.selector.pane[0] = self.selector.get_selection_summary()
        if event is not None:
            self.update_states_list()

    def create_pane(self, threaded=True):
        """A method that creates the UI component of DataExplorer."""
        self.story_board.update_ui()
        super().create_pane()
        if threaded:
            if not self.notebook:
                self.header.append(self.shutdown)
        self.pane.append(self.message_box)
