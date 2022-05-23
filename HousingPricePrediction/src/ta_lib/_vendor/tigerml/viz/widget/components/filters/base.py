import tigerml.core.dataframe as td
from tigerml.viz.widget.states import StatefulUI

from .wrapper import FilterWrapper


class FilterPanel(StatefulUI):
    """A class for FilterPanel object (parent for objects of type FilterCondition and FilterWrapper)."""

    def __init__(self, parent, data, dtypes, initial_state={}):
        super().__init__(children=[], dynamic_ui=True)
        self.parent = parent
        self.data = data
        self.dypes = dtypes
        self.initial_state = initial_state
        self._create_data_dict()
        self.add_filters = self.Button(
            name="+ Add", css_classes=["button", "tertiary"], width=50
        )
        self.add_filters.on_click(self._initiate_filters)

    @property
    def filter_group(self):
        """A property that tracks if FilterPanel has at least one child."""
        if self.children:
            return self.children[0]
        else:
            return None

    def create_ui(self, **kwargs):
        """A method to create the FilterPanel UI object."""
        return self.Row(
            self.SectionHeader(
                '<h3 class="section_header">Filters: </h3>', valign="middle"
            ),
            self.Column(
                "" if self.filter_group else self.add_filters,
                self.Spacer(height=1),
                self.Column(self.filter_group.show())
                if self.filter_group
                else self.Column(""),
                self.Spacer(height=1),
            ),
            css_classes=["filter_module"],
            **kwargs
        )

    def _initiate_filters(self, event=None):
        self.children = [
            FilterWrapper(self, self.data, self.data_dict, self.initial_state)
        ]  # to be assigned
        # explicitly to automatically update the child states
        self.filter_group.initiate_filters()
        self.pane[1][0] = ""
        self.pane[1][2][0] = self.filter_group.show()

    def delete_filters(self):
        """A method to delete all the child Filter objects."""
        self.children = []
        self.pane[1][2][0] = ""
        self.pane[1][0] = self.add_filters

    def _create_data_dict(self):
        def get_min_max(x, data):
            if x["Dtypes"] == "datetime":
                return data[x["Columns"]].min(), data[x["Columns"]].max()
            elif x["Dtypes"] in ["category", "bool", "string"]:
                return sorted(data[x["Columns"]].dropna().unique().tolist())
            elif x["Dtypes"] == "numeric":
                if data[x["Columns"]].nunique() < 0.05 * len(data):
                    return sorted(data[x["Columns"]].dropna().unique().tolist())
                else:
                    return data[x["Columns"]].min(), data[x["Columns"]].max()

        data_dict = td.DataFrame(
            list(self.dypes.items()), columns=["Columns", "Dtypes"]
        )
        data_dict["Values"] = data_dict.apply(
            lambda x: get_min_max(x, self.data), axis=1
        )
        self.data_dict = data_dict

    def show(self, **kwargs):
        """A method that returns the FilterPanel UI object."""
        self.pane = self.create_ui(**kwargs)
        return self.pane

    def refresh_ui(self):
        """A method that reloads the FilterPanel UI components."""
        if self.filter_group:
            self.pane[1][0] = ""
            self.pane[1][2][0] = self.filter_group.show(refresh=True)
        else:
            self.pane[1][0] = self.add_filters
            self.pane[1][2][0] = ""

    @property
    def has_changes(self):
        """A property that tracks the changes in the state of FilterPanel UI components."""
        return self.has_state_change()

    def get_filters(self):
        """A method returns the boolean series for the set filter configuration."""
        if self.filter_group:
            return self.filter_group.get_filter()
        else:
            return None

    def create_child(self, child_state):
        """A method to create the UI components for FilterPanel's children."""
        self.children = [FilterWrapper(self, self.data, self.data_dict)]
        self.filter_group.set_state(child_state)

    def compute(self):
        """A method to compute the filter result."""
        return self.parent.filter_data(recompute=True)

    def describe(self):
        """A method that provides the description of the FilterPanel in widget_builder."""
        if self.filter_group:
            return self.Row(
                "<b>Filters: </b>",
                self.filter_group.describe(),
                css_classes=["description"],
            )
        return self.Column()
