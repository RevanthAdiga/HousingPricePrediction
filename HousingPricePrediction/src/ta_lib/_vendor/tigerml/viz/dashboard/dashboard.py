import panel as pn

from tigerml.core.utils import dict_depth


class Dashboard:
    """A class for the Dashboard sub-module in viz module."""

    def __init__(self, widgets, name="", nav_types=None):
        self.widgets = widgets
        self.name = name
        n_levels = dict_depth(self.widgets)
        self.nav_types = [None] * n_levels
        if nav_types:
            assert (
                len(nav_types) == n_levels
            ), "number of levels in widgets and navigation options passed do not match"
            self.nav_types = nav_types

    @classmethod
    def _create_tabbed_nav(cls, names, elements):
        tabs = pn.Tabs()
        for idx, name in enumerate(names):
            tabs.append((name, elements[idx]))
        return tabs

    @classmethod
    def _create_menu_nav(cls, names, elements):
        pass

    @classmethod
    def _create_ui_for_level(cls, widgets, nav_types):
        current_nav_type = nav_types[0]
        elements = [
            cls._create_ui_for_level(value, nav_types[1:])
            if isinstance(value, dict)
            else value
            for value in widgets.values()
        ]
        if current_nav_type == "tabs":
            return cls._create_tabbed_nav(list(widgets.keys()), elements)
        elif current_nav_type == "menu":
            return cls._create_menu_nav(list(widgets.keys()), elements)
        else:
            return pn.Column(elements)

    def create_ui(self):
        """A method that creates the UI components of the Dashboard."""
        self.dashboard = self._create_ui_for_level(self.widgets, self.nav_types)

    def show(self):
        """A method that returns the UI components of the Dashboard."""
        pass

    def open(self, port=5006):
        """A method that launches Dashboard as a standalone widget."""
        self.create_ui()
        title = self.name
        if not title:
            title = "TigerML Dashboard"
        self.dashboard.show(title=title, port=port)
