from tigerml.viz.backends.panel import PanelBackend
from tigerml.viz.widget.states import StatefulUI


class Sorter(StatefulUI):
    """A class for sorting by single column Operation in DataProcessor."""

    def __init__(self, parent, columns):
        super().__init__()
        self.parent = parent
        self.process_columns = columns.copy()
        self.select_col = self.Select(name="Select col to sort by", options=self.process_columns, width=200)  # noqa
        self.sort_type = self.ToggleGroup(options=["ASC", "DESC"], name="Sort type", width=100,
                                          behavior="radio", css_classes=["tabs"])
        self.del_button = PanelBackend.Button(width=40, height=36, css_classes=["icon-button", "icon-delete"])
        self.del_button.on_click(self.delete_level)
        self.current_state.update({"select_col.value": self.select_col.value, "sort_type.value": self.sort_type.value})
        self.pane = self.Row(self.select_col, self.sort_type, self.del_button)

    def delete_level(self, event=None):
        """A method to delete the Sorter."""
        self.parent.delete_sorter(self)

    @property
    def hide_delete(self):
        """Hides the delete button when only one Sorter is present."""
        return len(self.parent.children) < 2

    def get_pane(self):
        """A method that returns the Sorter pane."""
        if self.hide_delete:
            self.pane[2] = ""
        else:
            self.pane[2] = self.del_button
        return self.pane

    def get_exp(self):
        """A method that returns the sorting expression."""
        return self.select_col.value, self.sort_type.value


class Sorters(StatefulUI):
    """A class for sorting by multiple columns Operation in DataProcessor."""

    def __init__(self, data):
        super().__init__(children=[], dynamic_ui=True)
        self.name = "Sort"
        self.process_columns = data.columns.tolist()
        self.children = [Sorter(parent=self, columns=self.process_columns)]
        self.add_level = self.Button(name="Add level", width=60, css_classes=["button", "primary"])
        self.add_level.on_click(self.add_sorter)
        self.pane = self.Row()
        self.create_ui()

    def create_ui(self):
        """A method to create the UI components."""
        self.pane.clear()
        self.pane.extend([self.add_level, self.Column(*[child.get_pane() for child in self.children])])
        return self.pane

    def create_child(self, child_state):
        """A method to create a new child Sorter."""
        child = Sorter(parent=self, columns=self.process_columns)
        child.set_state(child_state)
        self.children.append(child)
        return child

    def add_sorter(self, event=None):
        """A method to add a new child Sorter."""
        existing_cols = [child.select_col.value for child in self.children]
        avbl_cols = [col for col in self.process_columns if col not in existing_cols]
        self.children += [Sorter(parent=self, columns=avbl_cols)]
        self.pane[1] = self.Column(*[child.get_pane() for child in self.children])

    def delete_sorter(self, sorter):
        """A method to delete a particular child Sorter."""
        value = sorter.select_col.value
        self.children.remove(sorter)
        for child in self.children:
            child.select_col.options += [value]
        self.pane[1] = self.Column(*[child.get_pane() for child in self.children])

    # def get_config(self):
    #     column_list = []
    #     sort_type_list = []
    #     for child in self.children:
    #         exp = child.get_exp()
    #         column_list += [exp[0]]
    #         if exp[1] == 'ASC':
    #             sort_type_list += [True]
    #         else:
    #             sort_type_list += [False]
    #     return {'columns': column_list, 'type': sort_type_list}

    def exec_func(self, data):
        """A method to execute the operation."""
        processed_data = data.copy()
        children_state = self.get_ui_state()["children"]
        child_states = [[child[x] for x in child if ".value" in x] for child in children_state]
        cols = [child[0] for child in child_states]
        asc = [child[1] == "ASC" for child in child_states]
        processed_data = processed_data.sort_values(cols, ascending=asc)
        return processed_data
