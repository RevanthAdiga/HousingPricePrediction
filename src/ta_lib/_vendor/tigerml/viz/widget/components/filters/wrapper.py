from tigerml.viz.widget.states import StatefulUI

from .condition import FilterCondition

COMBINER = {"AND": " & ", "OR": " | "}


class FilterWrapper(StatefulUI):
    """A class for individual Filter group object in FilterPanel."""

    def __init__(self, parent, data, data_dict, children=[], initial_state={}):
        super().__init__(
            children=children, dynamic_ui=True, initial_state=initial_state
        )
        self.parent = parent
        self.data = data
        self.data_dict = data_dict.copy()
        self.filter_result = None
        self.make_group = self.Button(
            css_classes=["icon-button", "icon-split", "make_group"], width=24, height=24
        )
        # self.children = []
        self.make_group.on_click(self.convert_to_group)
        self.add_filter_button = self.Button(
            name="+",
            button_type="primary",
            css_classes=["add_button", "add_filter"],
            width=25,
            height=25,
        )
        self.add_filter_button.on_click(self.add_filter)
        self.combiner = self.Select(
            options=["AND", "OR"], width=60, css_classes=["combiner"]
        )
        self.group_controls = self.Column(
            self.combiner,
            self.add_filter_button,
            width=40,
            css_classes=["group_controls"],
        )
        self.del_button = self.Button(
            width=40, height=24, css_classes=["icon-button", "icon-cancel"]
        )
        self.del_button.on_click(self.remove_filter)
        self.pane = self.create_pane()
        self.current_state.update({"combiner.value": self.combiner.value})

    @property
    def type(self):
        """A property that tracks type of the FilterWrapper (whether Condition or a Group)."""
        if len(self.children) > 1:
            return "Group"
        else:
            return "Condition"

    def get_children_ui(self, refresh=False):
        """A method that returns UI components of the children of the FilterWrapper."""
        return self.Column(
            *[child.show(refresh=refresh) for child in self.children],
            css_classes=["conditions"],
        )

    def create_pane(self, refresh=False):
        """A method to create the FilterWrapper UI object."""
        if not self.children:
            self.children = [
                FilterCondition(parent=self, data=self.data, data_dict=self.data_dict)
            ]
        return self.Row(
            self.del_button,
            self.make_group if self.type == "Condition" else self.group_controls,
            self.get_children_ui(refresh=refresh),
            css_classes=["filter_wrapper"],
        )

    def initiate_filters(self):
        """A method to initiate the FilterWrapper object."""
        self.children = [
            FilterCondition(parent=self, data=self.data, data_dict=self.data_dict)
        ]
        self.pane[-1] = self.get_children_ui()

    def add_filter(self, event=None):
        """A method to add more FilterCondition to the FilterWrapper."""
        self.children += [
            FilterWrapper(parent=self, data=self.data, data_dict=self.data_dict)
        ]
        self.pane[-1].append(self.children[-1].show())

    def convert_to_group(self, event=None):
        """A method to convert FilterCondition into FilterWrapper object."""
        new_group = FilterWrapper(
            parent=self,
            data=self.data,
            data_dict=self.data_dict,
            children=self.children.copy(),
        )
        new_group.filter_result = self.filter_result
        self.children = [
            new_group,
            FilterWrapper(parent=self, data=self.data, data_dict=self.data_dict),
        ]
        self.pane[-1] = self.Column(
            *[child.show() for child in self.children], css_classes=["conditions"]
        )
        self.pane[1] = self.group_controls

    def remove_filter(self, event=None):
        """A method to delete the FilterWrapper object."""
        if self.parent.__class__.__name__ == "FilterPanel":
            self.parent.delete_filters()
            return
        parent_type = self.parent.type
        self.parent.delete_filter(self)
        if parent_type == "Group" and len(self.parent.children) == 1:
            remaining_condition = self.parent.children[0]
            self.parent.children = [] + remaining_condition.children
            for child in self.parent.children:
                child.parent = self.parent
            self.filter_result = remaining_condition.filter_result
            if self.parent.type == "Group":
                self.parent.pane[1] = self.parent.group_controls
                self.parent.combiner.value = remaining_condition.combiner.value
            else:
                self.parent.pane[1] = self.parent.make_group
            self.parent.pane[-1] = self.parent.get_children_ui()

    def delete_filter(self, filter_):
        """A method to delete a particular child FilterCondition object."""
        self.children.remove(filter_)
        self.pane[-1].remove(filter_.show())

    def get_filter(self):
        """
        Returns the consolidated filter result based on all conditions in the filter section.

        Loops through each condition in both filter group and filter condition objects and
        constructs the filter result series.
        """
        if not self.has_state_change():
            return self.filter_result
        self.filter_result = None
        for filter in self.children:
            current_filter = filter.get_filter()
            if current_filter is None:
                continue
            if self.filter_result is not None:
                if self.combiner.value == "AND":
                    self.filter_result = self.filter_result & current_filter
                else:
                    self.filter_result = self.filter_result | current_filter
            else:
                self.filter_result = current_filter
        # if self.filter_result is not None:
        #     self.filter_result = (self.filter_result)
        return self.filter_result

    def show(self, refresh=False):
        """A method that returns the FilterWrapper UI object."""
        if refresh:
            self.pane = self.create_pane(refresh=refresh)
        return self.pane

    def _create_child(self, child_type="", initial_state={}):
        if child_type == "group":
            new_filter = FilterWrapper(
                self,
                data=self.data,
                data_dict=self.data_dict,
                initial_state=initial_state,
            )
        else:
            new_filter = FilterCondition(
                self,
                data=self.data,
                data_dict=self.data_dict,
                initial_state=initial_state,
            )
        self.children.append(new_filter)
        return new_filter

    def create_child(self, child_state):
        """A method to create a FilterCondition child."""
        if child_state["children"]:
            child = self._create_child(child_type="group")
        else:
            child = self._create_child()
        child.set_state(child_state)

    def describe(self):
        """A method that provides the description of the FilterCWrapper object in widget_builder."""
        conditions = self.Column(
            *[child.describe() for child in self.children], css_classes=["filter_group"]
        )
        if len(conditions) > 1:
            # if 'combiner.value' in self.initial_state:
            #     for ind, cond in enumerate(conditions):
            #         if ind != 0:
            #             cond.insert(0, f'<b> {self.combiner.value} </b>')
            #     conditions = self.Row(conditions)
            # else:
            # self.combiner.width = 100
            if "combiner.value" in self.initial_state:
                combiner = self.Markdown(f"<b> {self.combiner.value} </b>", width=50)
            else:
                combiner = self.combiner
            conditions = self.Row(
                self.Column(combiner, css_classes=["combiner"]),
                conditions,
                css_classes=["live_combiner"],
            )
        return conditions
