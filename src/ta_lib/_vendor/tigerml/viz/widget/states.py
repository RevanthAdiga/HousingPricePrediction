import abc
import gc
import logging
from collections import OrderedDict

from ..backends.panel import PanelBackend

_LOGGER = logging.getLogger(__name__)


class StateMixin:
    """A class for managing and tracking the states of all the elements in viz module."""

    def __init__(self, children=None, dynamic_ui=False, initial_state={}):
        self.data_dependent = OrderedDict()
        self.children = children or []
        self.initial_state = initial_state
        self.current_state = {"no_of_children": len(self.children)}
        self.dynamic_ui = dynamic_ui  # True for filters and YExprs where children can be created, False for fixed.

    def set_state(self, passed_state):
        """A method to set the given state for an UI component."""
        if passed_state == self.get_state():
            return
        state = passed_state.copy()
        children_states = state.pop("children") if "children" in state else []
        # assert set(list(state.keys())) == set(
        # 	list(self.current_state.keys())), 'state passed does not have the right keys'
        for key in state:
            if "value" in key:
                if hasattr(self, key.replace(".value", "")):
                    exec(f"self.{key} = state[key]")
                else:
                    _LOGGER.warning(f'{self.__class__.__name__} does not have "{key.replace(".value", "")}" attribute. Omitting.')  # noqa
        # self.current_state = state
        # if children_states:
        if self.dynamic_ui:
            # for ind, child_state in enumerate(children_states):
            # 	if ind < len(self.children):
            # 		self.children[ind].set_state(child_state)
            # 	else:
            # 		child = self.create_child(child_state)
            # 		if child not in self.children:
            # 			self.children.append(child)
            # if len(children_states) < len(self.children):
            # 	deletable_children = self.children[len(children_states):]
            # 	self.children = self.children[:len(children_states)]
            # 	for child in deletable_children:
            # 		del child
            # 	gc.collect()
            self.children = []
            for child_state in children_states:
                self.create_child(child_state)
        else:
            for ind, child_state in enumerate(children_states):
                self.children[ind].set_state(child_state)

    def delete_child(self, child):
        """A method to delete the child of the UI component."""
        self.children.remove(child)

    @abc.abstractmethod
    def create_child(self, child_state):
        """An abstract method to be defined within each inherited class for child creation."""
        return

    def get_state(self, include_children=True):
        """A method to get the current state of an object."""
        state = self.current_state.copy()
        if include_children:
            state.update({"children": [child.get_state() for child in self.children]})
        return state

    def get_ui_state(self, include_children=True):
        """A method to get the current state of an UI component."""
        state = {}
        for key in [k for k in self.current_state.keys() if "." in k]:
            state[key] = eval(f"self.{key}")
        if include_children:
            state.update({"children": [child.get_ui_state() for child in self.children]})
            state.update({"no_of_children": len(self.children)})
        return state

    def update_state(self, partial_state):
        """A method to update the current state of an object."""
        pass

    def _get_property(self, prop):
        if prop == "no_of_children":
            return len(self.children)
        else:
            if hasattr(self, prop.split(".")[0]):
                return eval(f"self.{prop}")
            else:
                return None

    def save_current_state(self):
        """A method to save the current state of an object."""
        for key in self.current_state:
            self.current_state[key] = self._get_property(key)
        if self.children:
            for child in self.children:
                child.save_current_state()

    def has_state_change(self, *args, include_children=True):
        """A method to detect the changes in the state of an object."""
        if args:
            args = [a for a in args if a in self.current_state]
            changes = [self.current_state[arg] != self._get_property(arg) for arg in args]
        else:
            changes = [self.current_state[arg] != self._get_property(arg) for arg in self.current_state]
            if include_children:
                changes += [child.has_state_change() for child in self.children] + [self.current_state["no_of_children"] != len(self.children)]
        return any(changes)

    def _check_state_for_data(self, data):
        return all([self.current_state[f"{key}.value"] in data.columns for key in self.data_dependent]) and all([child.check_state_for_data(data) for child in self.children])  # noqa

    def _update_data_columns(self, data=None, dtypes=None):
        if data is not None:
            self.data = data
            self.dtypes = dtypes
        if self.data is not None:
            for key in self.data_dependent:
                exec(f"self.{key}.options = self.data_dependent[key](self.data, self.dtypes)")
            for child in self.children:
                if hasattr(child, "_update_data_columns"):
                    child._update_data_columns(self.data, self.dtypes)


class StatefulUI(StateMixin, PanelBackend):
    """A class for managing and tracking the states of all the UI components made of PanelBackend."""

    def __init__(self, children=None, dynamic_ui=False, initial_state={}):
        super().__init__(children=children, dynamic_ui=dynamic_ui, initial_state=initial_state)

    def _create_ui_from_config(self, ui_config, data=None, dtypes=None):
        ui_dict, data_dependent, stateful_list = self.create_ui_from_config(ui_config, data, dtypes, initial_state=self.initial_state)  # noqa
        for key in ui_dict:
            exec(f"self.{key} = ui_dict[key]")
        for key in data_dependent:
            exec(f"self.{key}.options = data_dependent[key](data=data, dtypes=dtypes)")
        self.data_dependent.update(data_dependent)
        for key in stateful_list:
            self.current_state[f"{key}.value"] = eval(f"self.{key}.value")
