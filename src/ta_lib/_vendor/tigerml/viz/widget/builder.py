import os

from .widget import VizWidget

HERE = os.path.dirname(os.path.abspath(__file__))


class WidgetBuilder(VizWidget):
    """A class for multiple WidgetBuilder sub-module in viz."""

    def __init__(self, *args, **kwargs):
        self.lockable_elements = self.Select(options=[], css_classes=["lockable_items"])
        self.unlock_element_btn = self.Button(width=0, css_classes=["unlock_btn"])
        self.lock_element_btn = self.Button(width=0, css_classes=["lock_btn"])
        self.unlock_element_btn.on_click(self.add_unlocked_class)
        self.lock_element_btn.on_click(self.remove_unlocked_class)
        super().__init__(*args, **kwargs)
        self.modify_child_creation(self.y_exprs, prefix="lock__children_2")
        self.modify_child_creation(self.filters.filter_group, prefix="lock__children_1")
        self.show_preview_btn = self.Button(name="Show Preview")
        self.save_state_btn = self.Button(name="Save State")
        self.show_preview_btn.on_click(self.show_preview)
        self.save_state_btn.on_click(self.save_state)
        self.preview_pane = self.Column()

    @classmethod
    def split_state(cls, element, element_state):
        """A method to split the state of each child element."""
        free_states = {}
        initial_state = {}
        if "children" in element_state:
            free_states["children"] = []
            initial_state["children"] = []
            for ind, child_state in enumerate(element_state["children"]):
                fs_local, in_local = cls.split_state(element.children[ind], child_state)
                free_states["children"].append(fs_local)
                initial_state["children"].append(in_local)
        for key in [k for k in element_state.keys() if ".value" in k]:
            el_name = key.replace(".value", "")
            el = eval(f"element.{el_name}")
            if "unlocked" in el.css_classes:
                free_states[key] = element_state.pop(key)
            else:
                initial_state[key] = element_state.pop(key)
        initial_state["no_of_children"] = element_state["no_of_children"]
        return free_states, initial_state

    def delete_preview(self, pane):
        """A method to delete the preview from UI."""
        self.preview_pane.remove(pane)

    def show_preview(self, event=None):
        """A method to display the preview on UI."""
        state = self.get_ui_state()
        free_states, initial_state = self.split_state(self, state)
        print(initial_state)
        from .viewer import WidgetView

        preview = WidgetView(initial_state, free_states, data=self.data, delete_callback=self.delete_preview)  # noqa
        preview.create_pane()
        # preview.debug.value = 'ON'
        self.preview_pane.append(preview.pane)

    def save_state(self, event=None):
        """A method to save the state of WidgetBuilder."""
        pass

    def _get_selected_element(self):
        self.el_path = self.lockable_elements.value.replace("lock__", "")
        elements = self.el_path.split("__")
        current_element = self
        for el in elements:
            if "children" in el:
                children = eval(f"current_element.children")
                ind = el.split("_")[-1]
                if "id" in ind:
                    current_element = [ch for ch in children if ch.id == ind][0]
                else:
                    current_element = children[int(ind)]
            else:
                current_element = eval(f"current_element.{el}")
        return current_element

    def refresh_element(self):
        """A method to reload the UI components of WidgetBuilder."""
        if "children" in self.el_path:
            self.refresh_ui()
        elif "split" in self.el_path:
            self.refresh_splitter()
        else:
            self.refresh_x()

    def add_unlocked_class(self, event=None):
        """A method to add new class to the WidgetBuilder."""
        current_element = self._get_selected_element()
        if "unlocked" not in current_element.css_classes:
            current_element.css_classes.append("unlocked")
            self.refresh_element()

    def remove_unlocked_class(self, event=None):
        """A method to remove a class from the WidgetBuilder."""
        current_element = self._get_selected_element()
        if "unlocked" in current_element.css_classes:
            current_element.css_classes.remove("unlocked")
            self.refresh_element()

    def modify_child_creation(self, obj, prefix="lock"):
        """A method to modify an existing child of the WidgetBuilder."""
        old_func = obj._create_child
        prefix = prefix + "__"

        def new_func(*args, **kwargs):
            child = old_func(*args, **kwargs)
            # from tigerml.core.utils import time_now_readable
            from datetime import datetime

            uid = "id{}".format(int(datetime.now().timestamp() * 10 ** 6))
            child.id = uid
            if hasattr(child, "_create_child"):
                self.modify_child_creation(child, prefix=f"{prefix}children_{uid}")
            self.make_states_lockable(child, child.get_state(), prefix=f"{prefix}children_{uid}")
            self.lock_widget[0] = ""
            self.lock_widget[0] = self.lockable_elements
            return child

        obj._create_child = new_func

    def make_states_lockable(self, element, element_states, prefix="lock"):
        """A method to lock the state of the children."""
        # print('Prefix is {}'.format(prefix))
        if prefix:
            prefix = prefix + "__"
        assert isinstance(element_states, dict)
        # keys = list(element_states.keys())
        if "children" in element_states:
            for ind, child_state in enumerate(element_states["children"]):
                self.make_states_lockable(element.children[ind], child_state, prefix=f"{prefix}children_{ind}")
        for key in [key for key in element_states.keys() if ".value" in key]:
            state_element = eval(f'element.{key.replace(".value", "")}')
            class_names = ["lockable", "no_lock"]
            if state_element.css_classes:
                if "lockable" not in state_element.css_classes:
                    for class_name in class_names:
                        state_element.css_classes.append(class_name)
            else:
                state_element.css_classes = class_names
            id_class = prefix + key.replace(".value", "")
            if id_class not in state_element.css_classes:
                state_element.css_classes.append(id_class)
            if id_class not in self.lockable_elements.options:
                self.lockable_elements.options.append(id_class)

    def _initiate(self):
        super()._initiate()
        self.make_states_lockable(self, self.get_state())

    def open(self, port=None):
        """A method to launch the WidgetBuilder as a standalone widget."""
        self.create_pane()
        script_path = f"{HERE}/../static_resources/builder_template.html"
        template_file = open(script_path, "r")
        template = template_file.read()
        template_file.close()
        import panel as pn

        self.tmpl = pn.Template(template)
        self.lock_widget = pn.Row(self.lockable_elements, self.unlock_element_btn, self.lock_element_btn, css_classes=["is_hidden"])  # noqa
        self.tmpl.add_panel("builder", self.pane)
        self.tmpl.add_panel("lock_selector", self.Column(self.lock_widget, self.Row(self.show_preview_btn, self.save_state_btn)))  # noqa
        self.tmpl.add_panel("preview_pane", self.preview_pane)

        script_path = f"{HERE}/../static_resources/locking_script.js"
        script_file = open(script_path, "r")
        script = "<script>{}</script>".format(script_file.read())
        script_file.close()
        self.tmpl.add_panel("builder_script", script)
        self.tmpl.show(title="Builder", port=port)
