import os

HERE = os.path.dirname(os.path.abspath(__file__))


class StateTracker:
    """Base class for state tracking operations of VizWidget in viz module."""

    def __init__(self, *args, **kwargs):
        self.all_states = []
        self.bookmarks = []
        self.state_index = -1

    def check_nav_options(self):
        """A method to disable and enable the navigation buttons based on the current state index."""
        if self.state_index - 1 < -len(self.all_states):
            self.back_button.disabled = True
        else:
            self.back_button.disabled = False
        if self.state_index + 1 > -1:
            self.forward_button.disabled = True
        else:
            self.forward_button.disabled = False

    def go_back(self, event=None):
        """A callback function to navigate backward in state."""
        self.state_index -= 1
        state = self.all_states[self.state_index]
        self.check_nav_options()
        self.load_state(state)

    def go_forward(self, event=None):
        """A callback function to navigate forward in state."""
        self.state_index += 1
        state = self.all_states[self.state_index]
        self.check_nav_options()
        self.load_state(state)

    def load_bookmark(self, event=None):
        """A callback function to load a saved Bookmark on the Widget."""
        bookmark_index = self.bmk_select.value
        bookmark_state = self.bookmarks[bookmark_index]["state"]
        self.load_state(bookmark_state)
        self.update_states_list()

    def delete_bookmark(self, event=None):
        """A callback function to delete a Bookmark."""
        bookmark_index = self.bmk_select.value
        self.bookmarks.pop(bookmark_index)
        self.refresh_bookmarks()

    def edit_bookmark(self, event=None):
        """A callback function to modify an existing Bookmark."""
        bookmark_index = self.bmk_select.value
        boomark = self.bookmarks[bookmark_index]
        boomark["name"] = self.bmk_name_input.value
        self.refresh_bookmarks()

    def get_bookmarks(self):
        """A method that generates the Bookmark for the current state."""

        bookmark_link_template = """
            <div class="bookmark">
                <a>{}</a>
                <div class="icon-link icon-edit"></div>
                <div class="icon-link icon-delete"></div>
            </div>
        """

        bookmark_links = "".join([bookmark_link_template.format(bookmark["name"]) for bookmark in self.bookmarks])  # noqa
        script_path = f"{HERE}/static_resources/bookmark_script.js"
        script_file = open(script_path, "r")
        custom_script = script_file.read()
        script_file.close()
        script = "<script>{}</script>".format(custom_script)
        return self.HTML(f'{script}<div class="bookmarks">{bookmark_links}</div>')

    def save_bookmarks(self, event=None):
        """A method to save all the Bookmarks in json format."""
        from tigerml.core.utils import time_now_readable
        import json

        path = "bookmarks_saved_at_{}.json".format(time_now_readable())
        json_file = open(path, "w", encoding="utf-8")
        json_file.write(json.dumps(self.bookmarks))
        json_file.close()

    def import_bookmarks(self, event=None):
        """A method to import an existing Bookmark in json format."""
        contents = self.import_file.value
        import json

        bookmarks = json.loads(contents)
        self.bookmarks += bookmarks
        self.refresh_bookmarks()

    def get_bookmarks_ui(self):
        """A method to generate the UI elements of the Bookmark."""
        self.bmk_select = self.Select(css_classes=["bookmark_input"])
        # self.bmk_button = self.Button(css_classes=['load_bookmark'])
        self.bmk_load = self.Button(css_classes=["load_bookmark"], width=0)
        self.bmk_name_edit = self.Button(css_classes=["edit_bookmark"], width=0)
        self.bmk_delete = self.Button(css_classes=["delete_bookmark"], width=0)
        self.bmk_name_input = self.TextInput(css_classes=["bookmark_name"], width=240)
        self.bmk_name_save = self.Button(css_classes=["icon-button", "icon-ok"], width=30, height=30)
        self.cancel = self.Button(css_classes=["icon-button", "icon-cancel"], width=30, height=30)
        self.bookmark_edit = self.Row(self.bmk_name_input, self.bmk_name_save, self.cancel, css_classes=["content"])
        edit_widget = self.Column(self.HTML('<div class="glass"></div>'), self.bookmark_edit,
                                  css_classes=["edit_bookmark_widget", "overlay", "is_hidden"], width=0, height=0)
        self.hidden_bmk_triggers = self.Row(self.bmk_select, self.bmk_load, self.bmk_name_edit, self.bmk_delete,
                                            css_classes=["is_hidden"], width=0, height=0)
        self.bmk_load.on_click(self.load_bookmark)
        self.bmk_delete.on_click(self.delete_bookmark)
        self.bmk_name_edit.on_click(self.edit_bookmark)
        self.import_button = self.Button(css_classes=["icon-button", "icon-import"], width=30, height=30)
        self.import_file = self.FileInput(accept=".json", css_classes=["is_hidden", "import_file"], width=0)
        self.save_button = self.Button(css_classes=["icon-button", "icon-save"], width=30, height=30)
        self.import_file.param.watch(self.import_bookmarks, "value")
        self.save_button.on_click(self.save_bookmarks)
        self.bookmarks_ops = self.Row(self.import_button, self.import_file, css_classes=["right"])
        self.bookmarks_header = self.Row(self.HTML("<h3>Bookmarks</h3>", css_classes=["section_header"], width=200),
                                         self.bookmarks_ops, css_classes=["full_width"])
        self.bookmarks_pane = self.Column(self.hidden_bmk_triggers, edit_widget, self.bookmarks_header,
                                          self.get_bookmarks(), css_classes=["bookmarks_widget"])
        return self.bookmarks_pane

    def add_to_bookmarks(self, event=None):
        """A method to add a particular state to Bookmarks."""
        state = self.get_state()
        self.bookmarks.append({"name": self.widget_title, "state": state})
        self.refresh_bookmarks()

    def update_states_list(self):
        """A method that updates the new state on the StateTracker."""
        if self.state_index != -1:
            self.all_states = self.all_states[: self.state_index + 1]  # All next states are deleted when some changes are made by User  # noqa
            self.state_index = (-1)  # User clicked on the button manually. So a new state gets created
        self.all_states.append(self.get_state())
        self.check_nav_options()

    def state_navigator(self):
        """A method that returns the Navigator pane."""
        self.back_button = self.Button(css_classes=["icon-button", "icon-left"], width=50, height=40)
        self.forward_button = self.Button(css_classes=["icon-button", "icon-right"], width=50, height=40)
        self.bookmark_button = self.Button(css_classes=["icon-button", "icon-star"], width=50, height=40)
        self.navigator = self.Row(self.back_button, self.forward_button, self.bookmark_button,
                                  css_classes=["full_width", "gray_bg", "margin_top"])
        self.back_button.on_click(self.go_back)
        self.forward_button.on_click(self.go_forward)
        self.bookmark_button.on_click(self.add_to_bookmarks)
        self.check_nav_options()
        return self.navigator

    def refresh_bookmarks(self):
        """A method that reloads the UI elements of Bookmarks."""
        self.bmk_select.options = list(range(0, len(self.bookmarks)))
        if len(self.bookmarks) > 0:
            if self.save_button not in self.bookmarks_ops:
                self.bookmarks_ops.insert(0, self.save_button)
        else:
            self.bookmarks_ops.remove(self.save_button)
        self.bookmarks_pane[3] = self.get_bookmarks()
