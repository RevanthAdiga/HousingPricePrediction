from ...states import StatefulUI


class DataLoaderUI(StatefulUI):
    """A class for all the UI components of DataLoader."""

    def __init__(self, initial_state={}):
        super().__init__(initial_state=initial_state)
        self._create_ui()

    def _create_ui(self):
        load_data_file = self.multichoice(
            options=self.data_files, name="Selected Files", css_classes=["files_dd"]
        )
        self._create_ui_from_config({"load_data_file": load_data_file})
        self.load_button = self.Button(
            name="Load File",
            css_classes=["secondary", "button"],
            disabled=len(self.selected_files) > 0,
        )
        if self.callback:
            self.load_button.on_click(self.pass_data_to)
        else:
            self.load_button.js_on_click(code="location.reload();")
        if not self.initial_state:
            self.select_files_button = self.Button(
                name="Select files from computer",
                css_classes=[
                    "tertiary",
                    "button",
                    "icon-button-prefix",
                    "icon-select_files",
                ],
            )
            self.file_loading = self.Row(self.select_files_button)
            self.file_selection = self.Row(self.load_data_file, self.load_button)
            self.select_files_button.on_click(self.show_selection_widget)
            self.select_files = self.FileSelector("/", file_pattern="*.[cxt][sl][vs]*")
            self.selected_files_summary = self.StaticText()
            self.add_selected_files = self.Button(
                name="Add Selected Files",
                css_classes=[
                    "secondary",
                    "button",
                    "is_hidden",
                    "icon-button-prefix",
                    "icon-add_files",
                ],
            )
            self.cancel_selection = self.Button(
                name="Cancel",
                css_classes=["button", "icon-button-prefix", "icon-cancel", "tertiary"],
            )
            self.select_files.link(
                self.add_selected_files, {"value": self.toggle_add_button}
            )
            self.add_selected_files.on_click(self.add_to_files)
            self.cancel_selection.on_click(self.end_selection)
            self.selection_widget = self.Column(
                self.select_files,
                self.selected_files_summary,
                self.Row(self.cancel_selection, self.add_selected_files),
            )
        # self.load_files = self.Column(self.Row(self.add_data_file, self.add_file, self.file_names), self.select_files)
        self.overlays = self.Column(height=0)
        self.data_loader = self.Column(
            self.get_loader_view(), self.overlays, css_classes=["is_visible"]
        )
        # self.data_loader = self.Column(self.minimised_view)

    def get_loader_view(self):
        """A method that returns the DataLoader UI component."""
        self.minimised_view = self.Row(
            self.file_loading if not self.initial_state else "",
            self.file_selection if self.selected_files else "",
        )
        return self.minimised_view

    def show_selection_widget(self, event=None):
        """A method that displays the DataLoader selections as pop-up."""
        self.overlays.append(self.Overlay(self.selection_widget, size="big"))

    def end_selection(self, event=None):
        """A method that clears the DataLoader selections pop-up."""
        self.overlays.clear()
        self.data_loader[0] = self.get_loader_view()
        # self.selection_widget.css_classes = ['is_hidden']
        # self.minimised_view.css_classes = ['is_visible']

    def toggle_add_button(self, target, event):
        """A method that toggles the css_classes for DataLoader selections pop-up."""
        if event.new:
            target.css_classes = [
                "secondary",
                "button",
                "is_visible",
                "icon-button-prefix",
                "icon-add_files",
            ]
        else:
            target.css_classes = [
                "secondary",
                "button",
                "is_hidden",
                "icon-button-prefix",
                "icon-add_files",
            ]
