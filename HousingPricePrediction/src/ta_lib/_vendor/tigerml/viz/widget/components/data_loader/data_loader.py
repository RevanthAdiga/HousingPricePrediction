import os

import tigerml.core.dataframe as td

from .ui import DataLoaderUI


class DataLoader(DataLoaderUI):
    """A class for all DataLoader operations along with its UI components."""

    def __init__(self, parent, callback=None, initial_state={}):
        self.parent = parent
        self.selected_files = []
        self.data_files = []
        self.file_map = {}
        self.current_file = None
        self.data = None
        self.callback = callback
        super().__init__(initial_state=initial_state)

    # @property
    # def selected_file(self):
    # 	return self.load_data_file.value

    def set_callback(self, callback):
        """A method to set the callback function for the DataLoader's load_data button."""
        self.callback = callback
        self.load_button.on_click(self.pass_data_to)

    def add_to_files(self, event=None):
        """A method to add files to DataLoader selections."""
        files = self.select_files.value
        self.selected_files = files.copy()
        self.end_selection()
        new_files = [val for val in files if val not in self.select_files]
        for file in new_files:
            if os.path.isdir(file):
                dir_files = os.listdir(file)
                applicable_files = [
                    os.path.join(file, x)
                    for x in dir_files
                    if x.split(".")[-1] in ["csv", "tsv", "xls", "xlsx"]
                ]
            else:
                applicable_files = [file]
            self.file_map.update({file: applicable_files})
        self.data_files = []
        for file in self.selected_files:
            self.data_files += self.file_map[file]
        self.load_data_file.options = self.data_files

    # @property
    # def has_changes(self):
    # 	return self.selected_file != self.current_file

    def read_file(self, file):
        """A method to read selected file."""
        if ".csv" in file:
            data = td.read_csv(file)
        elif ".tsv" in file:
            data = td.read_csv(file, delimiter="\t")
        elif ".xls" in file:
            data = td.read_excel(file, sheet_name=None)
        else:
            raise Exception(
                "Does not support the chosen file format - {}".format(
                    file.split(".")[-1]
                )
            )
        return data

    def compute(self):
        """A method to merge the data when multiple files are selected."""
        self.files = self.load_data_file.value
        full_data = None
        if self.has_state_change() and self.files:
            for file in self.files:
                data = self.read_file(file)
                data.convert_datetimes()
                data.categorize()
                if full_data is None:
                    full_data = data
                else:
                    if set(list(full_data.columns)) != set(list(data.columns)):
                        self.parent.message_box.show_message(
                            "Could not merge selected files. Pl select files with the same columns"
                        )
                        return
                    full_data = td.concat([full_data, data])
            self.data = full_data

    def pass_data_to(self, event=None):
        """A method to pass files to the DataLoader."""
        self.compute()
        self.current_file = self.files
        if self.callback is not None:
            self.callback(self.data)

    def show(self):
        """A method that returns the full DataLoader UI component."""
        return self.Row(
            self.SectionHeader(
                '<h3 class="section_header">Load Data: </h3>', valign="middle"
            ),
            self.data_loader,
            css_classes=["data_loader"],
        )
