import gc

from tigerml.viz.widget.states import StatefulUI

from .dp import DataProcessing


class DPOperation(StatefulUI):
    """A class for handling the Operations panel in DataProcessor sub-module."""

    def __init__(self, operator, parent=None):
        super().__init__(children=[operator])
        self.parent = parent
        self.operator = operator
        self.processed_data = None
        self.status = "in_progress"
        self.active = self.Toggle(width=24, height=24, value=True, css_classes=["icon-button", "icon-ok"])  # noqa
        self.active.param.watch(self.toggle_active, "value")
        self.current_state.update({"active.value": self.active.value, "operator.name": self.operator.name})
        self.view_button = self.Button(width=24, height=24, css_classes=["icon-button", "icon-view"])
        self.view_button.on_click(self.view_output_data)
        self.view_button.js_on_click(args={}, code=self.apply_glass)
        self.edit_button = self.Button(width=24, height=24, css_classes=["icon-button", "icon-edit"])
        self.edit_button.on_click(self.edit_operation)
        self.edit_button.js_on_click(args={}, code=self.apply_glass)
        self.add_button = self.Button(name="+", css_classes=["add_button"], width=30, height=30)
        self.add_button.on_click(self.create_node_after)
        self.add_button.js_on_click(args={}, code=self.apply_glass)
        self.delete_button = self.Button(name="Delete Operation", width=130, css_classes=["button", "tertiary"])
        self.delete_button.on_click(self.delete_operation_node)
        self.delete_button.js_on_click(args={}, code=self.remove_glass)
        self.save_edited_button = self.Button(name="Save Operation", width=130, css_classes=["button", "primary"])
        self.save_edited_button.on_click(self.save_edited_operation)
        self.save_edited_button.js_on_click(args={}, code=self.remove_glass)
        self.buttons = self.Row(self.active, self.view_button, self.edit_button, css_classes=["buttons"])
        self.inactive_flag = self.HTML("", css_classes=["is_inactive"], height=0, width=0)
        self.error_flag = self.HTML("", css_classes=["is_error"], height=0, width=0)
        self.create_pane()

    def create_pane(self):
        """A method that creates the UI of Operations panel."""
        self.error = False
        self.summary_ui = self.Row(self.get_operation_summary(), css_classes=["op_summary"])
        self.block_ui = self.Column(self.buttons, self.HTML("", css_classes=["node"]), self.summary_ui,
                                    css_classes=["block_ui"])
        self.pane = self.Row(self.block_ui, self.add_button, css_classes=["operation_block"], width=150)
        # if self.error:
        #     self.pane.insert(0, self.error_flag)
        if not self.active.value:
            self.pane.insert(0, self.inactive_flag)
        return self.pane

    # def execute_dp_op(self):
    #     dp = DataProcessing(self.input_data, parent=self.parent)
    #     dp.operations.value = dp.DP_OPTIONS[self.operator.name]
    #     dp.operation = self.operator
    #     dp.operations_wrapper[0] = dp.operation.pane
    #     dp.exec_func()
    #     if isinstance(dp.processed_data, str):
    #         self.error_msg = dp.processed_data
    #         self.err_traceback = dp.err_traceback
    #     return dp

    def toggle_active(self, event=None):
        """A method that toggles the activity status of a particular Operation."""
        # highlight the active path
        # self.active = event.new
        if self.active.value:
            if self.inactive_flag in self.pane:
                self.pane.remove(self.inactive_flag)
        else:
            self.pane.insert(0, self.inactive_flag)
        error = self.parent.check_error_in_workflow(self.parent.children.index(self))
        if not error and self.parent.parent_widget:
            self.parent.parent_widget.preprocess_data()

    def view_output_data(self, event=None):
        """A callback function to display the data."""
        try:
            data = self.output_data.copy()
            for col in data.columns:
                try:
                    non_na_element = data[col].dropna().unique()[0]
                except:
                    non_na_element = None
                if "Interval" in str(type(non_na_element)):
                    data[col] = data[col].astype(str)
            # row = self.Overlay(self.Table(data), size='full_screen')
            row = self.Overlay_inplace(self.Table(data))
            del data
            gc.collect()
        except:
            # show exception in overlay
            # row = self.Overlay(self.HTML(f"<pre> {self.parent.error_in_workflow['traceback']} </pre>"))
            row = self.Overlay_inplace(self.HTML(f"<pre> {self.parent.error_in_workflow['traceback']} </pre>"))
        self.parent.overlays.append(row)

    def edit_operation(self, event=None):
        """A method that provides for modifying a particular Operation."""
        op_input_data = self.input_data
        if self.operator.name == "Drop":
            self.operator.input_data_cols = op_input_data.columns.tolist()
        dp = DataProcessing(op_input_data, operation=self.operator, parent=self.parent)
        dp.exec_func()
        self.parent.dp_job = dp
        dp.buttons[0] = self.save_edited_button
        dp.buttons.append(self.delete_button)
        # row = self.Overlay(dp.pane, size='full_screen', closeable=True)
        row = self.Overlay_inplace(dp.pane)
        self.parent.overlays.append(row)

    @property
    def input_data(self):
        """A property that returns the input data given to a particular Operation."""
        step_index = self.parent.children.index(self)
        if step_index > 0:
            prev_active_steps = [child for child in self.parent.children[:step_index] if child.active.value]
            if prev_active_steps:
                return prev_active_steps[-1].output_data
            else:
                return self.parent.input_data
        else:
            return self.parent.input_data

    @property
    def output_data(self):
        """A property that returns the output data from a particular Operation."""
        try:
            if self.active.value:
                return self.operator.exec_func(self.input_data)
            else:
                return self.input_data
        except Exception as e:
            return str(type(e).__name__) + " : " + str(e)

    def process_data(self, data):
        """A method to trigger the particular Operation's exec_func.."""
        try:
            output_data = self.operator.exec_func(data)
            if isinstance(output_data, str):
                import traceback

                self.err_traceback = traceback.format_exc()
            return output_data
        except Exception as e:
            import traceback

            self.err_traceback = traceback.format_exc()
            return str(type(e).__name__) + " : " + str(e)

    def create_node_after(self, event=None):
        """A callback function to add a new node to the current branch."""
        self.parent.create_node_after(self)

    def create_branch(self, event=None):
        """A callback function to create a new branch."""
        self.parent.create_branch(self)

    def reach_state(self, event=None):
        """A callback function to reach a particular state of the WorkFlow."""
        self.parent.reach_state(self)

    def get_operation_summary(self):
        """A method that returns the summary of the Operation."""
        state = self.get_ui_state()
        type = state["operator.name"]
        opt_dict = state["children"][0]
        if type == "Custom":
            return opt_dict["operation_name.value"]
        elif type == "Sort":
            columns = [child["select_col.value"] for child in opt_dict["children"]]
        elif type == "Drop":
            columns = opt_dict["delete_cols.value"]
        elif type == "Bin":
            columns = opt_dict["bin_col.value"]
        elif type == "Create":
            columns = opt_dict["new_col_name.value"]
        if not isinstance(columns, list):
            columns = [columns]
        summary = "{} {}".format(type, ", ".join(columns))
        if type == "Sort":
            summary += " in {} order".format(", ".join(["ascending" if
                                                        child["sort_type.value"] == "ASC"
                                                        else "descending" for child in opt_dict["children"]]))
        elif type == "Bin":
            summary += " ({} bins)".format(opt_dict["bin_nos.value"])
        return summary

    def complete_operation(self, operator):
        """A method to safely wind up the Operation."""
        self.operator = operator
        self.children = [operator]
        # self.save_current_state()
        self.summary_ui[0] = self.get_operation_summary()
        self.status = "completed"

    def save_edited_operation(self, event=None):
        """A callback function to save the edited Operation."""
        # reset active flag and remove the associated component from UI if exist
        if self.inactive_flag in self.pane:
            self.pane.remove(self.inactive_flag)
        self.active.value = True
        # remove the error flag component from UI if exist
        if self.error_flag in self.pane:
            self.pane.remove(self.error_flag)
        self.complete_operation(self.parent.dp_job.operation)
        self.parent.overlays.clear()
        # trigger check for error functionality after editing an operation node
        node_index = self.parent.children.index(self)
        self.parent.check_error_in_workflow(from_dp_index=node_index)

    def delete_operation_node(self, event=None):
        """A callback function to delete a node on the WorkFlow branch."""
        # 1. delete the node selected for deletion
        node_index = self.parent.children.index(self)
        self.parent.children.pop(node_index)
        if self.parent.parent is None:
            self.parent.complete_workflow[0].pop(node_index + 1)
        else:
            self.parent.complete_workflow[0].pop(node_index)
        # 2. delete all the linked branches of the selected node
        if self in self.parent.branches.values():
            branches = [branch for branch in self.parent.branches if self.parent.branches[branch] == self]
            for branch in branches:
                del self.parent.branches[branch]
                gc.collect()
            self.parent.complete_workflow[1] = self.parent.get_branches_ui()
        # 3. delete the node branch itself if there are no other node remains on it
        branch_workflow = self.parent
        parent_workflow = self.parent.parent
        if not branch_workflow.children and branch_workflow.parent:
            del parent_workflow.branches[self.parent]
            del parent_workflow.branches_ui
            gc.collect()
            parent_workflow.complete_workflow[1] = parent_workflow.get_branches_ui()
        self.parent.overlays.clear()
        # trigger check for error functionality after deleting an operation node
        self.parent.check_error_in_workflow(from_dp_index=node_index)
