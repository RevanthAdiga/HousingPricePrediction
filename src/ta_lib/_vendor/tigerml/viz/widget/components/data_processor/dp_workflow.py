import gc
from collections import OrderedDict

from tigerml.viz.widget.states import StatefulUI

from .dp import DataProcessing
from .dp_operation import DPOperation


class DPWorkFlow(StatefulUI):
    """A class for managing the Operations WorkFlow in DataProcessor sub-module."""

    def __init__(self, initial_data, parent_widget=None, name=None, parent=None):
        super().__init__(children=[], dynamic_ui=True)
        self.parent_widget = parent_widget
        self.initial_data = initial_data.copy()
        self.output_data = self.initial_data
        self.active_data = self.output_data
        if name:
            self.name = name
        else:
            self.name = "dp_workflow_1"
        self.branches = OrderedDict()
        self.parent = parent
        self.child_num = 0
        self.active_child = None
        self.start_dp = self.Button(name="+", css_classes=["add_button"], width=30, height=30)  # noqa
        self.start_dp.on_click(self.show_dp)
        self.start_dp.js_on_click(args={}, code=self.apply_glass)
        self.overlays = self.Column(css_classes=["overlays"])
        self.complete_workflow = self.Column()
        self.create_dp_workflow()
        self.branches_ui = self.Column()
        self.error_in_workflow = {"node": -2, "error": "", "traceback": ""}
        self.complete_workflow.extend([self.dp_workflow, self.branches_ui])
        self.pane = self.Row(self.complete_workflow)
        if not self.parent:
            self.pane.insert(0, self.SectionHeader('<h3 class="section_header">Preprocessing: </h3>'))
        self.main_pane = self.Column(self.pane, self.overlays, css_classes=["dp_workflow"])

    def create_dp_workflow(self):
        """A method to create a new WorkFlow of Operations."""
        if hasattr(self, "dp_workflow"):
            self.dp_workflow.clear()
        self.dp_workflow = self.Row(*([child.create_pane() for child in self.children] + [self.start_dp]),
                                    css_classes=["dp_linear_flow"])
        if self.parent is None:
            self.dp_workflow.insert(0, self.source_node if hasattr(self, "source_node") else self.get_source_node())
        return self.dp_workflow

    def get_source_node(self):
        """A method to get the source node of the WorkFlow."""
        if not hasattr(self, "source_node"):
            self.add_button = self.Button(name="+", css_classes=["add_button"], width=30, height=30)
            self.add_button.on_click(self.create_node_after_source)
            self.add_button.js_on_click(args={}, code=self.apply_glass)
            view_source_btn = self.Button(width=24, height=24, css_classes=["icon-button", "icon-view"])
            view_source_btn.on_click(self.view_source)
            view_source_btn.js_on_click(args={}, code=self.apply_glass)
            # adding 'is_active' as css_classes to always keep source_node highlighted
            self.source_block_ui = self.Column(self.Row(view_source_btn, css_classes=["buttons"]),
                                               self.HTML("", css_classes=["node"]),
                                               self.Row("Source Data", css_classes=["op_summary"]),
                                               css_classes=["block_ui"])
            self.source_node = self.Row(self.source_block_ui, self.add_button,
                                        css_classes=["operation_block", "is_active"], width=150)
        return self.source_node

    @property
    def input_data(self):
        """A property that returns the input data of the WorkFlow."""
        if self.parent:
            branching_node = self.parent.branches[self]
            if branching_node:
                return branching_node.output_data
            else:
                return self.initial_data
        else:
            return self.initial_data

    # @property
    # def len_of_all_branches(self):
    #     branch_len = len(self.branches)
    #     if branch_len > 0:
    #         branch_len += sum([branch.len_of_all_branches for branch in self.branches])
    #     return branch_len

    def show_dp(self, event=None):
        """A callback function to create a new node on the WorkFlow."""
        if len(self.children) == 0:
            self.create_node_after()
        else:
            self.create_node_after(self.children[-1])

    def create_node_after(self, dp_op=None):
        """A method to create a new node after a particular node in the WorkFlow."""
        if dp_op:
            input_data = dp_op.output_data
            self.current_dp_op_index = self.children.index(dp_op) + 1
        else:
            input_data = self.input_data
            self.current_dp_op_index = 0
        self.dp_job = DataProcessing(input_data, parent=self)
        # row = self.Overlay(self.dp_job.pane, size='full_screen', closeable=True)
        row = self.Overlay_inplace(self.dp_job.pane)
        self.overlays.append(row)

    def create_node_after_source(self, event=None):
        """A method to create a new node right after the source data."""
        self.create_node_after()

    def create_child(self, child_state):
        """A method to create child for the WorkFlow based on the selected Operation type."""
        # import pdb
        # pdb.set_trace()
        from . import operations

        if child_state["operator.name"] == "Custom":
            operator = operations.CustomTransformer(self.data)
        elif child_state["operator.name"] == "Create":
            operator = operations.Transformer(self.data)
        elif child_state["operator.name"] == "Sort":
            operator = operations.Sorters(self.data)
        elif child_state["operator.name"] == "Drop":
            operator = operations.DropCols(self.data)
        elif child_state["operator.name"] == "Bin":
            operator = operations.Binner(self.data)
        # operator.set_state(child_state.pop('children')[0])
        child = DPOperation(operator, parent=self)
        operator.parent = child
        self.children.append(child)
        child.set_state(child_state)
        return child

    # def create_branch(self, dp_op=None):
    #     branch_names = list([branch.name for branch in self.branches])
    #     if branch_names:
    #         branch_num = int(branch_names[-1].replace('dp_branch_', '')) + 1
    #         branch_name = 'dp_branch_{}'.format(branch_num)
    #     else:
    #         branch_name = 'dp_branch_1'
    #     if dp_op:
    #         input_data = dp_op.output_data
    #     else:
    #         input_data = self.input_data
    #     branch_workflow = DPWorkFlow(input_data, name=branch_name, parent=self)
    #     if dp_op and self.branches:
    #         branching_indeces = [self.children.index(dp_op) if dp_op else -1 for dp_op in self.branches.values()]
    #         this_index = self.children.index(dp_op)
    #         insert_after = [2] * len(branching_indeces)
    #         for ind, b_ind in enumerate(branching_indeces):
    #             insert_after[ind] = 1 if b_ind < this_index and (ind == 0 or insert_after[ind-1] == 0) else 0
    #         insert_after_ind = insert_after.index(1)
    #         branches_list = list(self.branches.items())
    #         del self.branches
    #         gc.collect()
    #         self.branches = OrderedDict(branches_list[:insert_after_ind] + [(branch_workflow, dp_op)] +
    #                                     branches_list[insert_after_ind:])
    #         del self.branches_ui
    #         gc.collect()
    #         self.complete_workflow[1] = self.get_branches_ui()
    #     else:
    #         self.branches.update({branch_workflow: dp_op})
    #         self.branches_ui.append(self.get_branch_ui(branch_workflow, branch_index=self.len_of_all_branches))
    #     branch_workflow.show_dp()

    # def create_branch_from_source(self, event=None):
    #     self.create_branch()

    # def get_branch_ui(self, branch, branch_index):
    #     node = self.branches[branch]
    #     if node is None:
    #         node_index = -1
    #     else:
    #         node_index = self.children.index(node)
    #     if branch.parent and not branch.parent.parent:
    #         node_index += 1
    #     branch_ui = self.Row(self.Row('', width=150 * node_index), branch.main_pane, css_classes=[
    #         'branch_workflow', 'branch_{}'.format(branch_index)])
    #     return branch_ui

    # def get_branches_ui(self):
    #     self.branches_ui = self.Column()
    #     cum_branches = 0
    #     for index, branch in enumerate(self.branches.keys()):
    #         branch_index = index + cum_branches
    #         self.branches_ui.append(self.get_branch_ui(branch, branch_index=branch_index + 1))
    #         cum_branches += branch.len_of_all_branches
    #     return self.branches_ui

    # def set_active_false(self):
    #     for child in self.children:
    #         child.active.value = False
    #         # child.set_active_class()
    #     self.complete_workflow[0] = self.create_dp_workflow()
    #     for branch in self.branches:
    #         branch.set_active_false()

    def set_error_false(self):
        """A method to set the appropriate error flags in the WorkFlow."""
        for child in self.children:
            child.error = False
            if child.error_flag in child.pane:
                child.pane.remove(child.error_flag)
        # for branch in self.branches:
        #     branch.set_error_false()
        self.error_in_workflow = {"node": -2, "error": "", "traceback": ""}

    def view_source(self, event=None):
        """A callback function to display the source data to the WorkFlow."""
        # row = self.Overlay(self.Table(self.initial_data), size='full_screen')
        row = self.Overlay_inplace(self.Table(self.initial_data))
        self.overlays.append(row)

    def check_error_in_workflow(self, from_dp_index):
        """A method to track the error that occurred in the WorkFlow."""
        # reset error flag for all nodes
        self.set_error_false()
        for dp_op in self.children[from_dp_index:]:
            if not dp_op.active.value:
                continue
            processed_data = dp_op.process_data(dp_op.input_data)
            if isinstance(processed_data, str):
                dp_op.error = True
                dp_op.pane.insert(0, dp_op.error_flag)
                if from_dp_index == -1:
                    self.error_in_workflow["node"] = len(self.children) - 1
                else:
                    self.error_in_workflow["node"] = self.children.index(dp_op)
                self.error_in_workflow["error"] = processed_data
                self.error_in_workflow["traceback"] = dp_op.err_traceback
                break
        if self.error_in_workflow["node"] != -2:
            error_msg = "Error in Node-{}: {}".format(self.error_in_workflow["node"] + 1,
                                                      self.error_in_workflow["error"])
            if len(self.complete_workflow) == 2:
                self.complete_workflow.insert(2, error_msg)
            else:
                self.complete_workflow[2] = error_msg
            result = True
        else:
            if len(self.complete_workflow) == 3:
                self.complete_workflow.pop(2)
            result = False
        return result

    def complete_active_operation(self, operator, processed_data):
        """A method to safely wind up the active Operation in the WorkFlow."""
        self.output_data = processed_data
        dp_operation = DPOperation(operator=operator, parent=self)
        self.children.insert(self.current_dp_op_index, dp_operation)
        # trigger check for error functionality for current workflow
        self.check_error_in_workflow(from_dp_index=self.current_dp_op_index)
        dp_operation.complete_operation(operator)
        # +1 because of source node in the dp_workflow
        self.dp_workflow.insert(self.current_dp_op_index + 1, dp_operation.pane)
        self.overlays.clear()
        if self.parent_widget:
            self.parent_widget.preprocess_data()
        del self.dp_job.processed_data
        del self.dp_job
        gc.collect()

    def reach_state(self, operation_block):
        """A method to reach a particular state in the WorkFlow."""
        pass

    def show(self):
        """A method to display a particular state in the WorkFlow."""
        return self.main_pane

    def refresh_ui(self):
        """A method to reload the UI components the WorkFlow."""
        self.complete_workflow[0] = self.create_dp_workflow()
        self.check_error_in_workflow(0)

    def get_processed_data(self, recompute=False):
        """A method to compute the final data from the WorkFlow."""
        if self.has_state_change() or recompute:
            active_children = [child for child in self.children if child.active.value]
            self.output_data = self.initial_data.copy()
            for child in active_children:
                self.output_data = child.process_data(self.output_data)
            # self.save_current_state()
            return self.output_data
        else:
            return self.output_data

    def compute(self):
        """A method to compute the data that is to be passed to the Widget."""
        return self.parent_widget.preprocess_data(recompute=True)
