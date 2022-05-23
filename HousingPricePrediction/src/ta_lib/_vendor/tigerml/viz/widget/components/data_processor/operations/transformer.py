import numpy as np
import pandas as pd

from tigerml.viz.widget.states import StatefulUI


class Transformer(StatefulUI):
    """A class for FeatureEngineering of data in DataProcessor."""

    def __init__(self, data):
        super().__init__()
        self.name = "Create"
        self.input_data_cols = data.columns.tolist()
        self.pane = self.Column()
        self._initiate()
        self.current_state.update(self.opt_state())
        # self.pane = self.Row(self.transform_type)

    def create_ui(self):
        """A method to create the UI components."""
        self.pane.clear()
        self.pane.extend([self.transform_type, self.Row(self.new_col_name, self.col_selection_block),  # noqa
                          self.op_info_block, self.code_input])
        return self.pane

    def _initiate(self):
        self.transform_type = self.Select(name="Transformation type",
                                          options=["Column-wise transform", "Groupby transform",
                                                   "Groupby aggregated transform", "Row-wise transform"], width=200)
        self.transform_type.param.watch(self.opt_func, "value")
        self.new_col_name = self.TextInput(name="New column name", placeholder="Enter the column name to be added...",
                                           width=200)
        self.select_columns = self.MultiChoice(name="Select the cols to trans",
                                               options=self.input_data_cols, width=200)
        self.select_grpby_cols = self.MultiChoice(name="Select the cols to groupby",
                                                  options=self.input_data_cols, width=200)
        self.select_column = self.Select(name="Select the col to trans", options=self.input_data_cols, width=200)
        self.col_selection_block = self.Column(self.select_column, "")

        code_snippet = "# Edit the code below...\ndef func(x):\n\toutput = f(x)\n\treturn output"
        self.code_input = self.CodeInput(name="pre process code", value=code_snippet, height=400,
                                         width=500, language="python")
        op_info = "Note: Keep the main function name as 'func' and input & output for it would be a scalar."
        self.op_info_block = self.Row(op_info)
        self.create_ui()

    def opt_func(self, event=None):
        """A callback function that updates the Operations pane for different Transformations based on the selection."""
        if event.new == "Column-wise transform":
            op_info = "Note: Keep the main function name as 'func', "\
                      "input is a DataFrame with selected column and output for it should be a Series."
            self.op_info_block[0] = op_info
            self.col_selection_block[0] = self.select_column
            self.col_selection_block[1] = ""
            code_snippet = "# Edit the code below...\ndef func(x):\n\toutput = f(x)\n\treturn output"
            self.code_input.value = code_snippet

        elif event.new == "Groupby transform":
            op_info = "Note: Keep the main function name as 'func', "\
                      "input is a DataFrame with selected columns and output should be a Series."
            self.op_info_block[0] = op_info
            self.col_selection_block[0] = self.select_columns
            self.col_selection_block[1] = self.select_grpby_cols
            code_snippet = "# Edit the code below...\ndef func(df):\n\toutput = f(df)\n\treturn output"
            self.code_input.value = code_snippet

        elif event.new == "Groupby aggregated transform":
            op_info = "Note: Keep the main function name as 'func', "\
                      "input is a DataFrame with selected columns and output should be a scalar."
            self.op_info_block[0] = op_info
            self.col_selection_block[0] = self.select_columns
            self.col_selection_block[1] = self.select_grpby_cols
            code_snippet = "# Edit the code below...\ndef func(df):\n\toutput = f(df)\n\treturn output"
            self.code_input.value = code_snippet

        elif event.new == "Row-wise transform":
            op_info = "Note: Keep the main function name as 'func', "\
                      "input is a Row wise Dict of selected columns and output should be a scalar."
            self.op_info_block[0] = op_info
            self.col_selection_block[0] = self.select_columns
            self.col_selection_block[1] = ""
            code_snippet = "# Edit the code below...\ndef func(df):\n\toutput = f(df)\n\treturn output"
            self.code_input.value = code_snippet

    def opt_state(self):
        """A method that returns the state of the Operation."""
        state = {
            "new_col_name.value": self.new_col_name.value,
            "transform_type.value": self.transform_type.value,
            "select_column.value": self.select_column.value,
            "select_columns.value": self.select_columns.value,
            "select_grpby_cols.value": self.select_grpby_cols.value,
            "code_input.value": self.code_input.value,
        }
        return state

    def exec_func(self, data):
        """A method to execute the operation."""
        if self.new_col_name.value == "":
            return "Enter new column name."
        if self.new_col_name.value in data.columns:
            return "New column name cannot be an existing column name."
        processed_data = data.copy()
        import os

        HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = f"{HERE}/transform_code.py"
        f = open(file_path, "w")
        f.write(self.code_input.value)
        f.close()
        from importlib import reload
        from tigerml.viz.widget.components.data_processor import transform_code

        reload(transform_code)
        if self.transform_type.value == "Column-wise transform":
            """
            processed_data[self.new_col_name.value] = processed_data[self.select_cols.value].apply(
                lambda x: transform_code.func(x)) is equivalent to
            processed_data[self.new_col_name.value] = processed_data[[self.select_cols.value]].apply(
                lambda x: transform_code.func(x), axis=1) which is taken care in 'Row-wise transform' case.

            processed_data[self.new_col_name.value] = processed_data[[self.select_cols.value]].apply(
                lambda x: transform_code.func(x)) is equivalent to
            processed_data[self.new_col_name.value] = processed_data[[self.select_cols.value]].apply(
                lambda x: transform_code.func(x), axis=0) which will enable series level operations.
            """
            processed_data[self.new_col_name.value] = processed_data[[self.select_column.value]].apply(lambda x: transform_code.func(x))  # noqa
            return processed_data
        elif self.transform_type.value == "Groupby aggregated transform":

            def post_process_agg_result(df, result):
                result = pd.Series([result] * len(df))
                result.index = df.index
                return result.to_frame()

            processed_data[self.new_col_name.value] = processed_data.groupby(
                self.select_grpby_cols.value
            )[self.select_grpby_cols.value + self.select_columns.value].apply(
                lambda x: post_process_agg_result(x, transform_code.func(x))
            )

            """
            The above code is same as the following
            new_col_series = processed_data.groupby(self.select_grpby_cols.value)[
                self.select_grpby_cols.value + self.select_columns.value].apply(lambda x: transform_code.func(x))
            new_col_series.name = self.new_col_name.value
            processed_data = processed_data.merge(new_col_series, how='left',
                                                  left_on=self.select_grpby_cols.value,
                                                  right_index=True)
            """
            return processed_data
        elif self.transform_type.value == "Groupby transform":

            def post_process_trans_result(df, result):
                result.index = df.index
                return result.to_frame()

            processed_data[self.new_col_name.value] = processed_data.groupby(
                self.select_grpby_cols.value
            )[self.select_grpby_cols.value + self.select_columns.value].apply(
                lambda x: post_process_trans_result(x, transform_code.func(x))
            )
            return processed_data
        elif self.transform_type.value == "Row-wise transform":

            def generate_func_strings(cols):
                wrapper_func_str = ("\ndef wrapper_func(params, code=transform_code):\n"
                                    "\tx = {dict}\n"
                                    "\treturn code.func(x)")
                param = ""
                di = ""
                df_cols = "("
                for index in range(len(cols)):
                    param = param + "x" + str(index) + ", "
                    di = di + "'" + cols[index] + "':" + "x" + str(index) + ", "
                    df_cols = df_cols + "processed_data['" + cols[index] + "'], "
                param = param[:-2]
                di = di[:-2]
                df_cols = df_cols[:-2]
                df_cols += ")"
                wrapper_func_str = wrapper_func_str.replace("params", param)
                wrapper_func_str = wrapper_func_str.replace("dict", di)
                transform_func_str = "np.vectorize(wrapper_func)" + df_cols
                return wrapper_func_str, transform_func_str

            wrapper_func_, transform_func = generate_func_strings(self.select_columns.value)
            exec(wrapper_func_)
            processed_data[self.new_col_name.value] = eval(transform_func)
            return processed_data


class CustomTransformer(StatefulUI):
    """A class for completely transforming the data in DataProcessor."""

    def __init__(self, data):
        super().__init__()
        self.name = "Custom"
        self.operation_name = self.TextInput(name="Transformation name (optional)",
                                             placeholder="Enter the transformation name...",
                                             value="Custom transformation", width=200)
        self.op_info = "Note: Keep the main function name as 'func' and input & output for it would be a DataFrame."
        code_snippet = "# Edit the code below...\ndef func(df):\n\tdf = f(df)\n\treturn df"
        self.code_input = self.CodeInput(name="pre process code", value=code_snippet, height=400,
                                         width=500, language="python")
        self.current_state.update(self.opt_state())
        self.pane = self.Column()
        self.create_ui()

    def create_ui(self):
        """A method to create the UI components."""
        self.pane.clear()
        self.pane.extend([self.Row(self.op_info), self.operation_name, self.code_input])
        return self.pane

    def opt_state(self):
        """A method that returns the state of the Operation."""
        return {"operation_name.value": self.operation_name.value, "code_input.value": self.code_input.value}

    def exec_func(self, data):
        """A method to execute the operation."""
        processed_data = data.copy()
        import os

        HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = f"{HERE}/transform_code.py"
        f = open(file_path, "w")
        f.write(self.code_input.value)
        f.close()
        from importlib import reload
        from tigerml.viz.widget.components.data_processor import transform_code

        reload(transform_code)
        processed_data = transform_code.func(processed_data)
        return processed_data
