from tigerml.viz.widget.states import StatefulUI

from .constants import COMPARATORS, CONDITIONOPERATIONS, CONDITIONS, DTYPES


class FilterCondition(StatefulUI):
    """A class for individual FilterCondition object in FilterPanel."""

    def __init__(self, parent, data, data_dict, initial_state={}):
        super().__init__(initial_state=initial_state)
        self.parent = parent
        self.data = data
        self.data_dict = data_dict.copy()
        self.select_col = self.Select(
            options=self.data_dict["Columns"].values.tolist(),
            value=self.data_dict["Columns"][1],
            width=200,
        )
        self.select_col.param.watch(self.update_input_val, "value")
        self.condition = self.Select(
            options=CONDITIONS[self.data_dict["Dtypes"][1]], width=200
        )
        # self.condition.param.watch(self.update_input_val, 'value')
        self.input_val = ""
        self.pane = self.create_pane()
        self.select_col.value = self.select_col.values[0]
        self.current_state.update(
            {
                "select_col.value": self.select_col.value,
                "condition.value": self.condition.value,
                "input_val.value": self.input_val.value,
            }
        )

    def create_pane(self):
        """A method that creates UI components of FilterCondition object."""
        return self.Row(
            self.select_col,
            self.condition,
            self.input_val,
            css_classes=["filter_condition"],
        )

    @property
    def has_changes(self):
        """
        Tracks the changes in the filter condition.

        :return:
        bool : True if changed or False if not changed
        """
        return self.has_state_change(
            "select_col.value", "condition.value", "input_val.value"
        )

    def update_input_val(self, event=None):
        """A method to update the input value of the FilterCondition object."""
        dtype = self.data_dict[self.data_dict.Columns == event.new].Dtypes.values[0]
        values = self.data_dict[self.data_dict.Columns == event.new].Values.values[0]
        if dtype == "numeric":
            if isinstance(values, list):
                self.input_val = self.Select(
                    width=300, css_classes=["filter_multiselect"], options=values
                )
            else:
                self.input_val = self.TextInput(
                    width=300, placeholder="Enter value between  " + str(values)
                )
        elif "datetime" in dtype:
            min_date = values[0]
            max_date = values[1]
            self.input_val = self.DateRangeSlider(
                width=300, start=min_date, end=max_date, value=(min_date, max_date)
            )
        else:
            self.input_val = self.MultiChoice(
                width=300, css_classes=["filter_multiselect"], options=values
            )
        self.condition.options = []
        self.condition.options = CONDITIONS[dtype]
        self.condition.value = self.condition.options[0]
        self.pane[1] = self.condition
        self.pane[2] = self.input_val

    def delete_level(self, event=None):
        """A method to delete the FilterCondition object."""
        self.parent.delete_filter(self)

    def show(self, refresh=False):
        """A method that returns the FilterCondition UI object."""
        if refresh:
            self.pane = self.create_pane()
        return self.pane

    def get_filter(self):
        """
        Returns a boolean series for each individual condition.

        :parameter:
        CONDITIONS : contains the widget options for different data types
        COMPARATORS : variables in Comparators holds the display value of the condition widgets
        CONDITIONOPERATIONS : holds the python operation for the condition

        :return:
        filter_result : boolean series
        """
        if not self.has_changes and hasattr(self, "filter_result"):
            return self.filter_result
        negation = False
        list_input = False
        self.filter_result = None
        if self.input_val.value:
            lhs = 'self.data["{}"]'.format(self.select_col.value)
            col_dtype = self.data_dict[
                self.data_dict.Columns == self.select_col.value
            ].Dtypes.values[0]
            col_values = self.data_dict[
                self.data_dict.Columns == self.select_col.value
            ].Values.values[0]
            func = CONDITIONOPERATIONS[self.condition.value]
            if "isin" in func:
                list_input = True
            if col_dtype in [DTYPES.category, DTYPES.string]:
                if "!" in func:
                    func = func[1:]
                    negation = True
                if not list_input and col_dtype == DTYPES.string:
                    case = True
                    if func[0] == "i":
                        func = func[1:]
                        case = False
                    func = "str.{}".format(func)
                    value = '"{}"'.format(self.input_val.value)
                    if case is False:
                        if func.split(".")[-1] in [
                            COMPARATORS.startswith,
                            COMPARATORS.endswith,
                        ]:
                            func = "str.lower()." + func
                            value = value.lower()
                        else:
                            value = value + ", case=False"
                elif list_input:
                    value = self.input_val.value
                    if isinstance(value, str):
                        value = (
                            self.input_val.value.replace(", ", ",")
                            .replace(" ,", ",")
                            .split(",")
                        )
                else:
                    value = str(self.input_val.value)
                rhs = ".{}({})".format(func, value)
            elif DTYPES.datetime in col_dtype:
                start_date, end_date = self.input_val.value
                if start_date > col_values[0]:
                    self.filter_result = (
                        "(self.data['"
                        + self.select_col.value
                        + "']"
                        + ">= '"
                        + str(start_date)
                        + "'"
                    )
                if end_date < col_values[1]:
                    if self.filter_result is None:
                        self.filter_result = (
                            "(self.data['"
                            + self.select_col.value
                            + "']"
                            + "<= '"
                            + str(end_date)
                            + "'"
                        )
                    else:
                        self.filter_result += (
                            ") & ("
                            + "self.data['"
                            + self.select_col.value
                            + "']<= '"
                            + str(end_date)
                            + "')"
                        )
                        self.filter_result = "(" + self.filter_result
                self.filter_result += ")"
                self.filter_result = eval(self.filter_result)
                return self.filter_result
            elif list_input and isinstance(self.input_val.value, list):
                rhs = f".{func}({self.input_val.value})"
            elif list_input:
                rhs = f".{func}([{self.input_val.value}])"
            else:
                if isinstance(self.input_val.value, list):
                    input_value = str(self.input_val.value)[1:-1]
                else:
                    input_value = str(self.input_val.value)
                rhs = CONDITIONOPERATIONS[self.condition.value] + input_value
            filter_str = ("-" if negation else "") + lhs + rhs
            self.filter_result = eval(filter_str)
        return self.filter_result

    def describe(self):
        """A method that provides the description of the FilterCondition object in widget_builder."""
        description = self.Row(css_classes=["filter"])
        description.append(
            self.select_col
            if "select_col.value" not in self.initial_state
            else self.initial_state["select_col.value"]
        )
        description.append(
            self.condition
            if "condition.value" not in self.initial_state
            else self.initial_state["condition.value"]
        )
        description.append(
            self.input_val
            if "input_val.value" not in self.initial_state
            else str(self.initial_state["input_val.value"])
        )
        return description
