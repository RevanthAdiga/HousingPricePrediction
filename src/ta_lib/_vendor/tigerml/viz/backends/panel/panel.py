import os
from collections import OrderedDict
from functools import partial

import panel as pn

pn.extension("ace")

HERE = os.path.dirname(os.path.abspath(__file__))
style_path = f"{HERE}/../../static_resources/style.css"
zoom_style_path = f"{HERE}/../../static_resources/zoom_style.css"
tigerml_logo_path = f"{HERE}/../../static_resources/tigerml_logo.txt"
pn.config.css_files = [style_path]


def covert_args_to_kwargs(largs):
    lkwargs = {}
    largs = list(largs)
    for ind, larg in enumerate(largs):
        if isinstance(larg, dict):
            largs.remove(larg)
            lkwargs.update({f"__dummy__{ind}": larg})
    return lkwargs


def close_overlay(overlay_row, event=None):
    overlay_row.clear()
    return


class PanelBackend:
    """A backend support class for viz module using panel library (https://panel.holoviz.org/)."""

    Row = pn.Row
    Column = pn.Column
    Markdown = pn.pane.Markdown
    HTML = pn.pane.HTML
    Spacer = pn.Spacer
    Select = pn.widgets.Select
    Checkbox = pn.widgets.Checkbox
    Button = pn.widgets.Button
    MultiChoice = pn.widgets.MultiChoice
    WidgetBox = pn.WidgetBox
    ToggleGroup = pn.widgets.ToggleGroup
    FileSelector = pn.widgets.FileSelector
    FileInput = pn.widgets.FileInput
    StaticText = pn.widgets.StaticText
    TextInput = pn.widgets.TextInput
    DateRangeSlider = pn.widgets.DateRangeSlider
    DateSlider = pn.widgets.DateSlider
    TextAreaInput = pn.widgets.TextAreaInput
    Toggle = pn.widgets.Toggle
    JSON = pn.pane.JSON
    LiteralInput = pn.widgets.LiteralInput
    CodeInput = pn.widgets.Ace
    null_component = pn.pane.Markdown(width=0, height=0, css_classes=["is_hidden"])
    remove_glass = """var cols = document.getElementsByClassName('add_glass');
    for(var i = 0; i < cols.length; i++) {
    cols[i].classList.remove('glass_overlay');
    }
    """
    apply_glass = """var cols = document.getElementsByClassName('add_glass');
    for(var i = 0; i < cols.length; i++) {
    cols[i].classList.add('glass_overlay');
    }
    """
    toggle_glass = """var cols = document.getElementsByClassName('add_glass');
    for(var i = 0; i < cols.length; i++) {
    if (cols[i].className.includes('glass_overlay')) {
    cols[i].classList.remove('glass_overlay');
    }
    else {
    cols[i].classList.add('glass_overlay');
    }
    }
    """

    def Overlay(self, pane, size="big", closeable=True, close_callback=None):
        """A custom glass overlay object (centered) using panel components."""
        content = self.Column(pane, css_classes=["content", size])
        overlay = self.Column(
            self.HTML('<div class="glass"></div>'), content, css_classes=["overlay"]
        )
        row = self.Row(overlay)
        if closeable:
            close_btn = self.Button(
                css_classes=["icon-button", "icon-cancel"], width=50, height=40
            )
            if close_callback:
                close_btn.on_click(close_callback)
            else:
                close_btn.on_click(partial(close_overlay, row))
            content.insert(0, close_btn)
        return row

    def Overlay_inplace(self, pane, closeable=True, close_callback=None):
        """A custom glass overlay object (inplace) using panel components."""
        content = self.Column(pane, css_classes=["content"])
        overlay = self.Column(content, css_classes=["overlay_inplace"])
        row = self.Row(overlay)
        if closeable:
            close_btn = self.Button(
                css_classes=["icon-button", "icon-cancel"],
                width=50,
                height=40,
                background="white",
            )
            if close_callback:
                close_btn.on_click(close_callback)
            else:
                close_btn.on_click(partial(close_overlay, row))
                close_btn.js_on_click(args={}, code=self.remove_glass)
            overlay.insert(0, close_btn)
        return row

    def SectionHeader(self, content, valign="top"):
        """A custom SectionHeader object using panel components."""
        row = self.HTML(content, css_classes=["full_height"])
        if valign == "middle":
            row.css_classes.append("middle")
        return row

    def Table(self, data, width=None, height=600):
        """A custom Dataframe visualization object using panel components."""
        from tigerml.core.plots import hvPlot

        width = width or (len(data.columns) + 1) * 100
        return hvPlot(data).table(width=width, height=height)

    @classmethod
    def get_ui_dict(cls, *args, type, **kwargs):
        """A method that returns a dictionary of args and kwargs required for creating a panel UI component."""
        ui_dict = {}
        ui_dict["type"] = type
        ui_dict["args"] = args
        ui_dict["kwargs"] = kwargs
        return ui_dict

    def panel(self, obj, **kwargs):
        """A method that returns panel UI component of the given object."""
        return pn.panel(obj, **kwargs)

    def row(self, *largs, **lkwargs):
        """A method that returns panel Row UI dictionary with given arguments."""
        if largs:
            new_children = covert_args_to_kwargs(largs)
            lkwargs["children"] = new_children
        return self.get_ui_dict(*largs, type="row", **lkwargs)

    def column(self, *largs, **lkwargs):
        """A method that returns panel Column UI dictionary with given arguments."""
        if largs:
            new_children = covert_args_to_kwargs(largs)
            lkwargs["children"] = new_children
        return self.get_ui_dict(*largs, type="column", **lkwargs)

    def widgetbox(self, *largs, **lkwargs):
        """A method that returns panel UI dictionary object with given arguments."""
        if largs:
            new_children = covert_args_to_kwargs(largs)
            lkwargs["children"] = new_children
        return self.get_ui_dict(*largs, type="widgetbox", **lkwargs)

    def markdown(self, *args, **kwargs):
        """A method that returns panel Markdown UI dictionary with given arguments."""
        return self.get_ui_dict(*args, type="markdown", **kwargs)

    def html(self, *args, **kwargs):
        """A method that returns panel HTML UI dictionary with given arguments."""
        return self.get_ui_dict(*args, type="html", **kwargs)

    def button(self, *args, **kwargs):
        """A method that returns panel Button UI dictionary with given arguments."""
        return self.get_ui_dict(*args, type="button", **kwargs)

    def select(self, *args, **kwargs):
        """A method that returns panel Select UI dictionary with given arguments."""
        return self.get_ui_dict(*args, type="select", **kwargs)

    def multichoice(self, *args, **kwargs):
        """A method that returns panel Multichoice UI dictionary with given arguments."""
        return self.get_ui_dict(*args, type="multichoice", **kwargs)

    def checkbox(self, *args, **kwargs):
        """A method that returns panel Checkbox UI dictionary with given arguments."""
        return self.get_ui_dict(*args, type="checkbox", **kwargs)

    def togglegroup(self, *args, **kwargs):
        """A method that returns panel Togglegroup UI dictionary with given arguments."""
        return self.get_ui_dict(*args, type="togglegroup", **kwargs)

    def textinput(self, *args, **kwargs):
        """A method that returns panel Textinput UI dictionary with given arguments."""
        return self.get_ui_dict(*args, type="textinput", **kwargs)

    @classmethod
    def create_widget(cls, type, *args, **kwargs):
        """A method that returns panel UI components from given UI dictionary."""
        if "callback" in kwargs:
            callback = kwargs.pop("callback")
        else:
            callback = None
        if type == "row":
            element = pn.Row(*args, **kwargs)
        elif type == "column":
            element = pn.Column(*args, **kwargs)
        elif type == "widgetbox":
            element = pn.WidgetBox(*args, **kwargs)
        elif type == "markdown":
            element = pn.pane.Markdown(*args, **kwargs)
        elif type == "html":
            element = pn.pane.HTML(*args, **kwargs)
        elif type == "button":
            element = pn.widgets.Button(**kwargs)
        elif type == "select":
            element = pn.widgets.Select(**kwargs)
        elif type == "multichoice":
            element = pn.widgets.MultiChoice(**kwargs)
        elif type == "checkbox":
            element = pn.widgets.Checkbox(**kwargs)
        elif type == "togglegroup":
            element = pn.widgets.ToggleGroup(**kwargs)
        elif type == "textinput":
            element = pn.widgets.TextInput(**kwargs)
        else:
            raise Exception("Given widget type not recognised - {}".format(type))
        if callback:
            if type == "button":
                if isinstance(callback, list):
                    for func in callback:
                        element.on_click(func)
                else:
                    element.on_click(callback)
            else:
                if isinstance(callback, list):
                    for func in callback:
                        element.param.watch(func, "value")
                else:
                    element.param.watch(callback, "value")
        return element

    @classmethod
    def create_ui_from_config(cls, config, data=None, dtypes=None, initial_state={}):
        """A method that returns data dependent and independent UI components based on their initial configuration."""
        ui_dict = {}
        data_dependent = OrderedDict()
        stateful_elements = []
        for key in config:
            options_func = None
            if key in [
                key.replace(".value", "")
                for key in initial_state.keys()
                if ".value" in key
            ]:
                ui_element = cls.LiteralInput(
                    value=initial_state[f"{key}.value"],
                    disabled=True,
                    width=200,
                    css_classes=["fixed_value"],
                )
                stateful_elements.append(key)
            elif config[key] is None:
                ui_element = cls.null_component
            elif hasattr(config[key], "__module__") and config[
                key
            ].__module__.startswith("panel"):
                ui_element = config[key]
            elif isinstance(config[key], str):
                ui_element = pn.pane.Markdown(config[key])
            else:
                widget_type = config[key]["type"]
                largs = config[key]["args"]
                lkwargs = config[key]["kwargs"]
                if widget_type in ["row", "column", "widgetbox"]:
                    if "children" in lkwargs:
                        children_dict = lkwargs.pop("children")
                        (
                            children_ui,
                            ch_data_deps,
                            children_states,
                        ) = cls.create_ui_from_config(
                            children_dict, data, dtypes, initial_state
                        )
                        largs = [
                            children_ui[key] for key in children_dict.keys()
                        ] + list(largs)
                        ui_dict.update(
                            {
                                key: children_ui[key]
                                for key in children_ui
                                if "__dummy__" not in children_ui
                            }
                        )
                        data_dependent.update(ch_data_deps)
                        stateful_elements += children_states
                if "options" in lkwargs and lkwargs["options"].__class__.__name__ in [
                    "function",
                    "method",
                    "partial",
                ]:
                    options_func = lkwargs.pop("options")
                ui_element = cls.create_widget(widget_type, *largs, **lkwargs)
                if widget_type in [
                    "select",
                    "multichoice",
                    "checkbox",
                    "togglegroup",
                    "textinput",
                ]:
                    stateful_elements.append(key)
                if options_func is not None:
                    data_dependent.update({key: options_func})
            ui_dict.update({key: ui_element})
        return ui_dict, data_dependent, stateful_elements
