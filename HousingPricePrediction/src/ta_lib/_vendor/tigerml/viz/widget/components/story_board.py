import os
import webbrowser

from tigerml.core.utils import time_now_readable

from ...backends.panel import PanelBackend


class Comment(PanelBackend):
    """A class for individual Comment object in StoryBoard."""

    def __init__(self):
        self.comment_input = self.TextAreaInput(width=400)
        self.comment_content = ""
        self.add_comment = self.Button(
            name="Add Comment",
            css_classes=["tertiary", "button", "icon-button-prefix", "icon-comment"],
            width=150,
        )
        self.save_button = self.Button(
            name="Save Comment",
            css_classes=["secondary", "button", "icon-button-prefix", "icon-ok"],
            width=150,
        )
        self.cancel = self.Button(
            name="Cancel",
            css_classes=["tertiary", "button", "icon-button-prefix", "icon-cancel"],
            width=150,
        )
        self.edit_mode = self.Column(
            self.comment_input, self.Row(self.save_button, self.cancel)
        )
        self.normal_mode = self.Column(self.comment_content, self.add_comment)
        self.pane = self.Column(self.normal_mode, css_classes=["comment_module"])
        # bindings
        self.add_comment.on_click(self.go_to_edit_mode)
        self.save_button.on_click(self.save_comment)
        self.cancel.on_click(self.go_to_normal_mode)

    def go_to_edit_mode(self, event=None):
        """A callback function that enables comment editing."""
        self.comment_input.value = self.comment_content
        self.pane[0] = self.edit_mode

    def go_to_normal_mode(self, event=None):
        """A callback function that disables comment editing."""
        self.pane[0] = self.normal_mode

    def save_comment(self, event=None):
        """A callback function that saves the comment."""
        self.comment_content = self.comment_input.value
        self.normal_mode[0] = self.comment_content
        self.pane[0] = self.normal_mode

    @property
    def save_content(self):
        """A property that returns the comment."""
        return self.Column(self.comment_content)

    def show(self):
        """A method that returns the Comment's UI components."""
        return self.pane


class StoryItem(PanelBackend):
    """A class for individual Story object in StoryBoard."""

    def __init__(self, plot, parent, title="", context=None):
        self.plot = plot
        self.parent = parent
        self.title_text = title
        self.title = self.HTML(f"<h3>{title}</h3>")
        self.title_input = self.TextInput(css_classes=["story_title_input"])
        self.title_edit = self.Button(
            width=30, height=30, css_classes=["icon-edit", "icon-button", "is_hidden"]
        )
        self.title_confirm = self.Button(
            width=30, height=30, css_classes=["icon-ok", "icon-button"]
        )
        self.title_edit_cancel = self.Button(
            width=30, height=30, css_classes=["icon-cancel", "icon-button"]
        )
        self.title_controls = self.Column(self.title_edit)
        self.title_pane = self.Row(
            self.title, self.title_controls, css_classes=["story_item_title"]
        )
        self.title_edit.on_click(self.edit_mode_on)
        self.title_confirm.on_click(self.change_title)
        self.title_edit_cancel.on_click(self.edit_mode_off)
        self.context = context
        self.description = ""
        self.comment = Comment()
        self.delete_button = self.Button(
            name="Delete",
            css_classes=["tertiary", "button", "icon-delete", "icon-button-prefix"],
            width=100,
        )
        self.delete_button.on_click(self.delete_self)
        self.header = self.Row(
            self.title_pane,
            self.delete_button,
            css_classes=["section_header", "full_width", "gray_bg"],
        )
        self.content = self.Column(plot, self.description)

    def edit_mode_on(self, event=None):
        """A callback function that enables story editing."""
        self.title_input.value = self.title_text
        self.title_pane[0] = self.title_input
        self.title_controls[0] = self.Row(self.title_confirm, self.title_edit_cancel)

    def edit_mode_off(self, event=None):
        """A callback function that disables story editing."""
        self.title_pane[0] = self.title
        self.title_controls[0] = self.title_edit

    def change_title(self, event=None):
        """A callback function that enables story's title editing."""
        self.title_text = self.title_input.value
        self.title = self.HTML(f"<h3>{self.title_text}</h3>")
        self.edit_mode_off()

    def delete_self(self, event=None):
        """A callback function to delete the Story object."""
        self.parent.remove(self)

    @property
    def save_content(self):
        """A property that renders the Story object into html format and returns it."""
        if self.plot.__module__ == "panel.layout":
            plot = self.plot[1].object
            no_of_widgets = len(self.plot[0][1])
            import holoviews as hv

            hv.extension("bokeh")
            hv.output(widget_location="top_right")
            bokeh_renderer = hv.renderer("bokeh")
            plot_html = bokeh_renderer.static_html(plot)
            if plot.__module__ == "holoviews.core.layout":
                self.parent.custom_head_script = plot_html[
                    plot_html.index("</style>") + 8 : plot_html.index("</head>")
                ]
            head_tag = plot_html[
                plot_html.index("<head>") + 6 : plot_html.index("</head>")
            ]
            script_tags = []
            while "<script" in head_tag:
                tag = head_tag[
                    head_tag.index("<script") : head_tag.index("</script>") + 9
                ]
                script_tags += [tag]
                head_tag = head_tag.replace(tag, "")
            script = "".join([x for x in script_tags if "<script src=" not in x])
            html_body = plot_html[plot_html.index("</head>") + 7 :]
            plot_content = self.HTML(
                script
                + html_body[html_body.index("<body>") + 6 : html_body.index("</body>")],
                width=1020,
                height=520 + (no_of_widgets * 85),
            )
        else:
            import holoviews as hv

            hv.extension("bokeh")
            bokeh_renderer = hv.renderer("bokeh")
            plot_html = bokeh_renderer.static_html(self.plot)
            head_tag = plot_html[
                plot_html.index("<head>") + 6 : plot_html.index("</head>")
            ]
            script_tags = []
            while "<script" in head_tag:
                tag = head_tag[
                    head_tag.index("<script") : head_tag.index("</script>") + 9
                ]
                script_tags += [tag]
                head_tag = head_tag.replace(tag, "")
            script = "".join([x for x in script_tags if "<script src=" not in x])
            html_body = plot_html[plot_html.index("</head>") + 7 :]
            plot_content = self.HTML(
                script
                + html_body[html_body.index("<body>") + 6 : html_body.index("</body>")],
                width=1020,
                height=520,
            )
        final_content = self.Column(plot_content, self.description)
        return self.Column(
            self.Row(self.title, css_classes=["bold", "big", "full_width", "gray_bg"]),
            self.Row(final_content, self.comment.save_content),
            css_classes=["full_width"],
        )

    def show(self):
        """A method that returns the StoryItem's UI components."""
        return self.Column(
            self.header,
            self.Row(
                self.content, self.comment.show(), css_classes=["story_item_content"]
            ),
            css_classes=["full_width"],
        )


class StoryBoard(PanelBackend):
    """A class for containing all Story objects together in StoryBoard."""

    def __init__(self):
        self.items = []
        self.save_button = self.Button(
            name="SAVE THE BOARD",
            css_classes=[
                "right",
                "primary",
                "button",
                "icon-button-prefix",
                "icon-download-white",
            ],
        )
        self.save_button.on_click(self.save)
        self.header = self.Row(
            "# Story Board", self.save_button, css_classes=["full_width", "gray_bg"]
        )
        self.content = self.Column("")
        self.custom_head_script = None
        self.pane = self.Column(
            (self.header if self.items else ""),
            self.content,
            css_classes=["full_width", "story_board"],
        )

    def add_item(self, plot, title="", context=""):
        """A method to add a new StoryItem to StoryBoard."""
        item = StoryItem(plot, self, title, context)
        self.append(item)
        self.update_ui()

    def append(self, item):
        """A method to add an existing StoryItem to StoryBoard."""
        assert isinstance(item, StoryItem), "pass a story item to append"
        self.items.append(item)

    def remove(self, item):
        """A method to remove an existing StoryItem of the StoryBoard."""
        self.items.remove(item)
        self.update_ui()

    def compute_content(self):
        """A method to refresh the individual StoryItem's UI components."""
        self.content = self.Column(
            *[item.show() for item in self.items], css_classes=["full_width"]
        )

    def compute_save_content(self):
        """A method that renders individual Story objects into html format."""
        self.save_content = self.Column(
            *[item.save_content for item in self.items], css_classes=["full_width"]
        )

    def update_ui(self):
        """A method to refresh the StoryBoard UI components."""
        self.compute_content()
        self.pane[0] = self.header if self.items else self.Row("")
        self.pane[1] = self.content

    def save(self, event=None):
        """A property that renders the individual Story object and saves it as an html report."""
        self.compute_save_content()
        filename = "story_board_at_{}".format(time_now_readable())
        self.save_content.save(filename)

        if self.custom_head_script:
            file = open(filename + ".html", "r", encoding="utf-8")
            full_html = file.read()

            # commenting existing script inside head
            full_html = (
                full_html[: full_html.index("</style>") + 8]
                + "<!--"
                + full_html[full_html.index("</style>") + 8 :]
            )
            full_html = (
                full_html[: full_html.index("</head>")]
                + "-->"
                + full_html[full_html.index("</head>") :]
            )

            # adding custom script inside head
            # custom script is taken from the head_tag of static_html of single DMap_of_Layout
            custom_script = self.custom_head_script
            full_html = (
                full_html[: full_html.index("</head>")]
                + "\n"
                + custom_script
                + "\n"
                + full_html[full_html.index("</head>") :]
            )

            # rewriting the file with custom script
            with open(filename + ".html", "w", encoding="utf-8") as f:
                f.write(full_html)
        webbrowser.open("file://" + os.path.realpath(filename) + ".html")
        del self.save_content

    def show(self):
        """A method that returns the StoryBoard UI components."""
        self.update_ui()
        return self.pane
