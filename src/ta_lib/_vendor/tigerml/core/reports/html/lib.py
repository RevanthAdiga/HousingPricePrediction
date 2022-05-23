from tigerml.core.utils import time_now_readable

from .contents import HTMLText
from .Report import HTMLComponent, HTMLComponentGroup, HTMLDashboard, HTMLReport


def create_html_report(
    contents, columns=2, save=True, name="", path="", split_sheets=True
):
    if not name:
        name = "report_at_{}".format(time_now_readable())
    report = HTMLReport(name)
    assert isinstance(contents, dict), "contents should be in the form of a dict"
    if split_sheets:
        needs_folder = False
        for content in contents:
            content_name = content
            content = contents[content_name]
            if not isinstance(content, dict):
                content = {"": content}
            report_dict, needs_folder_local = create_html_dashboard(
                content, name=content_name, columns=columns
            )
            needs_folder = needs_folder or needs_folder_local
            report.append_dashboard(report_dict)
    else:
        report_dict, needs_folder = create_html_dashboard(
            contents, name=name, columns=columns
        )
        report.append_dashboard(report_dict)
    if save:
        report.save(path=path, needs_folder=needs_folder)
    else:
        return report


def create_html_dashboard(contents, name="", columns=2, flatten=False):
    dash = HTMLDashboard(name=name)
    # for content in contents:
    # 	content_name = content
    # 	content = contents[content_name]
    cg, needs_folder = create_component_group(
        contents, dash, columns=columns, flatten=flatten
    )
    dash.append(cg)
    return dash, needs_folder


def group_components(components, dashboard, name="", columns=2, flatten=False):
    if [component for component in components if isinstance(component, tuple)]:
        final_cg = HTMLComponentGroup(dashboard, name=name, columns=1)
        current_cg = None
        for component in components:
            if isinstance(component, tuple):
                # import pdb
                # pdb.set_trace()
                if current_cg:
                    import copy

                    old_cg = copy.copy(current_cg)
                    # old_cg.name = ''
                    # final_cg = HTMLComponentGroup(dashboard, name=name, columns=1)
                    final_cg.append(old_cg)
                new_cg = group_components(
                    component[1],
                    dashboard,
                    component[0],
                    columns=columns,
                    flatten=flatten,
                )
                final_cg.append(new_cg)
                current_cg = None
            else:
                if not current_cg:
                    current_cg = HTMLComponentGroup(dashboard, name="", columns=columns)
                current_cg.append(component)
        if current_cg:
            final_cg.append(current_cg)
    else:
        final_cg = HTMLComponentGroup(dashboard, name=name, columns=columns)
        for component in components:
            final_cg.append(component)

    # if final_cg != current_cg:
    # 	final_cg.append(current_cg)
    return final_cg


def create_component_group(contents, dashboard, name="", columns=2, flatten=False):
    from ..helpers import create_components

    needs_folder = False
    if isinstance(contents, str):
        components = [HTMLText(contents, name=dashboard.name)]
    else:
        components, needs_folder = create_components(
            contents, flatten=flatten, format="html"
        )
    # if len(contents) == 1:
    # 	columns = 1
    cg = group_components(
        components, dashboard, name=name, columns=columns, flatten=flatten
    )
    return cg, needs_folder
