def create_report(
    contents,
    name="",
    path="",
    format=".html",
    split_sheets=True,
    tiger_template=False,
    **kwargs
):
    if format == ".xlsx":
        from .excel import create_excel_report

        create_excel_report(
            contents, name=name, path=path, split_sheets=split_sheets, **kwargs
        )
    elif format == ".pptx":
        from .ppt.lib import create_ppt_report

        create_ppt_report(contents, name=name, path=path, tiger_template=tiger_template)
    if format == ".html":
        from .html import create_html_report

        create_html_report(contents, name=name, path=path, split_sheets=split_sheets)
