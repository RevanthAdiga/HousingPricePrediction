try:
    import pptx
except ModuleNotFoundError as e:
    Warning(
        "Please install python-pptx for Powerpoint "
        "reports. - conda install python-pptx"
    )
    raise e
from tigerml.core.utils import slugify

from .lib import create_ppt_report
from .Report import PptReport, Section
from .Slide import Slide, SlideComponent

prs = pptx.Presentation()


class slide_layouts:
    """Slide layouts class initializer."""

    def __init__(self):
        """Slide layout class initializer."""
        for layout in prs.slide_layouts:
            setattr(self, slugify(layout.name), layout)
        self.list = [slugify(layout.name) for layout in prs.slide_layouts]


layouts = slide_layouts()
