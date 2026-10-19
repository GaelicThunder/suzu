"""The dashboard object.

`View` is assembled from four mixins so each concern has its own file:

  core.py     state: construction, theme cycling, ramps, smoothing, tick()
  widgets.py  drawing primitives: bars, sparks, boxes, gauges, big digits
  panels.py   panel builders: cluster, lanes, model strip, experts, heatmap
  frames.py   full-screen composition: full / perf / pipes / layers / spec

Methods in one mixin call methods from another through `self`, exactly as they
did inside the single class.
"""
from .core import CoreMixin
from .widgets import WidgetsMixin
from .panels import PanelsMixin
from .frames import FramesMixin


class View(WidgetsMixin, PanelsMixin, FramesMixin, CoreMixin):
    """Live view of one engine on one port.

    Nothing is cached per terminal size: every frame is rebuilt from the size
    it is given, so a resize takes effect on the next tick.
    """
