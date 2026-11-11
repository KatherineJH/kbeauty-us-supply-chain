"""Visual layer: design tokens and fonts (theme), figure scaffolding (annotations), export, and one module per
chart family. Importing the package applies the theme, so notebook charts share the report style."""

from . import theme  # noqa: F401  (applies rcParams and registers the bundled fonts)
from . import annotations, comparison, decision, export, forecasting, formats, lineage, timeseries  # noqa: F401
