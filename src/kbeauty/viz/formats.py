"""Number and date formats used in chart text, so the same quantity reads the same everywhere."""

import matplotlib.dates as mdates
import matplotlib.ticker as mticker


def pct(x: float, digits: int = 1, signed: bool = True) -> str:
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def usd_man(x: float) -> str:
    """USD in units of 10,000 (만 달러), the unit used in the notebooks and case study."""
    return f"{x / 1e4:,.0f}만 달러"


def month_axis(ax, months=(1, 7), span=None) -> None:
    """Ticks at the given months, labelled '24.01'. With span=(first, last) ticks stop at the data, so space
    added on the right for direct labels gets no empty date ticks."""
    if span is None:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=months))
    else:
        import pandas as pd
        ticks = [d for d in pd.date_range(span[0], span[1], freq="MS") if d.month in months]
        ax.xaxis.set_major_locator(mticker.FixedLocator(mdates.date2num(ticks)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))


def percent_axis(ax, axis: str = "x", decimals: int = 0) -> None:
    fmt = mticker.PercentFormatter(1.0, decimals=decimals)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)
