#!/usr/bin/env python3
"""
Shared utility functions for plotting scripts.
"""

import calendar
from datetime import datetime, timedelta

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# Default font sizes
DEFAULT_FONTSIZES = {
    "tick_label": 10,
    "axis_label": 11,
    "legend": 10,
    "title": 12,
}

# List form [title, axis_label, tick_label] for use with set_axis_fontsizes()
FONTSIZES_LIST = [
    DEFAULT_FONTSIZES["title"],
    DEFAULT_FONTSIZES["axis_label"],
    DEFAULT_FONTSIZES["tick_label"],
]

# Color scheme from analyse_posteriors.py
COLORBLINDFR = {
    "main": ["#56b3e9", "#e0d316", "#0072b2", "#e69d00", "#cc79a7"],
    "additional": ["#EC681E", "#009e74", "#000000"],
}
COLORS = COLORBLINDFR["main"]

# Colors for leave-one-out variants
VARIANT_COLORS = {
    "all": COLORS[3],  # "#cc79a7"
    "nocasecounts": "purple",
    "nowastewater": "brown",
    "noseroprevalence": "red",
}


def decimal_year_to_matplotlib_date(decimal_year: float) -> float:
    """
    Convert one BEAST-style decimal year to a matplotlib date number.

    Matches the inversion used in ``convert_date_to_numerical_date`` /
    ``decimal_years_to_matplotlib_dates``.
    """
    t = float(decimal_year)
    y = int(np.floor(t))
    frac = t - y
    if frac <= 0:
        y -= 1
        frac = t - y
    diy = 366 if calendar.isleap(y) else 365
    day_offset = frac * diy - 1.0
    pydt = datetime(y, 1, 1) + timedelta(days=float(day_offset))
    return mdates.date2num(pydt)


def decimal_years_to_matplotlib_dates(decimal_years: np.ndarray) -> np.ndarray:
    """
    Convert BEAST-style decimal years to matplotlib date numbers (calendar x-axes).
    """
    dy = np.asarray(decimal_years, dtype=float)
    flat = dy.ravel()
    out_flat = np.empty(flat.shape[0], dtype=float)
    for i, t in enumerate(flat):
        out_flat[i] = decimal_year_to_matplotlib_date(t)
    return out_flat.reshape(dy.shape)


def configure_calendar_xaxis(ax: plt.Axes) -> None:
    """Matplotlib date numbers on x → tick labels ``MM-DD-YYYY`` (shared tree/overview plots)."""
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%Y"))


def set_axis_fontsizes(ax, fontsizes, xlabel=None, ylabel=None):
    """
    Set standard font sizes for axis labels and tick labels.

    Sets tick label fontsize to fontsizes[2] and ensures axis labels
    use fontsizes[1]. Optionally, you can override the current x and y labels.

    Args:
        ax: Matplotlib axis object
        fontsizes: List of font sizes [title, axis_label, tick_label]
        xlabel (str, optional): New x-axis label. If None, use current label.
        ylabel (str, optional): New y-axis label. If None, use current label.
    """
    # Set tick label fontsize
    ax.tick_params(labelsize=fontsizes[2])

    # Set axis labels with desired fontsize
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsizes[1])
    else:
        current_xlabel = ax.get_xlabel()
        if current_xlabel:
            ax.set_xlabel(current_xlabel, fontsize=fontsizes[1])

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsizes[1])
    else:
        current_ylabel = ax.get_ylabel()
        if current_ylabel:
            ax.set_ylabel(current_ylabel, fontsize=fontsizes[1])


def beautify_plot(
    ax,
    tick_label_fontsize=None,
    axis_label_fontsize=None,
    title_fontsize=None,
    remove_spines=True,
):
    """
    Beautify a plot by setting font sizes and removing spines.

    Args:
        ax: Matplotlib axis object
        tick_label_fontsize (int, optional): Font size for tick labels.
            Defaults to DEFAULT_FONTSIZES['tick_label'].
        axis_label_fontsize (int, optional): Font size for axis labels.
            Defaults to DEFAULT_FONTSIZES['axis_label'].
        title_fontsize (int, optional): Font size for title.
            Defaults to DEFAULT_FONTSIZES['title'].
        remove_spines (bool): If True, remove top and right spines. Default True.
    """
    # Use defaults if not specified
    if tick_label_fontsize is None:
        tick_label_fontsize = DEFAULT_FONTSIZES["tick_label"]
    if axis_label_fontsize is None:
        axis_label_fontsize = DEFAULT_FONTSIZES["axis_label"]
    if title_fontsize is None:
        title_fontsize = DEFAULT_FONTSIZES["title"]

    # Set tick label fontsize
    ax.tick_params(labelsize=tick_label_fontsize)

    # Set axis label font sizes (preserve existing labels)
    current_xlabel = ax.get_xlabel()
    if current_xlabel:
        ax.set_xlabel(current_xlabel, fontsize=axis_label_fontsize)

    current_ylabel = ax.get_ylabel()
    if current_ylabel:
        ax.set_ylabel(current_ylabel, fontsize=axis_label_fontsize)

    # Set title font size if title exists
    current_title = ax.get_title()
    if current_title:
        ax.set_title(current_title, fontsize=title_fontsize)

    # Remove top and right spines
    if remove_spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def configure_pdf_fonts():
    """
    Configure matplotlib settings for proper font embedding in PDF files.
    This ensures fonts are properly embedded and editable (e.g. in Illustrator).

    Sets:
    - pdf.fonttype to 42 (TrueType fonts, editable in Illustrator)
    - ps.fonttype to 42 (TrueType fonts for PostScript)
    - font.family to sans-serif with fallback fonts
    """
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType fonts (embedded)
    matplotlib.rcParams["ps.fonttype"] = 42  # TrueType fonts for PostScript
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "Helvetica",
        "sans-serif",
    ]


# Configure PDF fonts when module is imported
configure_pdf_fonts()


def save_figure_png_and_pdf(output_file):
    """
    Save figure as both PNG and PDF formats with proper font settings.

    Args:
        output_file (str or Path): Path to output file (with .png extension)
    """
    # Ensure PDF fonts are properly configured
    configure_pdf_fonts()

    # Convert Path to string if necessary
    output_file_str = str(output_file)

    # Save PNG
    plt.savefig(output_file_str, bbox_inches="tight", dpi=300)

    # Save PDF (replace .png with .pdf) with font embedding
    pdf_file = output_file_str.replace(".png", ".pdf")
    plt.savefig(
        pdf_file,
        bbox_inches="tight",
        dpi=300,
        format="pdf",
    )
    print(f"Plot saved to {output_file_str} and {pdf_file}")
