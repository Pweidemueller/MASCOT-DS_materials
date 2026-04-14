#!/usr/bin/env python3
"""
Shared utility functions for plotting scripts.
"""

import logging
import matplotlib.pyplot as plt
import matplotlib

from constants import COLORBLINDFR, COLORS, VARIANT_COLORS  # noqa: F401 — re-exported

logger = logging.getLogger(__name__)

# Default font sizes
DEFAULT_FONTSIZES = {
    "tick_label": 8,
    "axis_label": 10,
    "legend": 8,
    "title": 12,
}

# List form [title, axis_label, tick_label] for use with set_axis_fontsizes()
FONTSIZES_LIST = [
    DEFAULT_FONTSIZES["title"],
    DEFAULT_FONTSIZES["axis_label"],
    DEFAULT_FONTSIZES["tick_label"],
]


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
    logger.info("Plot saved to %s and %s", output_file_str, pdf_file)
