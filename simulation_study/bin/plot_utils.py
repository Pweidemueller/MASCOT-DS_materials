#!/usr/bin/env python3
"""
Shared utility functions for plotting scripts.
"""

import logging
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from constants import COLORBLINDFR, COLORS, VARIANT_COLORS  # noqa: F401 — re-exported

logger = logging.getLogger(__name__)

# Default font sizes
DEFAULT_FONTSIZES = {
    "tick_label": 8,
    "axis_label": 10,
    "legend": 8,
    "title": 10,
}

# List form [title, axis_label, tick_label] for use with set_axis_fontsizes()
FONTSIZES_LIST = [
    DEFAULT_FONTSIZES["title"],
    DEFAULT_FONTSIZES["axis_label"],
    DEFAULT_FONTSIZES["tick_label"],
]


# ---------------------------------------------------------------------------
# Trajectory utility functions (shared across simulate_datastreams, analyse_posteriors)
# ---------------------------------------------------------------------------


def get_outbreak_start_deme(df: pd.DataFrame) -> int:
    """Determine the deme where the outbreak starts from a trajectory DataFrame.

    The starting deme is the one where t == 0.0, population == "I", value == 1.0.

    Args:
        df: Trajectory DataFrame with columns including ``t``, ``population``,
            ``value``, and either ``index`` or ``Deme``.

    Returns:
        int: Deme index where the outbreak starts.

    Raises:
        ValueError: If the trajectory is empty, has no matching rows, or returns
            more than one candidate deme.
    """
    if df is None or df.empty:
        raise ValueError(
            "Trajectory data is empty or None; cannot determine outbreak start deme."
        )
    mask = (df["t"] == 0.0) & (df["population"] == "I") & (df["value"] == 1.0)
    candidates = df.loc[mask]
    if candidates.empty:
        raise ValueError(
            "No trajectory rows found with t == 0.0, population == 'I', and value == 1.0."
        )
    deme_col = "Deme" if "Deme" in candidates.columns else "index"
    if deme_col not in candidates.columns:
        raise ValueError("Trajectory data does not contain 'index' or 'Deme' column.")
    unique_demes = candidates[deme_col].dropna().unique()
    if len(unique_demes) != 1:
        raise ValueError(
            f"Expected exactly one starting deme, found {len(unique_demes)}: {unique_demes}."
        )
    start_deme = int(unique_demes[0])
    logger.info("Detected outbreak start deme: %s", start_deme)
    return start_deme


def t_first_infected_in_deme(traj_df: pd.DataFrame, deme: int) -> float:
    """Earliest time ``t`` where ``population == 'I'`` has >= 1 individual in the given deme.

    Args:
        traj_df: Trajectory DataFrame with columns ``population``, ``value``, ``t``,
            and either ``index`` or ``Deme``.
        deme: Deme index to query.

    Returns:
        float: Time (in years) of first infection in that deme.

    Raises:
        ValueError: If no infected rows exist for that deme.
    """
    col = "Deme" if "Deme" in traj_df.columns else "index"
    sub = traj_df[(traj_df["population"] == "I") & (traj_df[col] == deme)]
    sub = sub[sub["value"] >= 1].sort_values("t")
    if sub.empty:
        raise ValueError(f"No infected (I) trajectory rows for deme {deme}.")
    return float(sub["t"].min())


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
    # PDF embedding uses fontTools subsetter, which logs long glyph dumps at INFO.
    logging.getLogger("fontTools").setLevel(logging.WARNING)


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


def plot_data_csv_path(output_file, suffix: str = ""):
    """Companion CSV path for a plot: ``<stem>[_<suffix>]_data.csv`` next to the PNG.

    Args:
        output_file (str or Path): Plot path (typically ``.png``).
        suffix (str): Optional panel-specific suffix (e.g. ``"params_scatter"``)
            inserted before ``_data.csv`` when one figure has multiple data slices.
    """
    from pathlib import Path as _Path

    p = _Path(output_file)
    stem = p.stem
    sep = f"_{suffix}" if suffix else ""
    return p.with_name(f"{stem}{sep}_data.csv")


def save_plot_data_csv(df, output_file, *, suffix: str = "") -> None:
    """Write the DataFrame that was the input to a plot as a companion CSV.

    The file is written next to ``output_file`` with a ``_data.csv`` suffix
    (replacing the image extension). Use ``suffix`` to disambiguate multiple
    data slices sharing a single figure (e.g. panels of ``final_figure``).
    """
    csv_path = plot_data_csv_path(output_file, suffix=suffix)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Plot data saved to %s", csv_path)
