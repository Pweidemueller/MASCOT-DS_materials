#!/usr/bin/env python3
"""
Scatter plots of true parameter values vs posterior median, with HPD whiskers,
plus a bar plot of simulation-level 95% HPD coverage per parameter group.

Reads a validation CSV (e.g. all_params_*_hpd_validation.csv) and saves one figure
per logical parameter group. For scaling parameters (case counts and wastewater),
per-deme rows are saved as separate figures by outbreak role
(``caseCounts_scaling_start_deme.png``, ``caseCounts_scaling_secondary_deme.png``),
using the same start vs secondary assignment as prevalence panels
(``--sim_metadata_csv`` required for those splits).
Coverage for those groups still counts a simulation as covered only if every deme
row for that simulation has inHPD==1.

Optionally reads a migration-rates HPD validation CSV and saves (1) a two-panel
figure (bias and relative HPD width) with MASCOT vs MASCOT-DS boxplots per
direction, and (2) true-vs-median scatter plots (one file per direction).
Directions are relative to the outbreak start deme (from each simulation's
trajectory file), not fixed I0/I1 labels.

Optionally reads a prevalence HPD validation CSV and saves a two-row figure of
mean inHPD coverage vs per-simulation time index for start vs secondary deme,
after trajectories set first-infected times per deme; only time indices where
every simulation already has I in that deme are shown.

Optionally reads a combined Ne HPD validation CSV (MASCOT and MASCOT-DS with a
Model column) and saves Ne coverage vs time index (two deme panels, two colors),
plus mean bias (log Ne minus expected) and mean relative HPD width over time.

When the prevalence CSV includes HPD and expected columns, also saves prevalence
bias and relative HPD width over time (MASCOT-DS only; one line per panel).

When params, migration, and prevalence data are all available, assembles a
combined summary figure (14×10 in, 5×5 gridspec) showing per-group scatter,
coverage-over-time, and relative-bias panels. Per-simulation example prevalence
/ cumulative-incidence panels live in ``make_figure_individualsim.py``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import MODEL_MASCOT, MODEL_MASCOT_DS
from plot_utils import (
    COLORS,
    FONTSIZES_LIST,
    configure_pdf_fonts,
    save_figure_png_and_pdf,
    save_plot_data_csv,
)
from plot_utils import set_axis_fontsizes
from hpd_validation_timeseries import (
    MODEL_COLORS_DEFAULT,
    add_bias_and_hpd_width_columns,
    add_prevalence_bias_and_hpd_width_real_space,
    add_prevalence_deme_role_column,
    assert_matching_secondary_min_time_index,
    median_quantile_metric_by_time_index_role,
    median_quantile_metric_by_time_index_role_model,
    plot_ne_coverage_over_time,
    plot_prevalence_coverage_over_time,
    plot_two_panel_metric_multi_model,
    plot_two_panel_metric_single_series,
    prepare_validation_timeseries_df,
)

MIGRATION_PARAM_SUFFIX = re.compile(r"\.I(\d+)_to_I(\d+)$")

MIGRATION_DIRECTION_LABELS: tuple[str, ...] = (
    "start -> secondary",
    "secondary -> start",
)

# True-vs-inferred scatter: HPD whiskers slightly thicker than default dots; alpha on bars only
TRUE_VS_ESTIMATE_HPD_ELINEWIDTH = 0.72
TRUE_VS_ESTIMATE_HPD_ALPHA = 0.5

# Order and display labels for per-simulation parameter groups (top-to-bottom
# on horizontal summary panels; left-to-right on scatter panels).
PARAM_GROUP_ORDER: tuple[str, ...] = (
    "caseCounts.scaling",
    "caseCounts.dispersion",
    "wastewater.scaling",
    "wastewater.sigma",
)

PARAM_GROUP_TITLES: dict[str, str] = {
    "caseCounts.scaling": "CC scaling",
    "caseCounts.dispersion": "CC dispersion",
    "wastewater.scaling": "WW scaling",
    "wastewater.sigma": "WW sigma",
    # "seroprevalence.scaling": "SP scaling" — fixed to 1.0; not estimated
}

# Scaling parameters: true-vs-estimate scatters are one panel per outbreak role
# (start deme vs secondary deme; standalone files and nested subplots in plot_final_figure).
SCALING_PARAM_GROUPS: frozenset[str] = frozenset(
    {"caseCounts.scaling", "wastewater.scaling"}
)

# Order of nested panels / standalone files (matches prevalence rows).
SCALING_DEME_ROLE_ORDER: tuple[str, ...] = ("start", "secondary")

# Filename fragments for scaling outputs (see make_figure_individualsim._deme_label_for_filename).
SCALING_ROLE_FILENAME_STEM: dict[str, str] = {
    "start": "start_deme",
    "secondary": "secondary_deme",
}

# Short y-axis tick labels for the compact horizontal coverage / bias panels.
# Newlines keep labels vertically compact so they fit in narrow gridspec cells.
PARAM_GROUP_AXIS_LABELS: dict[str, str] = {
    "caseCounts.scaling": "CC\nscaling",
    "caseCounts.dispersion": "CC\ndispersion",
    "wastewater.scaling": "WW\nscaling",
    "wastewater.sigma": "WW\nsigma",
}

# Capitalised titles for the migration true-vs-estimate scatter panels.
MIGRATION_DIRECTION_TITLES: dict[str, str] = {
    "start -> secondary": "Start -> secondary",
    "secondary -> start": "Secondary -> start",
}

# Two-line y-axis labels for the migration coverage / bias summary panels.
MIGRATION_DIRECTION_AXIS_LABELS: dict[str, str] = {
    "start -> secondary": "start ->\nsecondary",
    "secondary -> start": "secondary ->\nstart",
}

# Parameter groups whose per-deme rows are collapsed into a single series
# (no legend, single color) in plot_final_figure. Remove an entry to restore
# per-deme styling for that group.
COLLAPSE_SERIES_GROUPS: frozenset[str] = frozenset(
    {
        "caseCounts.scaling",
        "wastewater.scaling",
    }
)

# Alpha for median-line+quantile-band time-series panels; mirrors hpd_validation_timeseries.
_TIME_SERIES_BAND_ALPHA = 0.22

# Grey reference band marking the acceptable coverage range (≈95% nominal) on
# coverage panels. Drawn as a horizontal span behind the data.
COVERAGE_REF_BAND: tuple[float, float] = (91.0, 99.0)
COVERAGE_REF_BAND_COLOR = "0.5"
COVERAGE_REF_BAND_ALPHA = 0.18


# ---------------------------------------------------------------------------
# Reusable axis-level helpers
# ---------------------------------------------------------------------------


# Figure-fraction offsets for panel labels, applied to the top-left corner of
# each axes' grid cell. Placing labels in figure coordinates (rather than axes
# coordinates) keeps them on an absolute grid: every label in a row shares the
# same y, and every label in a column shares the same x, regardless of how tall
# a panel is or how much room its title / y-axis label take up.
PANEL_LABEL_DX = 0.05
PANEL_LABEL_DY = 0.008


def add_panel_label(ax, label: str) -> None:
    """Draw a bold panel label above the top-left of an axes' grid cell.

    The label is positioned in figure coordinates from ``ax.get_position()`` so
    that labels align on an absolute grid (same y per row, same x per column)
    and always sit above and to the left of the axis title / labels. Call this
    before any later layout adjustment (e.g. ``tight_layout``).

    An empty ``label`` is a no-op, so callers can pass "" to skip a panel.
    """
    if not label:
        return
    fig = ax.get_figure()
    pos = ax.get_position()
    fig.text(
        pos.x0 - PANEL_LABEL_DX,
        pos.y1 + PANEL_LABEL_DY,
        label,
        fontsize=FONTSIZES_LIST[0] + 2,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def values_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    groups: list[str],
) -> dict[str, np.ndarray]:
    """Per-group numeric arrays dropped of NaN, keyed by ``groups`` order."""
    out: dict[str, np.ndarray] = {}
    for g in groups:
        sub = df[df[group_col] == g]
        out[g] = sub[value_col].dropna().to_numpy(dtype=float)
    return out


def coverage_percent_by_group(
    df: pd.DataFrame,
    group_col: str,
    groups: list[str],
) -> list[float]:
    """% of rows with ``inHPD == 1`` per group (NaN for empty groups)."""
    out: list[float] = []
    for g in groups:
        sub = df[df[group_col] == g]
        if sub.empty:
            out.append(float("nan"))
        else:
            out.append(100.0 * sub["inHPD"].mean())
    return out


def plot_horizontal_coverage_barplot(
    ax,
    labels: list[str],
    coverages: list[float],
    *,
    color: str = COLORS[3],
    xlabel: str = "Coverage (%)",
    show_ylabels: bool = True,
    xlim: tuple[float, float] = (0.0, 100.0),
) -> None:
    """Horizontal bars; ``labels[0]`` drawn at the top."""
    configure_pdf_fonts()
    y = np.arange(len(labels), dtype=float)
    ax.barh(y, coverages, color=color, edgecolor="white", linewidth=0.6, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if not show_ylabels:
        ax.tick_params(axis="y", labelleft=False)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=FONTSIZES_LIST[1])
    ax.tick_params(labelsize=FONTSIZES_LIST[2])
    # Light dashed gridlines along x-ticks so the reader can eyeball bar ends.
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_horizontal_bias_boxplot(
    ax,
    labels: list[str],
    values_by_label: dict[str, np.ndarray],
    *,
    color: str = COLORS[3],
    xlabel: str = "Bias",
    show_ylabels: bool = True,
    draw_zero_vline: bool = True,
    box_width: float = 0.5,
) -> None:
    """Horizontal boxplots (one per label); ``labels[0]`` drawn at the top."""
    configure_pdf_fonts()
    y = np.arange(len(labels), dtype=float)
    data_list = [
        values_by_label[lbl] if values_by_label[lbl].size > 0 else np.array([np.nan])
        for lbl in labels
    ]
    bp = ax.boxplot(
        data_list,
        positions=y,
        vert=False,
        patch_artist=True,
        widths=box_width,
        showfliers=True,
        whis=1.5,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor("0.25")
        patch.set_linewidth(0.6)
    for el in bp["medians"]:
        el.set_color("0.15")
        el.set_linewidth(0.9)
    for el in bp["whiskers"] + bp["caps"]:
        el.set_color("0.35")
        el.set_linewidth(0.6)
    if draw_zero_vline:
        ax.axvline(0.0, color="0.75", lw=0.7, ls="--", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    if not show_ylabels:
        ax.tick_params(axis="y", labelleft=False)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=FONTSIZES_LIST[1])
    ax.tick_params(labelsize=FONTSIZES_LIST[2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_vertical_coverage_barplot(
    ax,
    labels: list[str],
    coverages: list[float],
    *,
    color: str = COLORS[3],
    ylabel: str = "Coverage (%)",
    show_xlabels: bool = True,
    ylim: tuple[float, float] = (0.0, 100.0),
    xlabel_angle: float = 0.0,
    coverage_ref_band: tuple[float, float] | None = None,
) -> None:
    """Vertical bars with categorical x; intended to stack over a bias boxplot (shared x).

    ``coverage_ref_band`` (low, high) draws a grey horizontal span marking the
    acceptable coverage range behind the bars.
    """
    configure_pdf_fonts()
    x = np.arange(len(labels), dtype=float)
    if coverage_ref_band is not None:
        ax.axhspan(
            coverage_ref_band[0],
            coverage_ref_band[1],
            color=COVERAGE_REF_BAND_COLOR,
            alpha=COVERAGE_REF_BAND_ALPHA,
            linewidth=0,
            zorder=1,
        )
    ax.bar(x, coverages, color=color, edgecolor="none", linewidth=0, zorder=2)
    ax.set_xticks(x)
    if xlabel_angle > 0.0:
        ax.set_xticklabels(labels, rotation=xlabel_angle, ha="right")
    else:
        ax.set_xticklabels(labels, rotation=xlabel_angle)
    if not show_xlabels:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=FONTSIZES_LIST[1])
    ax.tick_params(labelsize=FONTSIZES_LIST[2])
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_vertical_bias_boxplot(
    ax,
    labels: list[str],
    values_by_label: dict[str, np.ndarray],
    *,
    color: str = COLORS[3],
    ylabel: str = "Bias",
    show_xlabels: bool = True,
    draw_zero_hline: bool = True,
    box_width: float = 0.5,
    xlabel_angle: float = 0.0,
) -> None:
    """Vertical boxplots (one per categorical x); intended to share x with a coverage bar above."""
    configure_pdf_fonts()
    x = np.arange(len(labels), dtype=float)
    data_list = [
        values_by_label[lbl] if values_by_label[lbl].size > 0 else np.array([np.nan])
        for lbl in labels
    ]
    bp = ax.boxplot(
        data_list,
        positions=x,
        vert=True,
        patch_artist=True,
        widths=box_width,
        showfliers=True,
        whis=1.5,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor("0.25")
        patch.set_linewidth(0.6)
    for el in bp["medians"]:
        el.set_color("0.15")
        el.set_linewidth(0.9)
    for el in bp["whiskers"] + bp["caps"]:
        el.set_color("0.35")
        el.set_linewidth(0.6)
    if draw_zero_hline:
        ax.axhline(0.0, color="0.75", lw=0.7, ls="--", zorder=0)
    ax.set_xticks(x)
    if xlabel_angle > 0.0:
        ax.set_xticklabels(labels, rotation=xlabel_angle, ha="right")
    else:
        ax.set_xticklabels(labels, rotation=xlabel_angle)
    if not show_xlabels:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel(ylabel, fontsize=FONTSIZES_LIST[1])
    ax.tick_params(labelsize=FONTSIZES_LIST[2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_time_series_coverage_on_axis(
    ax,
    agg: pd.DataFrame,
    role_key: str,
    *,
    color: str = COLORS[3],
    title: str = "",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    coverage_ref_band: tuple[float, float] | None = None,
) -> None:
    """Coverage(%) vs time_index for a single deme role.

    ``coverage_ref_band`` (low, high) draws a grey horizontal span marking the
    acceptable coverage range behind the line.
    """
    configure_pdf_fonts()
    sub = agg[agg["deme_role"] == role_key].sort_values("time_index")
    if coverage_ref_band is not None:
        ax.axhspan(
            coverage_ref_band[0],
            coverage_ref_band[1],
            color=COVERAGE_REF_BAND_COLOR,
            alpha=COVERAGE_REF_BAND_ALPHA,
            linewidth=0,
            zorder=1,
        )
    ax.plot(
        sub["time_index"],
        sub["coverage"] * 100.0,
        color=color,
        marker="o",
        ms=2.5,
        lw=0.9,
        zorder=2,
    )
    ax.set_ylim(0.0, 103.0)
    ax.set_yticks(np.linspace(0.0, 100.0, 6))
    ax.set_xlim(0.0, 103.0)
    if title:
        ax.set_title(title, fontsize=FONTSIZES_LIST[0])
    set_axis_fontsizes(
        ax,
        FONTSIZES_LIST,
        xlabel="Time index across outbreak" if show_xlabel else None,
        ylabel="Coverage (%) of\nprevalence trajectory" if show_ylabel else None,
    )
    # Dashed gridlines along y-ticks so the reader can eyeball point coverage.
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_time_series_metric_on_axis(
    ax,
    agg: pd.DataFrame,
    role_key: str,
    *,
    color: str = COLORS[3],
    ylabel: str = "",
    title: str = "",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    draw_zero_line: bool = False,
    band_alpha: float = _TIME_SERIES_BAND_ALPHA,
) -> None:
    """Median line + simulation-level quantile band vs time_index (single series)."""
    configure_pdf_fonts()
    sub = agg[agg["deme_role"] == role_key].sort_values("time_index")
    if not sub.empty:
        ax.fill_between(
            sub["time_index"],
            sub["q025"],
            sub["q975"],
            color=color,
            alpha=band_alpha,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            sub["time_index"],
            sub["median_value"],
            color=color,
            lw=0.9,
            zorder=2,
        )
    if draw_zero_line:
        ax.axhline(0.0, color="0.75", lw=0.7, ls="--", zorder=0)
    ax.set_xlim(0.0, 103.0)
    if title:
        ax.set_title(title, fontsize=FONTSIZES_LIST[0])
    set_axis_fontsizes(
        ax,
        FONTSIZES_LIST,
        xlabel="Time index across outbreak" if show_xlabel else None,
        ylabel=ylabel if show_ylabel else None,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def trajectory_stem_from_simulation_id(simulation: str) -> str:
    """
    Map pipeline ``Simulation`` id to the BEAST remaster trajectory basename.

    Example: ``22_2_simulation_datastreams_noclip`` -> ``22_2_simulation`` for
    ``22_2_simulation.traj`` under ``results/1_remaster_sim/``.
    """
    idx = simulation.find("_simulation")
    if idx == -1:
        raise ValueError(
            "Cannot derive trajectory stem from Simulation id "
            f"(missing '_simulation'): {simulation!r}"
        )
    return simulation[: idx + len("_simulation")]


def parse_migration_source_deme(parameter: str) -> int:
    """Deme index for (e.g. I0_to_I1 -> 0)."""
    m = MIGRATION_PARAM_SUFFIX.search(parameter)
    if not m:
        raise ValueError(
            f"Cannot parse migration deme pair from Parameter: {parameter!r}"
        )
    return int(m.group(1))


def migration_direction_label(parameter: str, outbreak_start_deme: int) -> str:
    """
    Label migration as start->secondary or secondary->start using outbreak start deme.

    The source deme of the rate (first index in I_src_to_I_dst) is compared to
    the deme where the outbreak begins (from trajectory).
    """
    src = parse_migration_source_deme(parameter)
    if src == outbreak_start_deme:
        return MIGRATION_DIRECTION_LABELS[0]
    return MIGRATION_DIRECTION_LABELS[1]


def load_trajectory_meta_from_csv(
    sim_metadata_csv: Path,
) -> dict[str, tuple[int, float, float]]:
    """Load per-simulation trajectory metadata from the consolidated CSV.

    The CSV has columns: ``simulation``, ``deme``, ``deme_type``, ``t_of_first_infect``,
    where ``simulation`` is the trajectory basename (e.g. ``1_2_simulation``).

    Returns:
        dict mapping trajectory stem -> ``(start_deme, t_first_I_start, t_first_I_secondary)``.
    """
    df = pd.read_csv(sim_metadata_csv)
    required = {"simulation", "deme", "deme_type", "t_of_first_infect"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sim metadata CSV missing columns: {missing}")
    out: dict[str, tuple[int, float, float]] = {}
    for stem, grp in df.groupby("simulation"):
        start_row = grp.loc[grp["deme_type"] == "start"]
        sec_row = grp.loc[grp["deme_type"] == "secondary"]
        if start_row.empty or sec_row.empty:
            raise ValueError(
                f"Simulation {stem!r}: expected one 'start' and one 'secondary' row."
            )
        out[str(stem)] = (
            int(start_row["deme"].iloc[0]),
            float(start_row["t_of_first_infect"].iloc[0]),
            float(sec_row["t_of_first_infect"].iloc[0]),
        )
    return out


def add_migration_direction_column(
    df: pd.DataFrame,
    starting_deme_by_sim: dict[str, int],
) -> pd.DataFrame:
    out = df.copy()
    if "Simulation" not in out.columns:
        raise ValueError("Migration dataframe must include a 'Simulation' column.")
    out["outbreak_start_deme"] = out["Simulation"].map(starting_deme_by_sim)
    if out["outbreak_start_deme"].isna().any():
        missing = out.loc[out["outbreak_start_deme"].isna(), "Simulation"].unique()
        raise ValueError(
            "Missing outbreak start deme for Simulation id(s) not in lookup: "
            f"{list(missing)}"
        )
    out["migration_direction"] = [
        migration_direction_label(p, int(d))
        for p, d in zip(out["Parameter"], out["outbreak_start_deme"])
    ]
    return out


def migration_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add bias (median - true) and relative HPD width ((upper-lower)/median)."""
    out = df.copy()
    out["bias"] = out["median"] - out["true_value"]
    out["relative_hpd_width"] = (out["hpd_upper"] - out["hpd_lower"]) / out["median"]
    return out


def plot_migration_bias_uncertainty_figure(
    df: pd.DataFrame,
    output_png: Path,
    starting_deme_by_sim: dict[str, int],
) -> None:
    """Two columns: bias and relative HPD width; shared y (migration direction)."""
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    df_plot = migration_derived_metrics(
        add_migration_direction_column(df, starting_deme_by_sim)
    )
    df_plot = df_plot[df_plot["Model"] == MODEL_MASCOT_DS]
    directions = list(MIGRATION_DIRECTION_LABELS)
    bias_by_dir = values_by_group(df_plot, "migration_direction", "bias", directions)
    width_by_dir = values_by_group(
        df_plot, "migration_direction", "relative_hpd_width", directions
    )
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 4.0), sharey=True)
    plot_horizontal_bias_boxplot(
        axes[0],
        directions,
        bias_by_dir,
        color=COLORS[3],
        xlabel="Migration rate\nbias",
        draw_zero_vline=True,
    )
    plot_horizontal_bias_boxplot(
        axes[1],
        directions,
        width_by_dir,
        color=COLORS[3],
        xlabel="Migration rate\nrel. HPD width",
        show_ylabels=False,
        draw_zero_vline=False,
    )
    fig.tight_layout()
    save_figure_png_and_pdf(output_png)
    save_plot_data_csv(df_plot, output_png)
    plt.close(fig)


def write_migration_csv_with_direction(
    df: pd.DataFrame,
    output_csv: Path,
    starting_deme_by_sim: dict[str, int],
) -> None:
    enriched = migration_derived_metrics(
        add_migration_direction_column(df, starting_deme_by_sim)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_csv, index=False)
    print(f"Wrote migration table with direction column to {output_csv}")


def parse_parameter(parameter: str) -> tuple[str, str]:
    """
    Return (group_id, series_label).

    caseCounts.scaling.Deme1:SimDataset -> ('caseCounts.scaling', 'Deme1')
    caseCounts.dispersion:SimDataset -> ('caseCounts.dispersion', '')
    """
    if ":SimDataset" in parameter:
        core = parameter.replace(":SimDataset", "")
    else:
        core = parameter
    m = re.match(r"^(.+)\.(Deme\d+)$", core)
    if m:
        return m.group(1), m.group(2)
    return core, ""


def add_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["Parameter"].map(parse_parameter)
    out["group_id"] = [p[0] for p in parsed]
    out["series_label"] = [p[1] for p in parsed]
    return out


def parse_deme_index_from_scaling_series_label(series_label: str) -> int | None:
    """Map ``Deme1`` / ``Deme2`` style suffix to 0-based deme index, else ``None``.

    Parameter names use 1-based deme labels (``Deme1``, ``Deme2``) while the
    trajectory metadata uses 0-based indices (0, 1), so the parsed number is
    decremented by one.
    """
    m = re.match(r"^Deme(\d+)$", str(series_label).strip())
    return int(m.group(1)) - 1 if m else None


def dataframe_has_scaling_deme_series_rows(df: pd.DataFrame) -> bool:
    """True if *df* has scaling-group rows with ``series_label`` like ``Deme1``."""
    mask = df["group_id"].isin(SCALING_PARAM_GROUPS)
    if not mask.any():
        return False
    for sl in df.loc[mask, "series_label"].fillna("").astype(str).unique():
        if parse_deme_index_from_scaling_series_label(sl) is not None:
            return True
    return False


def scaling_frame_has_deme_series_labels(g: pd.DataFrame) -> bool:
    """True if this parameter-group frame uses ``DemeN`` ``series_label`` suffixes."""
    for sl in g["series_label"].fillna("").astype(str).unique():
        if parse_deme_index_from_scaling_series_label(sl) is not None:
            return True
    return False


def scaling_rows_for_deme_role(
    g: pd.DataFrame,
    starting_deme_by_sim: dict[str, int],
    role: str,
) -> pd.DataFrame:
    """
    Rows where ``series_label`` is ``DemeN`` and *N* is the outbreak start deme
    (``role`` ``\"start\"``) or the other deme (``\"secondary\"``), per ``Simulation``.
    """
    out = g.copy()
    deme_nums = out["series_label"].map(parse_deme_index_from_scaling_series_label)
    valid = deme_nums.notna()
    out = out.loc[valid].copy()
    deme_nums = deme_nums.loc[valid]
    sim_str = out["Simulation"].astype(str)
    start_demes = sim_str.map(starting_deme_by_sim)
    if start_demes.isna().any():
        missing = out.loc[start_demes.isna(), "Simulation"].unique()
        raise ValueError(
            "Missing outbreak start deme for Simulation id(s) not in lookup: "
            f"{list(missing)}"
        )
    start_demes = start_demes.astype(int)
    row_roles = np.where(
        deme_nums.to_numpy(dtype=int) == start_demes.to_numpy(dtype=int),
        "start",
        "secondary",
    )
    out = out.assign(_scaling_deme_role=row_roles)
    return out[out["_scaling_deme_role"] == role].drop(columns="_scaling_deme_role")


def group_id_to_filename_stem(group_id: str) -> str:
    """Filesystem-safe stem, e.g. caseCounts.scaling -> caseCounts_scaling."""
    return group_id.replace(".", "_")


def simulation_hpd_coverage_percent(subdf: pd.DataFrame) -> float:
    """
    Percentage of simulations where the true value lies in the 95% HPD for all
    rows in this parameter group (e.g. both demes for scaling parameters).
    """
    coverage = subdf["inHPD"].mean()
    return 100.0 * coverage


def plot_hpd_coverage_barplot(
    ax,
    group_ids: list[str],
    percentages: list[float],
    label_for_group: dict[str, str],
) -> None:
    """Bar plot: parameter (x) vs % simulations with true value in 95% HPD (y)."""
    configure_pdf_fonts()
    labels = [label_for_group.get(gid, gid) for gid in group_ids]
    x = np.arange(len(group_ids), dtype=float)
    ax.bar(
        x,
        percentages,
        color=COLORS[3],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0.0, 100.0)
    set_axis_fontsizes(
        ax,
        FONTSIZES_LIST,
        xlabel=None,
        ylabel="Coverage",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_param_true_vs_estimate(
    ax,
    subdf: pd.DataFrame,
    title: str,
    *,
    series_colors: dict[str, str] | None = None,
    show_legend: bool = False,
) -> None:
    """
    x: true_value; y: median with vertical error bars from hpd_lower to hpd_upper.

    subdf must include columns true_value, median, hpd_lower, hpd_upper, series_label.
    Multiple non-empty series_label values are drawn as separate series (e.g. demes).

    If ``series_colors`` is set, keys are series_label values and values are colors
    (e.g. MASCOT vs MASCOT-DS for migration). Otherwise all series use COLORS[3].
    """
    configure_pdf_fonts()
    series_labels = subdf["series_label"].fillna("")
    unique_labels = sorted(s for s in series_labels.unique() if s != "")
    if not unique_labels:
        unique_labels = [""]

    for slabel in unique_labels:
        if slabel == "":
            part = subdf
            label = "posterior"
        else:
            part = subdf[series_labels == slabel]
            label = slabel
        if part.empty:
            continue
        x = part["true_value"].to_numpy(dtype=float)
        y = part["median"].to_numpy(dtype=float)
        lo = part["hpd_lower"].to_numpy(dtype=float)
        hi = part["hpd_upper"].to_numpy(dtype=float)
        yerr_lo = np.maximum(0.0, y - lo)
        yerr_hi = np.maximum(0.0, hi - y)
        if series_colors is None:
            color = COLORS[3]
        else:
            color = series_colors.get(slabel, COLORS[3])
        eb = ax.errorbar(
            x,
            y,
            yerr=np.vstack([yerr_lo, yerr_hi]),
            fmt="o",
            ms=1,
            capsize=0.0,
            elinewidth=TRUE_VS_ESTIMATE_HPD_ELINEWIDTH,
            color=color,
            markerfacecolor=color,
            markeredgecolor=None,
            label=label if len(unique_labels) > 1 or slabel != "" else None,
            linestyle="none",
        )
        # Alpha on HPD segments only (markers stay fully opaque)
        barlines = eb[2]
        if barlines is not None:
            for bar_line in barlines:
                bar_line.set_alpha(TRUE_VS_ESTIMATE_HPD_ALPHA)

    lims = [
        np.nanmin(
            np.concatenate(
                [
                    subdf["true_value"].to_numpy(dtype=float),
                    subdf["median"].to_numpy(dtype=float),
                    subdf["hpd_lower"].to_numpy(dtype=float),
                    subdf["hpd_upper"].to_numpy(dtype=float),
                ]
            )
        ),
        np.nanmax(
            np.concatenate(
                [
                    subdf["true_value"].to_numpy(dtype=float),
                    subdf["median"].to_numpy(dtype=float),
                    subdf["hpd_lower"].to_numpy(dtype=float),
                    subdf["hpd_upper"].to_numpy(dtype=float),
                ]
            )
        ),
    ]
    lims_x = [
        subdf["true_value"].to_numpy(dtype=float).min(),
        subdf["true_value"].to_numpy(dtype=float).max(),
    ]
    lims_y = [
        subdf["median"].to_numpy(dtype=float).min(),
        subdf["median"].to_numpy(dtype=float).max(),
    ]
    if not np.all(np.isfinite(lims)):
        return
    pad = 0.05 * (lims[1] - lims[0]) if lims[1] > lims[0] else 1.0
    lo_lim = lims[0] - pad
    hi_lim = lims[1] + pad
    ax.plot(
        [lims_x[0], lims_x[1]],
        [lims_x[0], lims_x[1]],
        color="0.5",
        lw=0.8,
        ls="--",
        zorder=0,
    )

    # ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color="0.5", lw=0.8, ls="--", zorder=0)
    # ax.set_xlim(lo_lim, hi_lim)
    # ax.set_ylim(lo_lim, hi_lim)
    # ax.set_aspect("equal", adjustable="box")

    ax.set_title(title, fontsize=FONTSIZES_LIST[0])
    set_axis_fontsizes(
        ax,
        FONTSIZES_LIST,
        xlabel="True value",
        ylabel="Inferred value",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend and (
        len(unique_labels) > 1 or (len(unique_labels) == 1 and unique_labels[0] != "")
    ):
        ax.legend(frameon=False, fontsize=FONTSIZES_LIST[2])


def migration_direction_filename_slug(direction: str) -> str:
    """Filesystem stem fragment, e.g. ``start -> secondary`` -> ``start_to_secondary``."""
    return direction.replace(" -> ", "_to_").replace(" ", "_")


def migration_frame_for_true_vs_estimate(
    df_mig: pd.DataFrame,
    starting_deme_by_sim: dict[str, int],
) -> pd.DataFrame:
    """
    Add ``migration_direction`` and ``series_label`` (from ``Model``) for scatter plots.
    """
    enriched = add_migration_direction_column(df_mig, starting_deme_by_sim)
    out = enriched.copy()
    out["series_label"] = out["Model"].astype(str)
    return out


def plot_migration_true_vs_estimate_scatters(
    df_mig: pd.DataFrame,
    output_dir: Path,
    starting_deme_by_sim: dict[str, int],
    *,
    filename_prefix: str = "migration_rates_true_vs_estimate",
) -> None:
    """
    One figure per migration direction (start vs secondary): true rate on x,
    posterior median on y with HPD whiskers; MASCOT vs MASCOT-DS colors.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = migration_frame_for_true_vs_estimate(df_mig, starting_deme_by_sim)
    frame = frame[frame["Model"] == MODEL_MASCOT_DS]
    model_colors = {
        MODEL_MASCOT: COLORS[4],
        MODEL_MASCOT_DS: COLORS[3],
    }
    for direction in MIGRATION_DIRECTION_LABELS:
        sub = frame[frame["migration_direction"] == direction]
        if sub.empty:
            continue
        slug = migration_direction_filename_slug(direction)
        out_png = output_dir / f"{filename_prefix}_{slug}.png"
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        plot_param_true_vs_estimate(
            ax,
            sub,
            title=f"Migration ({direction})",
            series_colors=model_colors,
            show_legend=True,
        )
        fig.tight_layout()
        save_figure_png_and_pdf(out_png)
        save_plot_data_csv(sub, out_png)
        plt.close(fig)


PREVALENCE_DEME_ROLE_TITLES: dict[str, str] = {
    "start": "Prevalence (start deme)",
    "secondary": "Prevalence (secondary deme)",
}


def plot_prevalence_true_vs_estimate_scatters(
    df_prev: pd.DataFrame,
    output_dir: Path,
    trajectory_meta: dict[str, tuple[int, float, float]],
    *,
    log_space: bool = False,
    filename_prefix: str = "prevalence_true_vs_estimate",
) -> None:
    """
    One scatter figure per deme role (start / secondary): true prevalence on x,
    posterior median on y, HPD whiskers. Rows where ``expectedlogPrev`` is NaN
    are dropped; no all-simulations-arrived filter is applied.

    When *log_space* is False (default) values are exponentiated to the natural
    prevalence scale; when True the log-space columns are used directly.
    """
    required = [
        "logPrevalence",
        "logPrevalence_hpd_lower",
        "logPrevalence_hpd_upper",
        "expectedlogPrev",
    ]
    missing = [c for c in required if c not in df_prev.columns]
    if missing:
        print(
            f"Skipping prevalence true-vs-estimate scatter: "
            f"CSV missing columns {missing}"
        )
        return

    starting_deme_by_sim = {s: m[0] for s, m in trajectory_meta.items()}
    df = add_prevalence_deme_role_column(df_prev, starting_deme_by_sim)
    df["expectedlogPrev"] = pd.to_numeric(df["expectedlogPrev"], errors="coerce")
    df = df.dropna(subset=["expectedlogPrev"])

    transform = (lambda x: x) if log_space else np.exp
    df["true_value"] = transform(df["expectedlogPrev"])
    df["median"] = transform(pd.to_numeric(df["logPrevalence"], errors="coerce"))
    df["hpd_lower"] = transform(
        pd.to_numeric(df["logPrevalence_hpd_lower"], errors="coerce")
    )
    df["hpd_upper"] = transform(
        pd.to_numeric(df["logPrevalence_hpd_upper"], errors="coerce")
    )
    df["series_label"] = ""

    scale_label = "log " if log_space else ""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for role_key, panel_title in PREVALENCE_DEME_ROLE_TITLES.items():
        title = f"{scale_label}{panel_title}"
        sub = df[df["deme_role"] == role_key]
        if sub.empty:
            continue
        suffix = "_log" if log_space else ""
        out_png = output_dir / f"{filename_prefix}_{role_key}{suffix}.png"
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        plot_param_true_vs_estimate(ax, sub, title=title)
        fig.tight_layout()
        save_figure_png_and_pdf(out_png)
        save_plot_data_csv(sub, out_png)
        plt.close(fig)


def run_time_series_hpd_validation_figures(
    *,
    combined_ne_csv: Path | None,
    df_ne_preloaded: pd.DataFrame | None,
    df_prev: pd.DataFrame | None,
    trajectory_meta: dict[str, tuple[int, float, float]],
    output_dir: Path,
) -> None:
    """
    Ne coverage (MASCOT + MASCOT-DS), Ne bias / relative HPD width on log scale
    (median across simulations with 2.5–97.5% quantile bands), and optionally
    prevalence bias / width when ``df_prev`` has the required columns.

    When both prevalence and combined Ne are present, checks that the minimum
    ``time_index`` in the secondary deme matches (MASCOT-DS vs prevalence).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    starting_deme_by_sim = {s: m[0] for s, m in trajectory_meta.items()}

    ne_path = combined_ne_csv
    df_ne_raw: pd.DataFrame | None = df_ne_preloaded
    if df_ne_raw is None and ne_path is not None and Path(ne_path).is_file():
        df_ne_raw = pd.read_csv(ne_path)

    df_prev_prepared: pd.DataFrame | None = None
    if df_prev is not None:
        df_prev_prepared = prepare_validation_timeseries_df(
            df_prev,
            trajectory_meta,
            starting_deme_by_sim,
            ne_reference_df=df_ne_raw,
        )

    df_ne_prepared: pd.DataFrame | None = None
    if df_ne_raw is not None:
        ne_need = [
            "Deme",
            "timesincestart",
            "inHPD",
            "Simulation",
            "Model",
            "logNe",
            "logNe_hpd_lower",
            "logNe_hpd_upper",
            "expectedlogNe",
        ]
        miss = [c for c in ne_need if c not in df_ne_raw.columns]
        if ne_path is not None:
            path_for_msg = ne_path
        else:
            path_for_msg = Path("<df_ne_preloaded>")
        if miss:
            raise ValueError(f"Combined Ne CSV missing columns {miss}: {path_for_msg}")
        df_ne_prepared = prepare_validation_timeseries_df(
            df_ne_raw, trajectory_meta, starting_deme_by_sim
        )

    if df_prev_prepared is not None and df_ne_prepared is not None:
        assert_matching_secondary_min_time_index(df_prev_prepared, df_ne_prepared)

    if df_ne_raw is not None and df_ne_prepared is not None:
        ne_cov_png = output_dir / "ne_coverage_over_time.png"
        plot_ne_coverage_over_time(
            df_ne_raw,
            ne_cov_png,
            trajectory_meta,
            models_colors=MODEL_COLORS_DEFAULT,
        )
        save_plot_data_csv(df_ne_prepared, ne_cov_png)
        df_ne_prepared = add_bias_and_hpd_width_columns(
            df_ne_prepared,
            expected_col="expectedlogNe",
            median_col="logNe",
            rel_hpd_width=False,
        )
        agg_ne_bias = median_quantile_metric_by_time_index_role_model(
            df_ne_prepared, "bias_logne"
        )
        df_ne_prepared.to_csv(
            output_dir / "ne_summary_logne_over_time.csv", index=False
        )
        ne_bias_png = output_dir / "ne_bias_logne_over_time.png"
        plot_two_panel_metric_multi_model(
            agg_ne_bias,
            ne_bias_png,
            ylabel="Bias (log Ne)",
            models_colors=MODEL_COLORS_DEFAULT,
            draw_zero_line=True,
        )
        save_plot_data_csv(agg_ne_bias, ne_bias_png)
        agg_ne_hpd_width = median_quantile_metric_by_time_index_role_model(
            df_ne_prepared, "hpd_width_logne"
        )
        ne_width_png = output_dir / "ne_hpd_width_logne_over_time.png"
        plot_two_panel_metric_multi_model(
            agg_ne_hpd_width,
            ne_width_png,
            ylabel="HPD width (log Ne)",
            models_colors=MODEL_COLORS_DEFAULT,
            draw_zero_line=False,
        )
        save_plot_data_csv(agg_ne_hpd_width, ne_width_png)

    if df_prev is not None:
        prev_bias_need = [
            "logPrevalence",
            "logPrevalence_hpd_lower",
            "logPrevalence_hpd_upper",
            "expectedlogPrev",
        ]
        miss_prev = [c for c in prev_bias_need if c not in df_prev.columns]
        if miss_prev:
            print(
                "Skipping prevalence bias / relative width figures: "
                f"CSV missing columns {miss_prev}"
            )
        else:
            df_prev_prepared_bias = prepare_validation_timeseries_df(
                df_prev,
                trajectory_meta,
                starting_deme_by_sim,
                ne_reference_df=df_ne_raw,
            )
            df_prev_prepared_bias = add_prevalence_bias_and_hpd_width_real_space(
                df_prev_prepared_bias
            )
            agg_p_bias = median_quantile_metric_by_time_index_role(
                df_prev_prepared_bias, "bias_prev_real"
            )
            prev_bias_png = output_dir / "prevalence_bias_prev_over_time.png"
            plot_two_panel_metric_single_series(
                agg_p_bias,
                prev_bias_png,
                ylabel="Bias of\nprevalence estimate",
                color=COLORS[3],
                draw_zero_line=True,
            )
            save_plot_data_csv(agg_p_bias, prev_bias_png)
            agg_p_rel = median_quantile_metric_by_time_index_role(
                df_prev_prepared_bias, "rel_hpd_width_prev_real"
            )
            prev_rel_png = output_dir / "prevalence_rel_hpd_width_prev_over_time.png"
            plot_two_panel_metric_single_series(
                agg_p_rel,
                prev_rel_png,
                ylabel="Relative HPD width of\nprevalence estimate",
                color=COLORS[3],
                draw_zero_line=False,
            )
            save_plot_data_csv(agg_p_rel, prev_rel_png)
            agg_p_bias_rel = median_quantile_metric_by_time_index_role(
                df_prev_prepared_bias, "bias_prev_real_rel"
            )
            prev_bias_rel_png = output_dir / "prevalence_bias_prev_rel_over_time.png"
            plot_two_panel_metric_single_series(
                agg_p_bias_rel,
                prev_bias_rel_png,
                ylabel="Relative bias of\nprevalence estimate",
                color=COLORS[3],
                draw_zero_line=True,
            )
            save_plot_data_csv(agg_p_bias_rel, prev_bias_rel_png)


PREV_ROLE_PANEL_TITLES: dict[str, str] = {
    "start": "Start deme",
    "secondary": "Secondary deme",
}


def _prepare_prevalence_scatter_frame(
    df_prev: pd.DataFrame,
    starting_deme_by_sim: dict[str, int],
    *,
    log_space: bool = True,
) -> pd.DataFrame:
    """Add ``deme_role`` and true/median/HPD columns (on log or natural scale)."""
    df = add_prevalence_deme_role_column(df_prev, starting_deme_by_sim)
    df["expectedlogPrev"] = pd.to_numeric(df["expectedlogPrev"], errors="coerce")
    df = df.dropna(subset=["expectedlogPrev"]).copy()
    transform = (lambda x: x) if log_space else np.exp
    df["true_value"] = transform(df["expectedlogPrev"])
    df["median"] = transform(pd.to_numeric(df["logPrevalence"], errors="coerce"))
    df["hpd_lower"] = transform(
        pd.to_numeric(df["logPrevalence_hpd_lower"], errors="coerce")
    )
    df["hpd_upper"] = transform(
        pd.to_numeric(df["logPrevalence_hpd_upper"], errors="coerce")
    )
    df["series_label"] = ""
    return df


def plot_final_figure(
    df_params: pd.DataFrame,
    df_mig: pd.DataFrame,
    df_prev: pd.DataFrame,
    trajectory_meta: dict[str, tuple[int, float, float]],
    output_png: Path,
    *,
    df_ne_reference: pd.DataFrame | None = None,
) -> None:
    """
    Assemble the combined summary figure via ``GridSpec`` (14×10 in, 5×5 grid).

    Layout (rows × cols, 0-indexed):

    * rows 0-1, col 0: prevalence true-vs-inferred scatter (start on row 0,
      secondary on row 1)
    * rows 0-1, cols 1-2: prevalence coverage(%) over time (start / secondary)
    * rows 0-1, cols 3-4: prevalence relative bias over time (start / secondary)
    * row 2, cols 0-1: migration scatter (start→secondary at col 0,
      secondary→start at col 1)
    * row 2, cols 2-5 (nested 1×2): migration coverage (left half, 1.5-col wide)
      and migration relative bias (right half, 1.5-col wide)
    * rows 3-4, cols 0-1: 2×2 parameter scatter (caseCounts row 3: scaling /
      dispersion; wastewater row 4: scaling / sigma)
    * row 3, cols 2-4: parameter coverage(%) bars (shared x with row 4)
    * row 4, cols 2-4: parameter relative bias boxplots

    ``df_params`` must already have ``group_id`` / ``series_label`` columns from
    :func:`add_group_columns`.
    """
    configure_pdf_fonts()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    starting_deme_by_sim = {s: m[0] for s, m in trajectory_meta.items()}

    # --- parameters: relative bias per row ((median - true) / true) -----
    df_params = df_params.copy()
    df_params["rel_bias"] = (df_params["median"] - df_params["true_value"]) / df_params[
        "true_value"
    ]

    # --- migration: restrict to MASCOT-DS and derive relative bias -----
    mig_enriched = add_migration_direction_column(df_mig, starting_deme_by_sim)
    mig_enriched = mig_enriched[mig_enriched["Model"] == MODEL_MASCOT_DS].copy()
    mig_enriched["rel_bias"] = (
        mig_enriched["median"] - mig_enriched["true_value"]
    ) / mig_enriched["true_value"]

    # --- prevalence: time-series aggregates and scatter frame ------------
    df_prev_prep = prepare_validation_timeseries_df(
        df_prev,
        trajectory_meta,
        starting_deme_by_sim,
        ne_reference_df=df_ne_reference,
    )
    prev_coverage_agg = (
        df_prev_prep.groupby(["time_index", "deme_role"], sort=True)
        .agg(coverage=("inHPD", "mean"))
        .reset_index()
    )
    df_prev_metrics = add_prevalence_bias_and_hpd_width_real_space(df_prev_prep)
    prev_rel_bias_agg = median_quantile_metric_by_time_index_role(
        df_prev_metrics, "bias_prev_real_rel"
    )
    prev_scatter_frame = _prepare_prevalence_scatter_frame(
        df_prev, starting_deme_by_sim, log_space=True
    )

    # --- figure & gridspec ----------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(nrows=4, ncols=10, hspace=0.9, wspace=1.3)

    # --- prevalence scatter (rows 0-1, col 0) ---------------------------
    for row_offset, role_key in enumerate(("start", "secondary")):
        ax = fig.add_subplot(gs[row_offset, 0:2])
        sub = prev_scatter_frame[prev_scatter_frame["deme_role"] == role_key]
        plot_param_true_vs_estimate(
            ax,
            sub,
            title=PREV_ROLE_PANEL_TITLES[role_key],
        )

    # --- prevalence coverage over time (rows 0-1, cols 1-2) -------------
    ax = fig.add_subplot(gs[0, 2:6])
    plot_time_series_coverage_on_axis(
        ax,
        prev_coverage_agg,
        "start",
        title=PREV_ROLE_PANEL_TITLES["start"],
        show_xlabel="start",
        coverage_ref_band=COVERAGE_REF_BAND,
    )
    ax = fig.add_subplot(gs[1, 2:6])
    plot_time_series_coverage_on_axis(
        ax,
        prev_coverage_agg,
        "secondary",
        title=PREV_ROLE_PANEL_TITLES["secondary"],
        show_xlabel="secondary",
        coverage_ref_band=COVERAGE_REF_BAND,
    )

    ax = fig.add_subplot(gs[0, 6:10])
    plot_time_series_metric_on_axis(
        ax,
        prev_rel_bias_agg,
        "start",
        ylabel="Relative bias",
        title=PREV_ROLE_PANEL_TITLES["start"],
        show_xlabel="start",
        draw_zero_line=True,
    )
    ax = fig.add_subplot(gs[1, 6:10])
    plot_time_series_metric_on_axis(
        ax,
        prev_rel_bias_agg,
        "secondary",
        ylabel="Relative bias",
        title=PREV_ROLE_PANEL_TITLES["secondary"],
        show_xlabel="secondary",
        draw_zero_line=True,
    )

    # --- migration scatter (row 2, cols 0-1) ----------------------------
    for col_idx, direction in enumerate(MIGRATION_DIRECTION_LABELS):
        ax = fig.add_subplot(gs[2 + col_idx, 0:2])
        sub = mig_enriched[mig_enriched["migration_direction"] == direction].copy()
        sub["series_label"] = ""
        plot_param_true_vs_estimate(
            ax, sub, title=MIGRATION_DIRECTION_TITLES[direction]
        )

    # --- migration coverage (1.5 width) + rel bias (1.5 width)
    #     nested 1×2 inside row 2, cols 2-4 -----------------------------
    mig_dirs = list(MIGRATION_DIRECTION_LABELS)
    mig_axis_labels = [MIGRATION_DIRECTION_AXIS_LABELS[d] for d in mig_dirs]
    # mig_inner = gs[2, 2:].subgridspec(1, 2, wspace=0.6)
    # ax_m_cov = fig.add_subplot(mig_inner[0, 0])
    ax_m_cov = fig.add_subplot(gs[2, 2:4])
    mig_coverage = coverage_percent_by_group(
        mig_enriched, "migration_direction", mig_dirs
    )
    plot_vertical_coverage_barplot(
        ax_m_cov,
        mig_axis_labels,
        mig_coverage,
        ylabel="Coverage (%)",
        coverage_ref_band=COVERAGE_REF_BAND,
    )
    # ax_m_bias = fig.add_subplot(mig_inner[0, 1])
    ax_m_bias = fig.add_subplot(gs[3, 2:4])
    rel_bias_by_dir = values_by_group(
        mig_enriched, "migration_direction", "rel_bias", mig_dirs
    )
    plot_vertical_bias_boxplot(
        ax_m_bias,
        mig_axis_labels,
        {mig_axis_labels[i]: rel_bias_by_dir[d] for i, d in enumerate(mig_dirs)},
        ylabel="Relative bias",
        show_xlabels=True,
    )

    # --- parameter scatter plots (2×2, rows 3-4, cols 0-1) --------------
    param_cells = {
        "caseCounts.scaling": (2, 4),
        "caseCounts.dispersion": (2, 6),
        "wastewater.scaling": (3, 4),
        "wastewater.sigma": (3, 6),
    }
    for gid, (r, c) in param_cells.items():
        ax = fig.add_subplot(gs[r, c : c + 2])
        sub = df_params[df_params["group_id"] == gid].copy()
        if gid in COLLAPSE_SERIES_GROUPS:
            sub["series_label"] = ""
        plot_param_true_vs_estimate(
            ax,
            sub,
            title=PARAM_GROUP_TITLES.get(gid, gid),
            show_legend=False,
        )

    # --- parameter vertical coverage (row 3) + bias (row 4), cols 2-4 ---
    param_groups = list(PARAM_GROUP_ORDER)
    param_axis_labels = [PARAM_GROUP_AXIS_LABELS[g] for g in param_groups]
    ax_p_cov = fig.add_subplot(gs[2, 8:])
    plot_vertical_coverage_barplot(
        ax_p_cov,
        param_axis_labels,
        coverage_percent_by_group(df_params, "group_id", param_groups),
        ylabel="Coverage (%)",
        show_xlabels=True,
        xlabel_angle=45.0,
        coverage_ref_band=COVERAGE_REF_BAND,
    )
    ax_p_bias = fig.add_subplot(gs[3, 8:])
    plot_vertical_bias_boxplot(
        ax_p_bias,
        param_axis_labels,
        {
            lbl: df_params[df_params["group_id"] == gid]["rel_bias"]
            .dropna()
            .to_numpy(dtype=float)
            for gid, lbl in zip(param_groups, param_axis_labels)
        },
        ylabel="Relative bias",
        show_xlabels=True,
        xlabel_angle=45.0,
    )

    save_figure_png_and_pdf(output_png)
    # Per-panel data CSVs. Different panels have incompatible schemas, so emit
    # one CSV per logical data slice rather than a single merged file.
    save_plot_data_csv(df_params, output_png, suffix="params")
    save_plot_data_csv(mig_enriched, output_png, suffix="migration")
    save_plot_data_csv(prev_scatter_frame, output_png, suffix="prevalence_scatter")
    save_plot_data_csv(prev_coverage_agg, output_png, suffix="prevalence_coverage")
    save_plot_data_csv(prev_rel_bias_agg, output_png, suffix="prevalence_rel_bias")
    plt.close(fig)


def plot_final_figure_reduced(
    df_params: pd.DataFrame,
    df_prev: pd.DataFrame,
    trajectory_meta: dict[str, tuple[int, float, float]],
    output_png: Path,
    *,
    df_ne_reference: pd.DataFrame | None = None,
) -> None:
    """
    Main reduced figure via ``GridSpec`` (4×3): start-deme prevalence plus the
    four datastream parameter groups.

    Layout (rows × cols, 0-indexed):

    * rows 0-1, col 0: start-deme prevalence true-vs-inferred scatter
    * row 0, cols 1-2: start-deme prevalence coverage(%) over time
    * row 1, cols 1-2: start-deme prevalence relative bias over time
    * row 2, col 0 / col 1: CC scaling / CC dispersion scatter
    * row 3, col 0 / col 1: WW scaling / WW sigma scatter
    * row 2, col 2: datastream parameter coverage(%) bars
    * row 3, col 2: datastream parameter relative-bias boxplots

    ``df_params`` must already have ``group_id`` / ``series_label`` columns from
    :func:`add_group_columns`.
    """
    configure_pdf_fonts()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    starting_deme_by_sim = {s: m[0] for s, m in trajectory_meta.items()}

    # --- parameters: relative bias per row ((median - true) / true) -----
    df_params = df_params.copy()
    df_params["rel_bias"] = (df_params["median"] - df_params["true_value"]) / df_params[
        "true_value"
    ]

    # --- prevalence: time-series aggregates and scatter frame ------------
    df_prev_prep = prepare_validation_timeseries_df(
        df_prev,
        trajectory_meta,
        starting_deme_by_sim,
        ne_reference_df=df_ne_reference,
    )
    prev_coverage_agg = (
        df_prev_prep.groupby(["time_index", "deme_role"], sort=True)
        .agg(coverage=("inHPD", "mean"))
        .reset_index()
    )
    df_prev_metrics = add_prevalence_bias_and_hpd_width_real_space(df_prev_prep)
    prev_rel_bias_agg = median_quantile_metric_by_time_index_role(
        df_prev_metrics, "bias_prev_real_rel"
    )
    prev_scatter_frame = _prepare_prevalence_scatter_frame(
        df_prev, starting_deme_by_sim, log_space=True
    )

    # --- figure & gridspec ----------------------------------------------
    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(nrows=4, ncols=3, hspace=0.9, wspace=0.45)

    # --- start-deme prevalence scatter (rows 0-1, col 0) ----------------
    ax = fig.add_subplot(gs[0:2, 0])
    sub = prev_scatter_frame[prev_scatter_frame["deme_role"] == "start"]
    plot_param_true_vs_estimate(ax, sub, title=PREV_ROLE_PANEL_TITLES["start"])
    add_panel_label(ax, "A")

    # --- start-deme prevalence coverage / relative bias (rows 0-1, cols 1-2)
    ax = fig.add_subplot(gs[0, 1:3])
    plot_time_series_coverage_on_axis(
        ax,
        prev_coverage_agg,
        "start",
        title="Start deme",
        coverage_ref_band=COVERAGE_REF_BAND,
    )
    add_panel_label(ax, "B")
    ax = fig.add_subplot(gs[1, 1:3])
    plot_time_series_metric_on_axis(
        ax,
        prev_rel_bias_agg,
        "start",
        ylabel="Relative bias of\nprevalence estimate",
        title="Start deme",
        draw_zero_line=True,
    )
    add_panel_label(ax, "C")

    # --- parameter scatters (rows 2-3, cols 0-1) ------------------------
    # Panel labels follow row-by-row reading order, interleaving with the
    # coverage (F) / bias (I) panels in col 2.
    param_cells = {
        "caseCounts.scaling": (2, 0, "D"),
        "caseCounts.dispersion": (2, 1, ""),
        "wastewater.scaling": (3, 0, ""),
        "wastewater.sigma": (3, 1, ""),
    }
    for gid, (r, c, panel_label) in param_cells.items():
        ax = fig.add_subplot(gs[r, c])
        sub = df_params[df_params["group_id"] == gid].copy()
        if gid in COLLAPSE_SERIES_GROUPS:
            sub["series_label"] = ""
        plot_param_true_vs_estimate(
            ax,
            sub,
            title=PARAM_GROUP_TITLES.get(gid, gid),
            show_legend=False,
        )
        if panel_label == "D":
            add_panel_label(ax, panel_label)

    # --- parameter coverage (row 2, col 2) + bias (row 3, col 2) --------
    param_groups = list(PARAM_GROUP_ORDER)
    param_axis_labels = [PARAM_GROUP_AXIS_LABELS[g] for g in param_groups]
    ax_p_cov = fig.add_subplot(gs[2, 2])
    plot_vertical_coverage_barplot(
        ax_p_cov,
        param_axis_labels,
        coverage_percent_by_group(df_params, "group_id", param_groups),
        ylabel="Coverage (%)",
        show_xlabels=True,
        xlabel_angle=45.0,
        coverage_ref_band=COVERAGE_REF_BAND,
    )
    add_panel_label(ax_p_cov, "E")
    ax_p_bias = fig.add_subplot(gs[3, 2])
    plot_vertical_bias_boxplot(
        ax_p_bias,
        param_axis_labels,
        {
            lbl: df_params[df_params["group_id"] == gid]["rel_bias"]
            .dropna()
            .to_numpy(dtype=float)
            for gid, lbl in zip(param_groups, param_axis_labels)
        },
        ylabel="Relative bias",
        show_xlabels=True,
        xlabel_angle=45.0,
    )
    add_panel_label(ax_p_bias, "F")

    save_figure_png_and_pdf(output_png)
    # Per-panel data CSVs. Different panels have incompatible schemas, so emit
    # one CSV per logical data slice rather than a single merged file.
    save_plot_data_csv(df_params, output_png, suffix="params")
    save_plot_data_csv(prev_scatter_frame, output_png, suffix="prevalence_scatter")
    save_plot_data_csv(prev_coverage_agg, output_png, suffix="prevalence_coverage")
    save_plot_data_csv(prev_rel_bias_agg, output_png, suffix="prevalence_rel_bias")
    plt.close(fig)


def plot_supplementary_figure(
    df_mig: pd.DataFrame,
    df_prev: pd.DataFrame,
    trajectory_meta: dict[str, tuple[int, float, float]],
    output_png: Path,
    *,
    df_ne_reference: pd.DataFrame | None = None,
) -> None:
    """
    Supplementary figure via ``GridSpec`` (4×3): secondary-deme prevalence plus
    migration rates.

    Layout (rows × cols, 0-indexed):

    * rows 0-1, col 0: secondary-deme prevalence true-vs-inferred scatter
    * row 0, cols 1-2: secondary-deme prevalence coverage(%) over time
    * row 1, cols 1-2: secondary-deme prevalence relative bias over time
    * row 2, cols 0-1: migration start->secondary true-vs-estimate scatter
    * row 3, cols 0-1: migration secondary->start true-vs-estimate scatter
    * row 2, col 2: migration coverage(%) bars
    * row 3, col 2: migration relative-bias boxplots
    """
    configure_pdf_fonts()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    starting_deme_by_sim = {s: m[0] for s, m in trajectory_meta.items()}

    # --- migration: restrict to MASCOT-DS and derive relative bias -----
    mig_enriched = add_migration_direction_column(df_mig, starting_deme_by_sim)
    mig_enriched = mig_enriched[mig_enriched["Model"] == MODEL_MASCOT_DS].copy()
    mig_enriched["rel_bias"] = (
        mig_enriched["median"] - mig_enriched["true_value"]
    ) / mig_enriched["true_value"]

    # --- prevalence: time-series aggregates and scatter frame ------------
    df_prev_prep = prepare_validation_timeseries_df(
        df_prev,
        trajectory_meta,
        starting_deme_by_sim,
        ne_reference_df=df_ne_reference,
    )
    prev_coverage_agg = (
        df_prev_prep.groupby(["time_index", "deme_role"], sort=True)
        .agg(coverage=("inHPD", "mean"))
        .reset_index()
    )
    df_prev_metrics = add_prevalence_bias_and_hpd_width_real_space(df_prev_prep)
    prev_rel_bias_agg = median_quantile_metric_by_time_index_role(
        df_prev_metrics, "bias_prev_real_rel"
    )
    prev_scatter_frame = _prepare_prevalence_scatter_frame(
        df_prev, starting_deme_by_sim, log_space=True
    )

    # --- figure & gridspec ----------------------------------------------
    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(nrows=4, ncols=3, hspace=0.9, wspace=0.45)

    # --- secondary-deme prevalence scatter (rows 0-1, col 0) ------------
    ax = fig.add_subplot(gs[0:2, 0])
    sub = prev_scatter_frame[prev_scatter_frame["deme_role"] == "secondary"]
    plot_param_true_vs_estimate(ax, sub, title=PREV_ROLE_PANEL_TITLES["secondary"])
    add_panel_label(ax, "A")

    # --- secondary-deme prevalence coverage / relative bias (rows 0-1, cols 1-2)
    ax = fig.add_subplot(gs[0, 1:3])
    plot_time_series_coverage_on_axis(
        ax,
        prev_coverage_agg,
        "secondary",
        title="Secondary deme",
        coverage_ref_band=COVERAGE_REF_BAND,
    )
    add_panel_label(ax, "B")
    ax = fig.add_subplot(gs[1, 1:3])
    plot_time_series_metric_on_axis(
        ax,
        prev_rel_bias_agg,
        "secondary",
        ylabel="Secondary deme",
        title="Relative bias of\nprevalence estimate",
        draw_zero_line=True,
    )
    add_panel_label(ax, "C")

    # --- migration scatters (rows 2-3, cols 0-1) ------------------------
    # Panel labels follow row-by-row reading order, interleaving with the
    # coverage (E) / bias (G) panels in col 2.
    mig_scatter_labels = ("D", "")
    for row_idx, direction in enumerate(MIGRATION_DIRECTION_LABELS):
        ax = fig.add_subplot(gs[2 + row_idx, 0:2])
        sub = mig_enriched[mig_enriched["migration_direction"] == direction].copy()
        sub["series_label"] = ""
        plot_param_true_vs_estimate(
            ax, sub, title=MIGRATION_DIRECTION_TITLES[direction]
        )
        add_panel_label(ax, mig_scatter_labels[row_idx])

    # --- migration coverage (row 2, col 2) + bias (row 3, col 2) --------
    mig_dirs = list(MIGRATION_DIRECTION_LABELS)
    mig_axis_labels = [MIGRATION_DIRECTION_AXIS_LABELS[d] for d in mig_dirs]
    ax_m_cov = fig.add_subplot(gs[2, 2])
    plot_vertical_coverage_barplot(
        ax_m_cov,
        mig_axis_labels,
        coverage_percent_by_group(mig_enriched, "migration_direction", mig_dirs),
        ylabel="Coverage (%)",
        show_xlabels=True,
        xlabel_angle=45.0,
        coverage_ref_band=COVERAGE_REF_BAND,
    )
    add_panel_label(ax_m_cov, "E")
    ax_m_bias = fig.add_subplot(gs[3, 2])
    rel_bias_by_dir = values_by_group(
        mig_enriched, "migration_direction", "rel_bias", mig_dirs
    )
    plot_vertical_bias_boxplot(
        ax_m_bias,
        mig_axis_labels,
        {mig_axis_labels[i]: rel_bias_by_dir[d] for i, d in enumerate(mig_dirs)},
        ylabel="Relative bias",
        show_xlabels=True,
        xlabel_angle=45.0,
    )
    add_panel_label(ax_m_bias, "F")

    save_figure_png_and_pdf(output_png)
    # Per-panel data CSVs. Different panels have incompatible schemas, so emit
    # one CSV per logical data slice rather than a single merged file.
    save_plot_data_csv(mig_enriched, output_png, suffix="migration")
    save_plot_data_csv(prev_scatter_frame, output_png, suffix="prevalence_scatter")
    save_plot_data_csv(prev_coverage_agg, output_png, suffix="prevalence_coverage")
    save_plot_data_csv(prev_rel_bias_agg, output_png, suffix="prevalence_rel_bias")
    plt.close(fig)


def plot_parameter_relative_bias_figure(
    df_params: pd.DataFrame, output_png: Path
) -> None:
    """Standalone vertical boxplot for relative bias across parameter groups."""
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    df_plot = df_params.copy()
    df_plot["rel_bias"] = (df_plot["median"] - df_plot["true_value"]) / df_plot[
        "true_value"
    ]
    param_groups = list(PARAM_GROUP_ORDER)
    param_axis_labels = [PARAM_GROUP_AXIS_LABELS[g] for g in param_groups]
    rel_bias_by_group = {
        lbl: df_plot[df_plot["group_id"] == gid]["rel_bias"]
        .dropna()
        .to_numpy(dtype=float)
        for gid, lbl in zip(param_groups, param_axis_labels)
    }

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    plot_vertical_bias_boxplot(
        ax,
        param_axis_labels,
        rel_bias_by_group,
        ylabel="Relative bias",
        show_xlabels=True,
    )
    fig.tight_layout()
    save_figure_png_and_pdf(output_png)
    save_plot_data_csv(df_plot, output_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="True vs estimated parameter figures with HPD error bars."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "results/3_analysis/all_params_datastreams_noclip_hpd_validation.csv"
        ),
        help="Path to HPD validation CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("figures/figure1/ds_param_true_vs_estimate"),
        help="Directory for PNG/PDF outputs (created if missing). "
        "For local test runs, use a path under sandbox/ rather than results/.",
    )
    parser.add_argument(
        "--migration_rates_csv",
        type=Path,
        default=Path(
            "results/3_analysis/all_migration_rates_datastreams_noclip_hpd_validation.csv"
        ),
        help="Migration rates HPD validation CSV (bias / relative uncertainty figure).",
    )
    parser.add_argument(
        "--migration_figure_out",
        type=Path,
        default=None,
        help="Output PNG stem for migration figure (default: output_dir/migration_rates_bias_uncertainty.png).",
    )
    parser.add_argument(
        "--migration_enriched_csv",
        type=Path,
        default=None,
        help="Write enriched CSV with outbreak_start_deme, migration_direction, bias, "
        "relative_hpd_width (default: output_dir/migration_rates_hpd_validation_with_direction.csv).",
    )
    parser.add_argument(
        "--sim_metadata_csv",
        type=Path,
        default=None,
        help="Concatenated per-simulation metadata CSV with columns simulation, deme, "
        "deme_type, t_of_first_infect (produced by simulate_datastreams.py and "
        "concatenated in the pipeline). Required for migration, prevalence, and Ne "
        "time-series figures.",
    )
    parser.add_argument(
        "--prevalence_csv",
        type=Path,
        default=Path(
            "results/3_analysis/all_prevalence_datastreams_noclip_hpd_validation.csv"
        ),
        help="Prevalence HPD validation CSV (coverage vs time index figure).",
    )
    parser.add_argument(
        "--combined_ne_csv",
        type=Path,
        default=None,
        help=(
            "Optional combined Ne HPD validation CSV with Model column (e.g. from "
            "combine_hpd_validation_ne_by_model.py). Writes ne_coverage_over_time, "
            "ne_bias_logne_over_time, and ne_hpd_width_logne_over_time in output_dir. "
            "Requires --sim_metadata_csv when this file exists."
        ),
    )
    parser.add_argument(
        "--final_figure_out",
        type=Path,
        default=None,
        help="Output PNG path for the reduced main figure (default: "
        "output_dir/final_figure_reduced.png).",
    )
    parser.add_argument(
        "--supplementary_figure_out",
        type=Path,
        default=None,
        help="Output PNG path for the supplementary figure (default: "
        "output_dir/supplementary_figure.png).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required = [
        "Parameter",
        "inHPD",
        "true_value",
        "hpd_lower",
        "hpd_upper",
        "median",
        "Simulation",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = add_group_columns(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Only the reduced main figure and the supplementary figure are produced.
    # Both require migration + prevalence HPD validation CSVs and the simulation
    # metadata CSV (for outbreak start/secondary deme assignment); per-subplot
    # data CSVs are emitted by the figure functions themselves.
    if not args.migration_rates_csv.exists():
        raise ValueError(f"Migration rates CSV not found: {args.migration_rates_csv}")
    df_mig = pd.read_csv(args.migration_rates_csv)
    mig_required = [
        "Parameter",
        "Model",
        "true_value",
        "hpd_lower",
        "hpd_upper",
        "median",
        "Simulation",
    ]
    mig_missing = [c for c in mig_required if c not in df_mig.columns]
    if mig_missing:
        raise ValueError(f"Migration CSV missing columns: {mig_missing}")

    if not args.prevalence_csv.exists():
        raise ValueError(f"Prevalence CSV not found: {args.prevalence_csv}")
    df_prev = pd.read_csv(args.prevalence_csv)
    prev_required = ["Deme", "timesincestart", "inHPD", "Simulation"]
    prev_missing = [c for c in prev_required if c not in df_prev.columns]
    if prev_missing:
        raise ValueError(f"Prevalence CSV missing columns: {prev_missing}")

    if args.sim_metadata_csv is None or not Path(args.sim_metadata_csv).is_file():
        raise ValueError(
            "Both figures require --sim_metadata_csv: path to the concatenated "
            "simulation metadata CSV (produced by simulate_datastreams.py and "
            "concatenated in the pipeline)."
        )
    stem_meta = load_trajectory_meta_from_csv(Path(args.sim_metadata_csv))

    ne_csv_ok = (
        args.combined_ne_csv is not None and Path(args.combined_ne_csv).is_file()
    )
    sims: set[str] = set(df_mig["Simulation"].astype(str).unique())
    sims |= set(df_prev["Simulation"].astype(str).unique())
    if ne_csv_ok:
        sims |= set(
            pd.read_csv(args.combined_ne_csv, usecols=["Simulation"])["Simulation"]
            .astype(str)
            .unique()
        )
    trajectory_meta = {
        sim: stem_meta[trajectory_stem_from_simulation_id(sim)] for sim in sims
    }

    df_ne_reference: pd.DataFrame | None = None
    if ne_csv_ok:
        df_ne_reference = pd.read_csv(args.combined_ne_csv)

    reduced_out = (
        args.final_figure_out
        if args.final_figure_out is not None
        else args.output_dir / "final_figure_reduced.png"
    )
    plot_final_figure_reduced(
        df_params=df,
        df_prev=df_prev,
        trajectory_meta=trajectory_meta,
        output_png=reduced_out,
        df_ne_reference=df_ne_reference,
    )
    supplementary_out = (
        args.supplementary_figure_out
        if args.supplementary_figure_out is not None
        else args.output_dir / "supplementary_figure.png"
    )
    plot_supplementary_figure(
        df_mig=df_mig,
        df_prev=df_prev,
        trajectory_meta=trajectory_meta,
        output_png=supplementary_out,
        df_ne_reference=df_ne_reference,
    )


if __name__ == "__main__":
    main()
