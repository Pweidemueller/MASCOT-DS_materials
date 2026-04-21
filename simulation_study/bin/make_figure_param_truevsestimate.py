#!/usr/bin/env python3
"""
Scatter plots of true parameter values vs posterior median, with HPD whiskers,
plus a bar plot of simulation-level 95% HPD coverage per parameter group.

Reads a validation CSV (e.g. all_params_*_hpd_validation.csv) and saves one figure
per logical parameter group. Rows like caseCounts.scaling.Deme1:SimDataset and
caseCounts.scaling.Deme2:SimDataset are combined into a single figure named
caseCounts_scaling. Coverage for combined groups counts a simulation as covered only
if every deme row for that simulation has inHPD==1.

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

# Parameter groups whose per-deme rows are collapsed into a single series
# (no legend, single color) in plot_final_figure. Remove an entry to restore
# per-deme styling for that group.
COLLAPSE_SERIES_GROUPS: frozenset[str] = frozenset(
    {
        "caseCounts.scaling",
        "wastewater.scaling",
    }
)

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


# ---------------------------------------------------------------------------
# Reusable axis-level helpers
# ---------------------------------------------------------------------------


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
    ax.set_yticklabels(labels if show_ylabels else [""] * len(labels))
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
    ax.set_yticklabels(labels if show_ylabels else [""] * len(labels))
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=FONTSIZES_LIST[1])
    ax.tick_params(labelsize=FONTSIZES_LIST[2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# Alpha for median-line+quantile-band time-series panels; mirrors hpd_validation_timeseries.
_TIME_SERIES_BAND_ALPHA = 0.22


def plot_time_series_coverage_on_axis(
    ax,
    agg: pd.DataFrame,
    role_key: str,
    *,
    color: str = COLORS[3],
    title: str = "",
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    """Coverage(%) vs time_index for a single deme role."""
    configure_pdf_fonts()
    sub = agg[agg["deme_role"] == role_key].sort_values("time_index")
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
        xlabel="Time index" if show_xlabel else None,
        ylabel="Coverage (%)" if show_ylabel else None,
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
        xlabel="Time index" if show_xlabel else None,
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
    if not np.all(np.isfinite(lims)):
        return
    pad = 0.05 * (lims[1] - lims[0]) if lims[1] > lims[0] else 1.0
    lo_lim = lims[0] - pad
    hi_lim = lims[1] + pad
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color="0.5", lw=0.8, ls="--", zorder=0)
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_ylim(lo_lim, hi_lim)
    ax.set_aspect("equal", adjustable="box")

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
        plot_ne_coverage_over_time(
            df_ne_raw,
            output_dir / "ne_coverage_over_time.png",
            trajectory_meta,
            models_colors=MODEL_COLORS_DEFAULT,
        )
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
        plot_two_panel_metric_multi_model(
            agg_ne_bias,
            output_dir / "ne_bias_logne_over_time.png",
            ylabel="Bias (log Ne)",
            models_colors=MODEL_COLORS_DEFAULT,
            draw_zero_line=True,
        )
        agg_ne_hpd_width = median_quantile_metric_by_time_index_role_model(
            df_ne_prepared, "hpd_width_logne"
        )
        plot_two_panel_metric_multi_model(
            agg_ne_hpd_width,
            output_dir / "ne_hpd_width_logne_over_time.png",
            ylabel="HPD width (log Ne)",
            models_colors=MODEL_COLORS_DEFAULT,
            draw_zero_line=False,
        )

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
            plot_two_panel_metric_single_series(
                agg_p_bias,
                output_dir / "prevalence_bias_prev_over_time.png",
                ylabel="Bias of\nprevalence estimate",
                color=COLORS[3],
                draw_zero_line=True,
            )
            agg_p_rel = median_quantile_metric_by_time_index_role(
                df_prev_prepared_bias, "rel_hpd_width_prev_real"
            )
            plot_two_panel_metric_single_series(
                agg_p_rel,
                output_dir / "prevalence_rel_hpd_width_prev_over_time.png",
                ylabel="Relative HPD width of\nprevalence estimate",
                color=COLORS[3],
                draw_zero_line=False,
            )
            agg_p_bias_rel = median_quantile_metric_by_time_index_role(
                df_prev_prepared_bias, "bias_prev_real_rel"
            )
            plot_two_panel_metric_single_series(
                agg_p_bias_rel,
                output_dir / "prevalence_bias_prev_rel_over_time.png",
                ylabel="Relative bias of\nprevalence estimate",
                color=COLORS[3],
                draw_zero_line=True,
            )


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
    Assemble the combined summary figure via ``GridSpec``.

    Layout (7 logical columns × 4 rows, expanded to 14 sub-columns so that the
    1.5-wide prevalence panels fit cleanly):

    * rows 0-1, cols 0-5 (user cols 1-3): empty placeholder for an external graphic
    * rows 0-1, cols 6-9 (user cols 4-5): 2×2 parameter scatter plots
      (CC scaling, CC dispersion on top; WW scaling, WW sigma on bottom)
    * rows 0-1, cols 10-11 (user col 6): parameter coverage (horizontal bars)
    * rows 0-1, cols 12-13 (user col 7): parameter bias (horizontal boxplot,
      shares y with coverage)
    * rows 2-3, cols 0-1 (user col 1): prevalence true-vs-estimate scatter
      (start on top, secondary on bottom)
    * rows 2-3, cols 2-4 (user cols 2 to 3.5, 1.5 wide): prevalence coverage
      over time (start on top, secondary on bottom)
    * rows 2-3, cols 5-7 (user cols 3.5 to 5, 1.5 wide): prevalence relative
      bias over time
    * rows 2-3, cols 8-9 (user col 5): migration scatter (start→secondary on
      top, secondary→start on bottom)
    * rows 2-3, cols 10-11 (user col 6): migration coverage (horizontal bars)
    * rows 2-3, cols 12-13 (user col 7): migration bias (horizontal boxplot,
      shares y with coverage)

    ``df_params`` must already have ``group_id`` / ``series_label`` columns from
    :func:`add_group_columns`.
    """
    configure_pdf_fonts()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    starting_deme_by_sim = {s: m[0] for s, m in trajectory_meta.items()}

    # --- parameters: relative bias per row ((median - true) / true) -----
    df_params = df_params.copy()
    df_params["rel_bias"] = (
        df_params["median"] - df_params["true_value"]
    ) / df_params["true_value"]

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
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(nrows=4, ncols=14, hspace=0.75, wspace=0.7)

    # Placeholder for external graphic (top-left)
    ax_placeholder = fig.add_subplot(gs[0:2, 0:6])
    ax_placeholder.set_axis_off()

    # --- parameter scatter plots (2×2) ----------------------------------
    param_cells = {
        "caseCounts.scaling": (0, slice(6, 8)),
        "caseCounts.dispersion": (0, slice(8, 10)),
        "wastewater.scaling": (1, slice(6, 8)),
        "wastewater.sigma": (1, slice(8, 10)),
    }
    for gid, (r, cslice) in param_cells.items():
        ax = fig.add_subplot(gs[r, cslice])
        sub = df_params[df_params["group_id"] == gid].copy()
        if gid in COLLAPSE_SERIES_GROUPS:
            sub["series_label"] = ""
        plot_param_true_vs_estimate(
            ax,
            sub,
            title=PARAM_GROUP_TITLES.get(gid, gid),
            show_legend=False,
        )

    # --- parameter coverage (horizontal) -------------------------------
    param_groups = list(PARAM_GROUP_ORDER)
    param_axis_labels = [PARAM_GROUP_AXIS_LABELS[g] for g in param_groups]
    ax_p_cov = fig.add_subplot(gs[0:2, 10:12])
    plot_horizontal_coverage_barplot(
        ax_p_cov,
        param_axis_labels,
        coverage_percent_by_group(df_params, "group_id", param_groups),
        xlabel="Coverage (%)",
    )

    # --- parameter relative bias (horizontal boxplot; own y labels, not sharey) ---
    ax_p_bias = fig.add_subplot(gs[0:2, 12:14])
    plot_horizontal_bias_boxplot(
        ax_p_bias,
        param_axis_labels,
        {
            lbl: df_params[df_params["group_id"] == gid]["rel_bias"]
            .dropna()
            .to_numpy(dtype=float)
            for gid, lbl in zip(param_groups, param_axis_labels)
        },
        xlabel="Relative bias",
    )

    # --- prevalence scatter (start / secondary) -------------------------
    for row_offset, role_key in enumerate(("start", "secondary")):
        ax = fig.add_subplot(gs[2 + row_offset, 0:2])
        sub = prev_scatter_frame[prev_scatter_frame["deme_role"] == role_key]
        plot_param_true_vs_estimate(
            ax,
            sub,
            title=PREV_ROLE_PANEL_TITLES[role_key],
        )

    # --- prevalence coverage over time ---------------------------------
    for row_offset, role_key in enumerate(("start", "secondary")):
        ax = fig.add_subplot(gs[2 + row_offset, 2:5])
        plot_time_series_coverage_on_axis(
            ax,
            prev_coverage_agg,
            role_key,
            title=PREV_ROLE_PANEL_TITLES[role_key],
            show_xlabel=(role_key == "secondary"),
        )

    # --- prevalence relative bias over time ----------------------------
    for row_offset, role_key in enumerate(("start", "secondary")):
        ax = fig.add_subplot(gs[2 + row_offset, 5:8])
        plot_time_series_metric_on_axis(
            ax,
            prev_rel_bias_agg,
            role_key,
            ylabel="Relative bias",
            title=PREV_ROLE_PANEL_TITLES[role_key],
            show_xlabel=(role_key == "secondary"),
            draw_zero_line=True,
        )

    # --- migration scatter (per direction) -----------------------------
    for row_offset, direction in enumerate(MIGRATION_DIRECTION_LABELS):
        ax = fig.add_subplot(gs[2 + row_offset, 8:10])
        sub = mig_enriched[mig_enriched["migration_direction"] == direction].copy()
        sub["series_label"] = ""
        plot_param_true_vs_estimate(
            ax, sub, title=MIGRATION_DIRECTION_TITLES[direction]
        )

    # --- migration coverage (horizontal) -------------------------------
    mig_dirs = list(MIGRATION_DIRECTION_LABELS)
    mig_axis_labels = [MIGRATION_DIRECTION_AXIS_LABELS[d] for d in mig_dirs]
    ax_m_cov = fig.add_subplot(gs[2:4, 10:12])
    plot_horizontal_coverage_barplot(
        ax_m_cov,
        mig_axis_labels,
        coverage_percent_by_group(mig_enriched, "migration_direction", mig_dirs),
        xlabel="Coverage (%)",
    )

    # --- migration relative bias (horizontal boxplot; own y labels) ----
    ax_m_bias = fig.add_subplot(gs[2:4, 12:14])
    rel_bias_by_dir = values_by_group(
        mig_enriched, "migration_direction", "rel_bias", mig_dirs
    )
    plot_horizontal_bias_boxplot(
        ax_m_bias,
        mig_axis_labels,
        {mig_axis_labels[i]: rel_bias_by_dir[d] for i, d in enumerate(mig_dirs)},
        xlabel="Relative bias",
    )

    save_figure_png_and_pdf(output_png)
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
        help="Output PNG path for the combined summary figure (default: "
        "output_dir/final_figure.png). Written only when params, migration, and "
        "prevalence data are all available.",
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

    coverage: list[tuple[str, float]] = []
    for group_id, g in df.groupby("group_id", sort=True):
        stem = group_id_to_filename_stem(group_id)
        out_png = args.output_dir / f"{stem}.png"
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        plot_param_true_vs_estimate(
            ax, g, title=PARAM_GROUP_TITLES.get(group_id, group_id)
        )
        fig.tight_layout()
        save_figure_png_and_pdf(out_png)
        plt.close(fig)
        coverage.append((group_id, simulation_hpd_coverage_percent(g)))

    gids = [c[0] for c in coverage]
    pcts = [c[1] for c in coverage]
    fig_bar, ax_bar = plt.subplots(figsize=(4.0, 2.8))
    plot_hpd_coverage_barplot(ax_bar, gids, pcts, PARAM_GROUP_TITLES)
    fig_bar.tight_layout()
    save_figure_png_and_pdf(args.output_dir / "ds_param_coverage.png")
    plt.close(fig_bar)

    df_mig: pd.DataFrame | None = None
    df_prev: pd.DataFrame | None = None
    if args.migration_rates_csv.exists():
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
    else:
        print(f"Skipping migration figure: file not found {args.migration_rates_csv}")

    if args.prevalence_csv.exists():
        df_prev = pd.read_csv(args.prevalence_csv)
        prev_required = ["Deme", "timesincestart", "inHPD", "Simulation"]
        prev_missing = [c for c in prev_required if c not in df_prev.columns]
        if prev_missing:
            raise ValueError(f"Prevalence CSV missing columns: {prev_missing}")
    else:
        print(
            f"Skipping prevalence coverage figure: file not found {args.prevalence_csv}"
        )

    ne_csv_ok = (
        args.combined_ne_csv is not None and Path(args.combined_ne_csv).is_file()
    )
    need_meta = (df_mig is not None) or (df_prev is not None) or ne_csv_ok
    trajectory_meta: dict[str, tuple[int, float, float]] | None = None
    starting_deme_by_sim: dict[str, int] | None = None
    if need_meta:
        if args.sim_metadata_csv is None or not Path(args.sim_metadata_csv).is_file():
            raise ValueError(
                "Migration, prevalence, and/or combined Ne time-series figures require "
                "--sim_metadata_csv: path to the concatenated simulation metadata CSV "
                "(produced by simulate_datastreams.py and concatenated in the pipeline)."
            )
        # Load stem-keyed metadata, then remap to HPD CSV Simulation IDs via stem mapping.
        stem_meta = load_trajectory_meta_from_csv(args.sim_metadata_csv)
        sims: set[str] = set()
        if df_mig is not None:
            sims |= set(df_mig["Simulation"].astype(str).unique())
        if df_prev is not None:
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
        starting_deme_by_sim = {s: m[0] for s, m in trajectory_meta.items()}

    if df_mig is not None and starting_deme_by_sim is not None:
        mig_png = (
            args.migration_figure_out
            if args.migration_figure_out is not None
            else args.output_dir / "migration_rates_bias_uncertainty.png"
        )
        plot_migration_bias_uncertainty_figure(df_mig, mig_png, starting_deme_by_sim)
        plot_migration_true_vs_estimate_scatters(
            df_mig, args.output_dir, starting_deme_by_sim
        )
        enriched = (
            args.migration_enriched_csv
            if args.migration_enriched_csv is not None
            else args.output_dir / "migration_rates_hpd_validation_with_direction.csv"
        )
        write_migration_csv_with_direction(df_mig, enriched, starting_deme_by_sim)

    df_ne_reference: pd.DataFrame | None = None
    if ne_csv_ok:
        df_ne_reference = pd.read_csv(args.combined_ne_csv)

    if df_prev is not None and trajectory_meta is not None:
        plot_prevalence_coverage_over_time(
            df_prev,
            args.output_dir / "prevalence_coverage_over_time.png",
            trajectory_meta,
            ne_reference_df=df_ne_reference,
        )
        plot_prevalence_true_vs_estimate_scatters(
            df_prev,
            args.output_dir,
            trajectory_meta,
            log_space=True,
        )

    if trajectory_meta is not None:
        run_time_series_hpd_validation_figures(
            combined_ne_csv=args.combined_ne_csv,
            df_ne_preloaded=df_ne_reference,
            df_prev=df_prev,
            trajectory_meta=trajectory_meta,
            output_dir=args.output_dir,
        )

    if df_mig is not None and df_prev is not None and trajectory_meta is not None:
        final_out = (
            args.final_figure_out
            if args.final_figure_out is not None
            else args.output_dir / "final_figure.png"
        )
        plot_final_figure(
            df_params=df,
            df_mig=df_mig,
            df_prev=df_prev,
            trajectory_meta=trajectory_meta,
            output_png=final_out,
            df_ne_reference=df_ne_reference,
        )


if __name__ == "__main__":
    main()
