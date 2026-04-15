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
from matplotlib.patches import Patch

from constants import MODEL_MASCOT, MODEL_MASCOT_DS
from plot_utils import COLORS, FONTSIZES_LIST, configure_pdf_fonts, save_figure_png_and_pdf
from plot_utils import set_axis_fontsizes
from hpd_validation_timeseries import (
    MODEL_COLORS_DEFAULT,
    add_bias_and_hpd_width_columns,
    add_prevalence_bias_and_hpd_width_real_space,
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
        raise ValueError(f"Cannot parse migration deme pair from Parameter: {parameter!r}")
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


def _values_for_model_direction(
    df: pd.DataFrame,
    value_col: str,
    model: str,
    direction: str,
) -> np.ndarray:
    sub = df[(df["Model"] == model) & (df["migration_direction"] == direction)]
    return sub[value_col].dropna().to_numpy(dtype=float)


def plot_migration_direction_model_boxplots(
    ax,
    df: pd.DataFrame,
    value_col: str,
    *,
    xlabel: str,
    show_y_ticklabels: bool = True,
    show_legend: bool = True,
    draw_zero_vline: bool = False,
) -> None:
    """
    Horizontal boxplots: x = value_col, y = migration direction.
    For each direction, MASCOT (COLORS[4]) and MASCOT-DS (COLORS[3]) are drawn
    side by side (small gap).
    """
    configure_pdf_fonts()
    direction_order = list(MIGRATION_DIRECTION_LABELS)
    models = [
        (MODEL_MASCOT, COLORS[4]),
        (MODEL_MASCOT_DS, COLORS[3]),
    ]
    n_dir = len(direction_order)
    group_centers = np.arange(n_dir, dtype=float)
    offset = 0.18
    box_width = 0.32

    legend_handles: list[Patch] = []
    for model_name, color in models:
        data_list = []
        positions = []
        for di, dlabel in enumerate(direction_order):
            vals = _values_for_model_direction(df, value_col, model_name, dlabel)
            data_list.append(vals if vals.size > 0 else np.array([np.nan]))
            pos = group_centers[di] + (-offset if model_name == MODEL_MASCOT else offset)
            positions.append(pos)
        bp = ax.boxplot(
            data_list,
            positions=positions,
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
        legend_handles.append(
            Patch(facecolor=color, edgecolor="0.25", linewidth=0.6, label=model_name)
        )

    if draw_zero_vline:
        ax.axvline(0.0, color="0.75", lw=0.7, ls="--", zorder=0)
    ax.set_yticks(group_centers)
    ax.set_yticklabels(direction_order)
    ax.set_xlabel(xlabel, fontsize=FONTSIZES_LIST[1])
    ax.tick_params(labelsize=FONTSIZES_LIST[2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(handles=legend_handles, frameon=False, fontsize=FONTSIZES_LIST[2], loc="best")


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
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 4.0), sharey=True)
    plot_migration_direction_model_boxplots(
        axes[0],
        df_plot,
        "bias",
        xlabel="Migration rate\nbias",
        show_y_ticklabels=True,
        show_legend=True,
        draw_zero_vline=True,
    )
    plot_migration_direction_model_boxplots(
        axes[1],
        df_plot,
        "relative_hpd_width",
        xlabel="Migration rate\nrel. HPD width",
        show_y_ticklabels=False,
        show_legend=False,
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
    coverage = subdf['inHPD'].mean()
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
    unique_labels = sorted(
        s for s in series_labels.unique() if s != ""
    )
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
        len(unique_labels) > 1
        or (len(unique_labels) == 1 and unique_labels[0] != "")
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
            raise ValueError(
                f"Combined Ne CSV missing columns {miss}: {path_for_msg}"
            )
        df_ne_prepared = prepare_validation_timeseries_df(
            df_ne_raw, trajectory_meta, starting_deme_by_sim
        )

    if df_prev_prepared is not None and df_ne_prepared is not None:
        assert_matching_secondary_min_time_index(
            df_prev_prepared, df_ne_prepared
        )

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
        df_ne_prepared.to_csv(output_dir / "ne_summary_logne_over_time.csv", index=False)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="True vs estimated parameter figures with HPD error bars."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/3_analysis/all_params_datastreams_noclip_hpd_validation.csv"),
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

    title_map = {
        "caseCounts.scaling": "CC scaling",
        "caseCounts.dispersion": "CC dispersion",
        "wastewater.scaling": "WW scaling",
        "wastewater.sigma": "WW sigma",
        "seroprevalence.scaling": "SP scaling",
    }

    coverage: list[tuple[str, float]] = []
    for group_id, g in df.groupby("group_id", sort=True):
        
        stem = group_id_to_filename_stem(group_id)
        out_png = args.output_dir / f"{stem}.png"
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        plot_param_true_vs_estimate(
            ax, g, title=title_map.get(group_id, group_id)
        )
        fig.tight_layout()
        save_figure_png_and_pdf(out_png)
        plt.close(fig)
        coverage.append((group_id, simulation_hpd_coverage_percent(g)))

    gids = [c[0] for c in coverage]
    pcts = [c[1] for c in coverage]
    fig_bar, ax_bar = plt.subplots(figsize=(4.0, 2.8))
    plot_hpd_coverage_barplot(ax_bar, gids, pcts, title_map)
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
        print(
            f"Skipping migration figure: file not found {args.migration_rates_csv}"
        )

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
        args.combined_ne_csv is not None
        and Path(args.combined_ne_csv).is_file()
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
                pd.read_csv(args.combined_ne_csv, usecols=["Simulation"])[
                    "Simulation"
                ]
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
        plot_migration_bias_uncertainty_figure(
            df_mig, mig_png, starting_deme_by_sim
        )
        plot_migration_true_vs_estimate_scatters(
            df_mig, args.output_dir, starting_deme_by_sim
        )
        enriched = (
            args.migration_enriched_csv
            if args.migration_enriched_csv is not None
            else args.output_dir / "migration_rates_hpd_validation_with_direction.csv"
        )
        write_migration_csv_with_direction(
            df_mig, enriched, starting_deme_by_sim
        )

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

    if trajectory_meta is not None:
        run_time_series_hpd_validation_figures(
            combined_ne_csv=args.combined_ne_csv,
            df_ne_preloaded=df_ne_reference,
            df_prev=df_prev,
            trajectory_meta=trajectory_meta,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
