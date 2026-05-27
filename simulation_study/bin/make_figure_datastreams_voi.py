#!/usr/bin/env python3
"""
Value-of-information / datastream comparison across MASCOT-DS variants.

Loads the per-variant prevalence, migration, and parameter HPD-validation CSVs
from ``simulation_study/results/3_analysis`` and produces seven comparison
figures, with one MASCOT-DS variant per color (from ``lab_palette``):

1. ``prevalence_relative_bias_over_time.png``
   Median (line) and 2.5–97.5% sim-level band of relative prevalence bias
   ``(exp(median) - exp(true)) / exp(true)`` per ``time_index`` × deme role.

2. ``prevalence_bias_logprev_over_time.png``
   Median + band of log-prevalence bias ``log(median) - log(true)``. Already
   scale-invariant (difference of logs = log of ratio), so no separate
   relative version is needed.

3. ``prevalence_hpd_width_logprev_over_time.png``
   Median + band of log-prevalence HPD width ``upper - lower``, mirroring
   the ``ne_hpd_width_logne_over_time`` figure.

   Plots 1-3 are also produced with a ``_no_treeonly`` suffix, repeating
   the same figures with the ``Tree only`` variant excluded so the other
   variants can be compared on a less stretched y-axis.

4. ``migration_rates_relative_bias.png``
   Boxplot per variant in two panels (start->secondary, secondary->start).

5. ``migration_rates_relative_hpd_width.png``
   Same layout as (4) for relative HPD width ``(upper - lower) / median``.

6. ``datastream_params_relative_bias.png``
   One subplot per parameter group (caseCounts.{scaling,dispersion},
   wastewater.{scaling,sigma}); boxplot per variant.

7. ``datastream_params_relative_hpd_width.png``
   Same layout as (6) for relative HPD width.

Each plot is accompanied by a ``<stem>_data.csv`` next to the PNG so the
figure can be reproduced from the saved values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lab_palette as lp
from constants import MODEL_MASCOT_DS
from hpd_validation_timeseries import (
    add_bias_and_hpd_width_columns,
    add_prevalence_bias_and_hpd_width_real_space,
    median_quantile_metric_by_time_index_role_model,
    plot_two_panel_metric_multi_model,
    prepare_validation_timeseries_df,
)
from make_figure_simstudy import (
    MIGRATION_DIRECTION_LABELS,
    MIGRATION_DIRECTION_TITLES,
    PARAM_GROUP_ORDER,
    PARAM_GROUP_TITLES,
    add_group_columns,
    add_migration_direction_column,
    load_trajectory_meta_from_csv,
    trajectory_stem_from_simulation_id,
)
from plot_utils import (
    FONTSIZES_LIST,
    configure_pdf_fonts,
    save_figure_png_and_pdf,
    save_plot_data_csv,
    set_axis_fontsizes,
)

# (variant key, display label, hex color) — order drives panel/legend order.
# Colors are drawn from lab_palette PALETTE_6, which is colorblind-tested.
VERSIONS: tuple[tuple[str, str, str], ...] = (
    ("datastreams", "All DS", lp.KYBURG_GOLD),
    ("datastreams_nocasecounts", "No CC", lp.UCSF_TEAL),
    ("datastreams_noseroprevalence", "No SP", lp.BASEL),
    ("datastreams_nowastewater", "No WW", lp.BRIDGE),
    ("datastreams_nomascotll", "No MASCOT-LL", lp.HUTCH),
    ("datastreams_onlytree", "Tree only", lp.RAIN),
)

VERSION_KEYS: list[str] = [v[0] for v in VERSIONS]
VERSION_LABELS: list[str] = [v[1] for v in VERSIONS]
LABEL_BY_KEY: dict[str, str] = {v[0]: v[1] for v in VERSIONS}
COLOR_BY_LABEL: dict[str, str] = {v[1]: v[2] for v in VERSIONS}
LABEL_COLOR_PAIRS: tuple[tuple[str, str], ...] = tuple((v[1], v[2]) for v in VERSIONS)


def _csv_path(analysis_dir: Path, kind: str, key: str) -> Path:
    """``kind`` is one of ``prevalence``, ``migration_rates``, ``params``."""
    return analysis_dir / f"all_{kind}_{key}_hpd_validation.csv"


def _starting_deme_for_sims(
    sim_ids: list[str],
    stem_meta: dict[str, tuple[int, float, float]],
) -> dict[str, int]:
    return {s: stem_meta[trajectory_stem_from_simulation_id(s)][0] for s in sim_ids}


def _trajectory_meta_for_sims(
    sim_ids: list[str],
    stem_meta: dict[str, tuple[int, float, float]],
) -> dict[str, tuple[int, float, float]]:
    return {s: stem_meta[trajectory_stem_from_simulation_id(s)] for s in sim_ids}


# ---------------------------------------------------------------------------
# Time-series prep & aggregation
# ---------------------------------------------------------------------------


def prepare_prevalence_metrics(
    df_prev: pd.DataFrame,
    stem_meta: dict[str, tuple[int, float, float]],
) -> pd.DataFrame:
    """Add ``time_index``, ``deme_role``, real-space rel bias and log HPD width."""
    sim_ids = df_prev["Simulation"].astype(str).unique().tolist()
    traj_meta = _trajectory_meta_for_sims(sim_ids, stem_meta)
    starting_deme = {s: m[0] for s, m in traj_meta.items()}
    df = prepare_validation_timeseries_df(df_prev, traj_meta, starting_deme)
    df = add_prevalence_bias_and_hpd_width_real_space(df)
    df = add_bias_and_hpd_width_columns(
        df,
        expected_col="expectedlogPrev",
        median_col="logPrevalence",
        rel_hpd_width=False,
    )
    return df


def aggregate_prevalence_across_versions(
    df_per_version: dict[str, pd.DataFrame],
    metric_col: str,
) -> pd.DataFrame:
    """Pool prepared per-version frames and aggregate by (time_index, role, Model).

    A ``Model`` column is set to the variant display label so the result
    plugs straight into ``plot_two_panel_metric_multi_model``.
    """
    pieces: list[pd.DataFrame] = []
    for key, df in df_per_version.items():
        sub = df[["time_index", "deme_role", metric_col]].copy()
        sub["Model"] = LABEL_BY_KEY[key]
        pieces.append(sub)
    pooled = pd.concat(pieces, ignore_index=True)
    return median_quantile_metric_by_time_index_role_model(pooled, metric_col)


# ---------------------------------------------------------------------------
# Migration & parameter prep
# ---------------------------------------------------------------------------


def prepare_migration_metrics(
    df_per_version: dict[str, pd.DataFrame],
    stem_meta: dict[str, tuple[int, float, float]],
) -> pd.DataFrame:
    """Concat MASCOT-DS rows with version label, direction, rel-bias, rel-width."""
    pieces: list[pd.DataFrame] = []
    for key, df in df_per_version.items():
        sims = df["Simulation"].astype(str).unique().tolist()
        starting_deme = _starting_deme_for_sims(sims, stem_meta)
        enriched = add_migration_direction_column(df, starting_deme)
        enriched = enriched[enriched["Model"] == MODEL_MASCOT_DS].copy()
        true_v = pd.to_numeric(enriched["true_value"], errors="coerce")
        med = pd.to_numeric(enriched["median"], errors="coerce")
        lo = pd.to_numeric(enriched["hpd_lower"], errors="coerce")
        hi = pd.to_numeric(enriched["hpd_upper"], errors="coerce")
        enriched["rel_bias"] = (med - true_v) / true_v
        enriched["rel_hpd_width"] = (hi - lo) / med
        enriched["Version"] = LABEL_BY_KEY[key]
        pieces.append(enriched)
    return pd.concat(pieces, ignore_index=True)


def prepare_param_metrics(
    df_per_version: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Concat datastream-parameter rows with version label and rel metrics.

    Rows with missing ``true_value`` (e.g. ``onlytree`` runs that don't
    estimate any datastream parameter) are kept and surface as NaN, which
    boxplot dropna handles automatically.
    """
    pieces: list[pd.DataFrame] = []
    for key, df in df_per_version.items():
        # ``onlytree`` does not estimate any datastream parameter, so its CSV
        # may omit the value columns entirely. Fill them with NaN so the
        # downstream rel-metric computation produces NaN (which boxplot drops).
        df_filled = df.copy()
        for col in ("true_value", "median", "hpd_lower", "hpd_upper"):
            if col not in df_filled.columns:
                df_filled[col] = np.nan
        enriched = add_group_columns(df_filled).copy()
        true_v = pd.to_numeric(enriched["true_value"], errors="coerce")
        med = pd.to_numeric(enriched["median"], errors="coerce")
        lo = pd.to_numeric(enriched["hpd_lower"], errors="coerce")
        hi = pd.to_numeric(enriched["hpd_upper"], errors="coerce")
        enriched["rel_bias"] = (med - true_v) / true_v
        enriched["rel_hpd_width"] = (hi - lo) / med
        enriched["Version"] = LABEL_BY_KEY[key]
        pieces.append(enriched)
    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------------
# Boxplot helper for plots 3-6
# ---------------------------------------------------------------------------


def plot_grouped_boxplots(
    df: pd.DataFrame,
    *,
    panel_col: str,
    panels: list[str],
    panel_titles: dict[str, str],
    value_col: str,
    versions: list[str],
    colors: dict[str, str],
    ylabel: str,
    output_png: Path,
    draw_zero_line: bool = False,
    figsize: tuple[float, float] | None = None,
    ncols: int | None = None,
    sharey: bool = False,
) -> None:
    """One boxplot per ``versions`` entry, faceted by ``panels``.

    Each panel is filtered by ``df[panel_col] == panel``; within a panel,
    one box per version is drawn at integer positions in ``versions`` order.
    """
    configure_pdf_fonts()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    n_panels = len(panels)
    if ncols is None:
        ncols = n_panels
    nrows = int(np.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (3.0 * ncols, 3.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for idx, panel in enumerate(panels):
        ax = axes_flat[idx]
        sub = df[df[panel_col] == panel]
        positions = np.arange(len(versions), dtype=float)
        data = [
            sub.loc[sub["Version"] == v, value_col].dropna().to_numpy(dtype=float)
            for v in versions
        ]
        # Empty arrays would crash boxplot; substitute a single NaN placeholder.
        data_for_bp = [d if d.size > 0 else np.array([np.nan]) for d in data]
        bp = ax.boxplot(
            data_for_bp,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            whis=1.5,
        )
        for patch, vname in zip(bp["boxes"], versions):
            patch.set_facecolor(colors[vname])
            patch.set_edgecolor("0.25")
            patch.set_linewidth(0.6)
        for el in bp["medians"]:
            el.set_color("0.15")
            el.set_linewidth(0.9)
        for el in bp["whiskers"] + bp["caps"]:
            el.set_color("0.35")
            el.set_linewidth(0.6)
        if draw_zero_line:
            ax.axhline(0.0, color="0.75", lw=0.7, ls="--", zorder=0)

        ax.set_xticks(positions)
        ax.set_xticklabels(versions, rotation=35, ha="right")
        ax.set_xlim(-0.7, len(versions) - 0.3)
        ax.set_title(
            panel_titles.get(panel, panel).replace("\n", " "),
            fontsize=FONTSIZES_LIST[0],
        )
        set_axis_fontsizes(ax, FONTSIZES_LIST, xlabel=None, ylabel=ylabel)
        ax.tick_params(labelsize=FONTSIZES_LIST[2])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    save_figure_png_and_pdf(output_png)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV helpers for sidecar data
# ---------------------------------------------------------------------------


def _migration_boxplot_long_form(
    df_mig_metrics: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    cols = [
        "Simulation",
        "Parameter",
        "migration_direction",
        "Version",
        value_col,
    ]
    return df_mig_metrics[cols].copy()


def _params_boxplot_long_form(
    df_params_metrics: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    sub = df_params_metrics[df_params_metrics["group_id"].isin(PARAM_GROUP_ORDER)]
    cols = [
        "Simulation",
        "Parameter",
        "group_id",
        "series_label",
        "Version",
        value_col,
    ]
    return sub[cols].copy()


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _plot_prevalence_set(
    prev_prepared: dict[str, pd.DataFrame],
    version_keys: list[str],
    output_dir: Path,
    suffix: str,
) -> None:
    """Produce the three prevalence-over-time plots for a subset of variants."""
    subset = {k: prev_prepared[k] for k in version_keys}
    color_by_key = {v[0]: v[2] for v in VERSIONS}
    label_color_pairs = tuple(
        (LABEL_BY_KEY[k], color_by_key[k]) for k in version_keys
    )

    agg_prev_relbias = aggregate_prevalence_across_versions(
        subset, "bias_prev_real_rel"
    )
    relbias_png = output_dir / f"prevalence_relative_bias_over_time{suffix}.png"
    plot_two_panel_metric_multi_model(
        agg_prev_relbias,
        relbias_png,
        ylabel="Relative bias of\nprevalence estimate",
        models_colors=label_color_pairs,
        draw_zero_line=True,
    )
    save_plot_data_csv(agg_prev_relbias, relbias_png)

    agg_prev_logbias = aggregate_prevalence_across_versions(subset, "bias_logprev")
    logbias_png = output_dir / f"prevalence_bias_logprev_over_time{suffix}.png"
    plot_two_panel_metric_multi_model(
        agg_prev_logbias,
        logbias_png,
        ylabel="Bias (log prevalence)",
        models_colors=label_color_pairs,
        draw_zero_line=True,
    )
    save_plot_data_csv(agg_prev_logbias, logbias_png)

    agg_prev_width = aggregate_prevalence_across_versions(subset, "hpd_width_logprev")
    width_png = output_dir / f"prevalence_hpd_width_logprev_over_time{suffix}.png"
    plot_two_panel_metric_multi_model(
        agg_prev_width,
        width_png,
        ylabel="HPD width (log prevalence)",
        models_colors=label_color_pairs,
        draw_zero_line=False,
    )
    save_plot_data_csv(agg_prev_width, width_png)


def run(
    analysis_dir: Path,
    output_dir: Path,
    sim_metadata_csv: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_meta = load_trajectory_meta_from_csv(sim_metadata_csv)

    # Load all per-version CSVs up front so we error early if any are missing.
    prev_by_version: dict[str, pd.DataFrame] = {}
    mig_by_version: dict[str, pd.DataFrame] = {}
    param_by_version: dict[str, pd.DataFrame] = {}
    for key in VERSION_KEYS:
        prev_by_version[key] = pd.read_csv(_csv_path(analysis_dir, "prevalence", key))
        mig_by_version[key] = pd.read_csv(
            _csv_path(analysis_dir, "migration_rates", key)
        )
        param_by_version[key] = pd.read_csv(_csv_path(analysis_dir, "params", key))

    # ---- 1 & 2: prevalence over time -------------------------------------
    prev_prepared: dict[str, pd.DataFrame] = {
        key: prepare_prevalence_metrics(df, stem_meta)
        for key, df in prev_by_version.items()
    }

    _plot_prevalence_set(prev_prepared, VERSION_KEYS, output_dir, suffix="")
    _plot_prevalence_set(
        prev_prepared,
        [k for k in VERSION_KEYS if k != "datastreams_onlytree"],
        output_dir,
        suffix="_no_treeonly",
    )

    # ---- 3 & 4: migration rates ------------------------------------------
    mig_metrics = prepare_migration_metrics(mig_by_version, stem_meta)

    mig_relbias_png = output_dir / "migration_rates_relative_bias.png"
    plot_grouped_boxplots(
        mig_metrics,
        panel_col="migration_direction",
        panels=list(MIGRATION_DIRECTION_LABELS),
        panel_titles=MIGRATION_DIRECTION_TITLES,
        value_col="rel_bias",
        versions=VERSION_LABELS,
        colors=COLOR_BY_LABEL,
        ylabel="Relative bias\n(migration rate)",
        output_png=mig_relbias_png,
        draw_zero_line=True,
        figsize=(8.0, 3.6),
        ncols=2,
        sharey=True,
    )
    save_plot_data_csv(
        _migration_boxplot_long_form(mig_metrics, "rel_bias"),
        mig_relbias_png,
    )

    mig_relwidth_png = output_dir / "migration_rates_relative_hpd_width.png"
    plot_grouped_boxplots(
        mig_metrics,
        panel_col="migration_direction",
        panels=list(MIGRATION_DIRECTION_LABELS),
        panel_titles=MIGRATION_DIRECTION_TITLES,
        value_col="rel_hpd_width",
        versions=VERSION_LABELS,
        colors=COLOR_BY_LABEL,
        ylabel="Relative HPD width\n(migration rate)",
        output_png=mig_relwidth_png,
        draw_zero_line=False,
        figsize=(8.0, 3.6),
        ncols=2,
        sharey=True,
    )
    save_plot_data_csv(
        _migration_boxplot_long_form(mig_metrics, "rel_hpd_width"),
        mig_relwidth_png,
    )

    # ---- 5 & 6: datastream parameters ------------------------------------
    param_metrics = prepare_param_metrics(param_by_version)
    param_panels = list(PARAM_GROUP_ORDER)
    param_panel_titles = {gid: PARAM_GROUP_TITLES.get(gid, gid) for gid in param_panels}

    param_relbias_png = output_dir / "datastream_params_relative_bias.png"
    plot_grouped_boxplots(
        param_metrics,
        panel_col="group_id",
        panels=param_panels,
        panel_titles=param_panel_titles,
        value_col="rel_bias",
        versions=VERSION_LABELS,
        colors=COLOR_BY_LABEL,
        ylabel="Relative bias",
        output_png=param_relbias_png,
        draw_zero_line=True,
        figsize=(11.0, 3.6),
        ncols=4,
        sharey=False,
    )
    save_plot_data_csv(
        _params_boxplot_long_form(param_metrics, "rel_bias"),
        param_relbias_png,
    )

    param_relwidth_png = output_dir / "datastream_params_relative_hpd_width.png"
    plot_grouped_boxplots(
        param_metrics,
        panel_col="group_id",
        panels=param_panels,
        panel_titles=param_panel_titles,
        value_col="rel_hpd_width",
        versions=VERSION_LABELS,
        colors=COLOR_BY_LABEL,
        ylabel="Relative HPD width",
        output_png=param_relwidth_png,
        draw_zero_line=False,
        figsize=(11.0, 3.6),
        ncols=4,
        sharey=False,
    )
    save_plot_data_csv(
        _params_boxplot_long_form(param_metrics, "rel_hpd_width"),
        param_relwidth_png,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis_dir",
        type=Path,
        default=Path("simulation_study/results/3_analysis"),
        help="Directory containing all_*_<variant>_hpd_validation.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Where to write the six PNGs/PDFs and their _data.csv sidecars.",
    )
    parser.add_argument(
        "--sim_metadata_csv",
        type=Path,
        default=None,
        help=(
            "Path to all_sim_metadata.csv (defaults to "
            "<analysis_dir>/all_sim_metadata.csv)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_meta = (
        args.sim_metadata_csv
        if args.sim_metadata_csv is not None
        else args.analysis_dir / "all_sim_metadata.csv"
    )
    run(args.analysis_dir, args.output_dir, sim_meta)


if __name__ == "__main__":
    main()
