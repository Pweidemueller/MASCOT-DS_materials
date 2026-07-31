#!/usr/bin/env python3
"""
Composite main + supplementary value-of-information figures (gridspec layout).

Reuses the per-variant metric preparation from ``make_figure_datastreams_voi``
and arranges the panels into three publication figures:

1. ``main_figure_voi.png`` (2x2):
   (0,0) start-deme log-prevalence bias over time
   (1,0) start-deme log-prevalence HPD width over time
   (0,1) start -> secondary migration-rate relative bias (boxplot per variant)
   (1,1) start -> secondary migration-rate relative HPD width (boxplot per
         variant), with the prior relative HPD width drawn as a dashed line.

2. ``supp_figure_voi_secondary.png`` (2x2): identical layout for the secondary
   deme prevalence and secondary -> start migration rate.

3. ``supp_figure_datastream_params.png`` (2x4): datastream-parameter relative
   bias (top row) and relative HPD width (bottom row), one column per parameter
   group. Each HPD-width panel carries the prior relative HPD width as a dashed
   line.

Prevalence panels use time in **days** (``timesincestart`` * 365), matching the
prevalence trajectory plots in ``analyse_posteriors.py``. Prior relative HPD
widths are sampled from the LogNormal priors declared in the MASCOT-DS template
XML, then summarised with the same 95% HPD definition used for the posteriors.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from make_figure_simstudy import (
    MIGRATION_DIRECTION_LABELS,
    PARAM_GROUP_ORDER,
    PARAM_GROUP_TITLES,
    load_trajectory_meta_from_csv,
)
from make_figure_datastreams_voi import (
    COLOR_BY_LABEL,
    LABEL_BY_KEY,
    VERSION_KEYS,
    VERSION_LABELS,
    VERSIONS,
    _csv_path,
    aggregate_prevalence_across_versions,
    prepare_migration_metrics,
    prepare_param_metrics,
    prepare_prevalence_metrics,
)
from plot_utils import (
    FONTSIZES_LIST,
    configure_pdf_fonts,
    save_figure_png_and_pdf,
    save_plot_data_csv,
    set_axis_fontsizes,
)

# Toggle the 2.5-97.5% across-simulation band on the prevalence panels. The
# quantiles are always computed; this only controls whether they are shaded.
# Set to True to draw the bands again.
SHOW_PREVALENCE_BAND = False

# Skyline (SkylinePrev) knots overlaid on the prevalence HPD-width panels. The
# coarse skyline has 11 knots spanning the same window as the per-simulation
# time-grid index, so we place them at equally spaced index positions rather
# than absolute times (which differ across simulations). Knots outside a panel's
# visible x-range are dropped (e.g. the secondary deme starts mid-window).
N_SKYLINE_KNOTS = 11
KNOT_INDEX_SPAN = (0, 100)

# Map each estimated datastream parameter group (and migration) to the BEAST
# parameter whose LogNormal prior governs it. Relative HPD width is scale
# invariant for a LogNormal, so per-deme duplicates share one entry.
PRIOR_PARAM_FOR_GROUP: dict[str, str] = {
    "caseCounts.scaling": "caseCounts.scaling.Deme1",
    "caseCounts.dispersion": "caseCounts.dispersion",
    "wastewater.scaling": "wastewater.scaling.Deme1",
    "wastewater.sigma": "wastewater.sigma",
}
PRIOR_PARAM_MIGRATION = "migrationRatesSkyline.t"

# Number of prior draws used to estimate the prior HPD width (large for a stable
# HPD); fixed seed keeps the dashed reference lines reproducible.
PRIOR_SAMPLE_N = 2_000_000
PRIOR_SAMPLE_SEED = 12345

# Shared panel titles / axis labels (kept identical across main and supplementary).
PREV_BIAS_YLABEL = "Bias of log prevalence"
PREV_WIDTH_YLABEL = "95% HPDI width of log prevalence"
MIG_BIAS_YLABEL = "Rel. bias of migration rate"
MIG_WIDTH_YLABEL = "Rel. 95% HPDI width of migration rate"
DEME_TITLE = {"start": "Start deme", "secondary": "Secondary deme"}
MIGRATION_TITLE = {
    MIGRATION_DIRECTION_LABELS[0]: "Start -> Secondary",
    MIGRATION_DIRECTION_LABELS[1]: "Secondary -> Start",
}
ONLYTREE_KEY = "datastreams_onlytree"


# ---------------------------------------------------------------------------
# Prior relative HPD width from the template XML
# ---------------------------------------------------------------------------


def _calculate_hpd(data: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """95% HPD ``(lower, upper, median)`` — same definition as analyse_posteriors."""
    data = np.sort(np.asarray(data, dtype=float))
    n = len(data)
    m = int((1 - alpha) * n)
    if m < 1:
        return float(data[0]), float(data[-1]), float(np.median(data))
    intervals = data[m:] - data[: n - m]
    k = int(np.argmin(intervals))
    return float(data[k]), float(data[k + m]), float(np.median(data))


def extract_lognormal_priors(xml_path: Path) -> dict[str, tuple[float, float]]:
    """Map BEAST parameter name -> (meanlog M, sdlog S) for each LogNormal prior.

    Handles both ``<prior x="@param">`` and the ``<prior><x arg="@param"/>`` form.
    """
    root = ET.parse(xml_path).getroot()
    priors: dict[str, tuple[float, float]] = {}
    for prior in root.iter("prior"):
        ln = prior.find("LogNormal")
        if ln is None:
            continue
        m_val = s_val = None
        for p in ln.findall("parameter"):
            if p.get("name") == "M":
                m_val = float(p.text)
            elif p.get("name") == "S":
                s_val = float(p.text)
        if m_val is None or s_val is None:
            continue
        target = prior.get("x")
        if target is None:
            x_el = prior.find("x")
            if x_el is not None:
                target = x_el.get("arg")
        if not target:
            continue
        name = target.lstrip("@").split(":")[0]
        priors[name] = (m_val, s_val)
    return priors


def prior_relative_hpd_width(
    m_val: float, s_val: float, rng: np.random.Generator
) -> float:
    """Relative HPD width ``(hpd_upper - hpd_lower) / median`` of a LogNormal prior."""
    samples = rng.lognormal(mean=m_val, sigma=s_val, size=PRIOR_SAMPLE_N)
    lo, hi, med = _calculate_hpd(samples)
    return (hi - lo) / med


def compute_prior_relative_hpd_widths(
    xml_path: Path,
) -> tuple[dict[str, float], float]:
    """Return (``{group_id: rel_hpd_width}``, migration ``rel_hpd_width``)."""
    priors = extract_lognormal_priors(xml_path)
    rng = np.random.default_rng(PRIOR_SAMPLE_SEED)
    group_widths: dict[str, float] = {}
    for group_id, beast_name in PRIOR_PARAM_FOR_GROUP.items():
        if beast_name in priors:
            m_val, s_val = priors[beast_name]
            group_widths[group_id] = prior_relative_hpd_width(m_val, s_val, rng)
    mig_width = float("nan")
    if PRIOR_PARAM_MIGRATION in priors:
        m_val, s_val = priors[PRIOR_PARAM_MIGRATION]
        mig_width = prior_relative_hpd_width(m_val, s_val, rng)
    return group_widths, mig_width


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------


def _label_color_pairs(version_keys: list[str]) -> list[tuple[str, str]]:
    color_by_key = {v[0]: v[2] for v in VERSIONS}
    return [(LABEL_BY_KEY[k], color_by_key[k]) for k in version_keys]


def _style_axis(ax) -> None:
    ax.tick_params(labelsize=FONTSIZES_LIST[2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_prevalence_panel(
    ax,
    agg: pd.DataFrame,
    role: str,
    label_color_pairs: list[tuple[str, str]],
    *,
    ylabel: str,
    title: str | None = None,
    draw_zero_line: bool = False,
    draw_knots: bool = False,
    show_band: bool = SHOW_PREVALENCE_BAND,
    band_alpha: float = 0.22,
) -> None:
    """Median line per variant vs ``time_index`` for one deme role.

    The 2.5-97.5% across-simulation band is drawn only when ``show_band`` is
    True (default from ``SHOW_PREVALENCE_BAND``); the quantiles are computed
    upstream regardless. ``time_index`` 0 is the earliest gridpoint (outbreak
    start / furthest in the past) and increases forward in time. With
    ``draw_knots``, the skyline knots are overlaid as faint dashed verticals.
    """
    sub_all = agg[agg["deme_role"] == role]
    for label, color in label_color_pairs:
        sub = sub_all[sub_all["Model"] == label].sort_values("time_index")
        if sub.empty:
            continue
        x = sub["time_index"].to_numpy(dtype=float)
        if show_band:
            ax.fill_between(
                x,
                sub["q025"],
                sub["q975"],
                color=color,
                alpha=band_alpha,
                lw=0,
                zorder=1,
            )
        ax.plot(
            x,
            sub["median_value"],
            color=color,
            lw=2.5,
            zorder=2,
            label=label,
            alpha=0.6,
        )
    if draw_zero_line:
        ax.axhline(0.0, color="0.75", lw=0.7, ls="--", zorder=0)
    if draw_knots:
        _draw_skyline_knots(ax)
    if title:
        ax.set_title(title, fontsize=FONTSIZES_LIST[0])
    set_axis_fontsizes(ax, FONTSIZES_LIST, xlabel="Time index", ylabel=ylabel)
    _style_axis(ax)


def _draw_skyline_knots(ax) -> None:
    """Overlay the 11 skyline knots as faint dashed verticals in index space.

    Anchored to the autoscaled x-range (so the lines never stretch the axis) and
    masked to that visible window, mirroring the calendar-date knot recipe but in
    the per-simulation-invariant time-grid index coordinate.
    """
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax)
    knot_positions = np.linspace(
        KNOT_INDEX_SPAN[0], KNOT_INDEX_SPAN[1], N_SKYLINE_KNOTS
    )
    for xv in knot_positions:
        if xmin <= xv <= xmax:
            ax.axvline(xv, color="0.55", lw=0.7, ls=(0, (4, 3)), zorder=0)


def draw_boxplot_panel(
    ax,
    df: pd.DataFrame,
    mask: pd.Series,
    value_col: str,
    versions: list[str],
    colors: dict[str, str],
    *,
    ylabel: str,
    title: str | None = None,
    prior_value: float | None = None,
    draw_zero_line: bool = False,
) -> None:
    """One box per variant; optional dashed prior reference and zero line."""
    sub = df[mask]
    positions = np.arange(len(versions), dtype=float)
    data = [
        sub.loc[sub["Version"] == v, value_col].dropna().to_numpy(dtype=float)
        for v in versions
    ]
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
    if prior_value is not None and np.isfinite(prior_value):
        ax.axhline(prior_value, color="0.15", lw=1.1, ls="--", zorder=6)
        # "Prior" label pinned to the top-left of the dashed line: x in axes
        # fraction (always at the left edge), y in data coords just above the
        # line. Uses the smallest fontsize in FONTSIZES_LIST.
        ax.text(
            0.01,
            prior_value,
            "Prior",
            transform=blended_transform_factory(ax.transAxes, ax.transData),
            color="0.15",
            fontsize=FONTSIZES_LIST[2],
            ha="left",
            va="bottom",
            zorder=7,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(versions, rotation=35, ha="right")
    ax.set_xlim(-0.7, len(versions) - 0.3)
    if title:
        ax.set_title(title, fontsize=FONTSIZES_LIST[0])
    set_axis_fontsizes(ax, FONTSIZES_LIST, xlabel=None, ylabel=ylabel)
    _style_axis(ax)


def _figure_legend(fig, label_color_pairs) -> None:
    handles = [
        Line2D([0], [0], color=color, lw=3.0, label=label)
        for label, color in label_color_pairs
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=FONTSIZES_LIST[2],
        bbox_to_anchor=(0.5, 0.0),
    )


def _add_panel_labels(
    fig,
    panels: list[tuple[object, str]],
    *,
    x_pad: float = 0.006,
    y_pad: float = 0.005,
) -> None:
    """Place bold panel letters at the top-left corner of each axis.

    Each letter sits just left of the y-axis region (tick labels + y-axis
    title, so the y-axis title reads to the right of the letter) and above any
    axis title (so the title sits below the letter). Mirrors the panel-label
    recipe in ``make_figure_individualsim.py``. ``x_pad``/``y_pad`` are offsets
    in figure-fraction coordinates.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    fontsize = FONTSIZES_LIST[0] + 4
    for ax, lab in panels:
        # Left edge of the whole y-axis (ticklabels + axis title) in fig coords.
        y_bbox = ax.yaxis.get_tightbbox(renderer)
        x_px = y_bbox.x0 if y_bbox is not None else ax.get_window_extent().x0
        x_fig = inv.transform((x_px, 0))[0] - x_pad
        # Top of the axes, lifted above the title if the panel carries one.
        y_fig = ax.get_position().y1
        if ax.title.get_text():
            title_top = inv.transform((0, ax.title.get_window_extent(renderer).y1))[1]
            y_fig = max(y_fig, title_top)
        fig.text(
            x_fig,
            y_fig + y_pad,
            lab,
            fontsize=fontsize,
            fontweight="bold",
            va="bottom",
            ha="left",
        )


# ---------------------------------------------------------------------------
# Composite figures
# ---------------------------------------------------------------------------


def _draw_prevalence_column(
    ax_bias,
    ax_width,
    agg_logbias: pd.DataFrame,
    agg_width: pd.DataFrame,
    role: str,
    label_color_pairs: list[tuple[str, str]],
) -> None:
    """Fill a (bias, HPD width) prevalence column for one deme role."""
    draw_prevalence_panel(
        ax_bias,
        agg_logbias,
        role,
        label_color_pairs,
        ylabel=PREV_BIAS_YLABEL,
        title=DEME_TITLE[role],
        draw_zero_line=True,
    )
    draw_prevalence_panel(
        ax_width,
        agg_width,
        role,
        label_color_pairs,
        ylabel=PREV_WIDTH_YLABEL,
        title=None,
        draw_zero_line=False,
        draw_knots=False,
    )


def _draw_migration_column(
    ax_bias,
    ax_width,
    mig_metrics: pd.DataFrame,
    migration_direction: str,
    mig_prior_width: float,
) -> None:
    """Fill a (relative bias, relative HPD width) migration column for one direction."""
    mask = mig_metrics["migration_direction"] == migration_direction
    draw_boxplot_panel(
        ax_bias,
        mig_metrics,
        mask,
        "rel_bias",
        VERSION_LABELS,
        COLOR_BY_LABEL,
        ylabel=MIG_BIAS_YLABEL,
        title=MIGRATION_TITLE[migration_direction],
        draw_zero_line=True,
    )
    draw_boxplot_panel(
        ax_width,
        mig_metrics,
        mask,
        "rel_hpd_width",
        VERSION_LABELS,
        COLOR_BY_LABEL,
        ylabel=MIG_WIDTH_YLABEL,
        title=None,
        prior_value=mig_prior_width,
        draw_zero_line=False,
    )


def make_main_figure(
    agg_logbias: pd.DataFrame,
    agg_width: pd.DataFrame,
    mig_metrics: pd.DataFrame,
    mig_prior_width: float,
    output_png: Path,
) -> None:
    """2x2: start-deme prevalence (Tree only excluded) + start -> secondary migration."""
    configure_pdf_fonts()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    # Tree only is so uncertain in prevalence it flattens the other variants, so
    # it is dropped from the prevalence column here (kept in the full supp fig)
    # but retained in the migration boxplots where it is a useful baseline.
    prev_pairs = _label_color_pairs([k for k in VERSION_KEYS if k != ONLYTREE_KEY])

    fig = plt.figure(figsize=(9.0, 7.0))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])
    _draw_prevalence_column(ax_a, ax_b, agg_logbias, agg_width, "start", prev_pairs)
    _draw_migration_column(
        ax_c,
        ax_d,
        mig_metrics,
        MIGRATION_DIRECTION_LABELS[0],
        mig_prior_width,
    )

    _figure_legend(fig, _label_color_pairs(VERSION_KEYS))
    # One letter per bias/HPDI-width column, on the top (bias) panel only.
    _add_panel_labels(fig, [(ax_a, "A"), (ax_c, "B")])
    save_figure_png_and_pdf(output_png)
    plt.close(fig)


def make_secondary_supp_figure(
    agg_logbias: pd.DataFrame,
    agg_width: pd.DataFrame,
    mig_metrics: pd.DataFrame,
    mig_prior_width: float,
    output_png: Path,
) -> None:
    """2x3: start-deme prevalence (all variants) + secondary-deme prevalence + secondary -> start migration."""
    configure_pdf_fonts()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    all_pairs = _label_color_pairs(VERSION_KEYS)
    # Start-deme column contrasts only the full model (All DS) against the
    # tree-only baseline; the secondary column keeps every variant.
    start_pairs = _label_color_pairs(["datastreams", ONLYTREE_KEY])

    fig = plt.figure(figsize=(13.0, 7.0))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[0, 2])
    ax_f = fig.add_subplot(gs[1, 2])
    _draw_prevalence_column(ax_a, ax_b, agg_logbias, agg_width, "start", start_pairs)
    _draw_prevalence_column(ax_c, ax_d, agg_logbias, agg_width, "secondary", all_pairs)
    _draw_migration_column(
        ax_e,
        ax_f,
        mig_metrics,
        MIGRATION_DIRECTION_LABELS[1],
        mig_prior_width,
    )

    _figure_legend(fig, all_pairs)
    # One letter per bias/HPDI-width column, on the top (bias) panel only.
    _add_panel_labels(fig, [(ax_a, "A"), (ax_c, "B"), (ax_e, "C")])
    save_figure_png_and_pdf(output_png)
    plt.close(fig)


def make_params_figure(
    param_metrics: pd.DataFrame,
    group_prior_widths: dict[str, float],
    output_png: Path,
) -> None:
    """2x4 figure: datastream-param relative bias (top) and relative HPD (bottom)."""
    configure_pdf_fonts()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    groups = list(PARAM_GROUP_ORDER)
    n = len(groups)

    fig = plt.figure(figsize=(3.0 * n, 6.4))
    gs = fig.add_gridspec(2, n, hspace=0.45, wspace=0.38)

    # One letter per group column, on the top (bias) panel only.
    letters = [chr(ord("A") + i) for i in range(n)]
    panels: list[tuple[object, str]] = []
    for col, gid in enumerate(groups):
        mask = param_metrics["group_id"] == gid
        title = PARAM_GROUP_TITLES.get(gid, gid)
        ax_bias = fig.add_subplot(gs[0, col])
        draw_boxplot_panel(
            ax_bias,
            param_metrics,
            mask,
            "rel_bias",
            VERSION_LABELS,
            COLOR_BY_LABEL,
            ylabel="Relative bias" if col == 0 else "",
            title=title,
            draw_zero_line=True,
        )
        ax_width = fig.add_subplot(gs[1, col])
        draw_boxplot_panel(
            ax_width,
            param_metrics,
            mask,
            "rel_hpd_width",
            VERSION_LABELS,
            COLOR_BY_LABEL,
            ylabel="Relative HPDI width" if col == 0 else "",
            title=None,
            prior_value=group_prior_widths.get(gid),
            draw_zero_line=False,
        )
        panels.append((ax_bias, letters[col]))

    # No figure legend on this supplementary panel figure.
    _add_panel_labels(fig, panels)
    save_figure_png_and_pdf(output_png)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    analysis_dir: Path,
    output_dir: Path,
    sim_metadata_csv: Path,
    template_xml: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_meta = load_trajectory_meta_from_csv(sim_metadata_csv)

    prev_by_version: dict[str, pd.DataFrame] = {}
    mig_by_version: dict[str, pd.DataFrame] = {}
    param_by_version: dict[str, pd.DataFrame] = {}
    for key in VERSION_KEYS:
        prev_by_version[key] = pd.read_csv(_csv_path(analysis_dir, "prevalence", key))
        mig_by_version[key] = pd.read_csv(
            _csv_path(analysis_dir, "migration_rates", key)
        )
        param_by_version[key] = pd.read_csv(_csv_path(analysis_dir, "params", key))

    # Prevalence metrics (plotted against the per-simulation time-grid index).
    prev_prepared = {
        key: prepare_prevalence_metrics(df, stem_meta)
        for key, df in prev_by_version.items()
    }
    agg_logbias = aggregate_prevalence_across_versions(prev_prepared, "bias_logprev")
    agg_width = aggregate_prevalence_across_versions(prev_prepared, "hpd_width_logprev")

    # Migration + parameter metrics.
    mig_metrics = prepare_migration_metrics(mig_by_version, stem_meta)
    param_metrics = prepare_param_metrics(param_by_version)

    # Prior relative HPD widths.
    group_prior_widths, mig_prior_width = compute_prior_relative_hpd_widths(
        template_xml
    )

    # 1. Main figure (start deme / start -> secondary).
    main_png = output_dir / "main_figure_voi.png"
    make_main_figure(
        agg_logbias,
        agg_width,
        mig_metrics,
        mig_prior_width,
        main_png,
    )
    save_plot_data_csv(agg_logbias, main_png, suffix="prevalence_logbias")
    save_plot_data_csv(agg_width, main_png, suffix="prevalence_hpd_width")
    save_plot_data_csv(
        mig_metrics[
            mig_metrics["migration_direction"] == MIGRATION_DIRECTION_LABELS[0]
        ][
            [
                "Simulation",
                "Parameter",
                "migration_direction",
                "Version",
                "rel_bias",
                "rel_hpd_width",
            ]
        ],
        main_png,
        suffix="migration",
    )

    # 2. Supplementary figure 1: start prevalence (all variants) + secondary
    #    prevalence + secondary -> start migration.
    supp1_png = output_dir / "supp_figure_voi_secondary.png"
    make_secondary_supp_figure(
        agg_logbias,
        agg_width,
        mig_metrics,
        mig_prior_width,
        supp1_png,
    )
    save_plot_data_csv(
        mig_metrics[
            mig_metrics["migration_direction"] == MIGRATION_DIRECTION_LABELS[1]
        ][
            [
                "Simulation",
                "Parameter",
                "migration_direction",
                "Version",
                "rel_bias",
                "rel_hpd_width",
            ]
        ],
        supp1_png,
        suffix="migration",
    )

    # 3. Supplementary figure 2 (datastream parameters).
    supp2_png = output_dir / "supp_figure_datastream_params.png"
    make_params_figure(param_metrics, group_prior_widths, supp2_png)
    params_long = param_metrics[param_metrics["group_id"].isin(PARAM_GROUP_ORDER)][
        [
            "Simulation",
            "Parameter",
            "group_id",
            "series_label",
            "Version",
            "rel_bias",
            "rel_hpd_width",
        ]
    ]
    save_plot_data_csv(params_long, supp2_png, suffix="params")
    prior_df = pd.DataFrame(
        [
            {"group_id": g, "prior_rel_hpd_width": w}
            for g, w in group_prior_widths.items()
        ]
        + [{"group_id": "migration", "prior_rel_hpd_width": mig_prior_width}]
    )
    save_plot_data_csv(prior_df, supp2_png, suffix="prior_rel_hpd_width")


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
        help="Where to write the composite figures and data sidecars.",
    )
    parser.add_argument(
        "--sim_metadata_csv",
        type=Path,
        default=None,
        help="Path to all_sim_metadata.csv (defaults to <analysis_dir>/all_sim_metadata.csv).",
    )
    parser.add_argument(
        "--template_xml",
        type=Path,
        default=Path("simulation_study/data/Mascot_datastreams_template_fixedtree.xml"),
        help="MASCOT-DS template XML used to read the LogNormal priors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_meta = (
        args.sim_metadata_csv
        if args.sim_metadata_csv is not None
        else args.analysis_dir / "all_sim_metadata.csv"
    )
    run(args.analysis_dir, args.output_dir, sim_meta, args.template_xml)


if __name__ == "__main__":
    main()
