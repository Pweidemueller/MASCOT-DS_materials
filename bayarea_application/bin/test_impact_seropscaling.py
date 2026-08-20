#!/usr/bin/env python3
"""
Supplementary figure: sensitivity of posterior estimates to the overall scale
of the seroprevalence datastream.

Compares three MASCOT-DS runs that are identical except for an artificial
rescaling of the seroprevalence observations (which sets the overall magnitude
of inferred prevalence):

  reference — unscaled seroprevalence
  serop05   — seroprevalence observations multiplied by 0.5
  serop2    — seroprevalence observations multiplied by 2

Each run directory must contain the standard post-`logcombiner` outputs:
``<run>.combined.log``, ``<run>.NeDynamics.DemeN.combined.log`` (one per deme,
including the ghost/outside deme), and ``<run>.xml``.

Produces one combined figure under ``--output-dir`` (PNG + PDF + CSV sidecar),
laid out on a 3-row x 4-column grid:

  Panel A (row 1, columns 1-3 — one per focal deme)
      Prevalence trajectories (linear y-axis). The three runs overlaid as
      solid/dashed/dotted median lines of the location's color; 95% HPD shown
      as a light shaded band in the same color, with a distinct hatch pattern
      per run (none for Reference, '//' for x0.5, '..' for x2) so the three
      overlapping bands stay distinguishable. No case counts or wastewater.

  Panel B (row 1, column 4)
      Relative migration from outside: for each focal deme, the outside-deme
      migration rate as a % of the total outside-origin migration rate
      (summed over the focal demes), per posterior sample. One dot + 95% HPD
      whisker per run at each location, offset horizontally so all three runs
      are visible at each location's x position.

  Panel C (row 2, columns 1-3 — one per focal deme)
      Migration events from the outside/ghost deme into that deme: one dot +
      95% HPD whisker per run. Runs on the x-axis, ordered
      Seroprevalence x0.5, Reference, Seroprevalence x2.

  Panel D (row 2, column 4)
      Total migration events among the focal demes (summed per posterior
      sample): one dot + 95% HPD whisker per run, same x-axis order as C.

  Panel E (row 3, columns 1-3 — one per focal deme)
      % of new cases attributable to introductions from other demes, the
      three runs overlaid (median line + hatched HPD band, same convention as
      Panel A) in the location's color, distinguished by line style. Column 4
      of this row holds the legend: line style, then hatch swatch, then the
      run label, one row per run.

Data loading and plotting reuse ``analyse_posteriors.py`` (deme identity,
colors, NeDynamics/introductions computation) and ``value_information_analysis.py``
(migration-events extraction, grid-shift XML parsing, HPD utilities, panel
styling/labelling) rather than reimplementing them.

The x-axis for the trajectory panels (A and D) is calendar date, computed from
each run's SplineGridRateShifts plus the most recent sample time in
``--state-time-csv`` (shared across all three runs since they use the same
sequence data).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import analyse_posteriors as ap
import value_information_analysis as via
from plot_utils import FONTSIZES_LIST, save_figure_png_and_pdf, set_axis_fontsizes

# ---------------------------------------------------------------------------
# Deme identity / colors — borrowed from analyse_posteriors (dynamically
# derived from COUNTIES_OF_INTEREST, so this stays correct if that changes).
# ---------------------------------------------------------------------------
FOCAL_DEME_LABELS: tuple[str, ...] = ap.FOCAL_DEME_LABELS
GHOST_LABEL: str = ap._GHOST_LABEL
ALL_DEME_LABELS: tuple[str, ...] = FOCAL_DEME_LABELS + (GHOST_LABEL,)
DEME_MAP: dict[str, str] = ap.DEME_MAP
DEME_COLORS: dict[str, str] = ap.DEME_COLORS

# ---------------------------------------------------------------------------
# The three runs being compared.
#
# ``VERSIONS`` drives the time-series panels (A, D) and the shared legend.
# ``CATEGORICAL_KEY_ORDER`` drives the x-axis order of the categorical
# dot-with-HPD panels (B, C), which is deliberately different (serop05,
# reference, serop2) per the figure spec.
# ---------------------------------------------------------------------------
VERSIONS: tuple[tuple[str, str, str], ...] = (
    ("reference", "Reference", "-"),
    ("serop05", "Seroprevalence ×0.5", "--"),
    ("serop2", "Seroprevalence ×2", ":"),
)
VERSION_INFO: dict[str, tuple[str, str]] = {k: (lbl, ls) for k, lbl, ls in VERSIONS}
CATEGORICAL_KEY_ORDER: tuple[str, ...] = ("serop05", "reference", "serop2")

# Hatch pattern per version for the HPD bands in panels A/D — same color, three
# overlapping bands, so hatching (not color) disambiguates which band is which.
#
# Hatch density (repeated characters, e.g. "////" vs "/") must be high here:
# matplotlib tiles a fixed number of hatch repetitions across the *whole axes
# bounding box*, not per patch — so a band that spans only a small fraction of
# a tall y-axis (as in the prevalence/introductions panels) gets very few
# hatch lines at the default density and can look unhatched.
HATCH_BY_KEY: dict[str, str | None] = {
    "reference": None,
    "serop05": "//",
    "serop2": "..",
}
_BAND_FACE_ALPHA = 0.30
_BAND_EDGE_ALPHA = 0.55

NEUTRAL_COLOR = "0.25"  # dot/whisker color for the location-agnostic aggregate panel

_SERIES_LW = via._SERIES_LW
_WHISKER_LW = via._WHISKER_LW


def _blend_over_white(color: str, alpha: float) -> tuple[float, float, float]:
    """Opaque RGB equivalent of *color* at *alpha* over a white background."""
    r, g, b = mcolors.to_rgb(color)
    return (alpha * r + (1 - alpha), alpha * g + (1 - alpha), alpha * b + (1 - alpha))


def _draw_hpd_band(ax: plt.Axes, x, lo, hi, color: str, key: str) -> None:
    """Shaded 95% HPD band with a per-version hatch pattern (see HATCH_BY_KEY).

    Drawn as two layers rather than one patch with both an alpha-blended
    facecolor and a hatch: matplotlib's PDF/SVG backends have a long-standing
    bug where a hatch is invisible whenever the *facecolor* itself carries
    alpha (confirmed empirically — edgecolor alpha alone is fine, and Agg/PNG
    is unaffected either way, which is why this looked fine as a PNG but was
    blank in the PDF export). The workaround: a plain translucent wash with a
    real alpha facecolor and no hatch, then a second, face-transparent
    (``facecolor="none"``) patch that carries only the hatch. The wash still
    uses genuine alpha, so overlapping bands blend as expected.
    """
    ax.fill_between(
        x,
        lo,
        hi,
        facecolor=mcolors.to_rgba(color, alpha=_BAND_FACE_ALPHA),
        edgecolor="none",
        zorder=2,
    )
    hatch = HATCH_BY_KEY[key]
    if hatch:
        ax.fill_between(
            x,
            lo,
            hi,
            facecolor=mcolors.to_rgba(color, alpha=0),
            edgecolor=color,
            hatch=hatch,
            linewidth=0.3,
            zorder=2,
        )


# ---------------------------------------------------------------------------
# Path helpers — every input is derived from a run's ``*.combined.log`` path.
# ---------------------------------------------------------------------------


def _run_stem(combined_log: Path) -> str:
    suffix = ".combined.log"
    if not combined_log.name.endswith(suffix):
        raise ValueError(f"Expected a '*.combined.log' file, got {combined_log}")
    return combined_log.name[: -len(suffix)]


def _xml_path(combined_log: Path) -> Path:
    return combined_log.parent / f"{_run_stem(combined_log)}.xml"


def _nedynamics_path(combined_log: Path, deme: str) -> Path:
    return (
        combined_log.parent
        / f"{_run_stem(combined_log)}.NeDynamics.{deme}.combined.log"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_version_data(
    paths: dict[str, Path], burnin_fraction: float
) -> dict[str, dict]:
    """Load everything needed for the figure, per version key."""
    data: dict[str, dict] = {}
    for key, combined_log in paths.items():
        if not combined_log.is_file():
            raise FileNotFoundError(f"Combined log not found: {combined_log}")
        xml_path = _xml_path(combined_log)
        if not xml_path.is_file():
            raise FileNotFoundError(f"BEAST XML not found: {xml_path}")

        nedynamics_paths: dict[str, Path] = {}
        for lab in ALL_DEME_LABELS:
            p = _nedynamics_path(combined_log, lab)
            if not p.is_file():
                raise FileNotFoundError(f"NeDynamics log not found: {p}")
            nedynamics_paths[lab] = p

        df = via.apply_burnin(via.load_log(combined_log), burnin_fraction)
        data[key] = {
            "combined_log": combined_log,
            "xml": xml_path,
            "grid_shifts": via.load_grid_shifts(xml_path),
            "mig_events": via.extract_migration_columns(df),
            "mig_rate_samples": ap.load_migration_rate_samples(
                combined_log, burnin_fraction
            ),
            "nedynamics_paths": nedynamics_paths,
        }
    return data


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def draw_legend_panel(ax: plt.Axes) -> None:
    """Legend for the empty grid cell (row D, col 4): one row per run, each
    showing the line style then the hatch swatch then the label — the same
    two encodings used together in panels A and D."""
    ax.axis("off")
    handle_pairs = []
    labels = []
    for key, label, ls in VERSIONS:
        line = Line2D([0, 1], [0, 0], color="0.15", lw=_SERIES_LW * 0.6, ls=ls)
        patch = Patch(
            facecolor=_blend_over_white("0.15", _BAND_FACE_ALPHA),
            edgecolor=_blend_over_white("0.15", _BAND_EDGE_ALPHA),
            hatch=HATCH_BY_KEY[key],
            linewidth=0.3,
        )
        handle_pairs.append((line, patch))
        labels.append(label)
    ax.legend(
        handles=handle_pairs,
        labels=labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.8)},
        loc="center",
        frameon=False,
        fontsize=FONTSIZES_LIST[1],
        handlelength=3.2,
        handleheight=1.8,
        labelspacing=1.8,
        borderpad=0,
    )


def _style_title_axis(ax: plt.Axes, title: str, xlabel, ylabel) -> None:
    ax.set_title(title, fontsize=FONTSIZES_LIST[0])
    set_axis_fontsizes(ax, FONTSIZES_LIST, xlabel=xlabel, ylabel=ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _format_date_ticks(ax: plt.Axes) -> None:
    via._format_date_axis(ax)
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(30)
        tick_label.set_ha("right")


def _dot_whisker(
    ax: plt.Axes, pos: float, med: float, lo: float, hi: float, color: str, ls: str
) -> None:
    ax.plot([pos, pos], [lo, hi], color=color, lw=_WHISKER_LW, ls=ls, zorder=2)
    ax.plot([pos], [med], marker="o", ms=8, color=color, mec=color, mew=0, zorder=3)


def _style_categorical_axis(ax: plt.Axes, order: tuple[str, ...]) -> None:
    positions = np.arange(len(order), dtype=float)
    ax.set_xticks(positions)
    ax.set_xticklabels([VERSION_INFO[k][0] for k in order], rotation=20, ha="right")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(bottom=0)


# ---------------------------------------------------------------------------
# Panel A — prevalence trajectories, one column per location.
# ---------------------------------------------------------------------------


def draw_prevalence_panel(
    ax: plt.Axes,
    lab: str,
    data: dict[str, dict],
    burnin_fraction: float,
    t_recent: float,
    records: list[dict],
    show_ylabel: bool,
) -> None:
    color = DEME_COLORS[lab]
    for key, label, ls in VERSIONS:
        entry = data[key]
        i_idx, log_I, _ = ap.load_nedynamics_arrays(
            entry["nedynamics_paths"][lab], burnin_fraction
        )
        summ = ap.summarise_logI_trajectory(log_I)
        x_dec = t_recent - entry["grid_shifts"][lab][i_idx]
        order = np.argsort(x_dec)
        x_plot = via.decimal_years_to_matplotlib_dates(x_dec[order])
        _draw_hpd_band(
            ax, x_plot, summ["hpd_lo_I"][order], summ["hpd_hi_I"][order], "grey", key
        )
        ax.plot(
            x_plot,
            summ["median_I"][order],
            color=color,
            lw=_SERIES_LW * 0.55,
            ls=ls,
            alpha=0.95,
            zorder=3,
        )
        for xv, med, lo, hi in zip(
            x_dec[order],
            summ["median_I"][order],
            summ["hpd_lo_I"][order],
            summ["hpd_hi_I"][order],
        ):
            records.append(
                {
                    "panel": "A",
                    "deme": lab,
                    "version": label,
                    "decimal_year": float(xv),
                    "median_I": float(med),
                    "hpd_lower_I": float(lo),
                    "hpd_upper_I": float(hi),
                }
            )
    _format_date_ticks(ax)
    _style_title_axis(
        ax, DEME_MAP.get(lab, lab), "Date", "Prevalence" if show_ylabel else None
    )


# ---------------------------------------------------------------------------
# Panel B — relative migration from outside (single subplot, x = location).
# ---------------------------------------------------------------------------


def draw_relative_migration_from_outside_panel(
    ax: plt.Axes, data: dict[str, dict], records: list[dict]
) -> None:
    """Each focal deme's share (%) of the total outside-origin migration rate,
    computed per posterior sample. One location per x position; the three
    runs are offset horizontally at each location so all are visible, ordered
    (serop05, reference, serop2) — same as CATEGORICAL_KEY_ORDER / panel B."""
    n_focal = len(FOCAL_DEME_LABELS)
    positions = np.arange(n_focal, dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(CATEGORICAL_KEY_ORDER))
    for key, offset in zip(CATEGORICAL_KEY_ORDER, offsets):
        label, ls = VERSION_INFO[key]
        rate_samples = data[key]["mig_rate_samples"]
        rate_arrays = [rate_samples[(GHOST_LABEL, lab)] for lab in FOCAL_DEME_LABELS]
        n = min(len(a) for a in rate_arrays)
        stacked = np.column_stack([a[:n] for a in rate_arrays])  # (n_samples, n_focal)
        totals = stacked.sum(axis=1, keepdims=True)
        pct_samples = 100.0 * stacked / totals
        for j, lab in enumerate(FOCAL_DEME_LABELS):
            col = pct_samples[:, j]
            med = float(np.median(col))
            lo, hi = via.hpd_bounds(col)
            _dot_whisker(ax, positions[j] + offset, med, lo, hi, DEME_COLORS[lab], ls)
            records.append(
                {
                    "panel": "D",
                    "deme": lab,
                    "version": label,
                    "median": med,
                    "hpd_lower": lo,
                    "hpd_upper": hi,
                }
            )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [DEME_MAP.get(lab, lab) for lab in FOCAL_DEME_LABELS], rotation=20, ha="right"
    )
    ax.set_xlim(-0.6, n_focal - 0.4)
    ax.set_ylim(bottom=0)
    _style_title_axis(
        ax,
        "Relative migration from outside",
        None,
        "Relative migration\nfrom outside (%)",
    )


# ---------------------------------------------------------------------------
# Panel C — outside -> local-deme migration events, one column per focal deme.
# ---------------------------------------------------------------------------


def draw_outside_migration_panel(
    ax: plt.Axes,
    lab: str,
    data: dict[str, dict],
    records: list[dict],
    show_ylabel: bool,
) -> None:
    color = DEME_COLORS[lab]
    direction = f"{GHOST_LABEL}_to_{lab}"
    positions = np.arange(len(CATEGORICAL_KEY_ORDER), dtype=float)
    for pos, key in zip(positions, CATEGORICAL_KEY_ORDER):
        label, ls = VERSION_INFO[key]
        samples = data[key]["mig_events"].get(direction)
        if samples is None or len(samples) == 0:
            continue
        med = float(np.median(samples))
        lo, hi = via.hpd_bounds(samples)
        _dot_whisker(ax, pos, med, lo, hi, color, ls)
        records.append(
            {
                "panel": "C",
                "deme": lab,
                "version": label,
                "median": med,
                "hpd_lower": lo,
                "hpd_upper": hi,
            }
        )
    _style_categorical_axis(ax, CATEGORICAL_KEY_ORDER)
    ylabel = "Migration events\n(Outside → deme)" if show_ylabel else None
    _style_title_axis(ax, DEME_MAP.get(lab, lab), None, ylabel)


# ---------------------------------------------------------------------------
# Panel D — total local migration events (single subplot).
# ---------------------------------------------------------------------------


def draw_local_migration_panel(
    ax: plt.Axes, data: dict[str, dict], records: list[dict]
) -> None:
    directions = via.LOCAL_MIG_DIRECTIONS
    positions = np.arange(len(CATEGORICAL_KEY_ORDER), dtype=float)
    for pos, key in zip(positions, CATEGORICAL_KEY_ORDER):
        label, ls = VERSION_INFO[key]
        mig_events = data[key]["mig_events"]
        arrs = [mig_events[d] for d in directions if d in mig_events]
        if not arrs:
            continue
        n = min(len(a) for a in arrs)
        total = np.vstack([a[:n] for a in arrs]).sum(axis=0)
        med = float(np.median(total))
        lo, hi = via.hpd_bounds(total)
        _dot_whisker(ax, pos, med, lo, hi, NEUTRAL_COLOR, ls)
        records.append(
            {
                "panel": "B",
                "deme": "all_focal",
                "version": label,
                "median": med,
                "hpd_lower": lo,
                "hpd_upper": hi,
            }
        )
    _style_categorical_axis(ax, CATEGORICAL_KEY_ORDER)
    _style_title_axis(ax, "Local migration events", None, "Migration events")


# ---------------------------------------------------------------------------
# Panel E — % new cases due to introductions, one column per focal deme.
# ---------------------------------------------------------------------------


def draw_introductions_panel(
    ax: plt.Axes,
    lab: str,
    data: dict[str, dict],
    burnin_fraction: float,
    t_recent: float,
    records: list[dict],
    show_ylabel: bool,
) -> None:
    color = DEME_COLORS[lab]
    sources = [s for s in ALL_DEME_LABELS if s != lab]
    for key, label, ls in VERSIONS:
        entry = data[key]
        source_paths = {s: entry["nedynamics_paths"][s] for s in sources}
        source_grid_shifts = {s: entry["grid_shifts"][s] for s in sources}
        mig_samples_by_source = {
            s: entry["mig_rate_samples"][(s, lab)] for s in sources
        }
        x_dec, pct_s, _, _ = ap.compute_introductions_pct_samples(
            local_path=entry["nedynamics_paths"][lab],
            local_grid_shifts=entry["grid_shifts"][lab],
            source_paths=source_paths,
            source_grid_shifts=source_grid_shifts,
            mig_samples_by_source=mig_samples_by_source,
            t_recent=t_recent,
            burnin_fraction=burnin_fraction,
        )
        med, lo, hi = ap.summarise_samples_per_time(pct_s)
        valid = ~np.isnan(med)
        x_plot = via.decimal_years_to_matplotlib_dates(x_dec[valid])
        _draw_hpd_band(ax, x_plot, lo[valid], hi[valid], "grey", key)
        ax.plot(
            x_plot,
            med[valid],
            color=color,
            lw=_SERIES_LW * 0.55,
            ls=ls,
            alpha=0.95,
            zorder=3,
        )
        for xv, m, l, h in zip(x_dec[valid], med[valid], lo[valid], hi[valid]):
            records.append(
                {
                    "panel": "E",
                    "deme": lab,
                    "version": label,
                    "decimal_year": float(xv),
                    "median_pct": float(m),
                    "hpd_lower": float(l),
                    "hpd_upper": float(h),
                }
            )
    _format_date_ticks(ax)
    ylabel = "% new cases due\nto introductions" if show_ylabel else None
    _style_title_axis(ax, DEME_MAP.get(lab, lab), "Date", ylabel)


# ---------------------------------------------------------------------------
# Combined figure — 3 rows x 4 columns.
# ---------------------------------------------------------------------------


def plot_combined_figure(
    data: dict[str, dict], burnin_fraction: float, t_recent: float, output_path: Path
) -> None:
    fig = plt.figure(figsize=(16.5, 11.0))
    gs = fig.add_gridspec(3, 4, hspace=0.65, wspace=0.38)
    records: list[dict] = []

    # Row 1 — Panel A (prevalence, cols 1-3)
    row_a_axes = []
    for j, lab in enumerate(FOCAL_DEME_LABELS):
        ax = fig.add_subplot(gs[0, j])
        draw_prevalence_panel(
            ax, lab, data, burnin_fraction, t_recent, records, show_ylabel=(j == 0)
        )
        row_a_axes.append(ax)
    ax_b = fig.add_subplot(gs[0, 3])

    # Panel B (aggregate local, col 4)
    draw_local_migration_panel(ax_b, data, records)

    # Row 2 — Panel C (outside -> local, cols 1-3)
    row_c_axes = []
    for j, lab in enumerate(FOCAL_DEME_LABELS):
        ax = fig.add_subplot(gs[1, j])
        draw_outside_migration_panel(ax, lab, data, records, show_ylabel=(j == 0))
        row_c_axes.append(ax)
    ax_d = fig.add_subplot(gs[1, 3])

    # Panel D (relative migration from outside, col 4)
    draw_relative_migration_from_outside_panel(ax_d, data, records)

    # Row 3 — Panel E: % introductions, one column per focal deme; col 4 holds
    # the legend (line style + hatch swatch + label, one row per run).
    row_e_axes = []
    for j, lab in enumerate(FOCAL_DEME_LABELS):
        ax = fig.add_subplot(gs[2, j])
        draw_introductions_panel(
            ax, lab, data, burnin_fraction, t_recent, records, show_ylabel=(j == 0)
        )
        row_e_axes.append(ax)
    ax_legend = fig.add_subplot(gs[2, 3])
    draw_legend_panel(ax_legend)

    fig.tight_layout()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    via._add_panel_label(row_a_axes[0], "A", renderer)
    via._add_panel_label(ax_b, "B", renderer)
    via._add_panel_label(row_c_axes[0], "C", renderer)
    via._add_panel_label(ax_d, "D", renderer)
    via._add_panel_label(row_e_axes[0], "E", renderer)

    save_figure_png_and_pdf(output_path)
    plt.close(fig)
    via._save_sidecar_csv(pd.DataFrame(records), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-log",
        type=Path,
        default=Path(
            "results_1000seq_fakecasecounts/"
            "SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams/"
            "SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams.combined.log"
        ),
        help="Combined log for the unscaled-seroprevalence (reference) run.",
    )
    parser.add_argument(
        "--serop05-log",
        type=Path,
        default=Path(
            "results_1000seq_fakecasecounts/"
            "SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams_serop05/"
            "SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams_serop05.combined.log"
        ),
        help="Combined log for the seroprevalence x0.5 run.",
    )
    parser.add_argument(
        "--serop2-log",
        type=Path,
        default=Path(
            "results_1000seq_fakecasecounts/"
            "SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams_serop2/"
            "SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams_serop2.combined.log"
        ),
        help="Combined log for the seroprevalence x2 run.",
    )
    parser.add_argument(
        "--state-time-csv",
        type=Path,
        default=Path(
            "results_1000seq_fakecasecounts/"
            "SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams/"
            "SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_state_time.csv"
        ),
        help=(
            "state_time.csv giving the most recent sample time (shared across "
            "all three runs, since they use the same sequence data); used to "
            "convert trajectory grid indices to calendar dates."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the output PNG, PDF, and sidecar CSV.",
    )
    parser.add_argument(
        "--burnin-fraction",
        type=float,
        default=0.0,
        help=(
            "Additional fraction of posterior samples to discard from the start "
            "of each combined log. Combined logs already have burnin applied by "
            "logcombiner, so the default of 0.0 is usually correct."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "reference": args.reference_log,
        "serop05": args.serop05_log,
        "serop2": args.serop2_log,
    }
    print("Loading combined logs, NeDynamics logs, and grid shifts…")
    data = load_version_data(paths, args.burnin_fraction)
    t_recent = ap.load_most_recent_sample_decimal_year(args.state_time_csv)
    print(f"  Most recent sample: {t_recent:.4f} (decimal year)")

    print("Building combined figure…")
    plot_combined_figure(
        data,
        args.burnin_fraction,
        t_recent,
        args.output_dir / "seropscaling_summary.png",
    )

    print(f"\nDone. Figure written to {args.output_dir}/")


if __name__ == "__main__":
    main()
