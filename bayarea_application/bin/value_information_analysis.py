#!/usr/bin/env python3
"""
Value-of-information analysis for Bay Area MASCOT-DS ablation study.

Uses real Bay Area SARS-CoV-2 combined BEAST posteriors (no ground truth).
Compares MASCOT-DS variants (ablation of individual data streams, plus two
no-seroprevalence variants with one nuisance scaler fixed at its full-data
posterior median) by HPD width and deviation from the full-datastream
reference ("All DS").

Input files expected per variant under <results_dir>/<prefix>_<variant_key>/:
  - <run>.combined.log          (main BEAST combined log with migration rates
    and datastream scaling parameters)
  - <run>.NeDynamics.DemeN.combined.log  (log-prevalence time series, one per deme)

Produces one main figure and several supplementary figures:

value_of_information_main.png — 2×3 GridSpec, Santa Clara as the worked example:
  (0,0) prevalence diff from All DS, main ablations (no seroprev variants)
  (1,0) prevalence 95% HPD width, main ablations
  (0,1) prevalence diff from All DS, the three seroprevalence variants
  (1,1) prevalence 95% HPD width, the three seroprevalence variants (+ All DS ref)
  (0,2) total local migration-events posteriors (summed over all directions among
        the focal demes), all variants except No phylogeny (see below)
  (1,2) total local migration-events relative HPD width, all variants except
        No phylogeny

supp_prevalence.png — prevalence diff + HPD width for the counties not shown in
  the main figure (Sacramento, San Francisco), plus a third column showing the
  total migration-from-background posteriors (top) and relative HPD width
  (bottom), summed over all Outside → focal-deme directions (excluding No
  phylogeny).

supp_migration_events.png — migration-events posteriors (dot + 95% HPD whiskers)
  for all estimated directions (the *_to_Outside directions fixed to 0 are
  omitted).

supp_migration_hpd.png — relative 95% HPD width for all estimated directions.

The "No phylogeny" variant (MASCOT-DS coalescent log-likelihood removed) is
excluded from every migration-events panel above, raw and aggregated: without
that likelihood, migration events and node heights (see below) are essentially
unconstrained by the data and would swamp the other variants' scale. It still
appears in the prevalence and datastream-parameter figures.

supp_datastream_parameters.png — posteriors of the datastream scaling / nuisance
  parameters (NeScaler, case-count scaling + dispersion, seroprevalence scaling,
  wastewater scaling + sigma) per deme, compared across variants.

node_height_relative_hpd_width.png — standalone figure: boxplot of internal-node
  height 95% relative HPD width (= HPD width / height_median), one box per
  variant, from each variant's <run>.combined.mcc.trees. No phylogeny is
  excluded (see above). The number of boxplot points differs slightly across
  variants because low-support clades are dropped (see
  extract_internal_node_relative_hpd_widths); node_low_support_counts.png
  shows exactly how many are dropped per variant.

node_low_support_counts.png — standalone figure: bar chart of the number of
  internal nodes per variant whose clade support was too low for
  treeannotator to report a height HPD/median at all (see
  extract_internal_node_relative_hpd_widths). Same variant set as the
  node-height HPD-width figure above.

"Total local migration events" and "total migration from background" are computed
per posterior sample (element-wise sum of the relevant migrationEvents columns),
then summarised by median and 95% credible interval — so the uncertainty reflects
the joint posterior of the summed count, not a sum of marginal intervals.

A <stem>_data.csv sidecar is written next to each PNG.

The three no-seroprevalence variants share the seroprevalence hue (Basel); they
are drawn in three shades of it (light → dark) so they stay distinguishable when
plotted together.
"""

from __future__ import annotations

import argparse
import re
import string
from pathlib import Path
from xml.etree import ElementTree as ET

import baltic as bt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import lab_palette as lp
from filter_gisaid_metadata import COUNTIES_OF_INTEREST
from plot_utils import (
    FONTSIZES_LIST,
    configure_pdf_fonts,
    decimal_years_to_matplotlib_dates,
    save_figure_png_and_pdf,
    set_axis_fontsizes,
)

# ---------------------------------------------------------------------------
# Variant configuration — order drives legend and panel order.
# Colors drawn from lab_palette (colorblind-tested).
# ---------------------------------------------------------------------------

VERSIONS: tuple[tuple[str, str, str], ...] = (
    ("datastreams", "All DS", lp.KYBURG_GOLD),
    ("datastreams_nocasecounts", "No CC", lp.UCSF_TEAL),
    ("datastreams_noseroprevalence", "No SP", lp.BASEL),
    ("datastreams_noseroprevalence_fixedNeScalers", "No SP (fix Ne)", lp.BASEL),
    ("datastreams_noseroprevalence_fixedCaseCountsScaling", "No SP (fix CC)", lp.BASEL),
    ("datastreams_nowastewater", "No WW", lp.BRIDGE),
    ("datastreams_nomascotll", "No phylogeny", lp.HUTCH),
    ("datastreams_onlytree", "Phylogeny only", lp.RAIN),
)

REFERENCE_KEY = "datastreams"
REFERENCE_LABEL = "All DS"

VERSION_KEYS: list[str] = [v[0] for v in VERSIONS]
VERSION_LABELS: list[str] = [v[1] for v in VERSIONS]
LABEL_BY_KEY: dict[str, str] = {v[0]: v[1] for v in VERSIONS}
COLOR_BY_LABEL: dict[str, str] = {v[1]: v[2] for v in VERSIONS}

# ---------------------------------------------------------------------------
# Publication colors / styles — the three no-seroprevalence variants share the
# seroprevalence hue (Basel, mid shade). To keep them distinguishable they are
# separated by line style (and by face hatching in the bar/box panels) rather
# than by colour:
#   No SP           → solid line, solid fill
#   No SP (fix Ne)  → dashed line, '//' hatch
#   No SP (fix CC)  → dotted line, '..' hatch
# Every other variant keeps its lab_palette colour, a solid line, and a solid fill.
# ---------------------------------------------------------------------------

NOSEROPREV_KEYS: tuple[str, ...] = (
    "datastreams_noseroprevalence",
    "datastreams_noseroprevalence_fixedNeScalers",
    "datastreams_noseroprevalence_fixedCaseCountsScaling",
)

# Publication color per variant key (the seroprev trio all share mid Basel).
PUB_COLOR_BY_KEY: dict[str, object] = {
    key: (lp.BASEL if key in NOSEROPREV_KEYS else color) for key, _, color in VERSIONS
}

# Line style per variant key (default solid); used in every line panel and on
# the dot-with-HPD-line migration markers.
LINESTYLE_BY_KEY: dict[str, str] = {
    NOSEROPREV_KEYS[1]: "--",
    NOSEROPREV_KEYS[2]: ":",
}

# Face hatch per variant key (default none); used on bar/box faces so the
# same-coloured seroprev trio stays distinguishable.
HATCH_BY_KEY: dict[str, str | None] = {
    NOSEROPREV_KEYS[1]: "//",
    NOSEROPREV_KEYS[2]: "..",
}


def _ls(key: str) -> str:
    """Line style for a variant key (solid by default)."""
    return LINESTYLE_BY_KEY.get(key, "-")


def _hatch(key: str) -> str | None:
    """Face hatch pattern for a variant key (none by default)."""
    return HATCH_BY_KEY.get(key)


# (key, label, color) triples used by the publication figures.
PUB_VERSIONS: list[tuple] = [(k, lbl, PUB_COLOR_BY_KEY[k]) for k, lbl, _ in VERSIONS]
MAIN_VERSIONS: list[tuple] = [v for v in PUB_VERSIONS if v[0] not in NOSEROPREV_KEYS]
SEROPREV_VERSIONS: list[tuple] = [v for v in PUB_VERSIONS if v[0] in NOSEROPREV_KEYS]
_ALLDS_TRIPLE: tuple = next(v for v in PUB_VERSIONS if v[0] == REFERENCE_KEY)

# Removing the MASCOT-DS coalescent log-likelihood leaves migration events (and
# node heights, see the standalone height-HPD figure) essentially unconstrained
# by the data, so this variant is excluded from every migration-events panel
# (raw per-direction and aggregated) while still appearing in the prevalence
# and datastream-parameter figures.
NO_MASCOTLL_KEY = "datastreams_nomascotll"
MIGRATION_KEYS: list[str] = [k for k in VERSION_KEYS if k != NO_MASCOTLL_KEY]
MIGRATION_VERSIONS: list[tuple] = [v for v in PUB_VERSIONS if v[0] != NO_MASCOTLL_KEY]

# ---------------------------------------------------------------------------
# Dynamic deme mapping — mirrors analyse_posteriors.py / create_mascot_xml.py.
# Counties are sorted alphabetically; outside/ghost deme is always last.
# ---------------------------------------------------------------------------

_sorted_focal = sorted(COUNTIES_OF_INTEREST)
_n_focal = len(_sorted_focal)

DEME_MAP: dict[str, str] = {f"Deme{i + 1}": c for i, c in enumerate(_sorted_focal)}
DEME_MAP[f"Deme{_n_focal + 1}"] = "Outside"
FOCAL_DEME_LABELS: tuple[str, ...] = tuple(f"Deme{i + 1}" for i in range(_n_focal))
GHOST_DEME_LABEL = f"Deme{_n_focal + 1}"
ALL_DEME_LABELS: tuple[str, ...] = FOCAL_DEME_LABELS + (GHOST_DEME_LABEL,)

# County → deme-label lookup, so the worked example survives deme renumbering.
_DEME_FOR_COUNTY: dict[str, str] = {county: deme for deme, county in DEME_MAP.items()}

# Worked example for the main figure: Santa Clara prevalence.
EXAMPLE_DEME: str = _DEME_FOR_COUNTY.get("Santa Clara", FOCAL_DEME_LABELS[-1])

I_COLUMN_RE = re.compile(r"^I_(\d+)$")
MIGRATION_COLUMN_RE = re.compile(r"^migrationEvents\.(Deme\d+)_to_(Deme\d+)$")

# All 12 directions (4 demes × 3 destinations each), in row-major order
# matching a (n_demes × n_demes-1) grid laid out as rows = source, cols = dest.
MIGRATION_DIRECTIONS: list[str] = [
    f"{src}_to_{dst}"
    for src in ALL_DEME_LABELS
    for dst in ALL_DEME_LABELS
    if src != dst
]

# Aggregate migration metrics (summed per posterior sample).
#   "total local"      = all directions among the focal demes
#   "total background" = all Outside → focal-deme directions
LOCAL_MIG_DIRECTIONS: list[str] = [
    f"{src}_to_{dst}"
    for src in FOCAL_DEME_LABELS
    for dst in FOCAL_DEME_LABELS
    if src != dst
]
BACKGROUND_MIG_DIRECTIONS: list[str] = [
    f"{GHOST_DEME_LABEL}_to_{dst}" for dst in FOCAL_DEME_LABELS
]
LOCAL_AGG_KEY = "total_local"
BACKGROUND_AGG_KEY = "total_background"


def _dir_title(direction: str) -> str:
    """Human-readable title for a migration direction key, e.g. 'Deme1_to_Deme2'."""
    src, dst = direction.split("_to_")
    return f"{DEME_MAP.get(src, src)} → {DEME_MAP.get(dst, dst)}"


def _is_zero_direction(raw_mig: RawMig, direction: str) -> bool:
    """True if all displayed migration variants have ~zero samples for this
    direction (the excluded no-phylogeny variant is not considered)."""
    for key in MIGRATION_KEYS:
        samples = raw_mig.get(key, {}).get(direction)
        if samples is not None and np.any(np.abs(samples) > 1e-10):
            return False
    return True


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _run_dir(results_dir: Path, variant_prefix: str, variant_key: str) -> Path:
    return results_dir / f"{variant_prefix}_{variant_key}"


def _combined_log_path(run_dir: Path) -> Path:
    return run_dir / f"{run_dir.name}.combined.log"


def _nedynamics_log_path(run_dir: Path, deme: str) -> Path:
    return run_dir / f"{run_dir.name}.NeDynamics.{deme}.combined.log"


def _mcc_trees_path(run_dir: Path) -> Path:
    return run_dir / f"{run_dir.name}.combined.mcc.trees"


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------


def load_log(path: Path) -> pd.DataFrame:
    """Read a BEAST tab-separated log, skipping '#'-prefixed comment lines."""
    return pd.read_csv(path, sep="\t", comment="#")


def apply_burnin(df: pd.DataFrame, burnin_fraction: float) -> pd.DataFrame:
    n = len(df)
    drop = int(np.floor(n * burnin_fraction))
    return df.iloc[drop:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prevalence utilities
# ---------------------------------------------------------------------------


def extract_i_columns(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract I_* columns from a NeDynamics log.

    Returns (sorted_indices, samples_matrix) where samples_matrix has shape
    (n_samples, n_timepoints), sorted by ascending grid index.
    Index 0 = present (most recent); high index = further in the past.
    """
    matched = sorted(
        [(int(m.group(1)), col) for col in df.columns if (m := I_COLUMN_RE.match(col))],
        key=lambda x: x[0],
    )
    if not matched:
        raise ValueError(f"No I_* columns found (first cols: {list(df.columns)[:5]}…).")
    indices = np.array([t for t, _ in matched])
    cols = [col for _, col in matched]
    return indices, df[cols].to_numpy(dtype=float)


def prevalence_summary(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column (timepoint) median and 95% HPD across samples (see
    ``hpd_bounds``)."""
    n_t = samples.shape[1]
    med = np.median(samples, axis=0)
    lo = np.empty(n_t)
    hi = np.empty(n_t)
    for j in range(n_t):
        lo[j], hi[j] = hpd_bounds(samples[:, j])
    return med, lo, hi


# ---------------------------------------------------------------------------
# Migration utilities
# ---------------------------------------------------------------------------


def extract_migration_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return {DemeX_to_DemeY: posterior_samples} for migrationEvents columns
    (integer counts) from the main combined log."""
    result = {}
    for col in df.columns:
        m = MIGRATION_COLUMN_RE.match(col)
        if m:
            result[f"{m.group(1)}_to_{m.group(2)}"] = df[col].to_numpy(dtype=float)
    return result


def hpd_bounds(samples: np.ndarray, ci: float = 0.95) -> tuple[float, float]:
    """Highest posterior density interval: the narrowest interval spanning a
    ``ci`` fraction of the samples (shortest sliding window over the sorted
    array), not an equal-tailed percentile interval. Matches
    ``analyse_posteriors.calculate_hpd_mcmc`` so HPD widths from the two
    scripts are directly comparable.
    """
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    m = int(ci * n)
    if m < 1 or m >= n:
        return float(x[0]), float(x[-1])
    intervals = x[m:] - x[: n - m]
    min_idx = int(np.argmin(intervals))
    return float(x[min_idx]), float(x[min_idx + m])


def _summed_migration_samples(
    raw_mig: "RawMig", directions: list[str]
) -> dict[str, np.ndarray]:
    """Per-variant per-sample sum of migrationEvents across *directions*.

    All directions of a variant come from the same combined log, so they share
    row order; summing element-wise gives the joint posterior of the total count.
    """
    out: dict[str, np.ndarray] = {}
    for key in VERSION_KEYS:
        arrs = [
            s
            for d in directions
            if (s := raw_mig.get(key, {}).get(d)) is not None and len(s) > 0
        ]
        if not arrs:
            continue
        n = min(len(a) for a in arrs)
        out[key] = np.vstack([a[:n] for a in arrs]).sum(axis=0)
    return out


def build_aggregate_migration(
    raw_mig: "RawMig", directions: list[str], agg_key: str
) -> tuple["RawMig", "MigSummary"]:
    """Build raw-samples and summary dicts for a summed-events aggregate.

    Returns (raw_like, summary_like) keyed by variant → {agg_key: ...}, so the
    aggregate can be passed straight to the standard migration panel helpers.
    """
    summed = _summed_migration_samples(raw_mig, directions)
    raw_like: "RawMig" = {key: {agg_key: s} for key, s in summed.items()}
    summary_like: "MigSummary" = {
        key: {agg_key: (float(np.median(s)), *hpd_bounds(s))}
        for key, s in summed.items()
    }
    return raw_like, summary_like


# ---------------------------------------------------------------------------
# Node-height HPD utilities (MCC consensus trees)
# ---------------------------------------------------------------------------

# Same trait keys as analyse_posteriors.py's HPD_KEY / node.traits convention.
HEIGHT_HPD_KEY = "height_95%_HPD"
HEIGHT_MEDIAN_KEY = "height_median"


def extract_internal_node_relative_hpd_widths(
    tree_path: Path,
) -> tuple[np.ndarray, int, int]:
    """Relative 95% HPD width of node height for every internal node of an MCC
    consensus tree, i.e. ``(height_95%_HPD upper − lower) / height_median``.

    Returns ``(widths, n_low_support, n_total)`` where ``n_total`` is the
    number of internal nodes in the tree and ``n_low_support`` is how many of
    them were skipped.

    Nodes missing either trait (or, degenerately, with a non-positive/
    non-finite median) are skipped. This is *not* about polytomies — MCC
    trees from these runs are strictly bifurcating. Instead, treeannotator
    only writes ``height_95%_HPD``/``height_median`` for a clade when it has
    enough posterior tree samples containing that clade to form a
    distribution. Clades at the minimum non-zero posterior-support bin (i.e.
    observed in only a handful of the summarised posterior trees) get a
    single point ``height=`` estimate and no HPD/median at all, so they are
    dropped here. Checked directly on
    ``SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams``:
    every dropped node had ``posterior`` at the minimum observed value
    (0.002), while every retained node had ``posterior`` at least twice that.
    This is why the number of boxplot points in
    ``node_height_relative_hpd_width.png`` differs slightly across variants —
    see ``node_low_support_counts.png`` for the per-variant drop count.
    """
    tree = bt.loadNexus(
        str(tree_path),
        treestring_regex=r"tree\s+\S+\s*=",
        absoluteTime=False,
        verbose=False,
    )
    widths = []
    n_total = 0
    n_low_support = 0
    for node in tree.Objects:
        if node.branchType != "node":
            continue
        n_total += 1
        hpd = node.traits.get(HEIGHT_HPD_KEY)
        median = node.traits.get(HEIGHT_MEDIAN_KEY)
        if hpd is None or median is None or len(hpd) != 2:
            n_low_support += 1
            continue
        median = float(median)
        if not np.isfinite(median) or median <= 0:
            n_low_support += 1
            continue
        lo, hi = float(hpd[0]), float(hpd[1])
        rel = (hi - lo) / median
        if np.isfinite(rel):
            widths.append(rel)
        else:
            n_low_support += 1
    return np.array(widths, dtype=float), n_low_support, n_total


def load_all_node_hpd_widths(
    results_dir: Path,
    variant_prefix: str,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, int]]]:
    """Relative node-height HPD widths for every variant's MCC tree, plus how
    many internal nodes per variant were dropped for low clade support (see
    ``extract_internal_node_relative_hpd_widths``).

    Excludes the no-phylogeny variant: with the MASCOT-DS coalescent
    log-likelihood removed, node heights are essentially unconstrained by the
    data (see MIGRATION_KEYS, excluded from migration panels for the same
    reason).

    Returns ``(node_hpd_widths, node_support_counts)`` where
    ``node_support_counts[key] = (n_low_support, n_total)``.
    """
    data: dict[str, np.ndarray] = {}
    counts: dict[str, tuple[int, int]] = {}
    for key in MIGRATION_KEYS:
        run_dir = _run_dir(results_dir, variant_prefix, key)
        tree_path = _mcc_trees_path(run_dir)
        if not tree_path.exists():
            print(f"  [warn] missing MCC tree: {tree_path.name}")
            continue
        widths, n_low_support, n_total = extract_internal_node_relative_hpd_widths(
            tree_path
        )
        counts[key] = (n_low_support, n_total)
        if len(widths) == 0:
            print(f"  [warn] no annotated internal nodes in: {tree_path.name}")
            continue
        data[key] = widths
    return data, counts


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# variant_key → deme → (indices, median, lower, upper)
PrevData = dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]
# variant_key → direction → (median, lower, upper)
MigSummary = dict[str, dict[str, tuple[float, float, float]]]
# variant_key → direction → raw MCMC samples
RawMig = dict[str, dict[str, np.ndarray]]
# variant_key → column name → raw MCMC samples
ParamData = dict[str, dict[str, np.ndarray]]


# ---------------------------------------------------------------------------
# Data loading across all variants
# ---------------------------------------------------------------------------


def load_all_prevalence(
    results_dir: Path,
    variant_prefix: str,
    demes: list[str],
    burnin_fraction: float,
) -> PrevData:
    """Load and summarise NeDynamics posteriors for all variants × demes."""
    data: PrevData = {}
    for key in VERSION_KEYS:
        run_dir = _run_dir(results_dir, variant_prefix, key)
        data[key] = {}
        for deme in demes:
            log_path = _nedynamics_log_path(run_dir, deme)
            if not log_path.exists():
                print(f"  [warn] missing NeDynamics log: {log_path.name}")
                continue
            df = apply_burnin(load_log(log_path), burnin_fraction)
            indices, samples = extract_i_columns(df)
            med, lo, hi = prevalence_summary(samples)
            data[key][deme] = (indices, med, lo, hi)
    return data


def load_all_migration(
    results_dir: Path,
    variant_prefix: str,
    burnin_fraction: float,
) -> tuple[MigSummary, RawMig]:
    """Load migration rate posteriors for all variants.

    Returns (summaries, raw_samples).
    """
    summaries: MigSummary = {}
    raw: RawMig = {}
    for key in VERSION_KEYS:
        run_dir = _run_dir(results_dir, variant_prefix, key)
        log_path = _combined_log_path(run_dir)
        if not log_path.exists():
            print(f"  [warn] missing combined log: {log_path.name}")
            summaries[key] = {}
            raw[key] = {}
            continue
        df = apply_burnin(load_log(log_path), burnin_fraction)
        mig = extract_migration_columns(df)
        raw[key] = mig
        summaries[key] = {
            direction: (float(np.median(s)), *hpd_bounds(s))
            for direction, s in mig.items()
        }
    return summaries, raw


# Datastream scaling / nuisance parameter columns to extract from combined logs.
PARAM_COLUMN_RE = re.compile(
    r"^(NeScaler\.Deme\d+|"
    r"caseCounts\.scaling\.Deme\d+:SimDataset|caseCounts\.dispersion:SimDataset|"
    r"seroprevalence\.scaling\.Deme\d+:SimDataset|"
    r"wastewater\.scaling\.Deme\d+:SimDataset|wastewater\.sigma:SimDataset)$"
)


def load_all_datastream_params(
    results_dir: Path,
    variant_prefix: str,
    burnin_fraction: float,
) -> ParamData:
    """Load datastream scaling / nuisance parameter posteriors for all variants.

    Only columns matching :data:`PARAM_COLUMN_RE` are kept. Variants that do not
    estimate a given parameter (e.g. seroprevalence scaling under a
    no-seroprevalence variant) simply omit that column.
    """
    data: ParamData = {}
    for key in VERSION_KEYS:
        run_dir = _run_dir(results_dir, variant_prefix, key)
        log_path = _combined_log_path(run_dir)
        if not log_path.exists():
            data[key] = {}
            continue
        df = apply_burnin(load_log(log_path), burnin_fraction)
        data[key] = {
            c: df[c].to_numpy(dtype=float)
            for c in df.columns
            if PARAM_COLUMN_RE.match(c)
        }
    return data


# ---------------------------------------------------------------------------
# Prior specifications (parsed from the reference variant's BEAST XML)
# ---------------------------------------------------------------------------

_Z975 = 1.959963984540054  # standard-normal 97.5th percentile


def _lognormal_stats(meanlog_or_mean: float, sdlog: float, mean_in_real_space: bool):
    """Summary stats of a LogNormal given BEAST's (M, S, meanInRealSpace).

    Returns (mean, median, q2.5, q97.5, relative_95_width) where the relative
    width = (q97.5 − q2.5) / median depends only on S.
    """
    if mean_in_real_space:
        mean = meanlog_or_mean
        meanlog = np.log(meanlog_or_mean) - sdlog * sdlog / 2.0
    else:
        meanlog = meanlog_or_mean
        mean = np.exp(meanlog + sdlog * sdlog / 2.0)
    median = np.exp(meanlog)
    q025 = np.exp(meanlog - sdlog * _Z975)
    q975 = np.exp(meanlog + sdlog * _Z975)
    return mean, median, q025, q975, (q975 - q025) / median


def load_prior_specs(xml_path: Path) -> dict[str, float]:
    """Parse LogNormal prior means from a BEAST XML.

    Returns prior_median_by_target, keyed by the parameter ids the priors apply to
    (the ``x="@..."`` reference, stripped of the leading ``@``). ``M`` references
    such as ``@NeScaler.MEAN.t:SimDataset`` are resolved to the referenced
    parameter's value.
    """
    means: dict[str, float] = {}
    medians: dict[str, float] = {}
    if not xml_path.exists():
        print(f"  [warn] reference XML not found, prior lines disabled: {xml_path}")
        return means
    root = ET.parse(str(xml_path)).getroot()

    def localname(elem) -> str:
        return elem.tag.split("}")[-1]

    def get_param(lognormal, name):
        if lognormal.get(name) is not None:
            return lognormal.get(name)
        for p in lognormal:
            if p.get("name") == name:
                return (p.text or "").strip()
        return None

    def resolve(val):
        if val is None:
            return None
        val = val.strip()
        if val.startswith("@"):
            el = root.find(f".//*[@id='{val[1:]}']")
            return float((el.text or "").strip()) if el is not None else None
        return float(val)

    for pr in root.iter():
        if localname(pr) != "prior" or not pr.get("x"):
            continue
        lognormal = next((c for c in pr if localname(c) == "LogNormal"), None)
        if lognormal is None:
            continue
        M = resolve(get_param(lognormal, "M"))
        S = resolve(get_param(lognormal, "S"))
        if M is None or S is None:
            continue
        mirs = lognormal.get("meanInRealSpace", "false") == "true"
        mean, median, _, _, _ = _lognormal_stats(M, S, mirs)
        means[pr.get("x").lstrip("@")] = mean
        medians[pr.get("x").lstrip("@")] = median
    return medians


def _prior_median_for(prior_medians: dict[str, float], colname: str) -> float | None:
    """Look up a column's prior median, tolerating the NeScaler ``.t:SimDataset`` id."""
    for cand in (colname, f"{colname}.t:SimDataset"):
        if cand in prior_medians:
            return prior_medians[cand]
    return None


# ---------------------------------------------------------------------------
# Calendar-time conversion for prevalence trajectories
# (mirrors analyse_posteriors.trajectory_indices_to_decimal_year)
# ---------------------------------------------------------------------------


def load_most_recent_sample_decimal_year(state_time_csv: Path) -> float | None:
    """Latest sampling time (decimal year) from a ``*_state_time.csv`` ('time' col)."""
    if not state_time_csv.is_file():
        return None
    df = pd.read_csv(state_time_csv)
    if "time" not in df.columns:
        return None
    return float(pd.to_numeric(df["time"], errors="coerce").max())


def _load_rate_shifts(xml_path: Path, tag: str, base_id: str) -> dict[str, np.ndarray]:
    """Parse per-deme rate-shift arrays (time before present) from a BEAST XML.

    Focal demes share *base_id*; the ghost deme uses the ``.<GHOST>`` suffix.
    *tag* is the element local name (``gridRateShifts`` or ``rateShifts``).
    """
    out: dict[str, np.ndarray] = {}
    if not xml_path.exists():
        return out
    root = ET.parse(str(xml_path)).getroot()
    base: np.ndarray | None = None
    for elem in root.iter():
        if elem.tag.split("}")[-1] != tag or not (elem.text or "").strip():
            continue
        eid = elem.get("id", "")
        vals = np.array([float(v) for v in elem.text.split()], dtype=float)
        if eid == base_id:
            base = vals
        elif eid == f"{base_id}.{GHOST_DEME_LABEL}":
            out[GHOST_DEME_LABEL] = vals
    if base is not None:
        for d in FOCAL_DEME_LABELS:
            out[d] = base
    return out


def load_grid_shifts(xml_path: Path) -> dict[str, np.ndarray]:
    """``SplineGridRateShifts`` (time-before-present per NeDynamics grid index)."""
    return _load_rate_shifts(xml_path, "gridRateShifts", "SplineGridRateShifts")


def load_sky_shifts(xml_path: Path) -> dict[str, np.ndarray]:
    """``SkygrowthRateShifts`` — the skyline/spline knot times (before present)."""
    return _load_rate_shifts(xml_path, "rateShifts", "SkygrowthRateShifts")


def _prev_x(
    deme: str,
    indices: np.ndarray,
    grid_shifts: dict[str, np.ndarray] | None,
    t_recent: float | None,
) -> tuple[np.ndarray, bool]:
    """Plotting x for a prevalence trajectory.

    Returns (x, is_date). When grid shifts and the most-recent sample time are
    available, x is matplotlib date numbers (index 0 = present); otherwise x is
    the raw grid index and is_date is False.
    """
    if grid_shifts and t_recent is not None and deme in grid_shifts:
        gs = grid_shifts[deme]
        if int(np.max(indices)) < len(gs):
            x_dec = t_recent - gs[np.asarray(indices, dtype=int)]
            return decimal_years_to_matplotlib_dates(x_dec), True
    return np.asarray(indices, dtype=float), False


def _format_date_axis(ax: plt.Axes) -> None:
    """Compact calendar-date ticks for a matplotlib-date x-axis."""
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _dates_to_decimal_year(dates: pd.Series) -> pd.Series:
    """Calendar dates → decimal year (mirrors
    analyse_posteriors.convert_date_to_numerical_date)."""
    dates = pd.to_datetime(dates)
    years = dates.dt.year
    year_start = pd.to_datetime(years.astype(str) + "-01-01")
    next_year_start = pd.to_datetime((years + 1).astype(str) + "-01-01")
    days_in_year = (next_year_start - year_start).dt.days
    return years + (dates.dt.dayofyear / days_in_year)


_DATASTREAM_CSVS = ("case_counts.csv", "wastewater.csv", "seroprevalence.csv")


def load_datastream_window(
    results_dir: Path, subdir: str = "datastreams_demes"
) -> tuple[float, float] | None:
    """Decimal-year span [min, max] over which the datastreams (case counts,
    wastewater, seroprevalence) are sampled — the same window for every variant.

    Read from ``<results_dir>/<subdir>/{case_counts,wastewater,seroprevalence}.csv``
    (the shared datastream tables). Returns None if none are found.
    """
    base = results_dir / subdir
    lo: float | None = None
    hi: float | None = None
    for fname in _DATASTREAM_CSVS:
        path = base / fname
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "date" not in df.columns or df.empty:
            continue
        dy = _dates_to_decimal_year(df["date"])
        lo = float(dy.min()) if lo is None else min(lo, float(dy.min()))
        hi = float(dy.max()) if hi is None else max(hi, float(dy.max()))
    if lo is None:
        return None
    return (lo, hi)


def _window_mask(
    x_plot: np.ndarray, is_date: bool, window: tuple[float, float] | None
) -> np.ndarray:
    """Boolean mask keeping trajectory points inside the datastream date window.

    Only applies in calendar-date mode; otherwise keeps all points.
    """
    if window is None or not is_date:
        return np.ones(len(x_plot), dtype=bool)
    lo_num, hi_num = decimal_years_to_matplotlib_dates(np.array(window, dtype=float))
    return (x_plot >= lo_num) & (x_plot <= hi_num)


# ---------------------------------------------------------------------------
# CSV sidecar helper
# ---------------------------------------------------------------------------


def _save_sidecar_csv(df: pd.DataFrame, png_path: Path) -> None:
    csv_path = png_path.with_name(f"{png_path.stem}_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Data → {csv_path.name}")


# ---------------------------------------------------------------------------
# Shared axis styling
# ---------------------------------------------------------------------------


# Line weights for the publication figures — deliberately heavy so the overlaid
# variant traces stay readable at print size. Series lines and the migration
# dot-with-HPD whiskers share these; thin guide lines (zero line, prior median,
# knot markers) keep their own hairline weights.
_SERIES_LW = 3.4  # prevalence median / HPD-width traces + legend proxies
_WHISKER_LW = 3.0  # migration dot-with-HPD whiskers and parameter markers


def _style_ax(ax: plt.Axes, title: str, xlabel, ylabel) -> None:
    ax.set_title(title, fontsize=FONTSIZES_LIST[0])
    set_axis_fontsizes(ax, FONTSIZES_LIST, xlabel=xlabel, ylabel=ylabel)
    ax.tick_params(labelsize=FONTSIZES_LIST[2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Panel letter labels (A, B, …) — mirrors analyse_posteriors.add_panel_label.
# The letter sits at the top-left of each axes' grid cell: horizontally just
# left of the y-axis label / tick labels (so the y-axis title reads to its
# right), vertically just above the cell's top edge (so the panel title reads
# below it). Placing labels in figure coordinates keeps a row's letters on a
# shared baseline regardless of individual panel heights.
# ---------------------------------------------------------------------------
_PANEL_LABEL_DX = 0.006  # shift left of the panel's left-most extent (fig frac)
_PANEL_LABEL_DY = 0.030  # shift above the cell's top edge (fig frac)


def _add_panel_label(ax: plt.Axes, label: str, renderer=None) -> None:
    """Draw one bold panel letter above the top-left of *ax*'s grid cell."""
    if not label:
        return
    fig = ax.get_figure()
    ss = ax.get_subplotspec()
    cell = ss.get_position(fig) if ss is not None else ax.get_position()
    x_left = cell.x0
    y_top = cell.y1
    if renderer is not None:
        try:
            yb = ax.yaxis.get_tightbbox(renderer)
            if yb is not None and yb.width > 0:
                x_left = min(x_left, yb.transformed(fig.transFigure.inverted()).x0)
        except Exception:
            pass
    fig.text(
        x_left - _PANEL_LABEL_DX,
        y_top + _PANEL_LABEL_DY,
        label,
        fontsize=FONTSIZES_LIST[0] + 2,  # ~2 pt larger than the title size
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _add_panel_labels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    """Label *axes* A, B, C, … in the given order (typically column-major).

    Requires a laid-out figure, so call after any ``tight_layout``/final axis
    formatting: a canvas draw supplies the renderer used to clear each y-axis.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, letter in zip(axes, string.ascii_uppercase):
        _add_panel_label(ax, letter, renderer)


def _legend_handles(versions: list[tuple]) -> list[Line2D]:
    """Line proxies for a shared variant legend (style mirrors the panels)."""
    return [
        Line2D([0], [0], color=c, lw=_SERIES_LW, ls=_ls(k), label=lbl)
        for k, lbl, c in versions
    ]


def _annotate_zero_panel(ax: plt.Axes) -> None:
    """Grey out an axis and label it as model-fixed zero."""
    ax.set_facecolor("#f2f2f2")
    ax.text(
        0.5,
        0.5,
        "rate fixed\nto 0",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=FONTSIZES_LIST[2],
        color="0.5",
        style="italic",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---------------------------------------------------------------------------
# Panel primitives — each draws onto a supplied Axes for a single deme /
# direction / parameter, given an explicit list of (key, label, color) versions.
# ---------------------------------------------------------------------------

_PREV_XLABEL_INDEX = "Time grid index (past → present)"
_PREV_XLABEL_DATE = "Date"

# Toggle: draw the skyline/spline knot times (SkygrowthRateShifts) as dashed
# vertical lines on the prevalence HPD-width panels. Set True to bring them back.
SHOW_PREV_HPD_KNOTS = False


def _finish_prev_axis(ax: plt.Axes, is_date: bool) -> None:
    """Order/format the prevalence x-axis (calendar dates or reversed index)."""
    if is_date:
        _format_date_axis(ax)  # dates already ascend left→right (past → present)
    else:
        ax.invert_xaxis()  # index 0 = present on the right


def _panel_prev_diff(
    ax: plt.Axes,
    prev_data: PrevData,
    deme: str,
    versions: list[tuple],
    records: list[dict],
    tag: str,
    grid_shifts: dict[str, np.ndarray] | None = None,
    t_recent: float | None = None,
    window: tuple[float, float] | None = None,
) -> None:
    """Median log-prevalence difference from the All-DS reference (zero line)."""
    ax.axhline(
        0.0,
        color=PUB_COLOR_BY_KEY[REFERENCE_KEY],
        lw=1.1,
        ls="--",
        zorder=0,
    )
    ref = prev_data.get(REFERENCE_KEY, {}).get(deme)
    if ref is None:
        ax.text(0.5, 0.5, "reference missing", transform=ax.transAxes, ha="center")
        return
    _, ref_med, _, _ = ref
    is_date = False
    for key, label, color in versions:
        if key == REFERENCE_KEY:
            continue  # reference is the zero line
        entry = prev_data.get(key, {}).get(deme)
        if entry is None or len(entry[1]) != len(ref_med):
            continue
        indices, med, _, _ = entry
        x, is_date = _prev_x(deme, indices, grid_shifts, t_recent)
        diff = med - ref_med
        m = _window_mask(x, is_date, window)
        ax.plot(x[m], diff[m], color=color, lw=_SERIES_LW, alpha=0.75, ls=_ls(key))
        for idx, d in zip(np.asarray(indices)[m], diff[m]):
            records.append(
                {
                    "panel": f"prev_diff:{deme}:{tag}",
                    "version": label,
                    "grid_index": int(idx),
                    "value": float(d),
                }
            )
    _finish_prev_axis(ax, is_date)


def _panel_prev_hpd_width(
    ax: plt.Axes,
    prev_data: PrevData,
    deme: str,
    versions: list[tuple],
    records: list[dict],
    tag: str,
    grid_shifts: dict[str, np.ndarray] | None = None,
    t_recent: float | None = None,
    window: tuple[float, float] | None = None,
    sky_shifts: dict[str, np.ndarray] | None = None,
) -> None:
    """95% HPD width of log-prevalence over the time grid.

    When *sky_shifts* (SkygrowthRateShifts) are supplied, the skyline/spline knot
    times are drawn as dashed vertical lines (converted to forward calendar time).
    """
    is_date = False
    for key, label, color in versions:
        entry = prev_data.get(key, {}).get(deme)
        if entry is None:
            continue
        indices, _, lo, hi = entry
        x, is_date = _prev_x(deme, indices, grid_shifts, t_recent)
        width = hi - lo
        m = _window_mask(x, is_date, window)
        ax.plot(x[m], width[m], color=color, lw=_SERIES_LW, alpha=0.75, ls=_ls(key))
        for idx, w in zip(np.asarray(indices)[m], width[m]):
            records.append(
                {
                    "panel": f"prev_hpd:{deme}:{tag}",
                    "version": label,
                    "grid_index": int(idx),
                    "value": float(w),
                }
            )

    # Spline knots (SkygrowthRateShifts) as dashed vertical lines in forward time.
    # Toggle via SHOW_PREV_HPD_KNOTS.
    if (
        SHOW_PREV_HPD_KNOTS
        and sky_shifts
        and is_date
        and t_recent is not None
        and deme in sky_shifts
    ):
        knot_x = decimal_years_to_matplotlib_dates(t_recent - sky_shifts[deme])
        km = _window_mask(knot_x, True, window)
        for xv in np.asarray(knot_x)[km]:
            ax.axvline(xv, color="0.55", lw=0.7, ls=(0, (4, 3)), zorder=0)

    _finish_prev_axis(ax, is_date)


def _panel_migration_dotwhisker(
    ax: plt.Axes,
    raw_mig: RawMig,
    direction: str,
    versions: list[tuple],
    records: list[dict],
    show_xticklabels: bool = True,
) -> None:
    """Migration-events posterior as a median dot with a vertical line spanning
    the 95% HPD, one marker per variant."""
    positions = np.arange(len(versions), dtype=float)
    for pos, (key, label, color) in zip(positions, versions):
        samples = raw_mig.get(key, {}).get(direction)
        if samples is None or len(samples) == 0:
            records.append(
                {
                    "panel": f"mig_events:{direction}",
                    "version": label,
                    "median": np.nan,
                    "hpd_lower": np.nan,
                    "hpd_upper": np.nan,
                }
            )
            continue
        med = float(np.median(samples))
        lo, hi = hpd_bounds(samples)
        ax.plot(
            [pos, pos],
            [lo, hi],
            color=color,
            lw=_WHISKER_LW,
            alpha=0.85,
            ls=_ls(key),
            zorder=2,
        )
        ax.plot(
            [pos],
            [med],
            marker="o",
            ms=6.5,
            color=color,
            mec="0.2",
            mew=0.7,
            zorder=3,
        )
        records.append(
            {
                "panel": f"mig_events:{direction}",
                "version": label,
                "median": med,
                "hpd_lower": lo,
                "hpd_upper": hi,
            }
        )

    ref = raw_mig.get(REFERENCE_KEY, {}).get(direction)
    if ref is not None and len(ref) > 0:
        ax.axhline(
            float(np.median(ref)),
            color=PUB_COLOR_BY_KEY[REFERENCE_KEY],
            lw=0.9,
            ls="--",
            zorder=0,
            alpha=0.6,
        )

    ax.set_xticks(positions)
    if show_xticklabels:
        ax.set_xticklabels(
            [lbl for _, lbl, _ in versions],
            rotation=40,
            ha="right",
            fontsize=FONTSIZES_LIST[2] - 1,
        )
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.7, len(versions) - 0.3)
    ax.set_ylim(bottom=0)


def _panel_migration_hpd(
    ax: plt.Axes,
    mig_summary: MigSummary,
    direction: str,
    versions: list[tuple],
    records: list[dict],
    show_xticklabels: bool = True,
) -> None:
    """Relative 95% HPD width (= HPD width / median) bars, one per variant."""
    positions = np.arange(len(versions), dtype=float)
    for pos, (key, label, color) in zip(positions, versions):
        entry = mig_summary.get(key, {}).get(direction)
        if entry is not None:
            med, lo, hi = entry
            rel = (hi - lo) / med if med > 0 else np.nan
        else:
            rel = np.nan
        ax.bar(
            pos,
            rel if not np.isnan(rel) else 0,
            width=0.7,
            color=color,
            edgecolor="0.25",
            linewidth=0.6,
            hatch=_hatch(key),
        )
        records.append(
            {
                "panel": f"mig_hpd:{direction}",
                "version": label,
                "value": float(rel) if not np.isnan(rel) else np.nan,
            }
        )

    ref_entry = mig_summary.get(REFERENCE_KEY, {}).get(direction)
    if ref_entry is not None:
        ref_med, ref_lo, ref_hi = ref_entry
        ref_rel = (ref_hi - ref_lo) / ref_med if ref_med > 0 else np.nan
        if not np.isnan(ref_rel):
            ax.axhline(
                ref_rel,
                color=PUB_COLOR_BY_KEY[REFERENCE_KEY],
                lw=1.4,
                ls="--",
                zorder=0,
                alpha=0.9,
            )

    ax.set_xticks(positions)
    if show_xticklabels:
        ax.set_xticklabels(
            [lbl for _, lbl, _ in versions],
            rotation=40,
            ha="right",
            fontsize=FONTSIZES_LIST[2] - 1,
        )
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.7, len(versions) - 0.3)
    ax.set_ylim(bottom=0)


def _param_data_range(
    params: ParamData, colname: str, exclude_keys: frozenset
) -> tuple[float, float] | None:
    """1st–99th percentile range of *colname* across the variants that have it."""
    arrs = [
        params[key][colname]
        for key, _, _ in PUB_VERSIONS
        if key not in exclude_keys and colname in params.get(key, {})
    ]
    if not arrs:
        return None
    return (
        min(float(np.percentile(a, 1)) for a in arrs),
        max(float(np.percentile(a, 99)) for a in arrs),
    )


def _draw_param_markers(
    ax: plt.Axes,
    params: ParamData,
    colname: str,
    records: list[dict],
    exclude_keys: frozenset,
    show_xticklabels: bool,
) -> None:
    """Median dot with a vertical line spanning the 95% HPD of *colname*, one
    marker per variant at fixed (shared) x positions."""
    drawn = [
        (i, key, label, color)
        for i, (key, label, color) in enumerate(PUB_VERSIONS)
        if key not in exclude_keys and colname in params.get(key, {})
    ]
    for pos, key, label, color in drawn:
        s = params[key][colname]
        med = float(np.median(s))
        lo, hi = hpd_bounds(s)
        ax.plot(
            [pos, pos],
            [lo, hi],
            color=color,
            lw=_WHISKER_LW,
            alpha=0.85,
            ls=_ls(key),
            zorder=2,
        )
        ax.plot(
            [pos],
            [med],
            marker="o",
            ms=6.5,
            color=color,
            mec="0.2",
            mew=0.7,
            zorder=3,
        )
        records.append(
            {
                "parameter": colname,
                "version": label,
                "median": med,
                "hpd_lower": lo,
                "hpd_upper": hi,
            }
        )

    # Tick every variant position (shared axis); labels written once (bottom row).
    ax.set_xticks(np.arange(len(PUB_VERSIONS)))
    if show_xticklabels:
        ax.set_xticklabels(
            [lbl for _, lbl, _ in PUB_VERSIONS],
            rotation=40,
            ha="right",
            fontsize=FONTSIZES_LIST[2] - 1,
        )
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.7, len(PUB_VERSIONS) - 0.3)


def _add_break_marks(ax_top: plt.Axes, ax_bot: plt.Axes) -> None:
    """Diagonal // break marks across the gap of a broken y-axis pair."""
    kwargs = dict(
        marker=[(-1, -0.5), (1, 0.5)],
        markersize=7,
        linestyle="none",
        color="0.2",
        mec="0.2",
        mew=1,
        clip_on=False,
    )
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kwargs)


def _param_panel(
    fig: plt.Figure,
    spec,
    params: ParamData,
    colname: str,
    records: list[dict],
    exclude_keys: frozenset,
    show_xticklabels: bool,
    prior_median: float | None,
    title: str,
    ylabel: str,
) -> None:
    """Draw one datastream-parameter panel into grid cell *spec*.

    When *prior_median* lies near the posterior it is a dashed grey line. When it
    lies far outside, the cell is split into a broken y-axis: a small upper
    segment carrying the prior line and the lower segment carrying the boxes,
    so the (much smaller) estimates stay readable.
    """
    rng = _param_data_range(params, colname, exclude_keys)
    if rng is None:
        ax = fig.add_subplot(spec)
        ax.set_visible(False)
        return
    data_lo, data_hi = rng
    span = (data_hi - data_lo) or (abs(data_hi) * 0.1 or 1.0)
    far = prior_median is not None and not (
        data_lo - span <= prior_median <= data_hi + span
    )

    if not far:
        ax = fig.add_subplot(spec)
        _draw_param_markers(
            ax, params, colname, records, exclude_keys, show_xticklabels
        )
        if prior_median is not None:
            ax.axhline(prior_median, color="0.45", lw=1.1, ls="--", zorder=3)
            lo, hi = min(data_lo, prior_median), max(data_hi, prior_median)
            pad = (hi - lo) * 0.08 or 1.0
            ax.set_ylim(lo - pad, hi + pad)
        else:
            pad = span * 0.08
            ax.set_ylim(data_lo - pad, data_hi + pad)
        _style_ax(ax, title, None, ylabel)
        return

    # Broken y-axis: small top segment for the prior line, large bottom for markers.
    sub = spec.subgridspec(2, 1, height_ratios=[1, 5], hspace=0.12)
    ax_top = fig.add_subplot(sub[0])
    ax_bot = fig.add_subplot(sub[1])

    _draw_param_markers(
        ax_bot, params, colname, records, exclude_keys, show_xticklabels
    )
    pad = span * 0.12
    ax_bot.set_ylim(data_lo - pad, data_hi + pad)

    ax_top.set_xlim(ax_bot.get_xlim())
    ax_top.set_xticks([])
    ax_top.axhline(prior_median, color="0.45", lw=1.1, ls="--", zorder=3)
    half = max(abs(prior_median) * 0.04, span * 0.5)
    ax_top.set_ylim(prior_median - half, prior_median + half)
    ax_top.set_yticks([prior_median])
    ax_top.set_yticklabels([f"{prior_median:.3g}"], fontsize=FONTSIZES_LIST[2])

    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    _add_break_marks(ax_top, ax_bot)

    ax_top.set_title(title, fontsize=FONTSIZES_LIST[0])
    set_axis_fontsizes(ax_bot, FONTSIZES_LIST, xlabel=None, ylabel=ylabel)
    ax_bot.tick_params(labelsize=FONTSIZES_LIST[2])


# ---------------------------------------------------------------------------
# Main figure — 2×3 GridSpec, Santa Clara worked example.
# ---------------------------------------------------------------------------


def plot_main_figure(
    prev_data: PrevData,
    mig_summary: MigSummary,
    raw_mig: RawMig,
    output_png: Path,
    grid_shifts: dict[str, np.ndarray] | None = None,
    t_recent: float | None = None,
    window: tuple[float, float] | None = None,
    sky_shifts: dict[str, np.ndarray] | None = None,
) -> None:
    configure_pdf_fonts()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    deme = EXAMPLE_DEME
    name = DEME_MAP.get(deme, deme)
    records: list[dict] = []
    local_raw, local_summary = build_aggregate_migration(
        raw_mig, LOCAL_MIG_DIRECTIONS, LOCAL_AGG_KEY
    )
    prev_xlabel = (
        _PREV_XLABEL_DATE if (grid_shifts and t_recent) else _PREV_XLABEL_INDEX
    )

    fig = plt.figure(figsize=(11.5, 7.5))
    gs = fig.add_gridspec(
        2,
        3,
        hspace=0.06,
        wspace=0.25,
        left=0.065,
        right=0.985,
        top=0.93,
        bottom=0.15,
    )

    diff_ylabel = "Log prev. bias relative to 'All DS'"
    width_ylabel = "Log prev. 95% HPDI width"

    # Titles label the column (variant group / direction) on the top row only;
    # the metric lives on the y-axis, so bottom-row panels carry no title.

    # Column 0 — main ablations (no seroprevalence variants).
    ax = fig.add_subplot(gs[0, 0])
    _panel_prev_diff(
        ax,
        prev_data,
        deme,
        MAIN_VERSIONS,
        records,
        "main",
        grid_shifts,
        t_recent,
        window,
    )
    _style_ax(ax, name, None, diff_ylabel)
    ax.tick_params(labelbottom=False)

    ax = fig.add_subplot(gs[1, 0])
    _panel_prev_hpd_width(
        ax,
        prev_data,
        deme,
        MAIN_VERSIONS,
        records,
        "main",
        grid_shifts,
        t_recent,
        window,
        sky_shifts,
    )
    _style_ax(ax, "", prev_xlabel, width_ylabel)

    # Column 1 — the three seroprevalence variants.
    ax = fig.add_subplot(gs[0, 1])
    _panel_prev_diff(
        ax,
        prev_data,
        deme,
        SEROPREV_VERSIONS,
        records,
        "seroprev",
        grid_shifts,
        t_recent,
        window,
    )
    _style_ax(ax, f"{name} - no SP versions", None, diff_ylabel)
    ax.tick_params(labelbottom=False)

    ax = fig.add_subplot(gs[1, 1])
    _panel_prev_hpd_width(
        ax,
        prev_data,
        deme,
        [_ALLDS_TRIPLE] + SEROPREV_VERSIONS,
        records,
        "seroprev",
        grid_shifts,
        t_recent,
        window,
        sky_shifts,
    )
    _style_ax(ax, "", prev_xlabel, width_ylabel)

    # Column 2 — total local migration events (summed over focal directions).
    # No phylogeny is excluded (see MIGRATION_VERSIONS).
    ax = fig.add_subplot(gs[0, 2])
    _panel_migration_dotwhisker(
        ax,
        local_raw,
        LOCAL_AGG_KEY,
        MIGRATION_VERSIONS,
        records,
        show_xticklabels=False,
    )
    _style_ax(ax, "Total local migration events", None, "Migration events")

    ax = fig.add_subplot(gs[1, 2])
    _panel_migration_hpd(ax, local_summary, LOCAL_AGG_KEY, MIGRATION_VERSIONS, records)
    _style_ax(ax, "", None, "Migration events\n95% relative HPDI width")

    fig.legend(
        handles=_legend_handles(PUB_VERSIONS),
        loc="lower center",
        ncol=8,
        frameon=False,
        fontsize=FONTSIZES_LIST[2],
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.2,
        handlelength=1.6,
    )
    # Panels created column-major (top,bottom per column), so the top-row axes
    # (one per column) sit at indices 0, 2, 4; only those get A, B, C labels.
    top_row_axes = list(fig.axes)[0::2]
    _add_panel_labels(fig, top_row_axes)
    save_figure_png_and_pdf(output_png)
    plt.close(fig)
    _save_sidecar_csv(pd.DataFrame(records), output_png)


# ---------------------------------------------------------------------------
# Supplementary figure 1 — prevalence for the non-example counties.
# ---------------------------------------------------------------------------


def plot_supp_prevalence(
    prev_data: PrevData,
    raw_mig: RawMig,
    output_png: Path,
    grid_shifts: dict[str, np.ndarray] | None = None,
    t_recent: float | None = None,
    window: tuple[float, float] | None = None,
    sky_shifts: dict[str, np.ndarray] | None = None,
) -> None:
    """Prevalence diff + HPD width for the non-example counties (Sacramento, San
    Francisco), plus a third column with the total migration-from-background
    posteriors (top) and relative HPD width (bottom)."""
    configure_pdf_fonts()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    demes = [d for d in FOCAL_DEME_LABELS if d != EXAMPLE_DEME]
    records: list[dict] = []
    prev_xlabel = (
        _PREV_XLABEL_DATE if (grid_shifts and t_recent) else _PREV_XLABEL_INDEX
    )

    bg_raw, bg_summary = build_aggregate_migration(
        raw_mig, BACKGROUND_MIG_DIRECTIONS, BACKGROUND_AGG_KEY
    )

    ncol = len(demes) + 1
    fig, axes = plt.subplots(2, ncol, figsize=(5.0 * ncol, 8.4), squeeze=False)
    for j, deme in enumerate(demes):
        name = DEME_MAP.get(deme, deme)
        # County name is the only title (the metric is on the y-axis); the two
        # rows share the x-axis, so only the bottom row is labelled.
        _panel_prev_diff(
            axes[0][j],
            prev_data,
            deme,
            PUB_VERSIONS,
            records,
            "supp",
            grid_shifts,
            t_recent,
            window,
        )
        _style_ax(axes[0][j], name, None, "Log prev. relative to 'All DS'")
        axes[0][j].tick_params(labelbottom=False)
        _panel_prev_hpd_width(
            axes[1][j],
            prev_data,
            deme,
            PUB_VERSIONS,
            records,
            "supp",
            grid_shifts,
            t_recent,
            window,
            sky_shifts,
        )
        _style_ax(axes[1][j], "", prev_xlabel, "Log prev. 95% HPDI width")

    # Third column — total migration from the background (Outside → focal) deme.
    # No phylogeny is excluded (see MIGRATION_VERSIONS).
    jb = len(demes)
    _panel_migration_dotwhisker(
        axes[0][jb],
        bg_raw,
        BACKGROUND_AGG_KEY,
        MIGRATION_VERSIONS,
        records,
        show_xticklabels=False,
    )
    _style_ax(axes[0][jb], "Total migration from background", None, "Migration events")
    _panel_migration_hpd(
        axes[1][jb],
        bg_summary,
        BACKGROUND_AGG_KEY,
        MIGRATION_VERSIONS,
        records,
        show_xticklabels=True,
    )
    _style_ax(axes[1][jb], "", None, "Migration events\n95% relative HPDI width")

    fig.legend(
        handles=_legend_handles(PUB_VERSIONS),
        loc="lower center",
        ncol=8,
        frameon=False,
        fontsize=FONTSIZES_LIST[2],
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.2,
        handlelength=1.6,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    # One label per column, on the top row only.
    top_row_axes = [axes[0][j] for j in range(ncol)]
    _add_panel_labels(fig, top_row_axes)
    save_figure_png_and_pdf(output_png)
    plt.close(fig)
    _save_sidecar_csv(pd.DataFrame(records), output_png)


# ---------------------------------------------------------------------------
# Supplementary figures 2a / 2b — migration, all estimated directions.
# ---------------------------------------------------------------------------


def _migration_grid(
    directions: list[str],
    output_png: Path,
    draw_panel,
    ylabel: str,
    versions: list[tuple],
) -> None:
    """Shared layout for a per-direction migration grid (events or HPD width).

    *draw_panel(ax, direction, records, show_xticklabels)* draws one direction.
    Directions fixed to 0 are assumed already filtered out by the caller.
    *versions* drives the legend (and must match what *draw_panel* actually draws).
    """
    configure_pdf_fonts()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    n = len(directions)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    records: list[dict] = []

    # Variant axis is written once per column, on the figure's bottom row.
    def bottom_row(i: int) -> bool:
        return i + ncol >= n

    fig = plt.figure(figsize=(4.6 * ncol, 3.5 * nrow + 1.4))
    gs = fig.add_gridspec(
        nrow,
        ncol,
        hspace=0.45,
        wspace=0.40,
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.13,
    )

    for i, direction in enumerate(directions):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        draw_panel(ax, direction, records, bottom_row(i))
        _style_ax(ax, _dir_title(direction), None, ylabel)

    fig.legend(
        handles=_legend_handles(versions),
        loc="lower center",
        ncol=8,
        frameon=False,
        fontsize=FONTSIZES_LIST[2],
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.2,
        handlelength=1.6,
    )
    save_figure_png_and_pdf(output_png)
    plt.close(fig)
    _save_sidecar_csv(pd.DataFrame(records), output_png)


def plot_supp_migration_events(
    raw_mig: RawMig,
    directions: list[str],
    output_png: Path,
) -> None:
    """Migration-events posteriors (dot + 95% HPD whiskers) for every estimated
    direction. Directions fixed to 0 are omitted. No phylogeny is excluded
    (see MIGRATION_VERSIONS)."""
    _migration_grid(
        directions,
        output_png,
        lambda ax, direction, records, show: _panel_migration_dotwhisker(
            ax, raw_mig, direction, MIGRATION_VERSIONS, records, show_xticklabels=show
        ),
        "Migration events",
        MIGRATION_VERSIONS,
    )


def plot_supp_migration_hpd(
    mig_summary: MigSummary,
    directions: list[str],
    output_png: Path,
) -> None:
    """Relative 95% HPD width (events / median) for every estimated direction.
    No phylogeny is excluded (see MIGRATION_VERSIONS)."""
    _migration_grid(
        directions,
        output_png,
        lambda ax, direction, records, show: _panel_migration_hpd(
            ax,
            mig_summary,
            direction,
            MIGRATION_VERSIONS,
            records,
            show_xticklabels=show,
        ),
        "Migration events\n95% relative HPDI width",
        MIGRATION_VERSIONS,
    )


# ---------------------------------------------------------------------------
# Supplementary figure 3 — datastream scaling / nuisance parameters.
# ---------------------------------------------------------------------------

# Each family becomes a row: a per-deme scaling parameter plus an optional
# scalar nuisance parameter (dispersion / sigma) that gets its own y-axis label.
DATASTREAM_FAMILIES: list[dict] = [
    {
        "label": "Ne scaler",
        "tmpl": "NeScaler.{deme}",
        "demes": list(ALL_DEME_LABELS),
        "scalar": None,
        "scalar_title": None,
        "scalar_ylabel": None,
        # The MASCOT log-likelihood is removed in this variant, so its Ne scaler
        # is meaningless here.
        "exclude": frozenset({"datastreams_nomascotll"}),
    },
    {
        "label": "Case-count scaling",
        "tmpl": "caseCounts.scaling.{deme}:SimDataset",
        "demes": list(FOCAL_DEME_LABELS),
        "scalar": "caseCounts.dispersion:SimDataset",
        "scalar_title": "dispersion",
        "scalar_ylabel": "Case-count\ndispersion",
        "exclude": frozenset(),
    },
    {
        "label": "Seroprevalence scaling",
        "tmpl": "seroprevalence.scaling.{deme}:SimDataset",
        "demes": list(FOCAL_DEME_LABELS),
        "scalar": None,
        "scalar_title": None,
        "scalar_ylabel": None,
        "exclude": frozenset(),
    },
    {
        "label": "Wastewater scaling",
        "tmpl": "wastewater.scaling.{deme}:SimDataset",
        "demes": list(FOCAL_DEME_LABELS),
        "scalar": "wastewater.sigma:SimDataset",
        "scalar_title": "σ",
        "scalar_ylabel": "Wastewater σ",
        "exclude": frozenset(),
    },
]


def _param_is_informative(params: ParamData, colname: str, tol: float = 1e-9) -> bool:
    """True if at least one variant estimates *colname* (posterior std > tol).

    Parameters that are fixed constants in every variant (e.g. seroprevalence
    scaling, pinned to 1.0) carry no value-of-information content and are dropped.
    """
    for key, _, _ in PUB_VERSIONS:
        s = params.get(key, {}).get(colname)
        if s is not None and len(s) > 1 and float(np.std(s)) > tol:
            return True
    return False


def plot_supp_datastream_params(
    params: ParamData,
    output_png: Path,
    prior_medians: dict[str, float] | None = None,
) -> None:
    configure_pdf_fonts()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    prior_medians = prior_medians or {}

    # Build per-family column lists (deme scalers + optional scalar), keeping only
    # informative columns and dropping families fixed in every variant.
    kept: list[dict] = []
    for fam in DATASTREAM_FAMILIES:
        cols: list[dict] = [
            {
                "col": fam["tmpl"].format(deme=d),
                "title": DEME_MAP.get(d, d),
                "ylabel": None,
            }
            for d in fam["demes"]
        ]
        if fam["scalar"]:
            cols.append(
                {
                    "col": fam["scalar"],
                    "title": fam["scalar_title"],
                    "ylabel": fam["scalar_ylabel"],
                }
            )
        cols = [c for c in cols if _param_is_informative(params, c["col"])]
        if cols:
            kept.append(
                {"label": fam["label"], "exclude": fam["exclude"], "cols": cols}
            )
        else:
            print(f"  [note] '{fam['label']}' fixed in all variants — omitted.")

    nrow = len(kept)
    ncol = max(len(fam["cols"]) for fam in kept)
    records: list[dict] = []

    fig = plt.figure(figsize=(4.4 * ncol, 3.7 * nrow + 0.6))
    gs = fig.add_gridspec(
        nrow,
        ncol,
        left=0.075,
        right=0.99,
        top=0.95,
        bottom=0.14,
        hspace=0.45,
        wspace=0.32,
    )

    for r, fam in enumerate(kept):
        is_bottom = r == nrow - 1
        for c in range(ncol):
            if c >= len(fam["cols"]):
                fig.add_subplot(gs[r, c]).set_visible(False)
                continue
            spec = fam["cols"][c]
            is_scalar = spec["ylabel"] is not None
            # Title: county name only on the top row of a deme column (the county
            # is shared down the column); scalar columns rely on their y-label.
            title = spec["title"] if (r == 0 and not is_scalar) else ""
            # y-axis label: family name on the first column; scalars get their own.
            if is_scalar:
                ylabel = spec["ylabel"]
            elif c == 0:
                ylabel = fam["label"]
            else:
                ylabel = ""
            _param_panel(
                fig,
                gs[r, c],
                params,
                spec["col"],
                records,
                exclude_keys=fam["exclude"],
                show_xticklabels=is_bottom,
                prior_median=_prior_median_for(prior_medians, spec["col"]),
                title=title,
                ylabel=ylabel,
            )

    handles = _legend_handles(PUB_VERSIONS)
    handles.append(
        Line2D([0], [0], color="0.45", lw=1.1, ls="--", label="Prior median")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=FONTSIZES_LIST[2],
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.2,
        handlelength=1.6,
    )
    save_figure_png_and_pdf(output_png)
    plt.close(fig)
    _save_sidecar_csv(pd.DataFrame(records), output_png)


# ---------------------------------------------------------------------------
# Standalone figure — relative node-height HPD width, MCC trees.
# ---------------------------------------------------------------------------


def plot_node_hpd_boxplot(
    node_hpd_widths: dict[str, np.ndarray],
    output_png: Path,
) -> None:
    """Boxplot of internal-node relative height-HPD width, one box per variant
    present in *node_hpd_widths* (variants with a missing/unannotated MCC tree
    are simply omitted from the x-axis)."""
    configure_pdf_fonts()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    versions = [v for v in PUB_VERSIONS if v[0] in node_hpd_widths]
    positions = np.arange(len(versions), dtype=float)

    fig, ax = plt.subplots(figsize=(1.1 * len(versions) + 2.0, 5.0))
    ref_widths = node_hpd_widths.get(REFERENCE_KEY)
    if ref_widths is not None and len(ref_widths) > 0:
        ax.axhline(
            float(np.median(ref_widths)),
            color=PUB_COLOR_BY_KEY[REFERENCE_KEY],
            lw=1.4,
            ls="--",
            zorder=0,
            alpha=0.9,
        )
    bp = ax.boxplot(
        [node_hpd_widths[key] for key, _, _ in versions],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="0.15", lw=1.6),
        boxprops=dict(edgecolor="0.25", linewidth=0.8),
        whiskerprops=dict(color="0.25", linewidth=0.8),
        capprops=dict(color="0.25", linewidth=0.8),
    )
    for patch, (key, _, color) in zip(bp["boxes"], versions):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        hatch = _hatch(key)
        if hatch:
            patch.set_hatch(hatch)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{lbl}\n(n={len(node_hpd_widths[key])})" for key, lbl, _ in versions],
        rotation=40,
        ha="right",
        fontsize=FONTSIZES_LIST[2],
    )
    ax.set_xlim(-0.7, len(versions) - 0.3)
    ax.set_ylim(bottom=0)
    _style_ax(
        ax,
        None,
        None,
        "Node height\n95% relative HPDI width",
    )
    save_figure_png_and_pdf(output_png)
    plt.close(fig)

    records = [
        {"version": lbl, "relative_hpd_width": float(w)}
        for key, lbl, _ in versions
        for w in node_hpd_widths[key]
    ]
    _save_sidecar_csv(pd.DataFrame(records), output_png)


def plot_node_low_support_counts(
    node_support_counts: dict[str, tuple[int, int]],
    output_png: Path,
) -> None:
    """Bar plot of the number of internal nodes per variant whose MCC-tree
    clade support was too low for treeannotator to report a height HPD/median
    (see ``extract_internal_node_relative_hpd_widths``).

    These are exactly the nodes excluded from ``plot_node_hpd_boxplot``, so
    the two figures explain each other: a variant with more low-support
    clades here ends up with fewer boxplot data points there.
    """
    configure_pdf_fonts()
    output_png.parent.mkdir(parents=True, exist_ok=True)

    versions = [v for v in PUB_VERSIONS if v[0] in node_support_counts]
    positions = np.arange(len(versions), dtype=float)
    n_low_support = [node_support_counts[key][0] for key, _, _ in versions]

    fig, ax = plt.subplots(figsize=(1.1 * len(versions) + 2.0, 5.0))
    bars = ax.bar(
        positions,
        n_low_support,
        width=0.6,
        edgecolor="0.25",
        linewidth=0.8,
    )
    for bar, (key, _, color) in zip(bars, versions):
        bar.set_facecolor(color)
        bar.set_alpha(0.75)
        hatch = _hatch(key)
        if hatch:
            bar.set_hatch(hatch)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [lbl for _, lbl, _ in versions],
        rotation=40,
        ha="right",
        fontsize=FONTSIZES_LIST[2],
    )
    ax.set_xlim(-0.7, len(versions) - 0.3)
    ax.set_ylim(bottom=0)
    _style_ax(
        ax,
        None,
        None,
        "Low-support internal nodes\n(no height HPD reported)",
    )
    save_figure_png_and_pdf(output_png)
    plt.close(fig)

    records = [
        {
            "version": lbl,
            "n_low_support": node_support_counts[key][0],
            "n_total_internal_nodes": node_support_counts[key][1],
        }
        for key, lbl, _ in versions
    ]
    _save_sidecar_csv(pd.DataFrame(records), output_png)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run(
    results_dir: Path,
    variant_prefix: str,
    focal_demes: list[str],
    include_ghost: bool,
    output_dir: Path,
    burnin_fraction: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prevalence is loaded for every focal deme (the supplement needs the
    # non-example counties; the ghost deme is not shown in these figures).
    demes = list(FOCAL_DEME_LABELS)

    print("Loading prevalence logs…")
    prev_data = load_all_prevalence(results_dir, variant_prefix, demes, burnin_fraction)

    print("Loading migration-events logs…")
    mig_summary, raw_mig = load_all_migration(
        results_dir, variant_prefix, burnin_fraction
    )

    print("Loading datastream-parameter logs…")
    params = load_all_datastream_params(results_dir, variant_prefix, burnin_fraction)

    print("Loading priors, grid shifts, and sample times from reference XML…")
    ref_run_dir = _run_dir(results_dir, variant_prefix, REFERENCE_KEY)
    ref_xml = ref_run_dir / f"{ref_run_dir.name}.xml"
    prior_medians = load_prior_specs(ref_xml)
    grid_shifts = load_grid_shifts(ref_xml)
    sky_shifts = load_sky_shifts(ref_xml)
    t_recent = load_most_recent_sample_decimal_year(
        ref_run_dir / f"{variant_prefix}_state_time.csv"
    )
    window = load_datastream_window(results_dir)
    if grid_shifts and t_recent is not None:
        print(f"  Prevalence x-axis = calendar date (t_recent={t_recent:.3f}).")
        if window is not None:
            print(
                f"  Restricting prevalence to datastream window "
                f"[{window[0]:.3f}, {window[1]:.3f}]."
            )
        else:
            print(
                "  [warn] datastream CSVs not found — prevalence not date-restricted."
            )
    else:
        print(
            "  [warn] grid shifts / state_time unavailable — prevalence x-axis "
            "falls back to grid index."
        )

    # Estimated directions = present in the data and not fixed to 0 everywhere
    # (only among the displayed migration variants; the no-phylogeny variant is
    # excluded from all migration panels, see MIGRATION_VERSIONS).
    observed_dirs = {d for key in MIGRATION_KEYS for d in raw_mig.get(key, {})}
    est_directions = [
        d
        for d in MIGRATION_DIRECTIONS
        if d in observed_dirs and not _is_zero_direction(raw_mig, d)
    ]

    print("\nMain figure…")
    plot_main_figure(
        prev_data,
        mig_summary,
        raw_mig,
        output_dir / "value_of_information_main.png",
        grid_shifts=grid_shifts,
        t_recent=t_recent,
        window=window,
        sky_shifts=sky_shifts,
    )

    print("Supplementary 1: prevalence (non-example counties) + background migration…")
    plot_supp_prevalence(
        prev_data,
        raw_mig,
        output_dir / "supp_prevalence.png",
        grid_shifts=grid_shifts,
        t_recent=t_recent,
        window=window,
        sky_shifts=sky_shifts,
    )

    print("Supplementary 2a: migration events (all estimated directions)…")
    plot_supp_migration_events(
        raw_mig,
        est_directions,
        output_dir / "supp_migration_events.png",
    )

    print("Supplementary 2b: migration relative HPD width (all estimated directions)…")
    plot_supp_migration_hpd(
        mig_summary,
        est_directions,
        output_dir / "supp_migration_hpd.png",
    )

    print("Supplementary 3: datastream parameters…")
    plot_supp_datastream_params(
        params,
        output_dir / "supp_datastream_parameters.png",
        prior_medians=prior_medians,
    )

    print("Loading MCC tree node-height HPD widths…")
    node_hpd_widths, node_support_counts = load_all_node_hpd_widths(
        results_dir, variant_prefix
    )

    print("Standalone: node-height relative HPD width…")
    plot_node_hpd_boxplot(
        node_hpd_widths,
        output_dir / "node_height_relative_hpd_width.png",
    )

    print("Standalone: low-support internal-node counts…")
    plot_node_low_support_counts(
        node_support_counts,
        output_dir / "node_low_support_counts.png",
    )

    print(f"\nDone. Figures written to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results_1000seq"),
        help=(
            "Directory containing per-variant subdirectories "
            "(default: results_1000seq)."
        ),
    )
    parser.add_argument(
        "--variant_prefix",
        type=str,
        default="SARSCoV2_Epsilon_BayArea_results_1000seq",
        help=(
            "Common prefix of variant subdirectory names. "
            "Each directory is <prefix>_<variant_key> "
            "(default: SARSCoV2_Epsilon_BayArea_results_1000seq)."
        ),
    )
    parser.add_argument(
        "--focal_demes",
        nargs="+",
        default=list(FOCAL_DEME_LABELS),
        help=(
            f"Focal deme labels for prevalence plots (default: {list(FOCAL_DEME_LABELS)})."
        ),
    )
    parser.add_argument(
        "--include_ghost",
        action="store_true",
        default=False,
        help="Also include the ghost/outside deme in prevalence plots.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for output PNGs, PDFs, and sidecar CSVs.",
    )
    parser.add_argument(
        "--burnin_fraction",
        type=float,
        default=0.0,
        help=(
            "Additional fraction of posterior samples to discard from the "
            "start of each combined log. Combined logs already have 20%% burnin "
            "applied by logcombiner, so the default of 0.0 is usually correct."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        results_dir=args.results_dir,
        variant_prefix=args.variant_prefix,
        focal_demes=args.focal_demes,
        include_ghost=args.include_ghost,
        output_dir=args.output_dir,
        burnin_fraction=args.burnin_fraction,
    )


if __name__ == "__main__":
    main()
