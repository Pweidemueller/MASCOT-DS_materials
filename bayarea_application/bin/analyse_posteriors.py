"""
Summarise NeDynamics posterior trajectories (I_* columns) per deme.

Reads one main BEAST combined log (for run association; file must exist) and
optional per-deme NeDynamics combined logs (``--nedynamics-log``). For each
deme with a log path, writes
``prevalence_trajectories_nedynamics_<DemeN>.png`` (median I and 95% HPD).
Grid indices are reversed for display so the x-axis runs from past (high index)
to present (low index). With ``--state-time-csv`` and ``--beast-xml``, the
x-axis uses decimal year from the latest sample time and SkygrowthRateShifts
parsed from the BEAST XML (which may be non-uniformly spaced).

Migration summary: downloads Plotly's US county GeoJSON (urllib), builds a
GeoPandas layer, and plots with Cartopy (Mercator, Natural Earth land/ocean —
no raster street map): every county intersecting the viewport,
COUNTY_FIPS_BY_DEME in grey, and local migration arrows (fixed offset in metres
in EPSG:3857, drawn in WGS84). The combined ``final_figure_gridspec`` map adds
an inset bar chart of relative outside→local inflow (Deme4→Deme1–3). Written as
``migration_rates_median.png``.

2×3 datastream overview (``datastream_overview.png``) when ``--state-time-csv``
is given: Deme 1–3 only, row 1 = NeDynamics prevalence with case-count bars and
wastewater on twin axes; row 2 = cumulative incidence HPD from BEAST logs +
seroprevalence. Case counts, seroprevalence, wastewater, and county populations
CSVs are required CLI inputs. Cumulative logs use SplineGridRateShifts from
``--mascot-datastream-log`` (default: main combined log).

When state time is available, ``final_figure_gridspec.png`` (and ``.pdf``)
combines MCC tree + migration map + datastream overview in one
:class:`matplotlib.gridspec.GridSpec` layout.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Literal
import json
import re
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from shapely.geometry import Point, box, shape
import baltic as bt

from filter_gisaid_metadata import COUNTIES_OF_INTEREST
from lab_palette import PALETTE_3, RAIN
from plot_utils import (
    beautify_plot,
    configure_calendar_xaxis,
    decimal_years_to_matplotlib_dates,
    decimal_year_to_matplotlib_date,
    DEFAULT_FONTSIZES,
    FONTSIZES_LIST,
)


I_COLUMN_PATTERN = re.compile(r"^I_(\d+)$")
TRANSMISSION_COLUMN_PATTERN = re.compile(r"^transmissionRate_(\d+)$")
MIGRATION_COLUMN_PATTERN = re.compile(
    r"^f_migrationRatesSkyline\.(Deme\d+)_to_(Deme\d+)$"
)
DEME_FROM_NAME_PATTERN = re.compile(r"\.NeDynamics\.(Deme\d+)", re.IGNORECASE)
CUMULATIVE_INCIDENCE_DEME_PATTERN = re.compile(
    r"\.cumulativeIncidence\.(Deme\d+)\b", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# California county reference data (FIPS, abbreviation, bounding box).
# Add rows here when new counties are used as focal demes.
# ---------------------------------------------------------------------------
_CA_COUNTY_INFO: dict[str, dict] = {
    "Alameda": {
        "fips": "06001",
        "short": "Ala",
        "bbox": (-122.37, -121.47, 37.45, 37.91),
    },
    "Sacramento": {
        "fips": "06067",
        "short": "Sac",
        "bbox": (-121.86, -121.02, 38.02, 38.74),
    },
    "San Francisco": {
        "fips": "06075",
        "short": "SF",
        "bbox": (-122.52, -122.36, 37.71, 37.83),
    },
    "San Mateo": {
        "fips": "06081",
        "short": "SM",
        "bbox": (-122.52, -122.08, 37.11, 37.71),
    },
    "Santa Clara": {
        "fips": "06085",
        "short": "SC",
        "bbox": (-122.20, -121.21, 36.89, 37.48),
    },
}

# ---------------------------------------------------------------------------
# Dynamic deme mapping — derived from COUNTIES_OF_INTEREST.
#
# Replicates create_mascot_xml.py's logic: normalise names (spaces → _),
# sort alphabetically, then number Deme1, Deme2, ….  The ghost / outside
# deme is appended last.
# ---------------------------------------------------------------------------
_sorted_focal = sorted(COUNTIES_OF_INTEREST)
_n_focal = len(_sorted_focal)
_GHOST_LABEL = f"Deme{_n_focal + 1}"

DEME_MAP: dict[str, str] = {
    f"Deme{i + 1}": county for i, county in enumerate(_sorted_focal)
}
DEME_MAP[_GHOST_LABEL] = "Outside Deme"

FOCAL_DEME_LABELS: tuple[str, ...] = tuple(f"Deme{i + 1}" for i in range(_n_focal))

# Consistent per-deme colors: focal demes use lab_palette PALETTE_3, outside = grey.
DEME_COLORS: dict[str, str] = {
    f"Deme{i + 1}": PALETTE_3[i % len(PALETTE_3)] for i in range(_n_focal)
}
DEME_COLORS[_GHOST_LABEL] = RAIN

# US county FIPS (state 06 + county) for Plotly GeoJSON ids
COUNTY_FIPS_BY_DEME: dict[str, str] = {}
for _i, _county in enumerate(_sorted_focal):
    if _county not in _CA_COUNTY_INFO:
        raise ValueError(
            f"County {_county!r} from COUNTIES_OF_INTEREST has no entry in "
            f"_CA_COUNTY_INFO (analyse_posteriors.py). Add its FIPS, short "
            f"name, and bounding box there."
        )
    COUNTY_FIPS_BY_DEME[f"Deme{_i + 1}"] = _CA_COUNTY_INFO[_county]["fips"]

# Short abbreviation lookup (used for migration bar labels)
COUNTY_SHORT: dict[str, str] = {
    county: _CA_COUNTY_INFO[county]["short"] for county in _sorted_focal
}

PLOTLY_COUNTIES_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

# Rate shift arrays parsed from the BEAST XML at runtime (populated in main).
# Focal demes share the base SkygrowthRateShifts; the ghost deme has its own.
DEME_SKYGROWTH_SHIFTS: dict[str, np.ndarray] = {}
DEME_SPLINEGRID_SHIFTS: dict[str, np.ndarray] = {}

# Migration-map viewport (lon/lat °) — computed from focal county bounding boxes.
_all_bboxes = [_CA_COUNTY_INFO[c]["bbox"] for c in _sorted_focal]
DEFAULT_MAP_LON_MIN = min(bb[0] for bb in _all_bboxes) - 0.25
DEFAULT_MAP_LON_MAX = max(bb[1] for bb in _all_bboxes) + 0.25
DEFAULT_MAP_LAT_MIN = min(bb[2] for bb in _all_bboxes) - 0.25
DEFAULT_MAP_LAT_MAX = max(bb[3] for bb in _all_bboxes) + 0.25

OUTPUT_FILENAME_MIGRATION_RATES = "migration_rates_median.png"
OUTPUT_FILENAME_DATASTREAM_OVERVIEW = "datastream_overview.png"
OUTPUT_FILENAME_FINAL_FIGURE_GRIDSPEC = "final_figure_gridspec.png"
OUTPUT_FILENAME_MIGRATION_PCT_FROM_OUTSIDE = "migration_pct_from_outside.png"
OUTPUT_FILENAME_MIGRATION_RATES_LOCAL = "migration_rates_local.png"
OUTPUT_FILENAME_INTRODUCTIONS = "introductions_pct.png"
OUTPUT_FILENAME_INTROS_VS_CASES = "introductions_vs_local_cases.png"
OUTPUT_FILENAME_LOCAL_PREVALENCE = "prevalence_trajectories_nedynamics_local.png"

# Overview figure colors
COLOR_CASE = "silver"
COLOR_WW = "dimgrey"
HPD_KEY = "height_95%_HPD"
MAX_KEY = "max"
HPD_LINEWIDTH = 3.0
HPD_ALPHA = 0.5


def _apply_burnin(df: pd.DataFrame, burnin_fraction: float) -> pd.DataFrame:
    """Drop the first ``burnin_fraction`` of rows; raise on empty input or bad fraction."""
    if not 0 <= burnin_fraction < 1:
        raise ValueError("burnin_fraction must be in [0, 1).")
    n = len(df)
    if n == 0:
        raise ValueError("DataFrame is empty")
    drop = int(np.floor(n * burnin_fraction))
    return df.iloc[drop:].reset_index(drop=True)


def _save_fig_png_pdf(fig: plt.Figure, output_path: Path, dpi: int = 150) -> None:
    """Save *fig* as both ``output_path`` (PNG) and its ``.pdf`` sibling, then close."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# Figure-fraction offsets for panel labels, applied to the top-left corner of
# each axes' grid cell. Placing labels in figure coordinates (rather than axes
# coordinates) keeps the vertical position on an absolute grid: every label in
# a row shares the same y, regardless of how tall a panel is. The horizontal
# position is pulled out to the left of the y-axis tick labels / axis label so
# the letter always sits left of everything in its panel.
PANEL_LABEL_DX = 0.006  # shift left of the panel's left-most extent (fig frac)
PANEL_LABEL_DY = 0.008  # shift above the cell's top edge (figure fraction)


def add_panel_label(ax, label: str, renderer=None) -> None:
    """Draw a bold panel label above the top-left of an axes' grid cell.

    * **Vertical** position is the top of the axes' *grid cell* (from its
      SubplotSpec), so labels in the same gridspec row align on an absolute
      grid even if an axes was later shrunk to fit (e.g. a Cartopy map fitting
      its aspect ratio inside a tall cell).
    * **Horizontal** position is the left edge of the axes' y-axis decorations
      (tick labels + axis label) when a *renderer* is supplied, so the letter
      clears the y-axis label instead of overlapping it. Without a renderer it
      falls back to the cell's left edge.

    Because the y-axis extent is only known once text has been laid out, call
    this after ``fig.canvas.draw()`` and pass the resulting renderer.

    An empty ``label`` is a no-op, so callers can pass "" to skip a panel.
    """
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
        x_left - PANEL_LABEL_DX,  # just left of the y-axis label / tick labels
        y_top + PANEL_LABEL_DY,  # just above the cell's top edge
        label,
        fontsize=FONTSIZES_LIST[0] + 2,  # ~2 pt larger than the title size
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _parse_xml_rateshifts_text(element: ET.Element) -> np.ndarray:
    """Extract whitespace-separated floats from an XML element's text."""
    if element.text is None:
        raise ValueError(f"Empty text in element {element.tag} id={element.get('id')}")
    return np.array([float(v) for v in element.text.strip().split()], dtype=float)


def extract_rate_shifts_from_xml(
    xml_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Parse SkygrowthRateShifts and SplineGridRateShifts from a BEAST2 XML.

    Returns (skygrowth_shifts, splinegrid_shifts) dicts keyed by deme label.
    Focal demes share the base ids (``SkygrowthRateShifts``,
    ``SplineGridRateShifts``); the ghost deme uses the ``.DemeN`` suffixed ids.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sky: dict[str, np.ndarray] = {}
    grid: dict[str, np.ndarray] = {}

    # Walk all elements; match by id attribute.
    for elem in root.iter():
        eid = elem.get("id", "")
        if elem.tag == "rateShifts" and eid == "SkygrowthRateShifts":
            vals = _parse_xml_rateshifts_text(elem)
            for d in FOCAL_DEME_LABELS:
                sky[d] = vals
        elif elem.tag == "gridRateShifts" and eid == "SplineGridRateShifts":
            vals = _parse_xml_rateshifts_text(elem)
            for d in FOCAL_DEME_LABELS:
                grid[d] = vals
        elif elem.tag == "rateShifts" and eid == f"SkygrowthRateShifts.{_GHOST_LABEL}":
            sky[_GHOST_LABEL] = _parse_xml_rateshifts_text(elem)
        elif (
            elem.tag == "gridRateShifts"
            and eid == f"SplineGridRateShifts.{_GHOST_LABEL}"
        ):
            grid[_GHOST_LABEL] = _parse_xml_rateshifts_text(elem)

    if not sky:
        raise ValueError(
            f"No SkygrowthRateShifts found in {xml_path}. "
            "Expected <rateShifts id='SkygrowthRateShifts' ...> elements."
        )
    return sky, grid


_DEME_MAP_COMMENT_RE = re.compile(r"<!--\s*deme_map:\s*(.+?)\s*-->")
_DEME_MAP_ENTRY_RE = re.compile(r"(Deme\d+)=(\S+)")


def read_deme_map_from_xml(xml_path: Path) -> dict[str, str] | None:
    """Return the deme→state mapping embedded by create_mascot_xml.py, or None.

    The comment ``<!-- deme_map: Deme1=Sacramento, Deme2=San_Francisco, ... -->``
    is written on the second line of the XML file.  Returns None if absent.
    """
    with open(xml_path, encoding="utf-8") as fh:
        for _ in range(5):  # only scan the first few lines
            line = fh.readline()
            if not line:
                break
            m = _DEME_MAP_COMMENT_RE.search(line)
            if m:
                return {
                    deme: state.rstrip(",")
                    for deme, state in _DEME_MAP_ENTRY_RE.findall(m.group(1))
                }
    return None


def validate_deme_map_from_xml(xml_path: Path) -> None:
    """Warn if the deme_map comment in *xml_path* conflicts with DEME_MAP.

    Mismatches indicate that COUNTIES_OF_INTEREST changed between XML
    generation and post-processing, which would silently produce wrong plots.
    No-op when the comment is absent (older XMLs).
    """
    xml_map = read_deme_map_from_xml(xml_path)
    if xml_map is None:
        return  # comment not present — old XML, skip silently
    mismatches = []
    for deme_label, xml_state in xml_map.items():
        # The ghost deme is always written as "background" in the XML comment
        # but displayed as "Outside Deme" in DEME_MAP — skip it.
        if deme_label not in FOCAL_DEME_LABELS:
            continue
        expected = DEME_MAP.get(deme_label)
        if expected is None:
            mismatches.append(
                f"  {deme_label}: XML has '{xml_state}', not in current DEME_MAP"
            )
        elif county_key(xml_state) != county_key(expected):
            mismatches.append(
                f"  {deme_label}: XML has '{xml_state}', current DEME_MAP has '{expected}'"
            )
    if mismatches:
        msg = "\n".join(mismatches)
        raise ValueError(
            f"Deme mapping in {xml_path} does not match the current COUNTIES_OF_INTEREST.\n"
            f"{msg}\n"
            "Re-generate the XML or update COUNTIES_OF_INTEREST in filter_gisaid_metadata.py."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot prevalence-style trajectories from NeDynamics I_* columns "
            "(median line and 95% HPD across samples)."
        ),
    )
    parser.add_argument(
        "combined_log",
        type=Path,
        help="Main BEAST combined.log path (must exist; used to tie the analysis to a run).",
    )
    parser.add_argument(
        "--nedynamics-log",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "NeDynamics combined logs (one per deme, in Deme1…DemeN order). "
            "The deme label is inferred from each filename."
        ),
    )
    parser.add_argument(
        "--burnin-fraction",
        type=float,
        default=0.0,
        help="Fraction of rows to drop from the start of each NeDynamics table after loading.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for the figure (created if missing).",
    )
    parser.add_argument(
        "--geojson-url",
        type=str,
        default=PLOTLY_COUNTIES_GEOJSON_URL,
        help="US county GeoJSON URL (default: Plotly counties-by-FIPS dataset).",
    )
    parser.add_argument(
        "--geojson-cache",
        type=Path,
        default=None,
        help=(
            "Optional path to cache the GeoJSON file. "
            "Default: <output-dir>/plotly_counties_geojson_fips.json"
        ),
    )
    parser.add_argument(
        "--beast-xml",
        type=Path,
        default=None,
        help=(
            "BEAST2 XML file used for the analysis. SkygrowthRateShifts and "
            "SplineGridRateShifts are extracted to map trajectory indices and "
            "cumulative-incidence grid points to actual time. Required when "
            "--state-time-csv is provided."
        ),
    )
    parser.add_argument(
        "--state-time-csv",
        type=Path,
        default=None,
        help=(
            "Optional sample table (e.g. *state_time.csv) with a 'time' column "
            "(decimal year). The maximum time sets the most recent sample; with "
            "rate shifts from --beast-xml, trajectory indices are mapped to "
            "decimal year on the prevalence plot."
        ),
    )
    parser.add_argument(
        "--case-counts-csv",
        type=Path,
        required=True,
        help="Case counts CSV (date, case_counts, deme county name) for datastream overview.",
    )
    parser.add_argument(
        "--seroprevalence-csv",
        type=Path,
        required=True,
        help="Seroprevalence CSV for datastream overview.",
    )
    parser.add_argument(
        "--wastewater-csv",
        type=Path,
        required=True,
        help="Wastewater CSV for datastream overview.",
    )
    parser.add_argument(
        "--county-populations-csv",
        type=Path,
        required=True,
        help="county, population CSV; row order defines Deme 1–3 indices (0-based 0..2).",
    )
    parser.add_argument(
        "--cumulative-incidence-log",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "BEAST cumulativeIncidence logs for focal demes (Deme1…DemeN order). "
            "Deme label is inferred from each filename."
        ),
    )
    parser.add_argument(
        "--mascot-datastream-log",
        type=Path,
        default=None,
        help=(
            "BEAST log whose header comments contain SplineGridRateShifts (for cum. incidence "
            "time mapping). Default: same as combined_log."
        ),
    )
    parser.add_argument(
        "--tree",
        type=Path,
        required=True,
        help="Path to NEXUS .trees file",
    )
    return parser.parse_args()


def build_deme_colormap(tree) -> dict[str, str]:
    """Map each distinct ``max`` deme value on the tree to DEME_COLORS."""
    values = sorted({k.traits[MAX_KEY] for k in tree.Objects if MAX_KEY in k.traits})
    return {d: DEME_COLORS.get(d, "#888888") for d in values}


def branch_colour(node, deme_colors: dict[str, str], fallback: str) -> str:
    deme = node.traits.get(MAX_KEY)
    if deme is None:
        return fallback
    return deme_colors.get(deme, fallback)


def youngest_tip_decimal_date(tree) -> float:
    """Same time origin BEAST uses for `height` (most recent tip, decimal year)."""
    return tree.root.absoluteTime + tree.treeHeight


def hpd_absolute_span(node, ref_youngest: float) -> tuple[float, float] | None:
    """
    Map `height_95%_HPD` onto decimal calendar time.

    Values are in BEAST units (time from the youngest tip). For any node,
    ``absoluteTime == ref_youngest - traits['height']`` (matches baltic after
    ``setAbsoluteTime`` because ``height + node.height == treeHeight``).
    """
    hpd = node.traits.get(HPD_KEY)
    if hpd is None or len(hpd) != 2:
        return None
    lo, hi = float(hpd[0]), float(hpd[1])
    t_lo = ref_youngest - lo
    t_hi = ref_youngest - hi
    return min(t_lo, t_hi), max(t_lo, t_hi)


def _draw_mcc_tree_deme_hpd(
    ax: plt.Axes,
    tree_path: Path,
    *,
    tree: object | None = None,
) -> None:
    if tree is None:
        tree = bt.loadNexus(
            str(tree_path),
            treestring_regex=r"tree\s+\S+\s*=",
            absoluteTime=True,
            verbose=False,
        )

    deme_colors = build_deme_colormap(tree)
    fallback = "#888888"

    root_time = tree.root.absoluteTime

    def x_attr(n):
        t = n.absoluteTime
        if t is None:
            t = root_time
        return decimal_year_to_matplotlib_date(t)

    y_attr = lambda n: n.y  # noqa: E731

    tree.plotTree(
        ax,
        x_attr=x_attr,
        y_attr=y_attr,
        colour=lambda n: branch_colour(n, deme_colors, fallback),
        width=0.9,
        zorder=2,
    )

    # Colored dots at tips, colored by deme.
    tips = tree.getExternal()
    for tip in tips:
        tx = x_attr(tip)
        ty = y_attr(tip)
        c = branch_colour(tip, deme_colors, fallback)
        ax.scatter(
            tx,
            ty,
            s=12,
            c=c,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )

    configure_calendar_xaxis(ax)
    ax.tick_params(axis="x", which="both", labelbottom=True)
    ax.set_ylabel("")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(left=False, labelleft=False)
    ax.tick_params(axis="x", labelsize=DEFAULT_FONTSIZES["tick_label"])
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=deme_colors.get(d, fallback),
            lw=3,
            label=DEME_MAP.get(d, d),
        )
        for d in sorted(deme_colors.keys())
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=DEFAULT_FONTSIZES["legend"],
        frameon=False,
    )


def plot_mcc_tree_deme_hpd(
    tree_path: Path,
    output_path: Path,
    figsize_width: float = 14.0,
    dpi: int = 120,
) -> None:
    tree = bt.loadNexus(
        str(tree_path),
        treestring_regex=r"tree\s+\S+\s*=",
        absoluteTime=True,
        verbose=False,
    )
    n_tips = len(tree.getExternal())
    fig_h = min(120.0, max(10.0, 0.032 * n_tips))
    fig, ax = plt.subplots(figsize=(figsize_width, fig_h), dpi=dpi)
    _draw_mcc_tree_deme_hpd(ax, tree_path, tree=tree)
    fig.tight_layout()
    _save_fig_png_pdf(fig, output_path, dpi=300)


def ensure_combined_log_exists(combined_log: Path) -> None:
    if not combined_log.is_file():
        raise FileNotFoundError(f"Combined log not found: {combined_log}")


def load_most_recent_sample_decimal_year(state_time_csv: Path) -> float:
    """
    Latest sampling time in decimal years from a state_time-style CSV
    (columns include ``time``).
    """
    if not state_time_csv.is_file():
        raise FileNotFoundError(f"State-time CSV not found: {state_time_csv}")
    df = pd.read_csv(state_time_csv)
    if "time" not in df.columns:
        raise ValueError(
            f"Expected a 'time' column in {state_time_csv}, got {list(df.columns)}"
        )
    times = df["time"].apply(pd.to_numeric, errors="raise")
    return float(times.max())


def deme_label_from_path(path: Path) -> str:
    match = DEME_FROM_NAME_PATTERN.search(path.name)
    if match:
        return match.group(1)
    return path.stem


def list_i_columns(columns: pd.Index) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    for col in columns:
        m = I_COLUMN_PATTERN.match(str(col))
        if m:
            pairs.append((int(m.group(1)), str(col)))
    pairs.sort(key=lambda x: x[0])
    return pairs


def list_transmission_columns(columns: pd.Index) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []
    for col in columns:
        m = TRANSMISSION_COLUMN_PATTERN.match(str(col))
        if m:
            pairs.append((int(m.group(1)), str(col)))
    pairs.sort(key=lambda x: x[0])
    return pairs


def load_nedynamics_arrays(
    path: Path,
    burnin_fraction: float,
    *,
    include_transmission: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read NeDynamics ``I_*`` (and optionally ``transmissionRate_*``) columns.

    Returns ``(grid_idx, log_I, beta)`` where ``log_I`` and ``beta`` have
    shape ``(n_samples, n_grid)`` post burn-in and ``grid_idx`` is the
    integer column suffix array (e.g. ``[0, 10, 20, …]``).  ``beta`` is
    ``None`` when ``include_transmission`` is False. When True, the
    ``I_*`` and ``transmissionRate_*`` grid indices must match.
    """
    header = pd.read_csv(path, sep="\t", nrows=0)
    i_pairs = list_i_columns(header.columns)
    if not i_pairs:
        raise ValueError(f"No I_<index> columns in {path}")
    i_idx = np.array([k for k, _ in i_pairs], dtype=int)
    i_cols = [c for _, c in i_pairs]

    t_cols: list[str] = []
    if include_transmission:
        t_pairs = list_transmission_columns(header.columns)
        if not t_pairs:
            raise ValueError(f"No transmissionRate_<index> columns in {path}")
        t_idx = np.array([k for k, _ in t_pairs], dtype=int)
        if not np.array_equal(i_idx, t_idx):
            raise ValueError(
                f"I_* and transmissionRate_* grid indices differ in {path}"
            )
        t_cols = [c for _, c in t_pairs]

    df = pd.read_csv(path, sep="\t", usecols=i_cols + t_cols).apply(
        pd.to_numeric, errors="raise"
    )
    df = _apply_burnin(df, burnin_fraction)
    log_I = df[i_cols].to_numpy(dtype=float)
    beta = df[t_cols].to_numpy(dtype=float) if t_cols else None
    return i_idx, log_I, beta


def summarise_logI_trajectory(log_I: np.ndarray) -> dict[str, np.ndarray]:
    """Median and 95% HPD per grid column for log-prevalence samples.

    Returns a dict with both log-scale (``median_logI``, ``hpd_lo_logI``,
    ``hpd_hi_logI``) and exponentiated (``median_I``, ``hpd_lo_I``,
    ``hpd_hi_I``) summaries.  Wraps :func:`summarise_samples_per_time`.
    """
    med, lo, hi = summarise_samples_per_time(log_I)
    return {
        "median_logI": med,
        "hpd_lo_logI": lo,
        "hpd_hi_logI": hi,
        "median_I": np.exp(med),
        "hpd_lo_I": np.exp(lo),
        "hpd_hi_I": np.exp(hi),
    }


def compute_introductions_pct_samples(
    *,
    local_path: Path,
    local_grid_shifts: np.ndarray,
    source_paths: dict[str, Path],
    source_grid_shifts: dict[str, np.ndarray],
    mig_samples_by_source: dict[str, np.ndarray],
    t_recent: float,
    burnin_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample % introductions plus raw introductions and total-new-infection arrays.

    Forward-time identity (per posterior sample ``s`` and local time ``t``,
    for target deme i):

        intros(s,t)        = Σ_{j≠i} m_fwd[j→i](s) · I_j(s,t)
        total_new_inf(s,t) = β_i(s,t) · I_i(s,t)
        pct(s,t)           = 100 · intros / total_new_inf

    Sources j include every other focal deme **and** the ghost/outside deme
    — i.e. all demes except the target itself.

    ``β_i`` is identified from deme i's prevalence trajectory, which already
    reflects every source of new deme-i infections — within-deme transmission
    plus imports from all other demes — so ``β_i · I_i`` is the total
    new-infection rate in deme i and is the correct denominator. The
    within-deme component (if needed) is ``β_i · I_i − Σ_{j≠i} m_fwd[j→i] · I_j``.

    ``I_*`` columns are stored as log-prevalence; we exponentiate. Each
    source's prevalence ``I_j`` is **linearly interpolated** onto the local
    deme's calendar dates (per sample). Outside any source's time range, all
    output arrays are ``NaN``.

    A sanity-check warning is issued if the numerator exceeds the denominator
    in any valid (sample, time) cell (Volz balance:
    ``Σ_{j≠i} m_fwd[j→i] · I_j ≤ β_i · I_i``).

    Returns
    -------
    x_dec : (n_local_grid,) decimal years, ascending (left = past → right = present).
    pct_samples : (n_samples, n_local_grid) percentages.
    intros_samples : (n_samples, n_local_grid) total introductions per unit time.
    total_new_inf_samples : (n_samples, n_local_grid) total new-infection rate β·I.
    """
    if not source_paths:
        raise ValueError("At least one source deme is required.")

    i_idx_l, logI_l, beta_l = load_nedynamics_arrays(
        local_path, burnin_fraction, include_transmission=True
    )
    n_l = len(logI_l)
    t_local = t_recent - local_grid_shifts[i_idx_l]
    I_local = np.exp(logI_l)
    n_grid_l = len(t_local)

    intros = np.zeros((n_l, n_grid_l), dtype=float)
    out_of_range = np.zeros((n_l, n_grid_l), dtype=bool)

    for src_label, src_path in source_paths.items():
        if src_label not in mig_samples_by_source:
            raise KeyError(f"Missing migration rate samples for source {src_label}")
        if src_label not in source_grid_shifts:
            raise KeyError(f"Missing grid shifts for source {src_label}")

        i_idx_s, logI_s, _ = load_nedynamics_arrays(src_path, burnin_fraction)
        m_s = mig_samples_by_source[src_label]
        n_s, n_m = len(logI_s), len(m_s)
        if not (n_l == n_s == n_m):
            raise ValueError(
                f"Post-burn-in sample counts differ: local={n_l}, "
                f"source[{src_label}]={n_s}, migration[{src_label}]={n_m}. "
                "Combined logs must be row-aligned."
            )

        t_src = t_recent - source_grid_shifts[src_label][i_idx_s]
        order_s = np.argsort(t_src)
        t_src_sorted = t_src[order_s]
        I_src_sorted = np.exp(logI_s[:, order_s])

        I_src_interp = np.empty((n_l, n_grid_l), dtype=float)
        for s in range(n_l):
            I_src_interp[s, :] = np.interp(
                t_local,
                t_src_sorted,
                I_src_sorted[s, :],
                left=np.nan,
                right=np.nan,
            )

        out_of_range |= np.isnan(I_src_interp)
        contrib = I_src_interp * m_s[:, None]
        # Accumulate; NaN cells get re-masked after the loop.
        intros += np.where(np.isnan(contrib), 0.0, contrib)

    total_new_inf = beta_l * I_local
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = 100.0 * intros / total_new_inf

    intros[out_of_range] = np.nan
    pct[out_of_range] = np.nan
    total_new_inf_masked = total_new_inf.copy()
    total_new_inf_masked[out_of_range] = np.nan

    # Sanity check: under Volz balance, Σ_{j≠i} m_fwd[j→i] · I_j ≤ β_i · I_i.
    with np.errstate(invalid="ignore"):
        violations = intros > total_new_inf_masked
    n_violations = int(np.count_nonzero(violations))
    if n_violations:
        n_valid = int(np.count_nonzero(~np.isnan(intros)))
        frac = 100.0 * n_violations / max(n_valid, 1)
        warnings.warn(
            f"%introductions sanity check: total introductions exceed β·I in "
            f"{n_violations}/{n_valid} ({frac:.2f}%) (sample, time) cells for "
            f"local={local_path.name}. Volz balance "
            f"(Σ m_fwd·I_j ≤ β_i·I_i) violated.",
            RuntimeWarning,
            stacklevel=2,
        )

    order_l = np.argsort(t_local)
    return (
        t_local[order_l],
        pct[:, order_l],
        intros[:, order_l],
        total_new_inf_masked[:, order_l],
    )


def summarise_samples_per_time(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-time-point median and 95% HPD across MCMC samples (NaN-safe).

    *samples* has shape ``(n_mcmc, n_grid)``.  Returns ``(median, hpd_lo,
    hpd_hi)``, each shape ``(n_grid,)``.  Time points with <2 valid samples
    are returned as ``NaN``.
    """
    n_t = samples.shape[1]
    med = np.full(n_t, np.nan)
    lo = np.full(n_t, np.nan)
    hi = np.full(n_t, np.nan)
    for j in range(n_t):
        col = samples[:, j]
        col = col[~np.isnan(col)]
        if col.size < 2:
            continue
        lo_j, hi_j, med_j = calculate_hpd_mcmc(col)
        med[j] = med_j
        lo[j] = lo_j
        hi[j] = hi_j
    return med, lo, hi


def _xlim_with_monthly_right_edge(
    x_dec_range: tuple[float, float],
) -> tuple[float, float]:
    """Convert a decimal-year range to matplotlib-date numbers, rounding the
    upper edge **up** to the first day of the next month.

    Ensures the date axis ends on a month boundary so a tick at that month
    start (e.g. 05-01-2021) is rendered by ``AutoDateLocator``.
    """
    x_lo_num, x_hi_num = decimal_years_to_matplotlib_dates(
        np.array([x_dec_range[0], x_dec_range[1]], dtype=float)
    )
    dt = mdates.num2date(x_hi_num)
    if not (dt.day == 1 and dt.hour == 0 and dt.minute == 0 and dt.second == 0):
        if dt.month == 12:
            nxt = datetime(dt.year + 1, 1, 1, tzinfo=dt.tzinfo)
        else:
            nxt = datetime(dt.year, dt.month + 1, 1, tzinfo=dt.tzinfo)
        x_hi_num = float(mdates.date2num(nxt))
    return float(x_lo_num), float(x_hi_num)


def compute_unified_x_dec_range(
    case_df: pd.DataFrame,
    ww_df: pd.DataFrame,
    sero_df: pd.DataFrame,
    t_recent: float,
) -> tuple[float, float]:
    """Decimal-year range anchored on the earliest datastream timestamp.

    Lower bound = minimum ``_decimal_year`` across the case-count, wastewater,
    and seroprevalence CSVs.  Upper bound = ``t_recent``.  Other series
    (e.g. % introductions) are cropped to this range so all panels share the
    same x-axis window.
    """
    x_lo = float(t_recent)
    for df in (case_df, ww_df, sero_df):
        if df is not None and not df.empty and "_decimal_year" in df.columns:
            x_lo = min(x_lo, float(df["_decimal_year"].min()))
    return x_lo, float(t_recent)


def compute_introductions_for_focal_demes(
    *,
    nedynamics_paths: dict[str, Path],
    mig_samples: dict[tuple[str, str], np.ndarray],
    t_recent: float,
    burnin_fraction: float,
    focal_labels: tuple[str, ...] = FOCAL_DEME_LABELS,
    ghost_label: str = _GHOST_LABEL,
) -> dict[str, dict]:
    """Return per-deme posterior summaries used by the introductions plots.

    For each focal deme ``lab``, ``out[lab]`` has keys:

    * ``x_dec`` — (n_grid,) decimal-year grid (ascending).
    * ``pct``, ``intros``, ``total_new_inf`` — each a ``(median, hpd_lo, hpd_hi)``
      tuple of (n_grid,) arrays.  ``pct`` is the % introductions metric,
      ``intros`` is the total introduction rate
      ``Σ_{j≠i} m_fwd[j→i] · I_j`` summed over every other deme (other focal
      demes + ghost), and ``total_new_inf`` is the total new-infection rate
      in the focal deme ``β · I_local`` (which already includes imports).
    * ``beta`` — ``(median, hpd_lo, hpd_hi)`` of the raw transmission rate β.
    * ``prevalence`` — ``(median_I, hpd_lo_I, hpd_hi_I)`` of focal-deme
      prevalence (HPD endpoints exponentiated from log-prevalence).

    The ghost deme also appears as ``out[ghost_label]`` with only ``x_dec``
    and ``prevalence`` (no migration/intros/β for the ghost).
    """
    outside_path = nedynamics_paths.get(ghost_label)
    if outside_path is None:
        raise ValueError(
            f"NeDynamics log for outside deme ({ghost_label}) is required to "
            "compute % introductions."
        )
    all_source_labels = tuple(focal_labels) + (ghost_label,)
    out: dict[str, dict] = {}
    for lab in focal_labels:
        local_path = nedynamics_paths.get(lab)
        if local_path is None:
            raise ValueError(
                f"NeDynamics log for focal deme {lab} is required to compute % introductions."
            )

        source_paths: dict[str, Path] = {}
        source_grid_shifts: dict[str, np.ndarray] = {}
        mig_samples_by_source: dict[str, np.ndarray] = {}
        for src in all_source_labels:
            if src == lab:
                continue
            src_path = nedynamics_paths.get(src)
            if src_path is None:
                raise ValueError(
                    f"NeDynamics log for source deme {src} is required to "
                    f"compute % introductions into {lab}."
                )
            key = (src, lab)
            if key not in mig_samples:
                raise KeyError(f"Missing migration rate samples for {src} -> {lab}")
            source_paths[src] = src_path
            source_grid_shifts[src] = DEME_SPLINEGRID_SHIFTS[src]
            mig_samples_by_source[src] = mig_samples[key]

        x_dec, pct_s, intros_s, total_new_inf_s = compute_introductions_pct_samples(
            local_path=local_path,
            local_grid_shifts=DEME_SPLINEGRID_SHIFTS[lab],
            source_paths=source_paths,
            source_grid_shifts=source_grid_shifts,
            mig_samples_by_source=mig_samples_by_source,
            t_recent=t_recent,
            burnin_fraction=burnin_fraction,
        )
        i_idx_l, logI_l, beta_l = load_nedynamics_arrays(
            local_path, burnin_fraction, include_transmission=True
        )
        t_local = t_recent - DEME_SPLINEGRID_SHIFTS[lab][i_idx_l]
        order_l = np.argsort(t_local)
        beta_summary = summarise_samples_per_time(beta_l[:, order_l])
        prev = summarise_logI_trajectory(logI_l[:, order_l])
        out[lab] = {
            "x_dec": x_dec,
            "pct": summarise_samples_per_time(pct_s),
            "intros": summarise_samples_per_time(intros_s),
            "total_new_inf": summarise_samples_per_time(total_new_inf_s),
            "beta": beta_summary,
            "prevalence": (prev["median_I"], prev["hpd_lo_I"], prev["hpd_hi_I"]),
        }

    i_idx_o, logI_o, _ = load_nedynamics_arrays(outside_path, burnin_fraction)
    t_outside = t_recent - DEME_SPLINEGRID_SHIFTS[ghost_label][i_idx_o]
    order_o = np.argsort(t_outside)
    prev_o = summarise_logI_trajectory(logI_o[:, order_o])
    out[ghost_label] = {
        "x_dec": t_outside[order_o],
        "prevalence": (prev_o["median_I"], prev_o["hpd_lo_I"], prev_o["hpd_hi_I"]),
    }
    return out


def crop_intros_data_to_x_range(
    intros_data: dict[str, dict],
    x_dec_range: tuple[float, float],
) -> dict[str, dict]:
    """Restrict every per-deme trajectory in ``intros_data`` to ``x_dec_range``.

    Each deme entry holds an ``x_dec`` grid plus per-grid arrays (the
    ``(median, lo, hi)`` summary tuples and the prevalence tuple), all aligned
    to that deme's ``x_dec``.  Masking every array to the
    ``[x_dec_range[0], x_dec_range[1]]`` window before plotting ensures the
    Matplotlib autoscaler never sees trajectory points that fall outside the
    plotted date range — otherwise off-window early-epidemic points (which the
    NeDynamics grid extends to, before the first datastream date) drag the
    log-scale prevalence axis floor well below the visible curve.
    """
    lo, hi = float(x_dec_range[0]), float(x_dec_range[1])
    cropped: dict[str, dict] = {}
    for lab, entry in intros_data.items():
        x_dec = np.asarray(entry["x_dec"], dtype=float)
        mask = (x_dec >= lo) & (x_dec <= hi)
        new_entry: dict = {}
        for key, val in entry.items():
            if key == "x_dec":
                new_entry[key] = x_dec[mask]
            elif isinstance(val, tuple):
                new_entry[key] = tuple(np.asarray(arr)[mask] for arr in val)
            else:
                new_entry[key] = np.asarray(val)[mask]
        cropped[lab] = new_entry
    return cropped


def _migration_column_names(columns: pd.Index) -> list[str]:
    return [str(c) for c in columns if MIGRATION_COLUMN_PATTERN.match(str(c))]


def load_migration_rate_samples(
    combined_log: Path,
    burnin_fraction: float,
) -> dict[tuple[str, str], np.ndarray]:
    """Full posterior samples per migration-rate column (after burn-in).

    Returns a dict mapping ``(from_deme, to_deme)`` to a 1-D array of MCMC
    samples.
    """
    header = pd.read_csv(combined_log, sep="\t", comment="#", nrows=0)
    cols = _migration_column_names(header.columns)
    if not cols:
        raise ValueError(
            f"No f_migrationRatesSkyline.Deme*_to_Deme* columns in {combined_log}"
        )
    df = pd.read_csv(combined_log, sep="\t", comment="#", usecols=cols).apply(
        pd.to_numeric, errors="raise"
    )
    df = _apply_burnin(df, burnin_fraction)

    samples: dict[tuple[str, str], np.ndarray] = {}
    for col in cols:
        m = MIGRATION_COLUMN_PATTERN.match(str(col))
        if m is None:
            continue
        fr, to = m.group(1), m.group(2)
        samples[(fr, to)] = df[col].to_numpy(dtype=float)
    return samples


def load_migration_rate_medians(
    combined_log: Path,
    burnin_fraction: float,
) -> dict[tuple[str, str], float]:
    """Median posterior value per migration-rate column.

    Thin wrapper around :func:`load_migration_rate_samples`.
    """
    samples = load_migration_rate_samples(combined_log, burnin_fraction)
    return {key: float(np.median(arr)) for key, arr in samples.items()}


def draw_ghost_inflow_migration(
    ax,
    median_rates: dict[tuple[str, str], float],
    *,
    rate_samples: dict[tuple[str, str], np.ndarray] | None = None,
    focal_labels: tuple[str, ...] = FOCAL_DEME_LABELS,
    ghost_label: str = _GHOST_LABEL,
    pops: dict[int, float] | None = None,
    orientation: str = "vertical",
) -> None:
    """Relative outside->local inflow: median dot + 95% HPD whiskers.

    When *rate_samples* is provided, the percentage is computed per posterior
    sample and summarised as median + 95% HPD.  Otherwise falls back to a
    single point from *median_rates* (no whiskers).

    When *pops* is provided, a cross (×) is drawn slightly offset from each
    deme tick showing that deme's fraction of the total focal-deme population
    (0–100 %).  Migration markers are shifted the other way so the two symbols
    don't overlap.

    *orientation* controls which axis the demes sit on:

    * ``"vertical"`` (default) — demes on the x-axis, percentage on the y-axis.
    * ``"horizontal"`` — demes on the y-axis, percentage on the x-axis. Used
      for the compact inset in the combined figure's tree panel.
    """
    horizontal = orientation == "horizontal"
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(
            f"orientation must be 'vertical' or 'horizontal', got {orientation!r}"
        )

    _MIGS_OFFSET = -0.15 if pops else 0.0
    _POPS_OFFSET = 0.15

    deme_labels = [DEME_MAP.get(lab, lab) for lab in focal_labels]
    pos = np.arange(len(focal_labels), dtype=float)
    colors = [DEME_COLORS[lab] for lab in focal_labels]
    fs = DEFAULT_FONTSIZES

    def _marker(value, p, color):
        """Median dot at (value, deme-pos), transposed for horizontal."""
        if horizontal:
            ax.plot(value, p, "o", color=color, ms=7, zorder=4)
        else:
            ax.plot(p, value, "o", color=color, ms=7, zorder=4)

    def _whisker(lo, hi, p, color):
        """95% HPD whisker spanning the value axis at deme-pos *p*."""
        if horizontal:
            ax.hlines(p, lo, hi, colors=color, linewidth=2.0, zorder=3)
        else:
            ax.vlines(p, lo, hi, colors=color, linewidth=2.0, zorder=3)

    if rate_samples is not None:
        # Per-sample percentage: 100 * rate_i / sum(rates)
        sample_arrays = []
        for lab in focal_labels:
            key = (ghost_label, lab)
            if key not in rate_samples:
                raise KeyError(f"Missing posterior samples for {ghost_label} -> {lab}")
            sample_arrays.append(rate_samples[key])
        stacked = np.column_stack(sample_arrays)  # (n_samples, n_demes)
        totals = stacked.sum(axis=1, keepdims=True)
        totals[totals <= 0] = np.nan
        pct_samples = 100.0 * stacked / totals  # (n_samples, n_demes)

        for j in range(len(focal_labels)):
            col_data = pct_samples[:, j]
            col_data = col_data[~np.isnan(col_data)]
            hpd_lo, hpd_hi, med = calculate_hpd_mcmc(col_data)
            _marker(med, pos[j] + _MIGS_OFFSET, colors[j])
            _whisker(hpd_lo, hpd_hi, pos[j] + _MIGS_OFFSET, colors[j])
    else:
        # Fallback: single median point, no whiskers.
        for lab in focal_labels:
            key = (ghost_label, lab)
            if key not in median_rates:
                raise KeyError(
                    f"Missing median migration rate for {ghost_label} -> {lab}"
                )
        rates = [median_rates[(ghost_label, lab)] for lab in focal_labels]
        total = float(np.sum(rates))
        if total <= 0:
            return
        pct = 100.0 * np.asarray(rates) / total
        for j in range(len(focal_labels)):
            _marker(pct[j], pos[j] + _MIGS_OFFSET, colors[j])

    # Population-size crosses: fraction of total focal population × 100.
    if pops:
        pop_values = [pops.get(j, 0.0) for j in range(len(focal_labels))]
        total_pop = sum(pop_values)
        if total_pop > 0:
            for j, col in enumerate(colors):
                pop_pct = 100.0 * pop_values[j] / total_pop
                p = pos[j] + _POPS_OFFSET
                if horizontal:
                    ax.plot(pop_pct, p, "x", color=col, ms=7, mew=2.0, zorder=4)
                else:
                    ax.plot(p, pop_pct, "x", color=col, ms=7, mew=2.0, zorder=4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if horizontal:
        ax.set_yticks(pos)
        ax.set_yticklabels(deme_labels, fontsize=fs["tick_label"])
        ax.set_xlabel("Relative migration from outside", fontsize=fs["axis_label"])
        ax.grid(axis="x", linestyle=":", alpha=0.45, zorder=0)
        ax.tick_params(axis="x", labelsize=fs["tick_label"])
        ax.set_ylim(-0.5, len(focal_labels) - 0.5)
        ax.invert_yaxis()  # first deme at the top, reading order top->bottom
    else:
        ax.set_xticks(pos)
        ax.set_xticklabels(
            deme_labels, rotation=22, ha="right", fontsize=fs["tick_label"]
        )
        ax.set_ylabel("Relative migration\nfrom outside", fontsize=fs["axis_label"])
        ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
        ax.tick_params(axis="y", labelsize=fs["tick_label"])
        ax.set_xlim(-0.5, len(focal_labels) - 0.5)


def plot_ghost_inflow_fraction_inset(
    ax_map,
    median_rates: dict[tuple[str, str], float],
    *,
    focal_labels: tuple[str, ...] = FOCAL_DEME_LABELS,
    ghost_label: str = _GHOST_LABEL,
) -> None:
    """
    Inset bar chart: each bar is 100 × rate(Deme4→Deme_i) / Σ_j rate(Deme4→Deme_j).

    ``ax_map`` is the Cartopy map axes; the inset is placed in the upper right in
    axes-relative coordinates.
    """
    ax_inset = ax_map.inset_axes([0.54, 0.69, 0.43, 0.30])
    ax_inset.set_facecolor("white")
    ax_inset.patch.set_alpha(0.94)
    draw_ghost_inflow_migration(
        ax_inset, median_rates, focal_labels=focal_labels, ghost_label=ghost_label
    )


def draw_local_migration_bars(
    ax,
    median_rates: dict[tuple[str, str], float],
    *,
    rate_samples: dict[tuple[str, str], np.ndarray] | None = None,
    focal_labels: tuple[str, ...] = FOCAL_DEME_LABELS,
) -> None:
    """Migration rates between local demes: median dot + 95% HPD whiskers.

    Colored by **origin** deme.
    """
    pairs: list[tuple[str, str]] = []
    for a in focal_labels:
        for b in focal_labels:
            if a != b:
                pairs.append((a, b))

    x_labels = [
        f"{COUNTY_SHORT.get(DEME_MAP.get(a, a), a)}\u2192{COUNTY_SHORT.get(DEME_MAP.get(b, b), b)}"
        for a, b in pairs
    ]
    x = np.arange(len(pairs), dtype=float)
    colors = [DEME_COLORS[a] for a, _b in pairs]
    fs = DEFAULT_FONTSIZES

    for j, (a, b) in enumerate(pairs):
        key = (a, b)
        if rate_samples is not None and key in rate_samples:
            samples = rate_samples[key]
            hpd_lo, hpd_hi, med = calculate_hpd_mcmc(samples)
            ax.plot(x[j], med, "o", color=colors[j], ms=7, zorder=4)
            ax.vlines(
                x[j],
                hpd_lo,
                hpd_hi,
                colors=colors[j],
                linewidth=2.0,
                zorder=3,
            )
        else:
            if key not in median_rates:
                raise KeyError(f"Missing migration rate for {a} -> {b}")
            ax.plot(
                x[j],
                median_rates[key],
                "o",
                color=colors[j],
                ms=7,
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=fs["tick_label"])
    ax.set_ylabel("Migration rate", fontsize=fs["axis_label"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax.tick_params(axis="y", labelsize=fs["tick_label"])
    ax.set_xlim(-0.5, len(pairs) - 0.5)
    ax.tick_params(axis="y", labelsize=DEFAULT_FONTSIZES["tick_label"])


def plot_migration_pct_from_outside(
    median_rates: dict[tuple[str, str], float],
    output_path: Path,
    *,
    rate_samples: dict[tuple[str, str], np.ndarray] | None = None,
    pops: dict[int, float] | None = None,
) -> None:
    """Standalone supplementary: relative outside→local inflow scatter."""
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    draw_ghost_inflow_migration(ax, median_rates, rate_samples=rate_samples, pops=pops)
    fig.tight_layout()
    _save_fig_png_pdf(fig, output_path)


def plot_migration_rates_local(
    median_rates: dict[tuple[str, str], float],
    output_path: Path,
    *,
    rate_samples: dict[tuple[str, str], np.ndarray] | None = None,
) -> None:
    """Standalone supplementary: local-to-local migration-rate scatter."""
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    draw_local_migration_bars(ax, median_rates, rate_samples=rate_samples)
    fig.tight_layout()
    _save_fig_png_pdf(fig, output_path)


def _draw_introductions_panel(
    ax,
    x_dec: np.ndarray,
    median: np.ndarray,
    hpd_lo: np.ndarray,
    hpd_hi: np.ndarray,
    *,
    color: str,
    county_name: str,
    show_xlabel: bool = False,
    show_ylabel: bool = True,
) -> None:
    """One % introductions subplot: median line + 95% HPD band over time."""
    fs = DEFAULT_FONTSIZES
    valid = ~np.isnan(median)
    if valid.any():
        x_plot = decimal_years_to_matplotlib_dates(np.asarray(x_dec, dtype=float))
        ax.fill_between(
            x_plot[valid],
            hpd_lo[valid],
            hpd_hi[valid],
            alpha=0.30,
            color=color,
            linewidth=0,
        )
        ax.plot(x_plot[valid], median[valid], color=color, lw=1.8)
    ax.set_title(county_name, fontsize=fs["title"])
    if show_ylabel:
        ax.set_ylabel("% new cases due\nto introductions", fontsize=fs["axis_label"])
    if show_xlabel:
        ax.set_xlabel("Date", fontsize=fs["axis_label"])
    ax.tick_params(labelsize=fs["tick_label"])
    configure_calendar_xaxis(ax)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    beautify_plot(ax, remove_spines=False)


def _overlay_prevalence_twin(
    ax,
    x_dec: np.ndarray,
    prev_median: np.ndarray,
    *,
    color: str = "0.4",
) -> plt.Axes:
    """Overlay local-deme prevalence (median) as a dotted line on a twin
    log-scale y-axis. Returns the twin axis. Shared by the standalone
    ``plot_introductions_panels`` and the combined ``final_figure_gridspec``."""
    fs = DEFAULT_FONTSIZES
    x_plot = decimal_years_to_matplotlib_dates(np.asarray(x_dec, dtype=float))
    ax2 = ax.twinx()
    ax2.plot(x_plot, prev_median, color=color, linestyle=":", lw=1.4)
    ax2.set_yscale("log")
    ax2.set_ylabel("Prevalence", color=color, fontsize=fs["axis_label"])
    ax2.tick_params(axis="y", labelsize=fs["tick_label"], colors=color)
    return ax2


def plot_introductions_panels(
    intros_data: dict[str, dict],
    output_path: Path,
    *,
    focal_labels: tuple[str, ...] = FOCAL_DEME_LABELS,
    ghost_label: str = _GHOST_LABEL,
    x_dec_range: tuple[float, float] | None = None,
) -> None:
    """Standalone stacked supplementary: one % introductions panel per focal deme,
    each overlaid with the local deme's prevalence on a twin axis (dotted), plus
    a bottom row showing the background-deme prevalence trajectory in grey.

    The twin-axis overlay and the bottom row are intentionally only applied here
    — ``_draw_introductions_panel`` (which is also reused inside
    ``final_figure_gridspec``) is left untouched.

    When ``x_dec_range`` is given, the x-axis limits are forced to that
    decimal-year range (so the panels can share limits with other figures).
    """
    fs = DEFAULT_FONTSIZES
    n = len(focal_labels)
    n_total = n + 1  # +1 for the bottom background-deme prevalence row
    fig, axes = plt.subplots(
        n_total, 1, figsize=(7.5, 2.4 * n_total), sharex=True, sharey=False
    )
    # Share the % introductions y-axis across focal panels only (not the ghost).
    for ax in axes[1:n]:
        ax.sharey(axes[0])

    prev_color = "0.4"  # neutral grey for prevalence overlays / ghost row
    twin_axes: list[plt.Axes] = []
    for i, lab in enumerate(focal_labels):
        ax = axes[i]
        x_dec = intros_data[lab]["x_dec"]
        med, lo, hi = intros_data[lab]["pct"]
        _draw_introductions_panel(
            ax,
            x_dec,
            med,
            lo,
            hi,
            color=DEME_COLORS[lab],
            county_name=DEME_MAP.get(lab, lab),
            show_xlabel=False,  # x-axis is labelled only on the bottom ghost row
            show_ylabel=True,
        )
        ax.tick_params(axis="x", labelbottom=False)

        # Twin axis: local-deme prevalence as a dotted grey line (log scale).
        prev_med, _, _ = intros_data[lab]["prevalence"]
        ax2 = _overlay_prevalence_twin(ax, x_dec, prev_med, color=prev_color)
        twin_axes.append(ax2)

    # Bottom row: background (ghost) deme prevalence in grey, same x-range.
    ax_ghost = axes[-1]
    ghost = intros_data[ghost_label]
    g_med, g_lo, g_hi = ghost["prevalence"]
    g_x_dec = np.asarray(ghost["x_dec"], dtype=float)
    g_x = decimal_years_to_matplotlib_dates(g_x_dec)
    ax_ghost.fill_between(g_x, g_lo, g_hi, alpha=0.30, color=prev_color, linewidth=0)
    ax_ghost.plot(g_x, g_med, color=prev_color, lw=1.6)
    ax_ghost.set_yscale("log")
    ax_ghost.set_title(DEME_MAP.get(ghost_label, ghost_label), fontsize=fs["title"])
    ax_ghost.set_ylabel("Prevalence", fontsize=fs["axis_label"])
    ax_ghost.set_xlabel("Date", fontsize=fs["axis_label"])
    ax_ghost.tick_params(labelsize=fs["tick_label"])
    configure_calendar_xaxis(ax_ghost)
    for label in ax_ghost.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    beautify_plot(ax_ghost, remove_spines=False)

    # Tight log y-limits on the ghost prevalence row: span the visible HPD
    # range only, with a small multiplicative margin — otherwise matplotlib's
    # auto-scale stretches the lower bound toward 0 and the trajectory ends
    # up squashed into the top of the panel.
    if x_dec_range is not None:
        vis_mask = (g_x_dec >= x_dec_range[0]) & (g_x_dec <= x_dec_range[1])
    else:
        vis_mask = np.ones(len(g_x_dec), dtype=bool)
    y_vals = np.concatenate([np.asarray(g_lo)[vis_mask], np.asarray(g_hi)[vis_mask]])
    y_vals = y_vals[np.isfinite(y_vals) & (y_vals > 0)]
    if y_vals.size:
        log_margin = 1.3  # ~30% breathing room on the log scale
        ax_ghost.set_ylim(y_vals.min() / log_margin, y_vals.max() * log_margin)

    if x_dec_range is not None:
        x_lo_num, x_hi_num = _xlim_with_monthly_right_edge(x_dec_range)
        for ax in axes:
            ax.set_xlim(x_lo_num, x_hi_num)
    fig.tight_layout()
    _save_fig_png_pdf(fig, output_path)


def _draw_intros_vs_cases_panel(
    ax,
    x_dec: np.ndarray,
    intros_summary: tuple[np.ndarray, np.ndarray, np.ndarray],
    total_new_inf_summary: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    intros_color: str,
    total_new_inf_color: str,
    county_name: str,
    show_xlabel: bool = False,
) -> plt.Axes:
    """One panel of the intros-vs-new-infections supplementary figure.

    Left y-axis: introductions ``Σ_{j≠i} m_fwd[j→i]·I_j`` (deme color).
    Right y-axis: total new-infection rate ``β·I_local`` (grey, dashed) —
    includes both within-deme transmission and imports.
    Both show median + 95% HPD band. Returns the right (twin) axis.
    """
    fs = DEFAULT_FONTSIZES
    x_plot = decimal_years_to_matplotlib_dates(np.asarray(x_dec, dtype=float))

    intros_med, intros_lo, intros_hi = intros_summary
    total_med, total_lo, total_hi = total_new_inf_summary
    valid_i = ~np.isnan(intros_med)
    valid_t = ~np.isnan(total_med)

    # Left axis — introductions.
    if valid_i.any():
        ax.fill_between(
            x_plot[valid_i],
            intros_lo[valid_i],
            intros_hi[valid_i],
            alpha=0.30,
            color=intros_color,
            linewidth=0,
        )
        ax.plot(
            x_plot[valid_i],
            intros_med[valid_i],
            color=intros_color,
            lw=1.8,
            label="Introductions",
        )
    ax.set_title(county_name, fontsize=fs["title"])
    ax.set_ylabel("Introductions /year", color=intros_color, fontsize=fs["axis_label"])
    ax.tick_params(axis="y", labelcolor=intros_color, labelsize=fs["tick_label"])
    ax.tick_params(axis="x", labelsize=fs["tick_label"])
    ax.set_yscale("log")

    # Right axis — total new-infection rate β·I (includes imports).
    ax2 = ax.twinx()
    if valid_t.any():
        ax2.fill_between(
            x_plot[valid_t],
            total_lo[valid_t],
            total_hi[valid_t],
            alpha=0.20,
            color=total_new_inf_color,
            linewidth=0,
        )
        ax2.plot(
            x_plot[valid_t],
            total_med[valid_t],
            color=total_new_inf_color,
            lw=1.4,
            linestyle="--",
            label="New infections",
        )
    ax2.set_ylabel(
        "New infections /year",
        color=total_new_inf_color,
        fontsize=fs["axis_label"],
        rotation=270,
        labelpad=15,
    )
    ax2.tick_params(
        axis="y", labelcolor=total_new_inf_color, labelsize=fs["tick_label"]
    )
    ax2.set_yscale("log")

    if show_xlabel:
        ax.set_xlabel("Date", fontsize=fs["axis_label"])
    configure_calendar_xaxis(ax)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    beautify_plot(ax, remove_spines=False)
    return ax2


def _draw_trajectory_band(
    ax,
    x_plot: np.ndarray,
    summary: tuple[np.ndarray, np.ndarray, np.ndarray],
    color: str,
    *,
    lw: float = 1.6,
    band_alpha: float = 0.30,
) -> None:
    """Plot a median line + 95% HPD band on *ax* using matplotlib date x values."""
    med, lo, hi = summary
    valid = ~np.isnan(med)
    if not valid.any():
        return
    ax.fill_between(
        x_plot[valid], lo[valid], hi[valid], alpha=band_alpha, color=color, linewidth=0
    )
    ax.plot(x_plot[valid], med[valid], color=color, lw=lw)


def plot_intros_vs_local_cases(
    intros_data: dict[str, dict],
    output_path: Path,
    *,
    focal_labels: tuple[str, ...] = FOCAL_DEME_LABELS,
    ghost_label: str = _GHOST_LABEL,
    x_dec_range: tuple[float, float] | None = None,
    total_new_inf_color: str = "0.30",
) -> None:
    """Supplementary 4×N grid: β, focal prevalence, ghost prevalence, intros vs new infections.

    One column per focal deme.  Rows (top to bottom):

      1. Raw transmission rate β (linear y, deme color).
      2. Raw focal-deme prevalence I_local (log y, deme color).
      3. Raw ghost-deme prevalence I_outside (log y) — identical in every column.
      4. Introductions ``Σ_{j≠i} m_fwd[j→i] · I_j`` summed over all other
         demes (left log y, deme color) and total new-infection rate
         ``β · I_local`` (right log y, grey dashed).

    All panels share the x-axis. Within each row, panels also share y so
    magnitudes are visually comparable across demes.
    """
    fs = DEFAULT_FONTSIZES
    n = len(focal_labels)
    fig, axes = plt.subplots(
        4,
        n,
        figsize=(3.4 * n + 1.0, 9.5),
        sharex=True,
        squeeze=False,
    )

    # Per-row y-sharing for the three raw-trajectory rows.
    for r in (0, 1, 2):
        for c in range(1, n):
            axes[r, c].sharey(axes[r, 0])
            axes[r, c].tick_params(axis="y", labelleft=False)

    ghost = intros_data[ghost_label]
    ghost_x_plot = decimal_years_to_matplotlib_dates(
        np.asarray(ghost["x_dec"], dtype=float)
    )
    ghost_color = DEME_COLORS[ghost_label]
    ghost_name = DEME_MAP.get(ghost_label, ghost_label)

    right_axes: list = []
    for c, lab in enumerate(focal_labels):
        d = intros_data[lab]
        x_plot = decimal_years_to_matplotlib_dates(np.asarray(d["x_dec"], dtype=float))
        deme_color = DEME_COLORS[lab]
        county_name = DEME_MAP.get(lab, lab)

        # Row 0 — β.
        ax = axes[0, c]
        _draw_trajectory_band(ax, x_plot, d["beta"], deme_color)
        ax.set_title(county_name, fontsize=fs["title"])
        if c == 0:
            ax.set_ylabel("Transmission rate β\n(/year)", fontsize=fs["axis_label"])
        ax.tick_params(axis="y", labelsize=fs["tick_label"])
        ax.tick_params(axis="x", labelbottom=False)
        beautify_plot(ax, remove_spines=False)

        # Row 1 — focal-deme prevalence (log y).
        ax = axes[1, c]
        _draw_trajectory_band(ax, x_plot, d["prevalence"], deme_color)
        ax.set_yscale("log")
        if c == 0:
            ax.set_ylabel("Local prevalence\nI", fontsize=fs["axis_label"])
        ax.tick_params(axis="y", labelsize=fs["tick_label"])
        ax.tick_params(axis="x", labelbottom=False)
        beautify_plot(ax, remove_spines=False)

        # Row 2 — ghost-deme prevalence (log y), same series in every column.
        ax = axes[2, c]
        _draw_trajectory_band(ax, ghost_x_plot, ghost["prevalence"], ghost_color)
        ax.set_yscale("log")
        if c == 0:
            ax.set_ylabel(f"{ghost_name}\nprevalence I", fontsize=fs["axis_label"])
        ax.tick_params(axis="y", labelsize=fs["tick_label"])
        ax.tick_params(axis="x", labelbottom=False)
        beautify_plot(ax, remove_spines=False)

        # Row 3 — intros (left y, log) + total new infections (right y, log).
        ax = axes[3, c]
        ax2 = _draw_intros_vs_cases_panel(
            ax,
            d["x_dec"],
            d["intros"],
            d["total_new_inf"],
            intros_color=deme_color,
            total_new_inf_color=total_new_inf_color,
            county_name="",
            show_xlabel=True,
        )
        right_axes.append(ax2)
        if c > 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        if c < n - 1:
            ax2.set_ylabel("")
            ax2.tick_params(axis="y", labelright=False)

    # Share y across columns for row 3 (left and right axes separately) so
    # magnitudes are comparable across demes.
    for c in range(1, n):
        axes[3, c].sharey(axes[3, 0])
        right_axes[c].sharey(right_axes[0])

    if x_dec_range is not None:
        x_lo_num, x_hi_num = _xlim_with_monthly_right_edge(x_dec_range)
        for ax in axes.flat:
            ax.set_xlim(x_lo_num, x_hi_num)

    fig.tight_layout()
    _save_fig_png_pdf(fig, output_path)


def _scale_linewidths(
    rates: list[float],
    lw_min: float = 1.0,
    lw_max: float = 5.0,
    absolute_min=None,
) -> list[float]:
    if not rates:
        return []
    lo = min(rates) if absolute_min is None else absolute_min
    hi = max(rates)
    if hi <= lo:
        return [0.5 * (lw_min + lw_max)] * len(rates)
    return [lw_min + (r - lo) / (hi - lo) * (lw_max - lw_min) for r in rates]


def _counties_gdf_from_geojson(data: dict) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame with a ``fips`` column from Plotly county GeoJSON."""
    rows: list[dict] = []
    for feat in data["features"]:
        fid = str(feat.get("id", ""))
        if not fid:
            continue
        rows.append({"fips": fid, "geometry": shape(feat["geometry"])})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _point3857_to_lonlat(p: np.ndarray) -> tuple[float, float]:
    g = gpd.GeoSeries([Point(float(p[0]), float(p[1]))], crs="EPSG:3857").to_crs(
        "EPSG:4326"
    )
    return float(g.iloc[0].x), float(g.iloc[0].y)


def load_plotly_counties_geojson(
    url: str,
    cache_path: Path | None,
) -> dict:
    """
    Load Plotly's US county GeoJSON (urllib); optional on-disk cache under
    output_dir to avoid repeated downloads (~3 MB).
    """
    if cache_path is not None and cache_path.is_file():
        text = cache_path.read_text(encoding="utf-8")
        return json.loads(text)
    req = urllib.request.urlopen(url, timeout=120)
    raw = req.read()
    data = json.loads(raw.decode("utf-8"))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
    return data


def _migration_map_central_lon(map_lon_min: float, map_lon_max: float) -> float:
    return 0.5 * (map_lon_min + map_lon_max)


def _draw_migration_rates_map(
    ax,
    median_rates: dict[tuple[str, str], float],
    *,
    map_lon_min: float,
    map_lon_max: float,
    map_lat_min: float,
    map_lat_max: float,
    geojson_cache_path: Path | None = None,
    geojson_url: str = PLOTLY_COUNTIES_GEOJSON_URL,
    focal_labels: tuple[str, ...] = FOCAL_DEME_LABELS,
    show_ghost_inflow_inset: bool = False,
    pops: dict[int, float] | None = None,
) -> None:
    """Draw migration arrows and counties onto an existing ``ax``.

    Focal counties are filled with ``DEME_COLORS``.
    """
    if map_lon_max <= map_lon_min or map_lat_max <= map_lat_min:
        raise ValueError(
            "map viewport requires map_lon_max > map_lon_min and "
            "map_lat_max > map_lat_min"
        )

    data = load_plotly_counties_geojson(geojson_url, geojson_cache_path)
    gdf = _counties_gdf_from_geojson(data)
    viewport = box(map_lon_min, map_lat_min, map_lon_max, map_lat_max)

    focal_fips = {COUNTY_FIPS_BY_DEME[lab] for lab in focal_labels}
    for lab in focal_labels:
        fid = COUNTY_FIPS_BY_DEME[lab]
        sub = gdf[gdf["fips"] == fid]
        if sub.empty:
            raise KeyError(f"Missing county GeoJSON feature for FIPS {fid} ({lab})")
        geom = sub.geometry.iloc[0]
        if not geom.intersects(viewport):
            raise ValueError(
                f"County FIPS {fid} ({lab}) does not intersect the built-in map viewport; "
                "edit DEFAULT_MAP_LON_MIN/MAX and DEFAULT_MAP_LAT_MIN/MAX in "
                "analyse_posteriors.py if needed."
            )

    gdf_view = gdf[gdf.intersects(viewport)].copy()
    others = gdf_view[~gdf_view["fips"].isin(focal_fips)]
    focal = gdf[gdf["fips"].isin(focal_fips)].copy()

    pos_3857: dict[str, np.ndarray] = {}
    for lab in focal_labels:
        fid = COUNTY_FIPS_BY_DEME[lab]
        c = focal[focal["fips"] == fid].geometry.iloc[0].centroid
        gcent = gpd.GeoDataFrame(geometry=[c], crs="EPSG:4326").to_crs(3857)
        pos_3857[lab] = np.array(
            [gcent.geometry.iloc[0].x, gcent.geometry.iloc[0].y], dtype=float
        )

    ax.set_extent(
        [map_lon_min, map_lon_max, map_lat_min, map_lat_max],
        crs=ccrs.PlateCarree(),
    )
    ax.set_facecolor("#c8d8e8")

    others.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        facecolor="#ececec",
        edgecolor="#6a6a6a",
        linewidth=0.35,
        alpha=0.78,
        zorder=3,
    )
    # Color focal counties by deme.
    for lab in focal_labels:
        fid = COUNTY_FIPS_BY_DEME[lab]
        sub = focal[focal["fips"] == fid]
        sub.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            facecolor=DEME_COLORS[lab],
            edgecolor="#2a2a2a",
            linewidth=0.85,
            alpha=0.55,
            zorder=4,
        )

    # Reveal water (SF Bay) that the county polygons paint over: county
    # boundaries are legal jurisdictions that extend across the Bay, so draw
    # true water features on top of the fills but below arrows/labels.
    for feat in (cfeature.OCEAN, cfeature.LAKES):
        ax.add_feature(
            feat.with_scale("10m"),
            facecolor="#c8d8e8",  # matches the axes background water tone
            edgecolor="none",
            zorder=4.5,
        )

    # txt_halo = [patheffects.withStroke(linewidth=3.0, foreground="white", alpha=0.92)]
    for lab in focal_labels:
        fid = COUNTY_FIPS_BY_DEME[lab]
        c = focal[focal["fips"] == fid].geometry.iloc[0].centroid
        ax.text(
            c.x,
            c.y,
            DEME_MAP.get(lab, lab),
            transform=ccrs.PlateCarree(),
            ha="center",
            va="center",
            fontsize=DEFAULT_FONTSIZES["tick_label"],
            color="0.12",
            fontweight="medium",
            zorder=7,
            # path_effects=txt_halo,
        )

    internal_pairs: list[tuple[str, str]] = []
    for a in focal_labels:
        for b in focal_labels:
            if a == b:
                continue
            internal_pairs.append((a, b))

    all_rates: list[float] = []
    for fr, to in internal_pairs:
        key = (fr, to)
        if key not in median_rates:
            raise KeyError(
                f"Missing median migration rate for {fr} -> {to} "
                f"(expected column f_migrationRatesSkyline.{fr}_to_{to})"
            )
        all_rates.append(median_rates[key])

    lws = _scale_linewidths(all_rates, absolute_min=0)
    lw_map = dict(zip(internal_pairs, lws))

    ll_transform = ccrs.PlateCarree()._as_mpl_transform(ax)
    # arrow_halo = [patheffects.withStroke(linewidth=2.5, foreground="white", alpha=0.85)]

    # Perpendicular half-gap in metres (EPSG:3857): same side-to-side distance for
    # every pair (was perp_frac * edge_length, so long edges looked too wide).
    arrow_pair_half_gap_m = 1400.0
    chord_inset = 0.08
    connection_style = "arc3,rad=0.15"

    focal_list = list(focal_labels)
    for i, a in enumerate(focal_list):
        for b in focal_list[i + 1 :]:
            pa = pos_3857[a]
            pb = pos_3857[b]
            edge = pb - pa
            elen = float(np.linalg.norm(edge))
            if elen == 0.0:
                continue
            u = edge / elen
            perp = np.array([-u[1], u[0]])
            off = arrow_pair_half_gap_m * perp

            t0 = chord_inset * elen
            t1 = (1.0 - chord_inset) * elen
            p_ab_s = pa + off + u * t0
            p_ab_e = pa + off + u * t1
            lon1, lat1 = _point3857_to_lonlat(p_ab_s)
            lon2, lat2 = _point3857_to_lonlat(p_ab_e)
            lw_ab = lw_map[(a, b)]
            arr_ab = FancyArrowPatch(
                (lon1, lat1),
                (lon2, lat2),
                transform=ll_transform,
                connectionstyle=connection_style,
                arrowstyle="-|>",
                mutation_scale=10.0 + 1.2 * lw_ab,
                linewidth=lw_ab,
                color="0.18",
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=5,
                # path_effects=arrow_halo,
            )
            ax.add_patch(arr_ab)

            p_ba_s = pa - off + u * t1
            p_ba_e = pa - off + u * t0
            lon3, lat3 = _point3857_to_lonlat(p_ba_s)
            lon4, lat4 = _point3857_to_lonlat(p_ba_e)
            lw_ba = lw_map[(b, a)]
            arr_ba = FancyArrowPatch(
                (lon3, lat3),
                (lon4, lat4),
                transform=ll_transform,
                connectionstyle=connection_style,
                arrowstyle="-|>",
                mutation_scale=10.0 + 1.2 * lw_ba,
                linewidth=lw_ba,
                color="0.18",
                shrinkA=0.0,
                shrinkB=0.0,
                zorder=5,
                # path_effects=arrow_halo,
            )
            ax.add_patch(arr_ba)

    if show_ghost_inflow_inset:
        plot_ghost_inflow_fraction_inset(
            ax,
            median_rates,
            focal_labels=focal_labels,
            ghost_label=_GHOST_LABEL,
        )

    # Degree labels come from gridlines on GeoAxes, not set_xticks/set_yticks.
    # ax.gridlines(
    #     draw_labels=False,
    #     linewidth=0.45,
    #     color="gray",
    #     alpha=0.5,
    #     linestyle="--",
    # )


def plot_migration_rates_map(
    median_rates: dict[tuple[str, str], float],
    output_path: Path,
    *,
    map_lon_min: float,
    map_lon_max: float,
    map_lat_min: float,
    map_lat_max: float,
    geojson_cache_path: Path | None = None,
    geojson_url: str = PLOTLY_COUNTIES_GEOJSON_URL,
    focal_labels: tuple[str, ...] = FOCAL_DEME_LABELS,
    pops: dict[int, float] | None = None,
) -> None:
    """
    GeoPandas + Cartopy: Natural Earth land/ocean (no OSM or terrain tiles),
    county polygons intersecting the viewport, focal demes in grey (optionally
    scaled by ``pops``). Straight arrow segments between focal demes (fixed lateral
    offset in metres); geometry in EPSG:3857 for offsets, then drawn in WGS84.
    Ghost deme rates are not shown.
    """
    central_lon = _migration_map_central_lon(map_lon_min, map_lon_max)
    fig = plt.figure(figsize=(10.0, 9.0))
    ax = plt.axes(projection=ccrs.Mercator(central_longitude=central_lon))
    _draw_migration_rates_map(
        ax,
        median_rates,
        map_lon_min=map_lon_min,
        map_lon_max=map_lon_max,
        map_lat_min=map_lat_min,
        map_lat_max=map_lat_max,
        geojson_cache_path=geojson_cache_path,
        geojson_url=geojson_url,
        focal_labels=focal_labels,
        pops=pops,
    )
    fig.tight_layout()
    _save_fig_png_pdf(fig, output_path, dpi=175)


def plot_prevalence_trajectories(
    deme_summaries: list[tuple[str, np.ndarray, dict[str, np.ndarray]]],
    deme: str,
    output_path: Path,
    *,
    x_domain: Literal["trajectory_index", "decimal_year"] = "trajectory_index",
    xlabel: str = "Trajectory index (left = past, right = present)",
) -> None:
    """
    One figure for ``deme``: x = trajectory index or calendar time, y = median and
    95% HPD of I.

    Parameters
    ----------
    deme_summaries
        List of (deme_label, x_array, summary_dict) where summary_dict has
        keys used for plotting (same length as x_array).
    deme
        The deme to plot the prevalence trajectory for.
    x_domain
        ``trajectory_index``: use ``x_array`` as-is (grid indices after reversal).
        ``decimal_year``: ``x_array`` is BEAST-style decimal years; converted to
        Matplotlib date numbers and the x-axis is formatted as calendar dates.
    xlabel
        X-axis label (e.g. ``Date`` when ``x_domain='decimal_year'``).
    """
    if x_domain not in ("trajectory_index", "decimal_year"):
        raise ValueError(
            f"x_domain must be 'trajectory_index' or 'decimal_year', got {x_domain!r}"
        )
    matching = [d for d in deme_summaries if d[0] == deme]
    if not matching:
        raise ValueError(
            f"No NeDynamics summary for {deme}; cannot write {output_path.name}"
        )
    label = matching[0][0]
    x_raw = matching[0][1]
    summ = matching[0][2]
    print(f"Prevalence traj, Deme4: {x_raw.max()}, {x_raw.min()}")
    if x_domain == "decimal_year":
        x_plot = decimal_years_to_matplotlib_dates(np.asarray(x_raw, dtype=float))
    else:
        x_plot = np.asarray(x_raw, dtype=float)

    print(f"Prevalence traj, Deme4: {x_plot.max()}, {x_plot.min()}")

    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.fill_between(
        x_plot,
        summ["hpd_lo_I"],
        summ["hpd_hi_I"],
        alpha=0.35,
        label="95% HPD",
    )
    ax.plot(x_plot, summ["median_I"], color="C0", lw=1.5, label="Median")
    ax.set_title(DEME_MAP[label])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Prevalence")
    ax.legend(loc="best", fontsize=DEFAULT_FONTSIZES["legend"])
    ax.grid(True, alpha=0.3)
    if x_domain == "decimal_year":
        configure_calendar_xaxis(ax)
        fig.autofmt_xdate()
    beautify_plot(ax, remove_spines=True)
    fig.tight_layout()
    _save_fig_png_pdf(fig, output_path)


def plot_local_prevalence_trajectories_stacked(
    deme_summaries: list[tuple[str, np.ndarray, dict[str, np.ndarray]]],
    output_path: Path,
    *,
    focal_labels: tuple[str, ...] = FOCAL_DEME_LABELS,
    x_domain: Literal["trajectory_index", "decimal_year"] = "trajectory_index",
    xlabel: str = "Trajectory index (left = past, right = present)",
) -> None:
    """Standalone investigation figure: focal-deme prevalence trajectories
    (median + 95% HPD of I), one stacked subplot per deme.

    Mirrors the single-panel background-deme figure from
    :func:`plot_prevalence_trajectories` (same data source, HPD band, calendar
    x-axis), but with one panel per local deme stacked vertically and coloured
    by deme.  Saved independently — it is not part of any composite figure.
    """
    if x_domain not in ("trajectory_index", "decimal_year"):
        raise ValueError(
            f"x_domain must be 'trajectory_index' or 'decimal_year', got {x_domain!r}"
        )
    summ_by_label = {lab: (x_arr, summ) for lab, x_arr, summ in deme_summaries}
    missing = [lab for lab in focal_labels if lab not in summ_by_label]
    if missing:
        raise ValueError(
            f"No NeDynamics summary for focal deme(s) {missing}; "
            f"cannot write {output_path.name}"
        )

    fs = DEFAULT_FONTSIZES
    n = len(focal_labels)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, lab in zip(axes, focal_labels):
        x_raw, summ = summ_by_label[lab]
        if x_domain == "decimal_year":
            x_plot = decimal_years_to_matplotlib_dates(np.asarray(x_raw, dtype=float))
        else:
            x_plot = np.asarray(x_raw, dtype=float)
        color = DEME_COLORS.get(lab, "C0")
        ax.fill_between(
            x_plot,
            summ["hpd_lo_I"],
            summ["hpd_hi_I"],
            alpha=0.35,
            color=color,
            linewidth=0,
            label="95% HPD",
        )
        ax.plot(x_plot, summ["median_I"], color=color, lw=1.5, label="Median")
        ax.set_title(DEME_MAP.get(lab, lab), fontsize=fs["title"])
        ax.set_ylabel("Prevalence", fontsize=fs["axis_label"])
        ax.tick_params(labelsize=fs["tick_label"])
        ax.legend(loc="best", fontsize=fs["legend"])
        ax.grid(True, alpha=0.3)
        beautify_plot(ax, remove_spines=True)
    axes[-1].set_xlabel(xlabel, fontsize=fs["axis_label"])
    if x_domain == "decimal_year":
        configure_calendar_xaxis(axes[-1])
        fig.autofmt_xdate()
    fig.tight_layout()
    _save_fig_png_pdf(fig, output_path)


def trajectory_indices_to_decimal_year(
    x_idx: np.ndarray,
    grid_shifts: np.ndarray,
    t_recent: float,
) -> np.ndarray:
    """Map NeDynamics grid indices to decimal year using SplineGridRateShifts.

    *x_idx* contains reversed column indices (after
    ``reverse_trajectory_for_time_forward``): e.g. ``[1000, 990, …, 10, 0]``
    where index 0 corresponds to the present. *grid_shifts* gives the actual
    time-before-present for each grid point (``SplineGridRateShifts`` from the
    BEAST XML), which need **not** be evenly spaced.
    """
    idx = np.round(np.asarray(x_idx, dtype=float)).astype(int)
    times_before_present = grid_shifts[idx]
    return t_recent - times_before_present


def reverse_trajectory_for_time_forward(
    x_idx: np.ndarray,
    summary: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    NeDynamics grid indices run backward in time (high index = start, low = recent).

    Reverse point order so plots read left-to-right as time moving forward.
    """
    x_rev = x_idx[::-1].copy()
    summ_rev = {k: np.asarray(v, dtype=float)[::-1].copy() for k, v in summary.items()}
    return x_rev, summ_rev


def read_beast_log_dataframe(path: Path) -> pd.DataFrame:
    """BEAST log as DataFrame (skip # comment lines); numeric columns coerced."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if not ln.startswith("#")]
    rows = [ln.split("\t") for ln in lines]
    if len(rows) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df.apply(pd.to_numeric, errors="coerce")


def extract_spline_grid_shifts(path: Path) -> np.ndarray:
    """Parse SplineGridRateShifts from BEAST log comment lines (same as MASCOT-DS XML)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") and 'id="SplineGridRateShifts"' in line:
                start_tag = "<gridRateShifts"
                end_tag = "</gridRateShifts>"
                start_idx = line.find(start_tag)
                if start_idx == -1:
                    continue
                tag_end_idx = line.find(">", start_idx)
                end_idx = line.find(end_tag, tag_end_idx)
                if tag_end_idx == -1 or end_idx == -1:
                    continue
                shifts_str = line[tag_end_idx + 1 : end_idx].strip()
                return np.array([float(x) for x in shifts_str.split()], dtype=float)
    raise ValueError(
        f"No SplineGridRateShifts comment found in {path}; "
        "use --mascot-datastream-log pointing to a datastream run log."
    )


def load_tree_height_after_burnin(path: Path, burnin_fraction: float) -> pd.DataFrame:
    df = read_beast_log_dataframe(path)
    if df.empty or "Sample" not in df.columns or "Tree.height" not in df.columns:
        raise ValueError(f"{path} must contain Sample and Tree.height columns.")
    if burnin_fraction < 0 or burnin_fraction >= 1:
        raise ValueError("burnin_fraction must be in [0, 1).")
    df = df.loc[df["Sample"] > df["Sample"].max() * burnin_fraction]
    return df[["Sample", "Tree.height"]].copy()


def parse_column_name_cumulative(col_name: str):
    if str(col_name).startswith("cumulativeIncidence_"):
        rest = str(col_name).split("_", 1)[1]
        return "cumulativeIncidence", float(rest)
    return None, None


def create_cumulative_incidence_long(df: pd.DataFrame) -> pd.DataFrame:
    wide_data: list[dict] = []
    for _, row in df.iterrows():
        sample = row["Sample"]
        by_gp: dict[float, dict] = {}
        for col_name, value in row.items():
            if col_name == "Sample":
                continue
            ptype, gridpoint = parse_column_name_cumulative(col_name)
            if ptype is None:
                continue
            if gridpoint not in by_gp:
                by_gp[gridpoint] = {"Sample": sample, "gridpoint": gridpoint}
            by_gp[gridpoint][ptype] = value
        for _, data in by_gp.items():
            wide_data.append(data)
    return pd.DataFrame(wide_data)


def timesincestart_from_gridpoints(
    gridpoints: np.ndarray, grid_shifts: np.ndarray
) -> np.ndarray:
    max_r = float(np.max(grid_shifts))
    idx = np.asarray(gridpoints, dtype=int)
    return max_r - grid_shifts[idx]


def calculate_hpd_mcmc(
    data: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, float]:
    x = np.sort(np.asarray(data, dtype=float))
    n = len(x)
    m = int((1 - alpha) * n)
    if m < 1:
        return float(x[0]), float(x[-1]), float(np.median(x))
    intervals = x[m:] - x[: n - m]
    min_idx = int(np.argmin(intervals))
    hpd_min = float(x[min_idx])
    hpd_max = float(x[min_idx + m])
    return hpd_min, hpd_max, float(np.median(x))


def hpd_by_time_grid(
    df: pd.DataFrame,
    n_gridpoints: int,
    column_name: str,
    deme_id: int,
) -> pd.DataFrame:
    min_time = float(df["timesincestart"].min())
    max_time = float(df["timesincestart"].max())
    time_grid = np.linspace(min_time, max_time, n_gridpoints)

    def assign_to_grid(time_val: float) -> float:
        j = int(np.argmin(np.abs(time_grid - time_val)))
        return float(time_grid[j])

    work = df.copy()
    work["timesinceroot_grid"] = work["timesincestart"].map(assign_to_grid)
    rows: list[dict] = []
    for grid_time, group in work.groupby("timesinceroot_grid"):
        lo, hi, med = calculate_hpd_mcmc(group[column_name].to_numpy(dtype=float))
        rows.append(
            {
                "Deme": deme_id,
                "timesincestart": float(grid_time),
                column_name: med,
                f"{column_name}_hpd_lower": lo,
                f"{column_name}_hpd_upper": hi,
            }
        )
    return pd.DataFrame(rows)


def process_cumulative_incidence_log_file(
    path: Path,
    deme_zero_based: int,
    tree_height: pd.DataFrame,
    grid_shifts: np.ndarray,
) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    raw = read_beast_log_dataframe(path)
    if raw.empty or "Sample" not in raw.columns:
        return pd.DataFrame()
    long_df = create_cumulative_incidence_long(raw)
    if long_df.empty or "cumulativeIncidence" not in long_df.columns:
        return pd.DataFrame()
    merged = long_df.merge(tree_height, on="Sample", how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged["timesincestart"] = timesincestart_from_gridpoints(
        merged["gridpoint"].to_numpy(dtype=float),
        grid_shifts,
    )
    n_gp = int(merged["gridpoint"].nunique())
    merged["Deme"] = int(deme_zero_based)
    return hpd_by_time_grid(merged, n_gp, "cumulativeIncidence", deme_zero_based)


def cuminc_timesincestart_to_decimal_year(
    ts: np.ndarray,
    t_recent: float,
    grid_shifts_max: float,
) -> np.ndarray:
    """Convert cumulative-incidence *timesincestart* to decimal year.

    *ts* values come from ``timesincestart_from_gridpoints`` which computes
    ``max(grid_shifts) - grid_shifts[idx]``: 0 at the most ancient grid point,
    ``max(grid_shifts)`` at the present. *grid_shifts_max* is
    ``max(SplineGridRateShifts)`` for the deme.
    """
    ts = np.asarray(ts, dtype=float)
    # time_before_present = grid_shifts_max - ts
    return t_recent - (grid_shifts_max - ts)


def print_prevalence_and_cuminc_summary(
    summ_by_label: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    cum_hpd: pd.DataFrame,
    pops: dict[int, float],
    x_dec_range: tuple[float, float],
    t_recent: float,
) -> None:
    """Print, per focal deme, the date/value of peak median prevalence and the
    median cumulative incidence (as % of county population, matching the
    right-hand panels of the datastream overview) at the end of the plotted
    time window.

    Restricts to ``x_dec_range`` (the shared x-axis window used across the
    datastream-overview and introductions figures) so the reported peak
    matches what is visible in the plots.
    """

    def fmt_date(dec_year: float) -> str:
        num = decimal_year_to_matplotlib_date(dec_year)
        return mdates.num2date(num).strftime("%Y-%m-%d")

    def fmt_date_long(dec_year: float) -> str:
        num = decimal_year_to_matplotlib_date(dec_year)
        dt = mdates.num2date(num)
        return f"{dt.day} {dt.strftime('%B %Y')}"

    x_lo, x_hi = float(x_dec_range[0]), float(x_dec_range[1])
    print(
        f"\nPrevalence peak & cumulative incidence summary "
        f"(plotted window {fmt_date(x_lo)} to {fmt_date(x_hi)}):"
    )
    peak_parts: list[str] = []
    peak_dates: list = []
    cum_parts: list[str] = []
    cum_end_dates: list[str] = []
    for d, lab in enumerate(FOCAL_DEME_LABELS):
        county_name = DEME_MAP.get(lab, lab)

        if lab in summ_by_label:
            x_dec, summary = summ_by_label[lab]
            x_dec = np.asarray(x_dec, dtype=float)
            mask = (x_dec >= x_lo) & (x_dec <= x_hi)
            if mask.any():
                median_I = np.asarray(summary["median_I"], dtype=float)[mask]
                hpd_lo_I = np.asarray(summary["hpd_lo_I"], dtype=float)[mask]
                hpd_hi_I = np.asarray(summary["hpd_hi_I"], dtype=float)[mask]
                x_masked = x_dec[mask]
                peak_idx = int(np.argmax(median_I))
                peak_date = x_masked[peak_idx]
                peak_dates.append(peak_date)
                peak_parts.append(
                    f"{county_name}: {median_I[peak_idx]:,.0f} "
                    f"[95% HPD interval: {hpd_lo_I[peak_idx]:,.0f}"
                    f"–{hpd_hi_I[peak_idx]:,.0f}] infections on "
                    f"{fmt_date_long(peak_date)}"
                )
            else:
                peak_parts.append(f"{county_name}: no prevalence data in plotted window")
        else:
            peak_parts.append(f"{county_name}: no NeDynamics summary available")

        sub_c = (
            cum_hpd[cum_hpd["Deme"] == d]
            if not cum_hpd.empty and "Deme" in cum_hpd.columns
            else pd.DataFrame()
        )
        pop = pops.get(d)
        cum_end_date: str | None = None
        if not sub_c.empty and lab in DEME_SPLINEGRID_SHIFTS and pop:
            ts = sub_c["timesincestart"].to_numpy(dtype=float)
            dec_x = cuminc_timesincestart_to_decimal_year(
                ts, t_recent, float(DEME_SPLINEGRID_SHIFTS[lab].max())
            )
            mask_c = dec_x <= x_hi
            if mask_c.any():
                dec_x_m = dec_x[mask_c]
                med_m = sub_c["cumulativeIncidence"].to_numpy(dtype=float)[mask_c]
                lo_m = sub_c["cumulativeIncidence_hpd_lower"].to_numpy(dtype=float)[
                    mask_c
                ]
                hi_m = sub_c["cumulativeIncidence_hpd_upper"].to_numpy(dtype=float)[
                    mask_c
                ]
                last_idx = int(np.argmax(dec_x_m))
                cum_end_date = fmt_date(dec_x_m[last_idx])
                cum_parts.append(
                    f"{100 * med_m[last_idx] / pop:.1f}% "
                    f"({100 * lo_m[last_idx] / pop:.1f}-"
                    f"{100 * hi_m[last_idx] / pop:.1f}%) for {county_name}"
                )
        if cum_end_date is None:
            cum_parts.append(f"no cumulative incidence data for {county_name}")
        else:
            cum_end_dates.append(cum_end_date)

    if peak_parts:
        n_counties = len(peak_parts)
        if peak_dates:
            span = (
                f"between {fmt_date_long(min(peak_dates))} and "
                f"{fmt_date_long(max(peak_dates))} "
            )
        else:
            span = ""
        sentence = "; ".join(peak_parts)
        print(
            f"\nMASCOT-DS estimated that prevalence peaked {span}"
            f"across the {n_counties} counties ({sentence})."
        )

    if cum_parts:
        if len(cum_parts) > 1:
            sentence = ", ".join(cum_parts[:-1]) + " and " + cum_parts[-1]
        else:
            sentence = cum_parts[0]
        end_date = cum_end_dates[-1] if cum_end_dates else fmt_date(x_hi)
        print(
            f"\nThe cumulative incidence at the end of the plotted period "
            f"({end_date}) was estimated to be {sentence}."
        )


def convert_date_to_numerical_date(date_series: pd.Series) -> pd.Series:
    date_series = pd.to_datetime(date_series)
    years = date_series.dt.year
    year_start = pd.to_datetime(years.astype(str) + "-01-01")
    next_year_start = pd.to_datetime((years + 1).astype(str) + "-01-01")
    days_in_year = (next_year_start - year_start).dt.days
    day_of_year = date_series.dt.dayofyear
    return years + (day_of_year / days_in_year)


def county_key(name: str) -> str:
    return str(name).strip().replace(" ", "_").lower()


def load_county_to_deme_zero_based(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    m: dict[str, int] = {}
    for i, row in df.iterrows():
        c = str(row["county"])
        m[county_key(c)] = int(i)
    return m


def load_county_populations_zero_based(path: Path) -> dict[int, float]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    result: dict[int, float] = {}
    for _, row in df.iterrows():
        key = county_key(str(row["county"]))
        for deme_idx, county_name in enumerate(_sorted_focal):
            if county_key(county_name) == key:
                result[deme_idx] = float(row["population"])
                break
        else:
            raise ValueError(
                f"county_populations.csv contains '{row['county']}' which has no match in "
                f"COUNTIES_OF_INTEREST ({sorted(COUNTIES_OF_INTEREST)}). "
                f"Re-run extract_county_populations.py with the current COUNTIES_OF_INTEREST."
            )
    missing = [_sorted_focal[i] for i in range(len(_sorted_focal)) if i not in result]
    if missing:
        raise ValueError(
            f"county_populations.csv is missing entries for: {missing}. "
            f"Re-run extract_county_populations.py."
        )
    return result


def load_case_counts_with_deme(
    path: Path, county_to_deme: dict[str, int]
) -> pd.DataFrame:
    if not path.is_file() or not county_to_deme:
        return pd.DataFrame()
    df = pd.read_csv(path)
    need = {"date", "case_counts", "deme"}
    if not need.issubset(df.columns):
        raise ValueError(
            f"Case counts CSV needs columns {need}, got {list(df.columns)}"
        )
    df = df.copy()
    df["_decimal_year"] = convert_date_to_numerical_date(pd.to_datetime(df["date"]))
    mapped = df["deme"].map(lambda x: county_to_deme.get(county_key(str(x))))
    df = df.loc[mapped.notna()].copy()
    df["Deme"] = mapped.loc[df.index].astype(int)
    return df


def load_wastewater_with_deme(
    path: Path, county_to_deme: dict[str, int]
) -> pd.DataFrame:
    if not path.is_file() or not county_to_deme:
        return pd.DataFrame()
    df = pd.read_csv(path)
    need = {"date", "wastewater", "deme"}
    if not need.issubset(df.columns):
        raise ValueError(f"Wastewater CSV needs columns {need}, got {list(df.columns)}")
    df = df.copy()
    df["_decimal_year"] = convert_date_to_numerical_date(pd.to_datetime(df["date"]))
    mapped = df["deme"].map(lambda x: county_to_deme.get(county_key(str(x))))
    df = df.loc[mapped.notna()].copy()
    df["Deme"] = mapped.loc[df.index].astype(int)
    return df


def load_seroprevalence_with_deme(
    path: Path, county_to_deme: dict[str, int]
) -> pd.DataFrame:
    if not path.is_file() or not county_to_deme:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" not in df.columns or "deme" not in df.columns:
        raise ValueError(
            f"Seroprevalence CSV needs date and deme; got {list(df.columns)}"
        )
    df = df.copy()
    if "seroprevalence" not in df.columns:
        ntest = "seroprevalence_numpeopletested"
        npos = "seroprevalence_numpeoplewithantibodies"
        if ntest not in df.columns or npos not in df.columns:
            raise ValueError(
                f"Seroprevalence CSV needs seroprevalence or {ntest}+{npos}"
            )
        nt = df[ntest].astype(float)
        na = df[npos].astype(float)
        df["seroprevalence"] = np.where(nt > 0, na / nt, np.nan)
    df["_decimal_year"] = convert_date_to_numerical_date(pd.to_datetime(df["date"]))
    mapped = df["deme"].map(lambda x: county_to_deme.get(county_key(str(x))))
    df = df.loc[mapped.notna()].copy()
    df["Deme"] = mapped.loc[df.index].astype(int)
    return df


def concat_cumulative_hpd(
    paths_and_demes: list[tuple[Path | None, int]],
    tree_height: pd.DataFrame,
    grid_shifts: np.ndarray,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for p, d in paths_and_demes:
        if p is None:
            continue
        h = process_cumulative_incidence_log_file(p, d, tree_height, grid_shifts)
        if not h.empty:
            parts.append(h)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def plot_prevalence_with_datastreams(
    ax: plt.Axes,
    x_dates: np.ndarray,
    summary: dict[str, np.ndarray],
    case_sub: pd.DataFrame,
    ww_sub: pd.DataFrame,
    fig: plt.Figure,
    ncols: int,
    *,
    deme_color: str = "#888888",
    show_prevalence_ylabel: bool = True,
    show_secondary_ylabels: bool = True,
    show_legend: bool = True,
) -> None:
    """
    ``x_dates`` must be matplotlib date numbers (see ``decimal_years_to_matplotlib_dates``).
    """
    fs = DEFAULT_FONTSIZES
    ax.fill_between(
        x_dates,
        summary["hpd_lo_I"],
        summary["hpd_hi_I"],
        alpha=0.35,
        color=deme_color,
        zorder=2,
    )
    ax.plot(
        x_dates,
        summary["median_I"],
        color=deme_color,
        lw=2,
        alpha=0.75,
        zorder=4,
        label="Prevalence",
    )
    if show_prevalence_ylabel:
        ax.set_ylabel("Prevalence", fontsize=fs["axis_label"])
    else:
        ax.set_ylabel("")
    ax.tick_params(labelsize=fs["tick_label"])

    ax2 = None
    ax3 = None
    use_twins = (case_sub is not None and not case_sub.empty) or (
        ww_sub is not None and not ww_sub.empty
    )
    if use_twins:
        ax.set_zorder(3)
        ax.patch.set_visible(False)

    if case_sub is not None and not case_sub.empty:
        ax2 = ax.twinx()
        ax2.set_zorder(1)
        t_b = decimal_years_to_matplotlib_dates(
            case_sub["_decimal_year"].to_numpy(dtype=float)
        )
        h = case_sub["case_counts"].to_numpy(dtype=float)
        order = np.argsort(t_b)
        t_b = t_b[order]
        h = h[order]
        if len(t_b) > 1:
            bw = float(
                min(np.diff(t_b).min() * 0.8, (t_b.max() - t_b.min()) / len(t_b) * 0.8)
            )
        else:
            bw = 0.02
        ax2.bar(t_b, h, width=bw, color=COLOR_CASE, label="Case counts", zorder=1)
        if show_secondary_ylabels:
            ax2.set_ylabel(
                "Case counts",
                rotation=270,
                labelpad=6.0,
                va="center",
                fontsize=fs["axis_label"],
            )
        else:
            ax2.set_ylabel("")
        ax2.tick_params(axis="y", labelsize=fs["tick_label"])

    if ww_sub is not None and not ww_sub.empty:
        ax3 = ax.twinx()
        ax3.set_zorder(2)
        ax3.patch.set_visible(False)
        if ax2 is not None:
            ax3.spines["right"].set_position(("outward", 45))
        ax3.plot(
            decimal_years_to_matplotlib_dates(
                ww_sub["_decimal_year"].to_numpy(dtype=float)
            ),
            ww_sub["wastewater"],
            color=COLOR_WW,
            lw=1,
            marker="o",
            ms=2,
            zorder=2,
            label="Wastewater",
        )
        if show_secondary_ylabels:
            ax3.set_ylabel(
                "Wastewater",
                rotation=270,
                labelpad=6.0,
                va="center",
                fontsize=fs["axis_label"],
            )
        else:
            ax3.set_ylabel("")
        ax3.tick_params(axis="y", labelsize=fs["tick_label"])

    handles = [
        Line2D([], [], color=deme_color, lw=2, alpha=0.75, label="Prev."),
        Patch(facecolor=COLOR_CASE, edgecolor=COLOR_CASE, label="CC"),
        Line2D(
            [],
            [],
            color=COLOR_WW,
            lw=1,
            marker="o",
            ms=2,
            linestyle="-",
            label="WW",
        ),
    ]
    ax.set_ylim(0, ax.get_ylim()[1])
    if ax2 is not None:
        ax2.set_ylim(0, ax2.get_ylim()[1])
    if ax3 is not None:
        ax3.set_ylim(0, ax3.get_ylim()[1])
    if show_legend:
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=fs["legend"],
            frameon=False,
        )


def plot_cumulative_with_seroprevalence(
    ax: plt.Axes,
    x_dates: np.ndarray,
    cum_med: np.ndarray,
    cum_lo: np.ndarray,
    cum_hi: np.ndarray,
    sero_sub: pd.DataFrame,
    popsize: float | None,
    *,
    deme_color: str = "#888888",
    show_cumulative_ylabel: bool = True,
    show_seroprevalence_ylabel: bool = True,
    show_legend: bool = True,
) -> None:
    """
    ``x_dates`` must be matplotlib date numbers (see ``decimal_years_to_matplotlib_dates``).
    """
    if popsize is None or float(popsize) <= 0:
        raise ValueError(
            "plot_cumulative_with_seroprevalence requires a positive popsize"
        )
    fs = DEFAULT_FONTSIZES
    ax.fill_between(
        x_dates,
        cum_lo / popsize,
        cum_hi / popsize,
        alpha=0.2,
        color=deme_color,
        zorder=2,
    )
    ax.plot(x_dates, cum_med / popsize, color=deme_color, lw=2, alpha=0.75, zorder=4)
    if show_cumulative_ylabel:
        ax.set_ylabel(
            "Cumulative incidence",
            fontsize=fs["axis_label"],
        )
    else:
        ax.set_ylabel("")
    ax2 = None
    if sero_sub is not None and not sero_sub.empty:
        ax2 = ax.twinx()
        ax2.plot(
            decimal_years_to_matplotlib_dates(
                sero_sub["_decimal_year"].to_numpy(dtype=float)
            ),
            sero_sub["seroprevalence"],
            color=COLOR_WW,
            lw=1,
            marker="o",
            ms=4,
        )
        if show_seroprevalence_ylabel:
            ax2.set_ylabel(
                "Seroprevalence",
                rotation=270,
                labelpad=6.0,
                va="center",
                fontsize=fs["axis_label"],
            )
        else:
            ax2.set_ylabel("")
    handles = [
        Line2D([], [], color=deme_color, lw=2, alpha=0.75, label="Cum. incidence"),
        Line2D(
            [],
            [],
            color=COLOR_WW,
            lw=1,
            marker="o",
            ms=4,
            linestyle="-",
            label="SP",
        ),
    ]
    ymax = float(cum_hi.max() / popsize) + 0.05
    if ax2 is not None:
        ymax = max(ymax, float(sero_sub["seroprevalence"].max()) + 0.05)
    ax.set_ylim(0, ymax)
    if ax2 is not None:
        ax2.set_ylim(0, ymax)
    if show_legend:
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=fs["legend"],
            frameon=False,
        )
    ax.tick_params(labelsize=fs["tick_label"])
    if ax2 is not None:
        ax2.tick_params(labelsize=fs["tick_label"])


def _draw_datastream_overview(
    fig: plt.Figure,
    axes: np.ndarray,
    *,
    summ_by_label: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    case_df: pd.DataFrame,
    ww_df: pd.DataFrame,
    sero_df: pd.DataFrame,
    cum_hpd: pd.DataFrame,
    pops: dict[int, float],
    t_recent: float,
    x_dec_range: tuple[float, float] | None = None,
) -> None:
    """Datastream overview: N rows (focal demes) × 2 cols (prevalence | cumulative).

    *axes* shape must be ``(N, 2)``.  Column 0 = prevalence + case counts +
    wastewater; column 1 = cumulative incidence + seroprevalence.  Rows are
    focal demes (top to bottom).
    """
    labels = FOCAL_DEME_LABELS
    nrows = len(labels)
    fs = DEFAULT_FONTSIZES

    for i, lab in enumerate(labels):
        county_name = DEME_MAP.get(lab, lab)
        dc = DEME_COLORS[lab]
        ax_prev = axes[i, 0]
        ax_cum = axes[i, 1]

        # Deme name as row title on the prevalence column.
        ax_prev.set_title(county_name, fontsize=fs["title"])

        # --- Prevalence (col 0) ---
        if lab not in summ_by_label:
            ax_prev.text(
                0.5,
                0.5,
                "No NeDynamics",
                ha="center",
                va="center",
                transform=ax_prev.transAxes,
                fontsize=fs["axis_label"],
            )
        else:
            x_dec, summary = summ_by_label[lab]
            d = i
            csub = (
                case_df[case_df["Deme"] == d] if not case_df.empty else pd.DataFrame()
            )
            wsub = ww_df[ww_df["Deme"] == d] if not ww_df.empty else pd.DataFrame()
            plot_prevalence_with_datastreams(
                ax_prev,
                decimal_years_to_matplotlib_dates(x_dec),
                summary,
                csub,
                wsub,
                fig,
                2,  # ncols for twin-axis offset calculation
                deme_color=dc,
                show_prevalence_ylabel=True,
                show_secondary_ylabels=True,
                show_legend=(i == 0),
            )

        # --- Cumulative incidence (col 1) ---
        d = i
        pop = pops.get(d)
        if cum_hpd.empty or "Deme" not in cum_hpd.columns:
            sub_c = pd.DataFrame()
        else:
            sub_c = cum_hpd[cum_hpd["Deme"] == d].sort_values("timesincestart")
        ssub = sero_df[sero_df["Deme"] == d] if not sero_df.empty else pd.DataFrame()
        if sub_c.empty:
            ax_cum.text(
                0.5,
                0.5,
                "No cumulative incidence log",
                ha="center",
                va="center",
                transform=ax_cum.transAxes,
                fontsize=fs["axis_label"],
            )
            ax_cum.set_ylabel("Cumulative incidence", fontsize=fs["axis_label"])
        else:
            ts = sub_c["timesincestart"].to_numpy(dtype=float)
            dec_x = cuminc_timesincestart_to_decimal_year(
                ts, t_recent, float(DEME_SPLINEGRID_SHIFTS[lab].max())
            )
            plot_cumulative_with_seroprevalence(
                ax_cum,
                decimal_years_to_matplotlib_dates(dec_x),
                sub_c["cumulativeIncidence"].to_numpy(dtype=float),
                sub_c["cumulativeIncidence_hpd_lower"].to_numpy(dtype=float),
                sub_c["cumulativeIncidence_hpd_upper"].to_numpy(dtype=float),
                ssub,
                pop,
                deme_color=dc,
                show_cumulative_ylabel=True,
                show_seroprevalence_ylabel=True,
                show_legend=(i == 0),
            )

    # Only the bottom row gets x-axis date labels.
    for i in range(nrows - 1):
        for c in range(2):
            axes[i, c].tick_params(axis="x", labelbottom=False)
    axes[nrows - 1, 0].set_xlabel("Date", fontsize=fs["axis_label"])
    axes[nrows - 1, 1].set_xlabel("Date", fontsize=fs["axis_label"])

    # x-axis lower bound. If x_dec_range is provided, use it verbatim so the
    # prevalence panels share limits with other figures (e.g. % introductions).
    if x_dec_range is not None:
        x_lo, x_hi = float(x_dec_range[0]), float(x_dec_range[1])
    else:
        x_lo = t_recent
        for df in (case_df, ww_df, sero_df):
            if df is not None and not df.empty and "_decimal_year" in df.columns:
                x_lo = min(x_lo, float(df["_decimal_year"].min()))
        x_hi = t_recent

    x_lo_num, x_hi_num = _xlim_with_monthly_right_edge((x_lo, x_hi))
    for ax in axes.ravel():
        ax.set_xlim(x_lo_num, x_hi_num)
    for i in range(nrows):
        for c in range(2):
            configure_calendar_xaxis(axes[i, c])
    for ax in axes.ravel():
        beautify_plot(ax, remove_spines=False)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")


def plot_datastream_overview(
    *,
    summ_by_label: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    case_df: pd.DataFrame,
    ww_df: pd.DataFrame,
    sero_df: pd.DataFrame,
    cum_hpd: pd.DataFrame,
    pops: dict[int, float],
    t_recent: float,
    output_path: Path,
    x_dec_range: tuple[float, float] | None = None,
) -> None:
    fig, axes = plt.subplots(_n_focal, 2, figsize=(12, 10), sharex="col")
    _draw_datastream_overview(
        fig,
        axes,
        summ_by_label=summ_by_label,
        case_df=case_df,
        ww_df=ww_df,
        sero_df=sero_df,
        cum_hpd=cum_hpd,
        pops=pops,
        t_recent=t_recent,
        x_dec_range=x_dec_range,
    )
    fig.tight_layout()
    _save_fig_png_pdf(fig, output_path)


def plot_final_figure_gridspec(
    *,
    tree_path: Path,
    median_rates: dict[tuple[str, str], float],
    rate_samples: dict[tuple[str, str], np.ndarray] | None = None,
    summ_by_label: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    case_df: pd.DataFrame,
    ww_df: pd.DataFrame,
    sero_df: pd.DataFrame,
    cum_hpd: pd.DataFrame,
    pops: dict[int, float],
    t_recent: float,
    output_path: Path,
    intros_data: dict[str, dict],
    map_lon_min: float = -123.62,
    map_lon_max: float = -120.68,
    map_lat_min: float = 34.95,
    map_lat_max: float = 39.98,
    geojson_cache_path: Path | None = None,
    geojson_url: str = PLOTLY_COUNTIES_GEOJSON_URL,
    figsize: tuple[float, float] = (16.0, 14.0),
    dpi: int = 150,
) -> None:
    """Assemble the combined final figure.

    Layout ((N_focal + 3) rows × 4 columns):

    * **Tree** — cols 0–1, all rows (left side, full height)
    * **Datastream overview** — cols 2–3, top N_focal rows (N deme-rows × 2 cols:
      prevalence + cases + wastewater | cumulative incidence + seroprevalence)
    * **Relative outside→local inflow** — col 2, row N_focal + 1
    * **% introductions panels** — col 2, rows N_focal + 2 : end (one row per
      focal deme, stacked; each with a dotted prevalence overlay on a twin axis)
    * **Map** — col 3, rows N_focal + 1 : end (right side)

    Grid row ``N_focal`` (between the datastream block and the inflow / map
    block) is intentionally left empty as a spacer.
    """
    # N focal-deme rows for the datastream block + 1 spacer row + 2 rows for the
    # inflow / introductions / map block in the right-hand columns.
    _n_gs_rows = _n_focal + 3
    fig = plt.figure(figsize=figsize, dpi=dpi)

    gs = GridSpec(
        _n_gs_rows,
        4,
        figure=fig,
        height_ratios=[1.0] * _n_gs_rows,
        # Column 3 (the map / cum-incidence column) is widened: the map spans
        # the tall N_focal+1:end block, so a wider cell lets the aspect-locked
        # Cartopy map fill more of its vertical space instead of letterboxing.
        width_ratios=[1.0, 1.0, 1.2, 1.6],
        # hspace here only affects the single visible row boundary between the
        # datastream-overview block (top rows) and the inflow / % introductions
        # / map block (bottom rows) — every other row gap is absorbed by a
        # row-spanning subplot. Kept generous so the top intros-panel title
        # clears the rotated "Date" tick labels of the prevalence panel above.
        hspace=0.90,
        wspace=0.40,
    )

    # --- Tree (cols 0-1, all rows, full height) ---
    ax_tree = fig.add_subplot(gs[:, 0:2])
    tree = bt.loadNexus(
        str(tree_path),
        treestring_regex=r"tree\s+\S+\s*=",
        absoluteTime=True,
        verbose=False,
    )
    ax_tree.set_aspect("auto")
    _draw_mcc_tree_deme_hpd(ax_tree, tree_path, tree=tree)

    # --- Relative outside->local inflow (col 2, row N_focal + 1) ---
    ax_inflow = fig.add_subplot(gs[_n_focal, 2])
    draw_ghost_inflow_migration(
        ax_inflow,
        median_rates,
        rate_samples=rate_samples,
        pops=pops,
    )

    # --- Map (col 3, rows N_focal + 1 : end) ---
    ax_map = fig.add_subplot(gs[_n_focal:_n_gs_rows, 3], projection=ccrs.PlateCarree())
    _draw_migration_rates_map(
        ax_map,
        median_rates,
        map_lon_min=map_lon_min,
        map_lon_max=map_lon_max,
        map_lat_min=map_lat_min,
        map_lat_max=map_lat_max,
        geojson_cache_path=geojson_cache_path,
        geojson_url=geojson_url,
        show_ghost_inflow_inset=False,
        pops=pops,
    )

    # Unified x-axis date range: prevalence panels and % introductions panels
    # share these limits, anchored on the earliest datastream timestamp
    # (upper bound rounded up to the next month start so the rightmost tick is
    # on a month boundary).
    x_dec_range = compute_unified_x_dec_range(case_df, ww_df, sero_df, t_recent)
    x_lo_num, x_hi_num = _xlim_with_monthly_right_edge(x_dec_range)

    # --- % introductions panels (col 2, rows N_focal + 2 : end): N_focal
    # stacked rows ---
    gs_intros = GridSpecFromSubplotSpec(
        _n_focal, 1, subplot_spec=gs[_n_focal + 1 : _n_gs_rows, 2], hspace=0.65
    )
    intros_axes: list = []
    for i in range(_n_focal):
        ax_i = fig.add_subplot(gs_intros[i, 0])
        intros_axes.append(ax_i)
    for i in range(1, _n_focal):
        intros_axes[i].sharex(intros_axes[0])
        intros_axes[i].sharey(intros_axes[0])
    for i, lab in enumerate(FOCAL_DEME_LABELS):
        x_dec = intros_data[lab]["x_dec"]
        med, lo, hi = intros_data[lab]["pct"]
        _draw_introductions_panel(
            intros_axes[i],
            x_dec,
            med,
            lo,
            hi,
            color=DEME_COLORS[lab],
            county_name=DEME_MAP.get(lab, lab),
            show_xlabel=(i == _n_focal - 1),
            show_ylabel=True,
        )
        # Dotted local-deme prevalence overlay (matches the standalone plot).
        prev_med, _, _ = intros_data[lab]["prevalence"]
        _overlay_prevalence_twin(intros_axes[i], x_dec, prev_med)
        if i < _n_focal - 1:
            intros_axes[i].tick_params(axis="x", labelbottom=False)
    for ax in intros_axes:
        ax.set_xlim(x_lo_num, x_hi_num)

    # --- Datastream overview (cols 2-3, top N_focal rows): N_focal rows × 2 cols ---
    gs_ds = GridSpecFromSubplotSpec(
        _n_focal,
        2,
        subplot_spec=gs[0:_n_focal, 2:4],
        hspace=0.30,
        wspace=0.70,
    )
    axes_ov = np.empty((_n_focal, 2), dtype=object)
    for r in range(_n_focal):
        for c in range(2):
            axes_ov[r, c] = fig.add_subplot(gs_ds[r, c])
    # Share x-axis within each column.
    for c in range(2):
        for r in range(1, _n_focal):
            axes_ov[r, c].sharex(axes_ov[0, c])

    _draw_datastream_overview(
        fig,
        axes_ov,
        summ_by_label=summ_by_label,
        case_df=case_df,
        ww_df=ww_df,
        sero_df=sero_df,
        cum_hpd=cum_hpd,
        pops=pops,
        t_recent=t_recent,
        x_dec_range=x_dec_range,
    )

    # Panel labels: draw once so y-axis label / tick extents are laid out, then
    # place each letter left of its panel's y-axis decorations (see
    # add_panel_label). Done after every panel exists so positions are final.
    fig.canvas.draw()
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        renderer = None
    add_panel_label(ax_tree, "A", renderer)
    add_panel_label(axes_ov[0, 0], "B", renderer)
    add_panel_label(ax_inflow, "C", renderer)
    add_panel_label(intros_axes[0], "D", renderer)
    add_panel_label(ax_map, "E", renderer)

    _save_fig_png_pdf(fig, output_path, dpi=dpi)


def nedynamics_path_label_pairs(args: argparse.Namespace) -> list[tuple[Path, str]]:
    """
    NeDynamics log paths with their deme label inferred from the filename.
    Verifies each file exists.
    """
    if args.nedynamics_log is None:
        return []
    out: list[tuple[Path, str]] = []
    for path in args.nedynamics_log:
        if not path.is_file():
            raise FileNotFoundError(f"NeDynamics log not found: {path}")
        lab = deme_label_from_path(path)
        out.append((path, lab))
    return out


def collect_deme_summaries(
    path_label_pairs: list[tuple[Path, str]],
    burnin_fraction: float,
) -> list[tuple[str, np.ndarray, dict[str, np.ndarray]]]:
    out: list[tuple[str, np.ndarray, dict[str, np.ndarray]]] = []
    for path, label in path_label_pairs:
        i_idx, log_I, _ = load_nedynamics_arrays(path, burnin_fraction)
        summ = summarise_logI_trajectory(log_I)
        x_idx, summ = reverse_trajectory_for_time_forward(
            np.asarray(i_idx, dtype=float), summ
        )
        out.append((label, x_idx, summ))
    return out


def main() -> None:
    args = parse_args()
    ensure_combined_log_exists(args.combined_log)
    for flag, p in (
        ("--case-counts-csv", args.case_counts_csv),
        ("--seroprevalence-csv", args.seroprevalence_csv),
        ("--wastewater-csv", args.wastewater_csv),
        ("--county-populations-csv", args.county_populations_csv),
    ):
        if not p.is_file():
            raise FileNotFoundError(f"{flag} path is not a file: {p}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse rate shifts from BEAST XML (required for time mapping).
    if args.beast_xml is not None:
        validate_deme_map_from_xml(args.beast_xml)
        sky, spg = extract_rate_shifts_from_xml(args.beast_xml)

        DEME_SKYGROWTH_SHIFTS.update(sky)
        DEME_SPLINEGRID_SHIFTS.update(spg)
        print(DEME_SPLINEGRID_SHIFTS)
    elif args.state_time_csv is not None:
        raise ValueError(
            "--beast-xml is required when --state-time-csv is provided "
            "(needed to map trajectory indices to calendar time)."
        )

    tree_path = args.output_dir / "mcc_tree_deme_height_hpd.png"

    plot_mcc_tree_deme_hpd(
        args.tree.resolve(),
        tree_path,
        figsize_width=15,
        dpi=300,
    )

    nedynamics_pairs = nedynamics_path_label_pairs(args)
    if not nedynamics_pairs:
        raise ValueError("Provide at least one NeDynamics log via --nedynamics-log.")

    deme_summaries = collect_deme_summaries(
        nedynamics_pairs,
        args.burnin_fraction,
    )
    prevalence_xlabel = "Trajectory index (left = past, right = present)"
    t_recent_decimal: float | None = None
    panel_data: tuple | None = None
    if args.state_time_csv is not None:
        t_recent_decimal = load_most_recent_sample_decimal_year(args.state_time_csv)
        deme_summaries = [
            (
                lab,
                trajectory_indices_to_decimal_year(
                    x_idx, DEME_SPLINEGRID_SHIFTS[lab], t_recent_decimal
                ),
                summ,
            )
            for lab, x_idx, summ in deme_summaries
        ]
        prevalence_xlabel = "Date"

    x_domain: Literal["trajectory_index", "decimal_year"] = (
        "decimal_year" if t_recent_decimal is not None else "trajectory_index"
    )
    # only plot for the outside deme
    prevalence_path = (
        args.output_dir / f"prevalence_trajectories_nedynamics_{_GHOST_LABEL}.png"
    )
    plot_prevalence_trajectories(
        deme_summaries,
        _GHOST_LABEL,
        prevalence_path,
        x_domain=x_domain,
        xlabel=prevalence_xlabel,
    )
    print(f"Wrote {prevalence_path.resolve()}")

    # Standalone investigation figure: focal-deme prevalence trajectories,
    # one stacked panel per deme (parallel to the background-deme figure above).
    local_prevalence_path = args.output_dir / OUTPUT_FILENAME_LOCAL_PREVALENCE
    plot_local_prevalence_trajectories_stacked(
        deme_summaries,
        local_prevalence_path,
        x_domain=x_domain,
        xlabel=prevalence_xlabel,
    )
    print(f"Wrote {local_prevalence_path.resolve()}")

    medians = load_migration_rate_medians(args.combined_log, args.burnin_fraction)
    mig_samples = load_migration_rate_samples(args.combined_log, args.burnin_fraction)

    pops_for_map: dict[int, float] | None = None
    intros_data: dict[str, dict] | None = None
    if t_recent_decimal is not None:
        summ_by_label = {l: (x_arr, s) for l, x_arr, s in deme_summaries}
        county_map = load_county_to_deme_zero_based(args.county_populations_csv)
        pops = load_county_populations_zero_based(args.county_populations_csv)
        pops_for_map = pops
        case_df = load_case_counts_with_deme(args.case_counts_csv, county_map)
        ww_df = load_wastewater_with_deme(args.wastewater_csv, county_map)
        sero_df = load_seroprevalence_with_deme(args.seroprevalence_csv, county_map)
        grid_log = (
            args.mascot_datastream_log
            if args.mascot_datastream_log is not None
            else args.combined_log
        )
        tree_h = load_tree_height_after_burnin(args.combined_log, args.burnin_fraction)
        cum_inputs: list[tuple[Path | None, int]] = []
        if args.cumulative_incidence_log is not None:
            for cum_path in args.cumulative_incidence_log:
                m = CUMULATIVE_INCIDENCE_DEME_PATTERN.search(cum_path.name)
                if m:
                    deme_idx = (
                        int(m.group(1).replace("Deme", "").replace("deme", "")) - 1
                    )
                else:
                    # Fallback: try the NeDynamics-style DemeN pattern
                    cum_lab = deme_label_from_path(cum_path)
                    deme_idx = int(cum_lab.replace("Deme", "")) - 1
                cum_inputs.append((cum_path, deme_idx))
        for p, deme_idx in cum_inputs:
            if p is not None and not p.is_file():
                raise FileNotFoundError(
                    f"Cumulative incidence log ({FOCAL_DEME_LABELS[deme_idx]}) "
                    f"does not exist or is not a file: {p}"
                )
        if any(p is not None for p, _ in cum_inputs):
            shifts = extract_spline_grid_shifts(grid_log)
            cum_hpd = concat_cumulative_hpd(cum_inputs, tree_h, shifts)
        else:
            cum_hpd = pd.DataFrame()

        # Compute % introductions before plotting so the unified date range
        # spans both the prevalence panels and the new % introductions panels.
        ned_paths_by_label = {lab: path for path, lab in nedynamics_pairs}
        intros_data = compute_introductions_for_focal_demes(
            nedynamics_paths=ned_paths_by_label,
            mig_samples=mig_samples,
            t_recent=t_recent_decimal,
            burnin_fraction=args.burnin_fraction,
        )
        x_dec_range = compute_unified_x_dec_range(
            case_df, ww_df, sero_df, t_recent_decimal
        )
        # Crop trajectories to the shared plotting window so the log-scale
        # prevalence twin axes autoscale to the visible data only (see
        # crop_intros_data_to_x_range).
        intros_data = crop_intros_data_to_x_range(intros_data, x_dec_range)

        print_prevalence_and_cuminc_summary(
            summ_by_label, cum_hpd, pops, x_dec_range, t_recent_decimal
        )

        ov_path = args.output_dir / OUTPUT_FILENAME_DATASTREAM_OVERVIEW
        plot_datastream_overview(
            summ_by_label=summ_by_label,
            case_df=case_df,
            ww_df=ww_df,
            sero_df=sero_df,
            cum_hpd=cum_hpd,
            pops=pops,
            t_recent=t_recent_decimal,
            output_path=ov_path,
            x_dec_range=x_dec_range,
        )
        print(f"Wrote {ov_path.resolve()}")

        intros_path = args.output_dir / OUTPUT_FILENAME_INTRODUCTIONS
        plot_introductions_panels(intros_data, intros_path, x_dec_range=x_dec_range)
        print(f"Wrote {intros_path.resolve()}")

        intros_vs_cases_path = args.output_dir / OUTPUT_FILENAME_INTROS_VS_CASES
        plot_intros_vs_local_cases(
            intros_data, intros_vs_cases_path, x_dec_range=x_dec_range
        )
        print(f"Wrote {intros_vs_cases_path.resolve()}")

        panel_data = (
            summ_by_label,
            case_df,
            ww_df,
            sero_df,
            cum_hpd,
            pops,
            t_recent_decimal,
        )

    mig_path = args.output_dir / OUTPUT_FILENAME_MIGRATION_RATES
    geo_cache = args.geojson_cache
    if geo_cache is None:
        geo_cache = args.output_dir / "plotly_counties_geojson_fips.json"
    plot_migration_rates_map(
        medians,
        mig_path,
        map_lon_min=DEFAULT_MAP_LON_MIN,
        map_lon_max=DEFAULT_MAP_LON_MAX,
        map_lat_min=DEFAULT_MAP_LAT_MIN,
        map_lat_max=DEFAULT_MAP_LAT_MAX,
        geojson_cache_path=geo_cache,
        geojson_url=args.geojson_url,
        pops=pops_for_map,
    )
    print(f"Wrote {mig_path.resolve()}")

    # Standalone supplementary scatter plots (moved out of the final figure).
    # NOTE: the relative outside->local inflow panel
    # (plot_migration_pct_from_outside) now lives in the main figure
    # (plot_final_figure_gridspec), so it is no longer emitted standalone here.
    local_rates_path = args.output_dir / OUTPUT_FILENAME_MIGRATION_RATES_LOCAL
    plot_migration_rates_local(
        medians,
        local_rates_path,
        rate_samples=mig_samples,
    )
    print(f"Wrote {local_rates_path.resolve()}")

    if panel_data is not None:
        (
            summ_by_label_p,
            case_df_p,
            ww_df_p,
            sero_df_p,
            cum_hpd_p,
            pops_p,
            t_recent_p,
        ) = panel_data
        if intros_data is None:
            raise RuntimeError(
                "intros_data must be computed when panel_data is available; "
                "this should not happen."
            )
        final_panel_path = args.output_dir / OUTPUT_FILENAME_FINAL_FIGURE_GRIDSPEC
        plot_final_figure_gridspec(
            tree_path=args.tree.resolve(),
            median_rates=medians,
            rate_samples=mig_samples,
            summ_by_label=summ_by_label_p,
            case_df=case_df_p,
            ww_df=ww_df_p,
            sero_df=sero_df_p,
            cum_hpd=cum_hpd_p,
            pops=pops_p,
            t_recent=t_recent_p,
            output_path=final_panel_path,
            intros_data=intros_data,
            map_lon_min=DEFAULT_MAP_LON_MIN,
            map_lon_max=DEFAULT_MAP_LON_MAX,
            map_lat_min=DEFAULT_MAP_LAT_MIN,
            map_lat_max=DEFAULT_MAP_LAT_MAX,
            geojson_cache_path=geo_cache,
            geojson_url=args.geojson_url,
            figsize=(23.0, 18.0),
        )
        print(f"Wrote {final_panel_path.resolve()}")


if __name__ == "__main__":
    main()
