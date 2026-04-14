"""Shared constants for the simulation study pipeline.

Single source of truth for model names, variant identifiers, colour palettes,
and file-naming patterns used across plotting and analysis scripts.
"""

# ---------------------------------------------------------------------------
# Model names (used in CSVs, plot legends, and channel metadata)
# ---------------------------------------------------------------------------
MODEL_MASCOT = "MASCOT"
MODEL_MASCOT_DS = "MASCOT-DS"

# ---------------------------------------------------------------------------
# Variant identifiers
# ---------------------------------------------------------------------------
# The "original" variant runs MASCOT without any datastreams.
VARIANT_ORIGINAL = "original"

# Datastream variants (all are subsets of MASCOT-DS)
DATASTREAM_VARIANTS = (
    "datastreams",
    "datastreams_nocasecounts",
    "datastreams_noseroprevalence",
    "datastreams_nowastewater",
    "datastreams_nomascotll",
    "datastreams_onlytree",
)

# Leave-one-out variant short names (without "datastreams_" prefix)
LEAVE_ONE_OUT_VARIANTS = ("nocasecounts", "nowastewater", "noseroprevalence")

VARIANT_LABELS = {
    "all": "All DS",
    "nocasecounts": "No CC",
    "nowastewater": "No WW",
    "noseroprevalence": "No SP",
}

# Suffixes stripped when extracting the base simulation name from variant-tagged
# strings.  Ordered longest-first so that e.g. "_datastreams_nocasecounts"
# matches before the shorter "_nocasecounts".
VARIANT_SUFFIXES = sorted(
    [
        "_datastreams_nocasecounts",
        "_datastreams_nowastewater",
        "_datastreams_noseroprevalence",
        "_datastreams",
        "_nocasecounts",
        "_nowastewater",
        "_noseroprevalence",
    ],
    key=len,
    reverse=True,
)

# ---------------------------------------------------------------------------
# Colour palette (colourblind-friendly)
# ---------------------------------------------------------------------------
COLORBLINDFR = {
    "main": ["#56b3e9", "#e0d316", "#0072b2", "#e69d00", "#cc79a7"],
    "additional": ["#EC681E", "#009e74", "#000000"],
}
COLORS = COLORBLINDFR["main"]

# Per-model plot colours (index into COLORS)
MODEL_COLORS = {
    MODEL_MASCOT: COLORS[4],
    MODEL_MASCOT_DS: COLORS[3],
}

# Per-variant plot colours (leave-one-out variants)
VARIANT_COLORS = {
    "all": COLORS[3],
    "nocasecounts": "purple",
    "nowastewater": "brown",
    "noseroprevalence": "red",
}
