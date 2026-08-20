#!/usr/bin/env bash
# =============================================================================
# 02_datastream_preparation.sh
#
# Phase 2: Filter epidemiological datastreams to the analysis date range.
#
# Takes the three datastreams (case counts, seroprevalence, wastewater) from
# data/ and writes per-deme CSV files covering the analysis window.
# The end date is inferred automatically from the most recent collection date
# in the final sequence metadata, so it tracks whatever date range Phase 1
# produced.
#
# Output: ${RESULTS_DIR}/datastreams_demes/
# =============================================================================
set -euo pipefail

# =============================================================================
# Usage
# =============================================================================
usage() {
    echo "Usage: $0 <RESULTS_DIR>"
    echo ""
    echo "  RESULTS_DIR    Output directory for all pipeline results"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

RESULTS_DIR="$1"

# =============================================================================
# CONFIG — Analysis parameters
# =============================================================================

# Start date for the datastream window. Chosen to precede the first case-count
# signal for the Epsilon variant in the Bay Area (late 2020 wave onset).
DATASTREAM_START_DATE="2020-10-01"

# =============================================================================
# CONFIG — Tool paths  (machine-specific)
# =============================================================================
CONDA_ENV="biopython_env"

# =============================================================================
# Derived paths
# =============================================================================
FINAL_METADATA="${RESULTS_DIR}/final_sequences/final_epsilon_sequences_metadata.csv"
DATASTREAMS_OUT="${RESULTS_DIR}/datastreams_demes"

# =============================================================================
# Pre-flight check
# =============================================================================
if [[ ! -f "${FINAL_METADATA}" ]]; then
    echo "ERROR: Final sequence metadata not found: ${FINAL_METADATA}"
    echo "Run 01_sequence_preparation.sh first."
    exit 1
fi

# =============================================================================
# Filter datastreams
# =============================================================================
echo "=== Filtering datastreams ==="
# The datastreams used are:
#   - Case counts:     data/covid19cases_test.csv
#   - Seroprevalence:  data/Nationwide_Commercial_Laboratory_Seroprevalence_Survey_20260302.csv
#   - Wastewater:      data/wastewatersurveillancecalifornia.csv
#
# All default paths are set in filter_datastreams_data.py.
# The seroprevalence filename is pinned to the final available NCHS survey
# download (retrieved 2026-03-02); see SEROPREVALENCE_FILENAME in that script.
#
# --metadata: the end date of the analysis window is inferred from the maximum
#   collection date in the final sequence metadata. This avoids hardcoding a
#   second date constant that could drift from the actual sequence window.
conda run -n "${CONDA_ENV}" python bin/filter_datastreams_data.py \
    -o "${DATASTREAMS_OUT}" \
    --start-date "${DATASTREAM_START_DATE}" \
    --metadata   "${FINAL_METADATA}"

echo ""
echo "=== Datastream preparation complete ==="
echo "Output: ${DATASTREAMS_OUT}/"
