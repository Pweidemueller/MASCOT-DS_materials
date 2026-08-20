#!/usr/bin/env bash
# =============================================================================
# 05_postprocessing_all.sh
#
# Wrapper: run 05_postprocessing.sh in parallel for all analysis variants.
#
# Each variant is a sensitivity analysis that removes one datastream from the
# full MASCOT-DS model:
#   datastreams           — full model (case counts + seroprevalence + wastewater)
#   datastreams_nocasecounts    — seroprevalence + wastewater only
#   datastreams_noseroprevalence — case counts + wastewater only
#   datastreams_nowastewater    — case counts + seroprevalence only
#   datastreams_nomascotll      — datastreams without MASCOT likelihood
#   datastreams_onlytree        — genetic data only (no datastreams)
#
# Usage:
#   ./scripts/05_postprocessing_all.sh <RESULTS_DIR> <STATE_TIME_CSV>
#
#   RESULTS_DIR    Parent directory holding variant subdirs and shared inputs
#                  (county_populations.csv, datastreams_demes/)
#   STATE_TIME_CSV Shared state_time.csv (same for all variants, generated
#                  by create_mascot_xml.py for the full-datastreams run)
#
# Each variant writes its postprocessing log to:
#   <RESULTS_DIR>/<VARIANT_DIR>/postprocessing.log
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
POSTPROCESS="${SCRIPT_DIR}/05_postprocessing.sh"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Conda env for Python steps (must match CONDA_ENV in 05_postprocessing.sh).
CONDA_ENV="biopython_env"

# =============================================================================
# Usage
# =============================================================================
usage() {
    echo "Usage: $0 <RESULTS_DIR> <STATE_TIME_CSV>"
    echo ""
    echo "  RESULTS_DIR    Directory containing BEAST variant subdirs and shared inputs"
    echo "  STATE_TIME_CSV Path to state_time.csv (shared across variants)"
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

RESULTS_DIR="${1%/}"   # strip trailing slash for consistent path construction
STATE_TIME_CSV="$2"

# =============================================================================
# CONFIG — Analysis variants
# One entry per sensitivity analysis; must match the BEAST run directory names
# under RESULTS_DIR.
# =============================================================================
VARIANT_NAMES=(
    SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams
    SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams_nocasecounts
    SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams_noseroprevalence
    SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams_noseroprevalence_fixedCaseCountsScaling
    SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams_noseroprevalence_fixedNeScalers
    SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams_nowastewater
    SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams_nomascotll
    SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams_onlytree
)

# =============================================================================
# Pre-flight checks
# =============================================================================
if [[ ! -f "${POSTPROCESS}" ]]; then
    echo "ERROR: 05_postprocessing.sh not found at ${POSTPROCESS}"
    exit 1
fi

if [[ ! -f "${STATE_TIME_CSV}" ]]; then
    echo "ERROR: STATE_TIME_CSV not found: ${STATE_TIME_CSV}"
    exit 1
fi

VARIANTS=()
for name in "${VARIANT_NAMES[@]}"; do
    d="${RESULTS_DIR}/${name}"
    if [[ ! -d "$d" ]]; then
        echo "ERROR: Variant directory not found: ${d}"
        exit 1
    fi
    VARIANTS+=("$d")
done

echo "=== Variants to process (${#VARIANTS[@]}) ==="
for v in "${VARIANTS[@]}"; do
    echo "  $(basename "$v")"
done
echo ""

# =============================================================================
# Launch all variants in parallel
# =============================================================================
PIDS=()
LOG_FILES=()

for BEAST_RUN_DIR in "${VARIANTS[@]}"; do
    RUN_LOG="${BEAST_RUN_DIR}/postprocessing.log"
    LOG_FILES+=("${RUN_LOG}")

    echo "Launching: $(basename "${BEAST_RUN_DIR}")"
    bash "${POSTPROCESS}" \
        "${BEAST_RUN_DIR}" \
        "${RESULTS_DIR}" \
        "${STATE_TIME_CSV}" \
        > "${RUN_LOG}" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#VARIANTS[@]} job(s) running in background."
echo "Monitor with: tail -f <variant_dir>/postprocessing.log"
echo ""

# =============================================================================
# Wait for all jobs and report
# =============================================================================
FAILED=()
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    RUN_NAME="$(basename "${VARIANTS[$i]}")"
    if wait "${PID}"; then
        echo "[OK]   ${RUN_NAME}"
    else
        echo "[FAIL] ${RUN_NAME}  (log: ${LOG_FILES[$i]})"
        FAILED+=("${RUN_NAME}")
    fi
done

echo ""

# =============================================================================
# Aggregate per-variant ESS tables into a single summary CSV
# =============================================================================
# Each variant's 05_postprocessing.sh writes a per-run CSV with columns
# (beast_run_name, parameter_name, ESS). Concatenate them here so all variants
# can be compared in one place. Only variants whose CSV was produced are
# included; missing files are skipped silently (e.g. if a variant failed
# before Step 5).
ESS_SUMMARY="${RESULTS_DIR}/ess_summary.csv"
echo "=== Aggregating ESS tables ==="
ESS_FILES=()
for v in "${VARIANTS[@]}"; do
    f="${v}/$(basename "${v}").ess.csv"
    [[ -f "$f" ]] && ESS_FILES+=("$f")
done

if [[ ${#ESS_FILES[@]} -eq 0 ]]; then
    echo "  No per-variant ESS CSVs found — skipping summary."
else
    # Header from the first file, then data rows (skipping each file's header).
    head -n 1 "${ESS_FILES[0]}" > "${ESS_SUMMARY}"
    for f in "${ESS_FILES[@]}"; do
        tail -n +2 "$f" >> "${ESS_SUMMARY}"
    done
    echo "  ESS summary: ${ESS_SUMMARY} (${#ESS_FILES[@]} variant(s))"
fi

# =============================================================================
# Cross-variant value-of-information analysis
# =============================================================================
# Compares all variants' prevalence and migration-rate posteriors against the
# full-datastreams reference. Reads <variant>/<variant>.combined.log and
# <variant>/<variant>.NeDynamics.DemeN.combined.log under RESULTS_DIR.
# The variant_prefix is the variant directory name with the trailing
# '_datastreams' suffix removed (e.g.
#   SARSCoV2_Epsilon_BayArea_results_1000seq_fakecasecounts_datastreams →
#   SARSCoV2_Epsilon_BayArea_results_1000seq).
echo "=== Value-of-information analysis ==="
VARIANT_PREFIX="${VARIANT_NAMES[0]%_datastreams}"
VOI_OUTPUT_DIR="${RESULTS_DIR}/value_of_information"
conda run -n "${CONDA_ENV}" python "${REPO_ROOT}/bin/value_information_analysis.py" \
    --results_dir "${RESULTS_DIR}" \
    --variant_prefix "${VARIANT_PREFIX}" \
    --output_dir "${VOI_OUTPUT_DIR}" \
    --burnin_fraction 0.0
echo "  VoI figures: ${VOI_OUTPUT_DIR}/"

echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "=== All variants completed successfully ==="
else
    echo "=== ${#FAILED[@]} variant(s) FAILED ==="
    for f in "${FAILED[@]}"; do
        echo "  ${f}"
    done
    exit 1
fi
