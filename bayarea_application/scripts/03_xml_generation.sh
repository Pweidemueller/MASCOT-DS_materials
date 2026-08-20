#!/usr/bin/env bash
# =============================================================================
# 03_xml_generation.sh
#
# Phase 3: Generate BEAST2 XML for the MASCOT-DS analysis.
#
# The primary block generates the published infer-tree XML (BEAST jointly
# infers the phylogeny and MASCOT-DS parameters from the alignment).
#
# A commented-out alternative block retains the fixed-tree configuration
# (conditions on the pre-computed timetree from Phase 1).
#
# Output: ${XML_OUTPUT_DIR}/
# =============================================================================
set -euo pipefail

# =============================================================================
# Usage
# =============================================================================
usage() {
    echo "Usage: $0 <RESULTS_DIR> <N_BACKGROUND>"
    echo ""
    echo "  RESULTS_DIR    Output directory for all pipeline results"
    echo "  N_BACKGROUND   Number of background sequences (0 = use ghost outside-deme)"
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

RESULTS_DIR="$1"
N_BACKGROUND="$2"

# =============================================================================
# CONFIG — Analysis parameters
# =============================================================================
XML_OUTPUT_DIR=${RESULTS_DIR}
# Base name shared by every variant; the variant suffix is appended per-iteration.
XML_BASENAME="SARSCoV2_Epsilon_BayArea_$(basename "${RESULTS_DIR}")"

# Becoming uninfectious rate (1/years)
BURATE=52.0
# MCMC chain length.
CHAIN_LENGTH=100000000

# Datastream variants to generate (all variants supported by create_mascot_xml.py
# except 'original'). Each gets its own dedicated subdirectory under RESULTS_DIR.
#
# RATE_SHIFT_REFERENCE_VARIANT is generated first; its computed max_age is then
# passed (via --max_age) to every other variant so that SkygrowthRateShifts /
# SplineGridRateShifts are identical across all variants. This is required for
# the posteriors to be directly comparable: any variant computing its own
# max_age from its (possibly reduced) datastream set would produce a different
# rate-shift grid.
RATE_SHIFT_REFERENCE_VARIANT="datastreams"
VARIANTS=(
    "${RATE_SHIFT_REFERENCE_VARIANT}"
    "datastreams_nocasecounts"
    "datastreams_noseroprevalence"
    "datastreams_nowastewater"
    "datastreams_nomascotll"
    "datastreams_onlytree"
)
# All non-reference variants reuse the reference variant's max_age. Listed
# explicitly (rather than "every variant except reference") so that any future
# additions to VARIANTS are forced to make a deliberate choice.
VARIANTS_REUSING_REFERENCE_MAX_AGE=(
    "datastreams_nocasecounts"
    "datastreams_noseroprevalence"
    "datastreams_nowastewater"
    "datastreams_nomascotll"
    "datastreams_onlytree"
)

# Coupled MCMC: 4 heated chains per run for better mixing.
USE_COUPLED_MCMC=false
COUPLED_CHAINS=4
DELTA_TEMPERATURE=0.1
RESAMPLE_EVERY=1000

# Anchor each deme's case-count datastream with a fake leading zero observation
# (passes --add_sentinel_zero_case_counts to create_mascot_xml.py). Set to
# false to omit the flag.
ADD_SENTINEL_ZERO_CASE_COUNTS=true

# =============================================================================
# CONFIG — Tool paths  (machine-specific)
# =============================================================================
CONDA_ENV="biopython_env"

# =============================================================================
# Derived paths
# =============================================================================
FINAL_DIR="${RESULTS_DIR}/final_sequences"
DATASTREAMS_DIR="${RESULTS_DIR}/datastreams_demes"

ALIGNMENT="${FINAL_DIR}/final_epsilon_sequences.fasta"
METADATA="${FINAL_DIR}/final_epsilon_sequences_metadata_with_ids.csv"
TIMETREE="${FINAL_DIR}/timetree_resolved.nexus"
POPULATIONS="${RESULTS_DIR}/county_populations.csv"

TEMPLATE_INFERTREE="MASCOTDS_templates/Mascot_datastreams_template_infertree.xml"

CASE_COUNTS="${DATASTREAMS_DIR}/case_counts.csv"
SEROPREVALENCE="${DATASTREAMS_DIR}/seroprevalence.csv"
WASTEWATER="${DATASTREAMS_DIR}/wastewater.csv"

# =============================================================================
# Pre-flight checks
# =============================================================================
echo "=== Pre-flight checks ==="
for F in "${ALIGNMENT}" "${METADATA}" "${POPULATIONS}" \
         "${CASE_COUNTS}" "${SEROPREVALENCE}" "${WASTEWATER}" \
         "${TEMPLATE_INFERTREE}"; do
    if [[ ! -f "${F}" ]]; then
        echo "ERROR: Required file not found: ${F}"
        echo "Run 01_sequence_preparation.sh and 02_datastream_preparation.sh first."
        exit 1
    fi
done
mkdir -p "${XML_OUTPUT_DIR}"
echo "All input files present."
echo ""

# =============================================================================
# Variant XMLs: BEAST infers the tree  (published analysis)
# =============================================================================
# --infer_tree: BEAST jointly estimates the phylogeny and MASCOT-DS parameters
#   from the alignment. This avoids conditioning the structured coalescent on a
#   single point-estimate timetree, properly propagating phylogenetic uncertainty.
#
# --ghost_outsidedeme: adds a 4th deme (Deme4) with no sequence observations or
#   datastream likelihoods. It absorbs lineages entering the Bay Area from the
#   rest of California/USA, acting as an unsampled reservoir. Without this, all
#   migration must be explained by the 3 focal demes alone.
#
# create_mascot_xml.py always disables clipTransRate on Spline elements (clipping
# was found to artificially suppress migration rate estimates) — no flag needed.
#
# create_mascot_xml.py also writes a state_time.csv alongside each XML, mapping
# each sample to its deme and decimal-year collection time — required by
# analyse_posteriors.py to calibrate figure x-axes.
# Build ghost-deme flag: when N_BACKGROUND=0, background lineages are modelled
# via the unsampled ghost outside-deme instead of explicit background sequences.
GHOST_ARGS=()
if [[ "${N_BACKGROUND}" -eq 0 ]]; then
    GHOST_ARGS=(--ghost_outsidedeme)
fi

COUPLED_ARGS=()
if [[ "${USE_COUPLED_MCMC}" == "true" ]]; then
    COUPLED_ARGS=(
        --coupled_mcmc
        --chains "${COUPLED_CHAINS}"
        --delta_temperature "${DELTA_TEMPERATURE}"
        --resample_every "${RESAMPLE_EVERY}"
        --log_heated_chains
        --optimise
    )
fi

SENTINEL_ARGS=()
if [[ "${ADD_SENTINEL_ZERO_CASE_COUNTS}" == "true" ]]; then
    SENTINEL_ARGS=(--add_sentinel_zero_case_counts)
fi

REFERENCE_MAX_AGE=""

GENERATED_XMLS=()
for VARIANT in "${VARIANTS[@]}"; do
    # Layout: ${RESULTS_DIR}/${XML_BASENAME}_${VARIANT}/${XML_BASENAME}_${VARIANT}.xml
    # create_mascot_xml.py appends "_${VARIANT}" to --xml_name, so the folder
    # name matches the XML basename exactly (minus the .xml extension).
    VARIANT_DIR="${XML_OUTPUT_DIR}/${XML_BASENAME}_${VARIANT}"
    mkdir -p "${VARIANT_DIR}"

    echo "=== Generating XML for variant: ${VARIANT} ==="
    echo "    Output dir: ${VARIANT_DIR}"

    # Decide whether this variant should reuse the reference variant's max_age.
    MAX_AGE_ARGS=()
    for REUSE_VARIANT in "${VARIANTS_REUSING_REFERENCE_MAX_AGE[@]}"; do
        if [[ "${VARIANT}" == "${REUSE_VARIANT}" ]]; then
            if [[ -z "${REFERENCE_MAX_AGE}" ]]; then
                echo "ERROR: cannot reuse max_age for '${VARIANT}' because the"
                echo "reference variant '${RATE_SHIFT_REFERENCE_VARIANT}' has"
                echo "not been generated yet (or its max_age sidecar is missing)."
                exit 1
            fi
            MAX_AGE_ARGS=(--max_age "${REFERENCE_MAX_AGE}")
            echo "    Reusing max_age=${REFERENCE_MAX_AGE} from '${RATE_SHIFT_REFERENCE_VARIANT}'"
            break
        fi
    done

    conda run -n "${CONDA_ENV}" python bin/create_mascot_xml.py \
        --infer_tree \
        ${GHOST_ARGS[@]+"${GHOST_ARGS[@]}"} \
        ${COUPLED_ARGS[@]+"${COUPLED_ARGS[@]}"} \
        --datastream_template "${TEMPLATE_INFERTREE}" \
        --alignment     "${ALIGNMENT}" \
        --metadata      "${METADATA}" \
        --xml_name      "${VARIANT_DIR}/${XML_BASENAME}" \
        --burate        "${BURATE}" \
        --chain_length  "${CHAIN_LENGTH}" \
        --case_counts   "${CASE_COUNTS}" \
        --seroprevalence "${SEROPREVALENCE}" \
        --wastewater    "${WASTEWATER}" \
        --population_csv "${POPULATIONS}" \
        --variant_type  "${VARIANT}" \
        --fixed_clock_rate 0.001 \
        --fixed_ne_scaler_mean 0.1 \
        ${SENTINEL_ARGS[@]+"${SENTINEL_ARGS[@]}"} \
        ${MAX_AGE_ARGS[@]+"${MAX_AGE_ARGS[@]}"}

    # Cache the reference variant's max_age (written by create_mascot_xml.py).
    if [[ "${VARIANT}" == "${RATE_SHIFT_REFERENCE_VARIANT}" ]]; then
        MAX_AGE_FILE="${VARIANT_DIR}/${XML_BASENAME}_max_age.txt"
        if [[ ! -f "${MAX_AGE_FILE}" ]]; then
            echo "ERROR: expected max_age sidecar not written: ${MAX_AGE_FILE}"
            exit 1
        fi
        REFERENCE_MAX_AGE=$(< "${MAX_AGE_FILE}")
        REFERENCE_MAX_AGE="${REFERENCE_MAX_AGE//[[:space:]]/}"
        echo "    Recorded reference max_age=${REFERENCE_MAX_AGE}"
    fi

    GENERATED_XMLS+=("${VARIANT_DIR}/${XML_BASENAME}_${VARIANT}.xml")
    echo ""
done

echo "=== XML generation complete ==="
for XML in "${GENERATED_XMLS[@]}"; do
    echo "  ${XML}"
done
