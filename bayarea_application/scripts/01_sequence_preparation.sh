#!/usr/bin/env bash
# =============================================================================
# 01_sequence_preparation.sh
#
# Phase 1: Prepare sequences for BEAST analysis.
#
# Steps:
#   1. Merge GISAID downloads for B.1.427 and B.1.429 into a single FASTA + TSV
#   2. Sample sequences per focal deme and (optionally) background
#   3. Extract county population sizes
#   4. Multiple-sequence alignment (MAFFT)
#   5. Replace ambiguous N bases with gaps
#   6. Trim alignment columns with poor coverage (trimAl)
#   7. Infer ML phylogeny (IQTree)
#   8. Time-calibrate the tree (treetime)
#   9. Iteratively remove clock-rate outliers until none remain
#  10. Add sequence_ID column to metadata (required by create_mascot_xml.py)
#  11. Resolve polytomies in the timetree
#
# Output directory: ${RESULTS_DIR}/
# =============================================================================
set -euo pipefail

# =============================================================================
# Usage
# =============================================================================
usage() {
    echo "Usage: $0 <RESULTS_DIR> <N_BACKGROUND>"
    echo ""
    echo "  RESULTS_DIR    Output directory for all pipeline results"
    echo "  N_BACKGROUND   Number of background sequences to sample (0 = ghost deme only)"
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

RESULTS_DIR="$1"
N_BACKGROUND="$2"

# =============================================================================
# CONFIG — Analysis parameters
# Do not change these values to reproduce the published analysis.
# =============================================================================

# If non-empty, reuse an existing GISAID merged directory and skip Step 1 (merge).
GISAID_MERGED_DIR=""

# If non-empty, only retain sequences whose 'variant' column matches this value.
VARIANT_FILTER=""

# Set to "true" to skip the manual inspection prompt in Step 9 and
# automatically accept every outlier removal batch.
AUTO_ACCEPT_OUTLIERS="true"

# Sampling: X sequences per focal deme.
N_DEMES=310
RANDOM_STATE=42
# Most-recent collection date included (Epsilon wave end in the Bay Area).
MOST_RECENT_DATE="2021-05-01"

# Treetime clock-filter threshold: flag sequences that deviate more than
# CLOCK_FILTER inter-quartile distances from the root-to-tip regression line.
CLOCK_FILTER=4

# trimAl gap threshold: retain alignment columns present in ≥ 90% of sequences.
TRIMAL_GAP_THRESHOLD=0.9

MAFFT_THREADS=4
IQTREE_THREADS=6

# =============================================================================
# CONFIG — Tool paths
# Defaults assume each tool is on your PATH (see README.md for install
# instructions). To point at a specific binary instead — e.g. on a cluster
# with environment modules, or a tool kept outside the environment — export
# the corresponding variable before running this script:
#   TRIMAL=/path/to/trimal IQTREE=/path/to/iqtree3 \
#     bash scripts/01_sequence_preparation.sh results 70
# =============================================================================
CONDA_ENV="${CONDA_ENV:-biopython_env}"
MAFFT="${MAFFT:-mafft}"
TRIMAL="${TRIMAL:-trimal}"
IQTREE="${IQTREE:-iqtree3}"
# treetime lives in its own venv (separate from biopython_env) because its
# numpy/scipy version requirements conflict with the main environment.
TREETIME="${TREETIME:-treetime}"

# =============================================================================
# Derived paths — built from RESULTS_DIR; do not edit by hand.
# =============================================================================
# Use the override directory if supplied, otherwise default to within RESULTS_DIR.
if [[ -n "${GISAID_MERGED_DIR}" ]]; then
    GISAID_MERGED="${GISAID_MERGED_DIR}"
else
    GISAID_MERGED="${RESULTS_DIR}/GISAID_merged"
fi
SAMPLED_DIR="${RESULTS_DIR}/sampled_sequences"
FINAL_DIR="${RESULTS_DIR}/final_sequences"
FINAL_FASTA_NAME="final_epsilon_sequences.fasta"

SAMPLED_FASTA="${SAMPLED_DIR}/sampled_sequences.fasta"
ALIGNED="${SAMPLED_DIR}/sampled_sequences_aligned.fasta"
ALIGNED_NON="${SAMPLED_DIR}/sampled_sequences_aligned_noN.fasta"
ALIGNED_TRIMMED="${SAMPLED_DIR}/sampled_sequences_aligned_noN_trimmed.fasta"

# =============================================================================
# Pre-flight checks
# =============================================================================
echo "=== Pre-flight checks ==="
if ! conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
    echo "ERROR: Conda environment '${CONDA_ENV}' not found."
    echo "Create it with: conda env create -f environment.yml (see README.md)."
    exit 1
fi
for TOOL in "${MAFFT}" "${TRIMAL}" "${IQTREE}" "${TREETIME}"; do
    if ! command -v "${TOOL}" >/dev/null 2>&1; then
        echo "ERROR: Tool not found on PATH: ${TOOL}"
        echo "Install it (see README.md) or set the matching environment variable to its path."
        exit 1
    fi
done
echo "All tools found. Starting pipeline."
echo ""

# =============================================================================
# Step 1: Merge GISAID downloads
# =============================================================================
echo "=== Step 1: Merge GISAID sequences ==="
if [[ -n "${GISAID_MERGED_DIR}" ]]; then
    echo "Skipping merge — reusing existing merged data: ${GISAID_MERGED}"
else
    # Reads data/GISAID_sequences/B1427/ and B1429/, merges TSVs and FASTAs,
    # excludes non-human sequences (flagged in *nonhuman*.tsv files), deduplicates.
    conda run -n "${CONDA_ENV}" python bin/merge_gisaid_sequences.py \
        --output_dir "${GISAID_MERGED}"
fi

# =============================================================================
# Step 2: Sample sequences per focal deme
# =============================================================================
echo ""
echo "=== Step 2: Sample sequences ==="
# Focal demes: defined by COUNTIES_OF_INTEREST in filter_gisaid_metadata.py.
# --n_background 0: the ghost outside-deme absorbs background lineages without
#   requiring explicit outgroup sequences — no background sampling needed.
# --most_recent_date caps the analysis window at the Epsilon wave peak.
VARIANT_ARGS=()
if [[ -n "${VARIANT_FILTER}" ]]; then
    VARIANT_ARGS=(--variant "${VARIANT_FILTER}")
fi
conda run -n "${CONDA_ENV}" python bin/filter_gisaid_metadata.py \
    "${GISAID_MERGED}/merged_metadata.tsv" \
    "${GISAID_MERGED}/merged_sequences.fasta" \
    --n_demes "${N_DEMES}" \
    --n_background "${N_BACKGROUND}" \
    --random_state "${RANDOM_STATE}" \
    --most_recent_date "${MOST_RECENT_DATE}" \
    --output_dir "${SAMPLED_DIR}" \
    ${VARIANT_ARGS[@]+"${VARIANT_ARGS[@]}"}

# =============================================================================
# Step 3: Extract county population sizes
# =============================================================================
echo ""
echo "=== Step 3: Extract county populations ==="
# Reads data/covid19cases_test.csv (gitignored raw data) and writes
# county_populations.csv used as PopSize priors in the BEAST XML.
conda run -n "${CONDA_ENV}" python bin/extract_county_populations.py \
    -o "${RESULTS_DIR}/county_populations.csv"

# =============================================================================
# Step 4: Multiple sequence alignment (MAFFT)
# =============================================================================
echo ""
echo "=== Step 4: MAFFT alignment ==="
"${MAFFT}" --auto --thread "${MAFFT_THREADS}" "${SAMPLED_FASTA}" > "${ALIGNED}"

# =============================================================================
# Step 5: Replace N ambiguity bases with gaps
# =============================================================================
echo ""
echo "=== Step 5: Replace N with gaps ==="
# trimAl treats N as an informative character, causing it to retain poorly-
# sequenced columns. Replacing N with - ensures these positions are masked.
conda run -n "${CONDA_ENV}" python bin/replaceN_withgaps.py \
    -i "${ALIGNED}" \
    -o "${ALIGNED_NON}"

# =============================================================================
# Step 6: Trim poorly-covered alignment columns (trimAl)
# =============================================================================
echo ""
echo "=== Step 6: trimAl ==="
# -gt 0.9: retain only columns present (non-gap) in ≥ 90% of sequences.
# This removes ambiguously-assembled ends common in SARS-CoV-2 amplicon data.
"${TRIMAL}" \
    -in  "${ALIGNED_NON}" \
    -out "${ALIGNED_TRIMMED}" \
    -gt "${TRIMAL_GAP_THRESHOLD}"

# =============================================================================
# Step 7: Initial ML tree (IQTree)
# =============================================================================
echo ""
echo "=== Step 7: IQTree (initial) ==="
"${IQTREE}" -s "${ALIGNED_TRIMMED}" -T "${IQTREE_THREADS}"

# =============================================================================
# Step 8: Initial time-calibrated tree (treetime)
# =============================================================================
echo ""
echo "=== Step 8: treetime (initial) ==="
# --stochastic-resolve: randomly breaks polytomies (used on first pass only;
#   later passes use --greedy-resolve for stability).
# --clock-filter ${CLOCK_FILTER}: flag sequences deviating > 4 IQDs from the
#   root-to-tip regression.
"${TREETIME}" \
    --tree   "${ALIGNED_TRIMMED}.treefile" \
    --aln    "${ALIGNED_TRIMMED}" \
    --dates  "${SAMPLED_DIR}/sampled_dates.csv" \
    --stochastic-resolve \
    --outdir "${SAMPLED_DIR}" \
    --clock-filter "${CLOCK_FILTER}"

# =============================================================================
# Step 9: Iterative outlier removal
# =============================================================================
echo ""
echo "=== Step 9: Iterative clock outlier removal ==="
# treetime writes outliers.tsv when any sequences fall outside the clock-filter
# threshold. This loop removes them, rebuilds the tree, and re-runs treetime
# until no outliers remain.
#
# MANUAL INSPECTION REQUIRED before each iteration:
#   - root_to_tip_regression.pdf  (is clock signal improving or fragmenting?)
#   - outliers.tsv                (are flagged sequences genuine outliers?)
#   - molecular_clock.txt         (R² and rate — sanity-check the estimate)
#
# The script pauses and asks for confirmation before removing each batch.

ITERATION=0
CURRENT_FASTA="${ALIGNED_TRIMMED}"
CURRENT_METADATA="${SAMPLED_DIR}/sampled_sequences_metadata.csv"
CURRENT_DATES="${SAMPLED_DIR}/sampled_dates.csv"
CURRENT_OUTDIR="${SAMPLED_DIR}"

while true; do
    OUTLIERS_FILE="${CURRENT_OUTDIR}/outliers.tsv"

    if [[ ! -f "${OUTLIERS_FILE}" ]]; then
        echo "No outliers.tsv found at ${OUTLIERS_FILE} — assuming clean."
        break
    fi

    OUTLIER_COUNT=$(awk 'NR>1' "${OUTLIERS_FILE}" | wc -l | tr -d ' ')

    echo ""
    echo "===== MANUAL INSPECTION REQUIRED (iteration ${ITERATION}) ====="
    echo "  Outliers flagged: ${OUTLIER_COUNT}"
    echo "  Review the following before continuing:"
    echo "    ${CURRENT_OUTDIR}/root_to_tip_regression.pdf"
    echo "    ${OUTLIERS_FILE}"
    echo "    ${CURRENT_OUTDIR}/molecular_clock.txt"
    echo ""
    if [[ "${AUTO_ACCEPT_OUTLIERS}" == "true" ]]; then
        CONFIRM="y"
        echo "AUTO_ACCEPT_OUTLIERS=true — automatically proceeding."
    else
        read -r -p "Remove these outliers and re-run IQTree + treetime? [y/n] " CONFIRM
    fi
    if [[ "${CONFIRM}" != "y" ]]; then
        echo "Stopping outlier loop at user request."
        echo "Current FASTA: ${CURRENT_FASTA}"
        break
    fi

    # After the first iteration the cleaned sequences go to FINAL_DIR.
    NEXT_OUTDIR="${FINAL_DIR}"
    conda run -n "${CONDA_ENV}" python bin/remove_outliers.py \
        -o "${NEXT_OUTDIR}" \
        -n "${FINAL_FASTA_NAME}" \
        "${OUTLIERS_FILE}" \
        "${CURRENT_FASTA}" \
        "${CURRENT_METADATA}" \
        "${CURRENT_DATES}"

    # move outlier file to renamed outlier file
    mv "${CURRENT_OUTDIR}/outliers.tsv" "${CURRENT_OUTDIR}/outliers_${ITERATION}.tsv"

    CURRENT_FASTA="${NEXT_OUTDIR}/${FINAL_FASTA_NAME}"
    CURRENT_METADATA="${NEXT_OUTDIR}/${FINAL_FASTA_NAME%.fasta}_metadata.csv"
    CURRENT_DATES="${NEXT_OUTDIR}/${FINAL_FASTA_NAME%.fasta}_dates.csv"
    CURRENT_OUTDIR="${NEXT_OUTDIR}"

    echo ""
    echo "--- Re-running IQTree (iteration $((ITERATION + 1))) ---"
    "${IQTREE}" -s "${CURRENT_FASTA}" -T "${IQTREE_THREADS}" -redo

    echo ""
    echo "--- Re-running treetime (iteration $((ITERATION + 1))) ---"
    # --greedy-resolve on subsequent passes (deterministic, faster than stochastic).
    "${TREETIME}" \
        --tree   "${CURRENT_FASTA}.treefile" \
        --aln    "${CURRENT_FASTA}" \
        --dates  "${CURRENT_DATES}" \
        --greedy-resolve \
        --outdir "${CURRENT_OUTDIR}" \
        --clock-filter "${CLOCK_FILTER}"

    ITERATION=$((ITERATION + 1))
done

# If the outlier loop never ran (no outliers on the first treetime pass),
# FINAL_DIR was never created. Copy the necessary files there under the
# canonical names so that Steps 10/11 and downstream scripts always find
# the same structure regardless of how many outlier rounds occurred.
if [[ "${ITERATION}" -eq 0 ]]; then
    echo ""
    echo "No outliers detected — copying clean sequences to ${FINAL_DIR}/"
    mkdir -p "${FINAL_DIR}"
    cp "${CURRENT_FASTA}"    "${FINAL_DIR}/${FINAL_FASTA_NAME}"
    cp "${CURRENT_METADATA}" "${FINAL_DIR}/${FINAL_FASTA_NAME%.fasta}_metadata.csv"
    cp "${CURRENT_DATES}"    "${FINAL_DIR}/${FINAL_FASTA_NAME%.fasta}_dates.csv"
    cp "${CURRENT_OUTDIR}/timetree.nexus" "${FINAL_DIR}/timetree.nexus"
fi

# =============================================================================
# Step 10: Append sequence_ID to metadata
# =============================================================================
echo ""
echo "=== Step 10: Append sequence_ID to metadata ==="
# create_mascot_xml.py uses a sequence_ID column to map sequences to demes.
# This step derives those IDs by matching FASTA headers to Accession IDs.
conda run -n "${CONDA_ENV}" python bin/append_fasta_header_meta.py \
    --metadata "${FINAL_DIR}/${FINAL_FASTA_NAME%.fasta}_metadata.csv" \
    --fasta    "${FINAL_DIR}/${FINAL_FASTA_NAME}" \
    --output   "${FINAL_DIR}/${FINAL_FASTA_NAME%.fasta}_metadata_with_ids.csv"

# =============================================================================
# Step 11: Resolve polytomies in the timetree
# =============================================================================
echo ""
echo "=== Step 11: Resolve polytomies ==="
# create_mascot_xml.py will error on polytomies when using a fixed timetree.
# dendropy's resolve_polytomies() inserts zero-length branches, which minimises
# structural assumptions about unresolved splits.
TIMETREE_NEXUS="${FINAL_DIR}/timetree.nexus"
TIMETREE_RESOLVED="${FINAL_DIR}/timetree_resolved.nexus"
conda run -n "${CONDA_ENV}" python -c "
import dendropy
tree = dendropy.Tree.get(path='${TIMETREE_NEXUS}', schema='nexus')
tree.resolve_polytomies()
tree.write(path='${TIMETREE_RESOLVED}', schema='nexus')
print('Wrote resolved timetree: ${TIMETREE_RESOLVED}')
"

echo ""
echo "=== Sequence preparation complete ==="
echo "Final sequences: ${FINAL_DIR}/${FINAL_FASTA_NAME}"
echo "Metadata with IDs: ${FINAL_DIR}/${FINAL_FASTA_NAME%.fasta}_metadata_with_ids.csv"
echo "Resolved timetree: ${FINAL_DIR}/timetree_resolved.nexus"
