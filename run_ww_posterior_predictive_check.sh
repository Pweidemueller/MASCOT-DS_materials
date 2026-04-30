#!/usr/bin/env bash
# Run the wastewater posterior predictive check across all simulations,
# then aggregate the per-sim scalars into a cross-sim summary.
#
# The per-sim work is in run_ww_ppc_single.sh so that the xargs command
# template stays short (macOS xargs -I{} caps substituted args at 255 bytes,
# which is why the previous inline version failed).
set -eu

BASE=simulation_study/results
mkdir -p sandbox/ww_ppc_allsims

# Parallelism. Set WW_PPC_JOBS=1 for sequential, or higher if the machine
# has spare cores.
: "${WW_PPC_JOBS:=4}"

ls "$BASE/2_mascot/" | xargs -n1 -P"$WW_PPC_JOBS" bash run_ww_ppc_single.sh \
  || echo "[warn] one or more sims failed; see sandbox/ww_ppc_allsims/<sim>/run.log"

conda run -n biopython_env python simulation_study/bin/analyse_ww_ppc_crosssim.py \
  --ppc_dirs sandbox/ww_ppc_allsims/*/ww_ppc \
  --output_dir sandbox/ww_ppc_crosssim_allsims
