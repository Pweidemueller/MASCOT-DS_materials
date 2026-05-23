#!/usr/bin/env bash
# Run the wastewater posterior predictive check across all simulations,
# then aggregate the per-sim scalars into a cross-sim summary.
#
# The per-sim work is in run_ww_ppc_single.sh so that the xargs command
# template stays short (macOS xargs -I{} caps substituted args at 255 bytes,
# which is why the previous inline version failed).
set -eu

BASE=simulation_study/results
mkdir -p results/6_ww_ppc_allsims

# Runtime selection:
#   WW_PPC_RUNTIME=conda      -> use `conda run -n biopython_env python`
#   WW_PPC_RUNTIME=apptainer  -> use `apptainer exec <image> python`
# Defaults to conda for local usage.
: "${WW_PPC_RUNTIME:=conda}"
: "${WW_PPC_CONDA_ENV:=biopython_env}"
: "${WW_PPC_APPTAINER_IMAGE:=/wynton/home/mueller/pweide/structured_sims/containers/biopython.sif}"

if [ "$WW_PPC_RUNTIME" = "conda" ]; then
  PYTHON_CMD=(conda run -n "$WW_PPC_CONDA_ENV" python)
elif [ "$WW_PPC_RUNTIME" = "apptainer" ]; then
  PYTHON_CMD=(apptainer exec "$WW_PPC_APPTAINER_IMAGE" python)
else
  echo "[error] WW_PPC_RUNTIME must be 'conda' or 'apptainer' (got: $WW_PPC_RUNTIME)" >&2
  exit 1
fi

# Export so child scripts can use the same runtime.
export WW_PPC_RUNTIME WW_PPC_CONDA_ENV WW_PPC_APPTAINER_IMAGE

# Parallelism. Set WW_PPC_JOBS=1 for sequential, or higher if the machine
# has spare cores.
: "${WW_PPC_JOBS:=4}"

ls "$BASE/2_mascot/" | xargs -n1 -P"$WW_PPC_JOBS" bash simulation_study/run_ww_ppc_single.sh \
  || echo "[warn] one or more sims failed; see sandbox/ww_ppc_allsims/<sim>/run.log"

"${PYTHON_CMD[@]}" simulation_study/bin/analyse_ww_ppc_crosssim.py \
  --ppc_dirs results/6_ww_ppc_allsims/*/ww_ppc \
  --output_dir results/6_ww_ppc_crosssim
