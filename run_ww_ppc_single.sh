#!/usr/bin/env bash
# Run the wastewater posterior predictive check for a single simulation.
# Usage: run_ww_ppc_single.sh <SIM_NAME>
set -eu
SIM="$1"
BASE=simulation_study/results_gridpointsnap
DS=$BASE/2_mascot/$SIM/datastreams
ORIG=$BASE/2_mascot/$SIM/original
RAW=$BASE/1_remaster_sim
OUT=sandbox/snap_ww_ppc_allsims/$SIM
mkdir -p "$OUT"
echo "=== $SIM ==="
conda run -n biopython_env python simulation_study/bin/make_figure_individualsim.py \
  --log_file_original          $ORIG/${SIM}_original.mascot_logs.combined.log \
  --log_file_datastream        $DS/${SIM}_datastreams.mascot_logs.combined.log \
  --case_counts_file           $RAW/${SIM}_casecounts.csv \
  --seroprevalence_file        $RAW/${SIM}_seroprevalence.csv \
  --wastewater_file            $RAW/${SIM}_wastewater.csv \
  --cumulative_incidence_deme1 $DS/${SIM}_datastreams.cumulativeIncidence.Deme1.combined.log \
  --cumulative_incidence_deme2 $DS/${SIM}_datastreams.cumulativeIncidence.Deme2.combined.log \
  --nedynamics_deme1           $DS/${SIM}_datastreams.NeDynamics.Deme1.combined.log \
  --nedynamics_deme2           $DS/${SIM}_datastreams.NeDynamics.Deme2.combined.log \
  --trajectory_file            $RAW/${SIM}.traj \
  --params_csv                 $RAW/${SIM}_parameters.csv \
  --deme_switches_csv          $RAW/${SIM}_deme_switches_groundtruth.csv \
  --ww_ppc_grid_mode                  interpolate \
  --burnin                     0.0 \
  --out_prefix                 $OUT/${SIM} \
  --spline_snap_diagnostics_out $OUT/spline_snap_diag \
  --skip_per_deme_figures \
  --ww_ppc_out                 $OUT/ww_ppc \
  --ww_ppc_n_samples           1000 > "$OUT/run.log" 2>&1
