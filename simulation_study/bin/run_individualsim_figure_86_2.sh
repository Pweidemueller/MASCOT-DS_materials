#!/usr/bin/env bash
#
# Produce the individual-simulation figures for simulation 86_2:
#   * per-deme combined prevalence + cumulative incidence panels
#       <SIM>_deme0_start_deme_combprevcuminc.{png,pdf}
#       <SIM>_deme<N>_secondary_deme_combprevcuminc.{png,pdf}
#     (--combprevcuminc_only suppresses the per-deme prevalence_log and Ne panels)
#   * the 4-panel start-deme dynamics summary figure
#       <SIM>_dynamics_summary.{png,pdf}
#
# Run from the repository root:
#   bash simulation_study/bin/run_individualsim_figure_86_2.sh
#
set -euo pipefail

SIM=86_2_simulation
BASE=simulation_study/results
DS=$BASE/2_mascot/$SIM/datastreams
ORIG=$BASE/2_mascot/$SIM/original
RAW=$BASE/1_remaster_sim
OUT=$BASE/individualsim

mkdir -p "$OUT"

conda run -n biopython_env python simulation_study/bin/make_figure_individualsim.py \
    --log_file_original          "$ORIG/${SIM}_original.mascot_logs.combined.log" \
    --log_file_datastream        "$DS/${SIM}_datastreams.mascot_logs.combined.log" \
    --case_counts_file           "$RAW/${SIM}_casecounts.csv" \
    --seroprevalence_file        "$RAW/${SIM}_seroprevalence.csv" \
    --wastewater_file            "$RAW/${SIM}_wastewater.csv" \
    --cumulative_incidence_deme1 "$DS/${SIM}_datastreams.cumulativeIncidence.Deme1.combined.log" \
    --cumulative_incidence_deme2 "$DS/${SIM}_datastreams.cumulativeIncidence.Deme2.combined.log" \
    --nedynamics_deme1           "$DS/${SIM}_datastreams.NeDynamics.Deme1.combined.log" \
    --nedynamics_deme2           "$DS/${SIM}_datastreams.NeDynamics.Deme2.combined.log" \
    --trajectory_file            "$RAW/${SIM}.traj" \
    --params_csv                 "$RAW/${SIM}_parameters.csv" \
    --deme_switches_csv          "$RAW/${SIM}_deme_switches_groundtruth.csv" \
    --burnin                     0.1 \
    --out_prefix                 "$OUT/${SIM}" \
    --combprevcuminc_only \
    --dynamics_figure_out        "$OUT/${SIM}_dynamics_summary.png"

echo "Figures written to $OUT"
