# Test MASCOTdatastreams on structured SIR outbreaks

## Run simulation study

## Make figures
### Individual simulation
```
bash simulation_study/bin/run_individualsim_figure_86_2.sh
```

### True vs estimate and value of information figures
```
nextflow run main.nf -entry ANALYSE_FROM_BEASTOUTPUTS \
    --outdir /Users/pweidemuller/Documents/git_repos/MASCOT-DS_materials/simulation_study/results
```


## Estimate efficiency/convergence times
```
conda run -n biopython_env python simulation_study/bin/sampler_efficiency.py \
    --output-dir simulation_study/results_individuallogs/sampler_efficiency
```
## Analyse sigma over-estimation
1. Per-simulation wastewater PPC + residual ingredients: uses the combined log which already had burnin removed
```
conda run -n biopython_env python simulation_study/bin/compute_ww_ppc_persim.py \
    --results_dir simulation_study/results \
    --output_dir  simulation_study/results/ww_ppc_full \
    --burnin      0.0 \
    --n_samples   1000 \
    --seed        0
    # optional subsetting (omit for the full 100-sim run):
    #   --limit 5
    #   --only_simids 7_2 20_2 8_2
```
2. Build the 3-panel supplementary figure (+ standalone mean-misfit plot)
```
conda run -n biopython_env python simulation_study/bin/make_figure_ww_sigma_supp.py \
    --summary_csv      simulation_study/results/ww_ppc_full/ww_ppc_persim_summary.csv \
    --per_obs_csv      simulation_study/results/ww_ppc_full/ww_ppc_per_obs.csv \
    --output_dir       simulation_study/results/ww_ppc_full \
    --panel_b_simid    45_2 \
    --panel_b_max_days 30 \
    --panel_b_deme     1
```