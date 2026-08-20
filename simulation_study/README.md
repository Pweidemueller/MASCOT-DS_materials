# MASCOT-DS Simulation Study

This folder contains the Nextflow pipeline and scripts used to test MASCOT-DS on simulated, structured two-deme SIR outbreaks.

## Study design

For each of 100 simulations, epidemiological parameters (transmission rates, sampling proportion, population sizes, ...) are randomly sampled, then used to simulate a SIR outbreak between two demes and its sampled phylogeny with [remaster](https://github.com/tgvaughan/remaster) (needs BEAST2 and remaster installed). From each simulated outbreak, case count, seroprevalence, and wastewater data streams are generated, and a BEAST2/MASCOT-DS XML is built for several data stream input **versions** — the full model (`datastreams`) and versions with one data stream held back or only the tree given (`datastreams_nocasecounts`, `datastreams_noseroprevalence`, `datastreams_nowastewater`, `datastreams_nomascotll`, `datastreams_onlytree`) plus the `original` MASCOT model without data streams. Each XML is run under BEAST2/MASCOT-DS with 3 seeds, logs are combined, and the resulting posteriors are compared against the known simulation truth to assess parameter recovery and the value of each data stream.

## Pipeline overview

The pipeline is implemented in [`main.nf`](main.nf) as two Nextflow entry points:

- **Default workflow** (`nextflow run main.nf`) — runs the full pipeline end to end: sample parameters → simulate outbreak (remaster) → generate data streams → build MASCOT-DS XMLs for all variants → run BEAST2 → combine logs/trees → annotate/plot trees. This is the compute-heavy step (100 simulations × 7 variants × 3 seeds of BEAST2 runs) and is what was run on the Wynton HPC cluster to produce the published results.
- **`ANALYSE_FROM_BEASTOUTPUTS` workflow** — skips simulation and inference entirely and only re-runs post-processing/figure generation, reading combined BEAST2 log files from an existing `--outdir` (e.g. the published results directory). This is the fast, cheap way to reproduce the paper's figures and tables without re-running any MCMC:
  ```
  nextflow run main.nf -entry ANALYSE_FROM_BEASTOUTPUTS \
      -profile docker \
      --outdir /path/to/published/results
  ```

Both entry points read tool paths, resource limits, and container images from [`nextflow.config`](nextflow.config) — see "Reproducing on your own machine" below.

## Installation

| Tool | Used for | Install |
|---|---|---|
| [Nextflow](https://www.nextflow.io/docs/latest/install.html) (≥22.10.0) | Pipeline orchestration | `curl -s https://get.nextflow.io \| bash` |
| [BEAST2](https://www.beast2.org/) (2.7.7) with [remaster](https://github.com/tgvaughan/remaster), [MASCOT](https://github.com/CompEvol/Mascot) and [MASCOT-DS](https://github.com/Pweidemueller/Mascot_datastreams) | Outbreak simulation and phylodynamic inference | See BEAST2/package manager installation instructions. **Not containerized** — always needs a local install (see below). |
| Docker or Apptainer/Singularity | Running the Python/R pre- and post-processing steps in a reproducible container, without installing dependencies yourself | [Docker](https://docs.docker.com/get-docker/) or [Apptainer](https://apptainer.org/docs/user/main/quick_start.html) |

If you'd rather not use containers, the Python/R steps can also be run from a local `biopython_env` conda environment (as in the `local` profile) — see [`../bayarea_application/environment.yml`](../bayarea_application/environment.yml) for the equivalent package list; a native R install with `tidyverse` and `ape`/`ggtree` covers the tree-plotting steps.

### Containers

The Python and R pre-/post-processing steps run in two containers built for **linux/amd64**, published on Docker Hub:

- Python/BioPython: https://hub.docker.com/repository/docker/phweide/biopython
- R/tidyverse: https://hub.docker.com/repository/docker/phweide/r-tidyverse

On an Apple Silicon/ARM machine, Docker will run these under amd64 emulation automatically (the `docker` profile below sets `--platform linux/amd64` explicitly).

## Reproducing on your own machine

Three profiles are defined in [`nextflow.config`](nextflow.config):

- **`docker`** — general-purpose profile for reproducing this study outside the author's compute cluster (wynton). Pulls both containers directly from Docker Hub the first time they're needed; requires only Docker and a local BEAST2 install. **Recommended starting point.**
- **`local`** and **`wynton`** — the author's own working profiles (macOS + conda, and the Wynton SGE cluster + pre-pulled `.sif` files), kept as-is for continuity with the original runs. Their hardcoded paths are specific to the author's machines and will need editing to reuse directly. The `wynton` profile lives in its own file, [`conf/wynton.config`](conf/wynton.config), so it's easy to copy as a starting point (SGE `clusterOptions`/`penv`, work directory, container/BEAST2 paths) if you're setting this pipeline up on a different HPC cluster.

To run with the `docker` profile, point it at your own BEAST2 install on the command line (BEAST2 is not containerized, so this is required regardless of profile):
```
nextflow run main.nf -profile docker \
    --beast_path /path/to/beast/bin/beast \
    --logcombiner_path /path/to/beast/bin/logcombiner \
    --treeannotator_path /path/to/beast/bin/treeannotator
```

**Running on a cluster without internet access on compute nodes**: Docker Hub can't be reached at task run time, so pre-pull both images into local `.sif` files on a login node (which does have internet access):
```
apptainer pull biopython.sif    docker://phweide/biopython:latest
apptainer pull r-tidyverse.sif  docker://phweide/r-tidyverse:latest
```
then point the pipeline at those files instead of Docker Hub, either on the command line:
```
nextflow run main.nf -profile wynton \
    --biopython_container  /path/to/biopython.sif \
    --rtidyverse_container /path/to/r-tidyverse.sif \
    --beast_path /path/to/beast/bin/beast \
    ...
```
or via a small local config file (e.g. `-c conf/mycluster.config`, not committed to the repo) that sets `params.biopython_container` / `params.rtidyverse_container` and the BEAST paths for your own cluster. [`conf/wynton.config`](conf/wynton.config) is a complete, concrete example of this pattern — copy it and adjust the executor and SGE-specific `clusterOptions`/`penv` for your own scheduler if it isn't SGE.

## Make figures

### Individual simulation
```
bash simulation_study/bin/run_individualsim_figure_86_2.sh
```

### True vs. estimate and value-of-information figures
```
nextflow run main.nf -entry ANALYSE_FROM_BEASTOUTPUTS \
    -profile docker \
    --outdir /path/to/results
```

## Estimate efficiency/convergence times
```
conda run -n biopython_env python simulation_study/bin/sampler_efficiency.py \
    --output-dir simulation_study/results_individuallogs/sampler_efficiency
```

## Analyse sigma over-estimation
1. Per-simulation wastewater PPC + residual ingredients (uses the combined log which already had burn-in removed):
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
2. Build the 3-panel supplementary figure:
```
conda run -n biopython_env python simulation_study/bin/make_figure_ww_sigma_supp.py \
    --summary_csv      simulation_study/results/ww_ppc_full/ww_ppc_persim_summary.csv \
    --per_obs_csv      simulation_study/results/ww_ppc_full/ww_ppc_per_obs.csv \
    --output_dir       simulation_study/results/ww_ppc_full \
    --panel_b_simid    45_2 \
    --panel_b_max_days 30 \
    --panel_b_deme     1
```
