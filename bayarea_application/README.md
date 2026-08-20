# MASCOTDS Application
This folder contains all templates and scripts to reproduce the results of the MASCOT-DS analysis of the SARS-CoV-2 winter 2020-21 wave in the Bay Area.

First, make sure to download all data described in [README_data](README_data.md).

For running any of the commands below, make sure you are in the `bayarea_application` folder.

## Installation
The scripts in this folder rely on the following software. Install each so that the executable is on your `PATH` (e.g. via conda, a package manager, or by adding its location to `PATH`); alternatively, point a script at a specific binary by setting the matching environment variable before running it (see the "CONFIG — Tool paths" block at the top of each script, e.g. `TRIMAL=/path/to/trimal bash scripts/01_sequence_preparation.sh ...`).

| Tool | Used for | Install |
|---|---|---|
| Conda/Mamba | Python environment management | https://github.com/conda-forge/miniforge |
| `biopython_env` (conda env) | All Python steps (sequence merging/filtering, postprocessing, plotting) | `conda env create -f environment.yml` |
| [MAFFT](https://mafft.cbrc.jp/alignment/software/) | Multiple sequence alignment | Download pre-built package for your OS, this analysis used v7.490 |
| [trimAl](http://trimal.cgenomics.org/) | Alignment column trimming | Download a prebuilt binary, we used v1.5 |
| [IQ-TREE](http://www.iqtree.org/) (v3) | Maximum-likelihood phylogeny inference | Download a release from https://github.com/iqtree/iqtree3/releases , this analysis used Release 3.0.1|
| [TreeTime](https://github.com/neherlab/treetime) | Time-calibration of the ML tree, clock-outlier filtering | `pip install phylo-treetime` (recommended in its own virtual environment — its numpy/scipy requirements can conflict with `biopython_env`), we used v0.11.4 |
| [BEAST2](https://www.beast2.org/) with [MASCOT](https://github.com/nicfel/Mascot) and [MASCOT-DS](https://github.com/Pweidemueller/Mascot_datastreams) | Phylodynamic inference | See BEAST2/MASCOT installation instructions |

Create the Python environment with:

```
conda env create -f environment.yml
conda activate biopython_env
```

## Produce a multiple sequence alignment from Epsilon sequences
Merges the GISAID fasta and tsv files into one big fasta and metadata file. Extracts 310 randomly chosen Epsilon sequences for the three local Bay Area counties, and 70 Epsilon sequences randomly chosen from across the US.

```
bash scripts/01_sequence_preparation.sh results 70
```

## Prepare epidemiological data streams
Filters and extracts case counts, wastewater concentrations and seroprevalence observations for the three counties (Sacramento, San Francisco, Santa Clara) within the time period of interest (2020-10-01 until sampling date of most recent sequence).

```
bash scripts/02_datastream_preparation.sh results
```

## Create MASCOT-DS XML files
Creates the MASCOT-DS XML files for the full data stream model and also the leave-on-out versions (data stream input versions), each saved into a separate subfolder in `results`.

```
bash scripts/03_xml_generation.sh results 70
```

## Run MCMC inference with BEAST2
Each XML file should be run with 3 different seeds. We used seeds 420001, 420002 and 420003. The BEAST2 command looks like this, we recommend running these on a compute cluster. Make sure to replace `beast` with a path to the BEAST2 binary if it is not on your PATH.

```
beast \
    -seed 420001 \
    -prefix "420001" \
    results/SARSCoV2_Epsilon_BayArea_results_datastreams/SARSCoV2_Epsilon_BayArea_results_datastreams.xml
```

Make sure all output files for a given data stream input version are stored in the respective subfolders, e.g. `results/SARSCoV2_Epsilon_BayArea_results_datastreams/`.

## Analysing BEAST2 run outputs
To analyse all data stream input versions jointly, just run the wrapper script. This will also output the value of information plots in `results/value_of_information`. The figures for the all data streams version as shown in the publication can be located in `results/SARSCoV2_Epsilon_BayArea_results_datastreams/00_figures`.

```
bash scripts/05_postprocessing_all.sh results results/SARSCoV2_Epsilon_BayArea_results_datastreams/SARSCoV2_Epsilon_BayArea_results_state_time.csv
```

If you want to analyse individual data stream input versions, run:
```
bash scripts/05_postprocessing.sh results/SARSCoV2_Epsilon_BayArea_results_datastreams results results/SARSCoV2_Epsilon_BayArea_results_datastreams/SARSCoV2_Epsilon_BayArea_results_state_time.csv
```

## Fix Ne or case count scaler in the abscence of seroprevalence data
Ideally the full datastream version is converged (ESS>200). We ran this when ESS values were not yet >200, but the combined log showed clear stationarity.
```
python bin/fix_noseroprev_params_from_posterior.py --input_xml results/SARSCoV2_Epsilon_BayArea_results_datastreams_noseroprevalence/SARSCoV2_Epsilon_BayArea_results_datastreams_noseroprevalence.xml --combined_log results/SARSCoV2_Epsilon_BayArea_results_datastreams/SARSCoV2_Epsilon_BayArea_results_datastreams.combined.log --output_dir results/SARSCoV2_Epsilon_BayArea_results_datastreams_noseroprevalence_fixNe
```

Then run MCMC inference on the these newly created XMLs using the instructions above (three seeds and then combining them with the `scripts/05_postprocessing.sh`).

## Test effect of seroprevalence scale
In order to test the impact of scale of seroprevalence observations, we also artificially increased (x2) and decreased (x0.5) the values. For this simply create two new folders in `results`: 
`SARSCoV2_Epsilon_BayArea_results_datastreams_serop2` and `SARSCoV2_Epsilon_BayArea_results_datastreams_serop05`. Copy `results/SARSCoV2_Epsilon_BayArea_results_datastreams/SARSCoV2_Epsilon_BayArea_results_datastreams.xml` into each subfolder. Then extract the seroprevalence value elements in the XML (`<parameter id="seroWithAntibodiesCounts.Deme1:SimDataset"...` for all three Demes) and multiply these values by 2 and 0.5 and paste that string back in.

Then run MCMC inference on the these newly created XMLs using the instructions above (three seeds and then combining them with the `scripts/05_postprocessing.sh`).

Create the supplementary figure in the `bioconda_env` environment:
```
python bin/test_impact_seropscaling.py --output-dir results/seropscaling
```