# Test MASCOTdatastreams on structured SIR outbreaks

TO DO
- [ ] in `create_birthdeath_simXML.py` the sampling_rate is a bit of a misnomer, the parameter is more like a sampling probability which then `gamma * sampling_rate` leads to a true rate as used in Remaster

This repository contains a Nextflow pipeline that simulates structured SIR outbreaks (ReMASTER), simulates additional datastreams (e.g. case counts) from simulated trajectories, and performs structured coalescent inference with BEAST2/MASCOT (with and without datastreams). This README explains the modeling logic and assumptions.

## Workflow logic
The pipeline in `main.nf` runs these steps:
- `MAKE_SIM_XML` -> generate a ReMASTER XML per contact matrix (via `bin/create_birthdeath_simXML.py`). Contact matrices are given in `data/contact_matrices/`.
- `RUN_REMASTER` -> simulate tree, trajectories, alignment with ReMASTER.
- `SIMULATE_DATASTREAMS` -> derive case counts, wastewater, serology from trajectories (via `bin/simulate_datastreams.py`).
- `MAKE_MASCOT_XML` -> build Mascot XMLs (original vs datastreams) from alignment/tree/case counts (via `bin/create_mascot_xml_fixedtree.py`).
- `RUN_MASCOT` -> run inference with Mascot; `COMBINE_LOGS/TREES` -> merge across seeds.
- Optional `ANALYSE_POSTERIORS` -> compare inferred Ne(t) (via `bin/analyse_posteriors.py`).

Defaults and paths live in `nextflow.config`. Simulation time is in years unless stated.

---

## Step 1 — Structured SIR simulation (ReMASTER)
Inputs: contact matrix C per dataset, per-contact probability q (or a direct mixing matrix M), target `R0`, recovery rate `gamma` (per year), initial `S` and `I` per deme, sampling rate `s`.

- Mixing: $M = q \cdot C$ if a contact matrix is used (`bin/create_birthdeath_simXML.py:load_mixing_matrix`).
- Population fraction per deme: $f_i = (S_i + I_i) / \Sigma_j (S_j + I_j)$.
- R0 calibration (spectral radius):
  - $K_0 = M \cdot diag(f) \cdot diag(1/\gamma)$; $\phi_0 = spectral_radius(K_0)$.
  - $\beta = (R_0 / \phi_0) \cdot M$ ensures the next-generation matrix has dominant eigenvalue $R_0$ (`get_beta_demes`).
- Mass-action reactions per deme i:
  - Within: $S[i] + I[i] \rightarrow 2 I[i]$ at rate parameter $\beta[i,i] / N_i$.
  - Between: $I[i] + S[j] \rightarrow I[i] + I[j]$ at rate parameter $\beta[i,j] / N_j$ (i ≠ j).
  - Removals split by sampling: $I[i] \rightarrow Rs$ at $\gamma \cdot s$, and $I[i] \rightarrow Rh[i]$ at $\gamma \cdot (1-s)$.
- Outputs: `*.traj` (trajectories of S/I/Rh/Rs), `*.trees` (typed trees), `*.nexus` (alignment).

---

## Step 2 — Datastream simulation from trajectories
Inputs: `*.traj` from Step 1. Internals in `bin/simulate_datastreams.py`.

- Sampling cadence: choose indices closest to a fixed day-grid (e.g., every 1, 3, 7 days) on $t_{days} = 365 \cdot t$.
- Wastewater (heuristic): delayed, lognormal-noisy transform of $I(t)$ scaled by population $N$.
- Serology (heuristic): lognormal-noisy transform of cumulative $I(t)$ scaled by $N$.
- Case counts: $cases(t) = floor(p_detect \cdot I(t))$ at sampled days; integer day times.
- Output: `<stem>_casecounts.csv` (columns: `case_counts`, `t_case_counts_fromsimstart`, `t_case_counts_frommostrecentsample`, `index`) and `<stem>_trajectories.png`.

---

## Step 3 — Mascot XMLs (original vs datastreams)
Inputs: alignment (`*.nexus`), tree (`*.trees`), and case counts CSV.

- `create_mascot_xml_fixedtree.py` rebuilds:
  - Alignment `<data>` from Nexus; `dateTrait` from tip times; `typeTrait` from deme states.
  - If datastreams: add `<data id="caseCounts.t:SimDataset" spec="mascotdatastreams.distribution.CaseCountData">` from the CSV.
- Templates: `data/Mascot_template_fixedtree.xml` (sequence only) and `data/Mascot_datastreams_template_fixedtree.xml` (adds Negative Binomial case-count likelihood with dispersion parameter).
- Outputs: `<xml_name>_original.xml` and `<xml_name>_datastreams.xml` plus `<xml_name>_state_time.csv`.

---

## Step 4 — Mascot inference and combining
Mascot is run per XML with three seeds; outputs are prefixed by seed. `LogCombiner` merges logs/trees per XML after a burn-in fraction (`params.log_burnin`, `params.tree_burnin`).

---

## Posterior analysis (optional)
`bin/analyse_posteriors.py` parses Mascot logs, extracts `SkylineNe.*.*`, removes 10% burn-in, interpolates to a common grid, computes means and 95% HPDs for `Ne(t)` (exp of logNe), and optionally overlays trajectories and case counts. Outputs figures and CSVs.

---

### Expected effective population size per deme (Volz 2012)
`bin/analyse_posteriors.py` can also compute an expected effective population size (coalescent effective size) trajectory for each deme directly from the simulation trajectories and the parameters CSV produced by `bin/create_birthdeath_simXML.py`.

- Formula (Volz 2012): for deme i at time t: $Ne_i(t) = I_i(t) / (2 · \beta_i · S_i(t))$, where $S_i(t)$ and $I_i(t)$ are the susceptible and infected counts in deme i at time t, and β_i is the within-deme transmission rate parameter used in the mass-action reaction for i.

- Important: The parameters CSV exports the rate parameters actually used in the reactions:
  - For within-deme infection, the exported value is $\beta_{ii} / N_i$ (where N_i is the total population size of deme i). This matches the XML reaction rate parameterization `S[i] + I[i] -> 2I[i]` at rate $\beta_{ii}/N_i$.
  - Therefore, the expected Ne computed by the script uses $\beta_i = \beta_{ii} / N_i$ from the CSV, i.e., $Ne_i(t) = I_i(t) / (2 · (\beta_{ii}/N_i) \cdot S_i(t))$.

- Inputs required:
  - Trajectory file (`*.traj`) containing columns `Sample, t, population, index, value` (with populations among `S`, `I`, `Rh`, `Rs`). Here, `t` is in days.
  - Parameters CSV with columns `parameter, deme, value` that includes rows where `parameter == "beta"` and `deme == "i->i"` for within-deme i. Demes are 0-based in the CSV.

- How to enable in the analysis step:
  - Run `bin/analyse_posteriors.py` and pass both the trajectory and parameters CSV paths:
    
    ```bash
    python bin/analyse_posteriors.py \
      --log_file_default path/to/default.log \
      --log_file_datastream path/to/datastream.log \
      --trajectory_file path/to/run.traj \
      --params_csv path/to/run_parameters.csv \
      --out_prefix results/out
    ```
---

## Creating a LaTeX presentation from results

The script `create_latex_presentation.py` automatically generates a LaTeX Beamer presentation containing all plots produced by the pipeline. Tree plots are grouped on the same slide (groundtruth on the left, MASCOT original in the middle, MASCOT datastreams on the right), and other plots are included separately.

### Generating the LaTeX file

Run the script to scan the results directory and generate `presentation.tex`:

```bash
python3 create_latex_presentation.py --results-dir results --output presentation.tex
```

The script will:
- Automatically find all PNG plots in the results directory
- Group plots by base name (e.g., `1_2_simulation`, `2_2_simulation`, etc.)
- Order them numerically
- Create slides with tree comparisons side-by-side

### Compiling to PDF

Compile the LaTeX file to PDF using `pdflatex`:

```bash
pdflatex -interaction=nonstopmode presentation.tex
```

For best results (especially if references or page numbers need to be resolved), run it twice:

```bash
pdflatex -interaction=nonstopmode presentation.tex
pdflatex -interaction=nonstopmode presentation.tex
```

The output will be `presentation.pdf` with all plots included.

---

## Assumptions and caveats
- Closed SIR within demes; mass-action within and between demes; rates per year; time in years.
- $\beta$ matrix scaled to match $R_0$ via spectral radius of the next-generation matrix built from $M$, $f$, and $\gamma$.
- $Rs$ pools sampled removals across demes to anchor "most recent sample" times; $Rh$ are non observed recoveries.
- Datastreams are heuristic (not calibrated surveillance models); parameters are set for signal, not realism.
- Mascot uses structured skyline on logNe with smoothness priors; datastreams add a Negative Binomial likelihood with dispersion.

---

## File map
- Workflow: `main.nf`; config/profiles: `nextflow.config`.
- ReMASTER XML generator: `bin/create_birthdeath_simXML.py`.
- Datastreams: `bin/simulate_datastreams.py`.
- Mascot XML builder: `bin/create_mascot_xml_fixedtree.py`; templates in `data/`.
- Posterior analysis: `bin/analyse_posteriors.py`.
- LaTeX presentation generator: `create_latex_presentation.py`.

If you need run instructions, consult `nextflow.config` and process tags in `main.nf`.

---

## TO DO
[ ] what exactly is RandomTree.t:SimDataset doing? How does changing the name="popSize" change the initial state and/or the next proposed steps?
[ ] check if the time scales of samples and mutation rates + priors are consistent: https://github.com/rbouckaert/DeveloperManual?tab=readme-ov-file#trees--clock-model-parameters 
[ ] make the sampling rate parameter a channel to try low vs. high sampling rates
[ ] start epidemic in low vs. high transmissibility deme
[ ] find a way to calculate the ESS of the posteriors and get them as a table to quickly check if analysis converged
[ ] find a way to automatically compare if MCMC fits the anticipate NE based on the simulation parameters (need to calculate what the NE should be based on the simulation parameters (transmission, death rates etc))