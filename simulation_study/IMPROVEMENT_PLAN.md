# Simulation Study — Improvement Plan

Working document for Phase 1a: refactor Python scripts and optimise the Nextflow
pipeline. Suggestions are grouped by area and prioritised. Check items off or
annotate as we work through them.

---

## 1. Nextflow pipeline optimisation — DONE

All items in this section have been implemented in `main.nf` and `nextflow.config`.
Needs cluster testing to verify end-to-end correctness.

### 1.1 Eliminate duplicated variant-parsing logic — DONE

The workflow block (lines 544–675) repeats the same `base_name` / `variant_type`
extraction regex **five times** — once each for `logs_grouped`,
`cumIncLogs_grouped`, `NeDynamics_grouped`, `trees_grouped`, and
`mascot_all_seeded`. The pattern is always:

```groovy
if (basename.endsWith('_original')) {
    base_name = basename.replaceAll(/_original$/, '')
    variant_type = 'original'
} else if (basename.contains('_datastreams')) {
    def matcher = basename =~ /^(.*)_(datastreams.*)$/
    ...
}
```

**Suggestion:** Extract a Groovy helper function `parseVariant(xmlname)` that
returns a `[base_name, variant_type]` tuple. Replace all five occurrences. This
cuts ~80 lines and makes variant-name changes a one-line fix.

**Implementation:** `parseVariant()` helper defined. All 5 regex blocks eliminated
by carrying `base_name`/`variant_type` through channels (see 1.5). The helper
remains available as a fallback but is no longer called in the main workflow.

### 1.2 Reduce intermediate empty-file creation — DONE

`make_empty_path()` creates on-disk sentinel files in `${projectDir}/empty_files/`
to represent "no data" for excluded datastream variants. The workflow creates
**dozens** of uniquely-named empty files (one per base × variant × datastream
type) just to avoid Nextflow filename collisions.

**Suggestion:** Replace with a single canonical empty file per datastream type
(e.g. `empty_casecounts.csv`, `empty_seroprevalence.csv`,
`empty_wastewater.csv`), using `stageAs` to rename when needed. Or better,
modify `create_mascot_xml_fixedtree.py` to accept `--no-casecounts` /
`--no-seroprevalence` / `--no-wastewater` flags instead of empty file paths —
this makes intent explicit and removes the empty-file machinery entirely.

**Implementation:** Empty-file machinery eliminated entirely for datastream files.
All variants receive the real datastream files; `--variant_type` tells the script
which to use vs ignore. `"original"` was added to the exclusion lists in
`create_mascot_xml_fixedtree.py` so it correctly skips all datastream loading.
For analysis, real files serve as ground truth regardless of MASCOT variant.
`emptyFile()` is retained only as a safety-net fallback for combined log files
that may not exist (e.g. original variant produces no NeDynamics logs), using
per-(base,variant) unique names to avoid Nextflow staging collisions.

### 1.3 Consolidate COMBINE_LOGS: single invocation per simulation — DONE

Currently `COMBINE_LOGS` is called separately for every `(xmlname, logtype)`
pair — that means 5 log types × 7 variants × 50 sims = **1,750 cluster jobs**,
each taking only seconds but each with scheduling overhead (SGE queue time, file
staging). The same applies to `CONCATENATE_HPD_VALIDATION` and `AGGREGATE_ESS`.

**Suggestion:** Merge all log types for a given `xmlname` into a single process
invocation that runs logcombiner in a loop. This trades trivial parallelism for
dramatically fewer jobs. Similarly, `CONCATENATE_HPD_VALIDATION` and
`AGGREGATE_ESS` do simple CSV header-concat that could be a shell one-liner
inside `ANALYSE_POSTERIORS` or `CALCULATE_ESS` respectively, rather than
standalone processes with cluster overhead.

**Implementation:** COMBINE_LOGS now runs one invocation per xmlname. The script
loops over log types (mascot_logs, NeDynamics.Deme1/2, cumulativeIncidence.Deme1/2)
using filename patterns. This reduces ~1,750 cluster jobs to ~350 (7 variants x 50 sims).
The separate cumIncLogs/NeDynamics channel parsing is eliminated — all log files are
extracted from RUN_MASCOT outputs and grouped per xmlname.

### 1.4 Simplify the `base_inputs` channel — DONE

The `base_inputs` channel (lines 729–747) re-creates the same variant list as
`datastream_variants` (lines 467–485) with slightly different tuple structure.
This is fragile — adding a variant requires changes in two places.

**Suggestion:** Derive `base_inputs` from `all_variants` or
`ds_outputs.datastreams` once, and join it with combined logs by `(base, variant)`
directly. This also removes the second batch of `make_empty_path()` calls.

**Implementation:** `base_inputs` now uses canonical empty files (3 shared files
instead of dozens of unique files). The variant list in `base_inputs` matches the
`datastream_variants` definition. The second batch of `make_empty_path()` calls
is eliminated.

### 1.5 Pass `base_name` and `variant_type` through channels, not re-parse from filenames — DONE

Multiple processes receive an XML file and then parse its filename to recover
`base_name` and `variant_type`. This is the source of the repeated regex logic
in 1.1.

**Suggestion:** Carry `base_name` and `variant_type` as val() elements through
the entire channel graph starting from `MAKE_MASCOT_XML`. The information is
already known at that point — there is no need to recover it from filenames
downstream.

**Implementation:** MAKE_MASCOT_XML output now includes `val("${nexus.baseName}")`
(base_name) and `val(variant_type)`. RUN_MASCOT carries both through all three
output emits. All downstream channel operations use the carried metadata — zero
regex parsing remains in the workflow.

### 1.6 Replace `storeDir` with persistent-publish + re-analysis workflow — DONE

**Problem:** The wynton scratch filesystem (where `workDir` lives) is purged
every 2 weeks. Nextflow's `-resume` relies on the work directory, so it cannot
recover expensive `RUN_MASCOT` results after a purge. `storeDir` was used as a
workaround but it cannot restore `val()` outputs — nearly every process in this
pipeline emits `val()` metadata — so it produces warnings and doesn't actually
cache correctly.

A naive "cache only RUN_MASCOT outputs to persistent storage and copy them back"
approach also fails: if scratch is purged and the pipeline re-runs, the
stochastic upstream processes (especially `RUN_REMASTER`) produce **different**
simulations. Pairing cached MASCOT posteriors with fresh simulation outputs
would produce meaningless analysis results.

**Suggestion:** Two changes:

**A. Drop `storeDir` everywhere.** It doesn't work with `val()` outputs and
gives false confidence. For runs where scratch survives, Nextflow's native
`-resume` handles caching correctly for all output types.

**B. Publish the full provenance chain to persistent storage and add an
`ANALYSE_FROM_PUBLISHED` workflow entrypoint.** This gives a safe way to
re-analyse BEAST outputs after scratch is purged, without re-running anything.

Concretely:

1. **Publish all outputs from `SAMPLE_SIMPARAMS` through `RUN_MASCOT`** to a
   persistent location (home directory or lab storage) using `publishDir` (default
   `overwrite: true` so re-runs with bug fixes or new stochastic realisations
   replace stale cached outputs). Use a structured directory layout:

   ```
   ${params.persistent_cache}/
   ├── 0_simparams/
   │   └── {simNb}_{ndemes}_sampled_parameters.csv
   │   └── {simNb}_{ndemes}_matrix.csv
   ├── 1_remaster/
   │   └── {simNb}_{ndemes}_simulation.{trees,traj,nexus,log}
   ├── 1_datastreams/
   │   └── {simNb}_{ndemes}_simulation_{casecounts,seroprevalence,wastewater}.csv
   ├── 1_groundtruth/
   │   └── {simNb}_{ndemes}_simulation_deme_switches_groundtruth.csv
   └── 2_mascot/{base_name}/{variant}/
       └── {seed}_{xmlname}.{log,SimDataset.trees,...}
   ```

2. **Add a second workflow entrypoint** `ANALYSE_FROM_PUBLISHED` that reads from
   the persistent cache directory, constructs channels from the published files
   (using `Channel.fromPath` with glob patterns), and feeds them into the
   downstream processes (`COMBINE_LOGS` onward). This workflow skips all
   simulation and inference — it only does post-processing and figure generation.

   ```groovy
   workflow ANALYSE_FROM_PUBLISHED {
       // Read published MASCOT outputs from persistent storage
       mascot_logs = Channel.fromPath("${params.persistent_cache}/2_mascot/*/*/*.log")
           .map { log -> /* parse base_name, variant, seed from path */ }
           .groupTuple(by: [0, 1])

       // Read simulation metadata for ground-truth comparison
       sim_params = Channel.fromPath("${params.persistent_cache}/1_remaster/*_parameters.csv")
           .map { /* parse simNb, link to trajectories */ }

       // ... feed into COMBINE_LOGS, ANALYSE_POSTERIORS, etc.
   }
   ```

3. **Usage:** Normal runs use the default `workflow { ... }` — everything
   executes end-to-end and publishes to persistent storage. When scratch is
   purged and you want to re-analyse or add new figure scripts:

   ```bash
   nextflow run main.nf -entry ANALYSE_FROM_PUBLISHED \
       --persistent_cache /wynton/home/mueller/pweide/mascot_cache
   ```

**Why this approach over alternatives:** `storeDir` cannot cache `val()` outputs
that nearly every process emits, and selectively caching only `RUN_MASCOT`
outputs breaks provenance because re-executed upstream processes produce
different stochastic simulations that no longer match the cached posteriors.

**Implementation:** All `storeDir` directives removed. `params.cacheDir` removed
from config. RUN_MASCOT now has `publishDir` enabled. An `ANALYSE_FROM_PUBLISHED`
workflow entrypoint was added that reads from the published results directory
(`--outdir`) and feeds into ANALYSE_POSTERIORS and downstream processes.

### 1.7 Commented-out `clip` variants add noise — DONE

The workflow has commented-out `_clip` variants throughout (e.g. lines 481, 734–744).
These are dead code from an earlier design iteration.

**Suggestion:** Remove all commented-out clip variants. If clipping needs to be
re-enabled later, it is a one-line change to add back. Keep instructions how to enable this in the README.md file for future reference.

**Implementation:** All commented-out `_clip` variant lines removed. The `_noclip`
suffix dropped from variant labels (e.g., `datastreams_noclip` → `datastreams`).
`clip_trans_rate` removed from MAKE_MASCOT_XML input; `--clip-trans-rate false`
hardcoded in the process script. The `collectMany` wrapper that existed only to
produce clip/noclip variants was eliminated.

### 1.8 Hardcoded conda path — DONE

`SAMPLE_SIMPARAMS`, `MAKE_SIM_XML`, `SIMULATE_DATASTREAMS`, `MAKE_MASCOT_XML`
all have `conda "/Users/pweidemuller/miniconda3/envs/biopython_env"` hardcoded
in the process definition. This only works on your local machine.

**Suggestion:** Move the conda path to a `params.conda_env` or profile-level
`conda` directive so it works for collaborators.

**Implementation:** All four hardcoded conda paths replaced with `params.conda_env`.
Default is `null` (for container-based profiles like wynton). Local profile sets
`params.conda_env = "/Users/pweidemuller/miniconda3/envs/biopython_env"`.

### 1.9 `ANALYSE_POSTERIORS` input tuple is unwieldy — DONE

The process takes a 15-element tuple (line 289). Adding a new input requires
updating the tuple in the process, the channel join, and the `base_inputs`
mapping.

**Suggestion:** Group related inputs. For example, pass all combined log files
as a single `path("*.combined.log")` glob and resolve them inside the script
by naming convention. Or use Nextflow's `val(meta)` map pattern (common in
nf-core) where a single map carries all metadata. Let's try the nextflow map pattern.

**Implementation:** ANALYSE_POSTERIORS input changed from 15-element tuple to
`val(meta)` map + 12 path elements. The meta map carries `[base, variant, simNb]`.
The channel join/combine operations use plain keys, then convert to meta right
before the process call. Process tag, publishDir, and script all use `meta.base`
and `meta.variant`.

---

## 2. Scripts not yet integrated into the pipeline — DONE

All items in this section have been implemented in `main.nf`, `nextflow.config`,
and `bin/combine_hpd_validation_ne_by_model.py`. Needs cluster testing to verify
end-to-end correctness.

| Script | Purpose | Integration point | Status |
|--------|---------|-------------------|--------|
| `quantify_informationcontent.py` | Compare HPD widths across leave-one-out variants to quantify information gain per datastream | After `CONCATENATE_HPD_VALIDATION`: needs all-variants CSVs as input | **DONE** — `QUANTIFY_INFORMATION_CONTENT` process |
| `hpd_validation_timeseries.py` | Time-series HPD coverage and bias/width plots | Library module (no CLI); used by `make_figure_param_truevsestimate.py` | **No process needed** — library imported by other scripts |
| `make_figure_param_truevsestimate.py` | True vs. estimated scatter plots with HPD whiskers, migration bias/uncertainty figures | After `CONCATENATE_HPD_VALIDATION`: needs params + migration CSVs and optionally trajectory files | **DONE** — `MAKE_FIGURE_TRUE_VS_ESTIMATE` process |
| `make_figure_individualsim.py` | Per-simulation, per-deme publication figures (prevalence, Ne, cumulative incidence) | After `ANALYSE_POSTERIORS`: same inputs as analysis, but one figure set per simulation | **DONE** — `MAKE_INDIVIDUAL_SIM_FIGURES` process |
| `combine_hpd_validation_ne_by_model.py` | Aggregate Ne HPD results tagging MASCOT vs MASCOT-DS | After `CONCATENATE_HPD_VALIDATION`: needs original + datastreams Ne CSVs | **DONE** — `COMBINE_HPD_NE_BY_MODEL` process; CLI refactored |

### Implementation details

- **`hpd_validation_timeseries.py`** has no CLI — it is a library imported by
  `make_figure_param_truevsestimate.py`. No process needed; just ensure it is
  available in the container / conda env.

- **`combine_hpd_validation_ne_by_model.py`** was refactored to support two modes:
  (1) explicit file paths via `--original-ne-csv` and `--datastreams-ne-csv` (used
  by the Nextflow process), and (2) legacy `--analysis-dir` directory scan (for
  standalone use). The old directory-scan mode referenced `_datastreams_noclip`
  paths which no longer exist after improvement 1.7.

- **`MAKE_INDIVIDUAL_SIM_FIGURES`** receives the same input tuple as
  `ANALYSE_POSTERIORS` (same `analysis_inputs` channel). Runs in parallel with
  analysis since it is a separate process.

- **`QUANTIFY_INFORMATION_CONTENT`** collects params, prevalence, and migration
  rate CSVs from all datastream variants (including leave-one-out variants) via
  `.collect()` channels. Runs as a single job after all variants are concatenated.

- **`MAKE_FIGURE_TRUE_VS_ESTIMATE`** needs params/migration/prevalence CSVs for the
  `datastreams` variant, the combined Ne CSV from `COMBINE_HPD_NE_BY_MODEL`, and
  all trajectory files staged in the working directory for outbreak start deme
  lookup (`--trajectory_dir .`).

- All new processes are wired into both the main `workflow` and the
  `ANALYSE_FROM_PUBLISHED` re-analysis entrypoint. Process resource configs
  added to `nextflow.config` (wynton profile).

---

## 3. Per-script code quality improvements — DONE

### 3.1 `create_mascot_xml_fixedtree.py` (1,398→1,178 lines) — DONE

- [x] **`replace_blocks_template()` decomposed** (~310→~60 lines): extracted
  `_replace_xml_element()`, `_inject_traits()`, `_inject_datastream_params()`
  (with `_DATASTREAM_PARAM_MAP` dict), `_configure_rate_shifts()`,
  `_configure_mcmc()`, `_write_xml()`, `_build_output_suffix()`.

- [x] **`main()` decomposed** (~400→~60 lines): extracted `_detect_exclusions()`,
  `_process_tree_data()`, `_build_all_datastreams()`, `_build_alignment_block()`,
  `_extract_newick()`, `_build_trait_block()`, `_build_type_trait_block()`.

- [x] **Datastream builders deduplicated**: extracted generic
  `_build_datastream_by_deme()` — the three type-specific functions now delegate
  to it.

- [x] **`remove_excluded_datastream_elements()` made data-driven** (~240→~80
  lines): `DATASTREAM_ELEMENTS` dict maps each type to its param_patterns,
  distribution_prefixes, prior_prefixes, etc. Single iteration loop.

- [x] **Typo fixed:** `WasterwaterSigmaScalerX` → `WastewaterSigmaScalerX`.

- [x] **Error message bug fixed:** wastewater function incorrectly said
  "Sero prevalence CSV missing" → "Wastewater CSV missing".

### 3.2 `analyse_posteriors.py` (3,633 lines) — DONE

- [x] **`prepare_skyline_plot_data()` decomposed** (~258→~53 lines): extracted
  `_load_input_files()`, `_parse_beast_logs()`, `_compute_hpd_intervals()`,
  `_validate_all_hpd()`. `prepare_skyline_plot_data()` is now a 4-step
  orchestrator. Per-deme NeDynamics and cumulative incidence processing
  deduplicated (copy-paste for deme1/deme2 → loop over `[(file, idx)]` pairs).

- [x] **Hardcoded interpolation methods** documented with inline comments:
  `"linear"` for original (relative rate shifts → piecewise constant skyline),
  `"cubic_normal_spline"` for datastreams (absolute rate shifts benefit from
  spline). These are intentional per-variant choices, not arbitrary defaults.

- [ ] **Multiple plotting functions** (`plot_skyline_ne()`,
  `plot_trajectory_compartments()`, `plot_datastream_params()`,
  `plot_migration_rates()`) are long. They are not urgent to refactor since they
  are leaf functions, but consider extracting repeated subplot-setup patterns.

### 3.3 `create_mascot_xml.py` — DONE (deleted)

Script and its templates (`Mascot_template.xml`, `Mascot_datastreams_template.xml`)
deleted. README updated to reference `create_mascot_xml_fixedtree.py` only.

### 3.4 `simulate_datastreams.py` (926→898 lines) — DONE

- [x] **`_simulate_and_concat()` helper** extracted: handles the common
  groupby → filter-by-first-detection → simulate → concat pattern. Each
  datastream type passes a closure with its specific parameters. Eliminated
  ~80 lines of triplicated loop + filter logic.

- [x] **`constrain_to_first_detection` logic unified**: the filtering is now
  handled entirely inside `_simulate_and_concat()` via the `constrain` and
  `first_detection` arguments — no per-datastream conditional blocks.

### 3.5 `create_birthdeath_simXML.py` (998 lines) — DONE

- [x] **Debug prints** removed (done in quick-fixes pass).

- [x] **`main()` decomposed** (~265→~55 lines): extracted `load_inputs()`,
  `compute_betas()`, `_export_parameters_csv()`, plus module-level helpers
  `_get_scalar()`, `_get_vector()`, `_get_matrix()` (were closures inside
  main). `main()` is now a 5-step orchestrator.

- [x] **TODO at line 723** resolved: the mixing-matrix / contact-matrix
  approach is correct — M encodes relative mixing intensities and the absolute
  transmission level is set by R0 calibration via spectral radius. Added
  clarifying comment replacing the TODO.

### 3.6 `quantify_informationcontent.py` (1,689→1,673 lines) — DONE

- [x] **`identify_variant()` and `identify_migration_variant()` merged**: single
  `identify_variant(filename, all_keyword=...)` function. The `all_keyword`
  parameter differentiates params/migration/prevalence file types.

- [x] **Generic `_load_and_tag_files()` helper** extracted: handles the common
  iterate → identify variant → load CSV → tag → collect pattern. The three
  `load_all_*_files()` functions now delegate to it.

- [x] **`_VARIANT_SUFFIXES` module-level constant** defined (sorted longest-first).
  `extract_simulation_name()` uses it instead of an inline list.

### 3.7 `plot_ess_heatmap.py` (550 lines) — DONE

- [x] **Debug print statements** at lines 301–315 removed.

- [x] **Duplicated filtering logic** extracted into `filter_all_inf_parameters()`
  helper.

### 3.8 `plot_hpd_validation.py` (809 lines) — DONE

- [x] **Debug print** at line 560 removed.

- [ ] **`create_migration_rates_median_plots()`** is 186 lines — long but
  acceptable for a plotting function; no urgent refactor needed.

### 3.9 `calculate_ess.py` (345 lines) — DONE

- [x] **Duplicate `import csv`** removed.

- Otherwise well-structured. No major issues.

### 3.10 `sample_simparameters.py` (336 lines) — no issues

Clean and well-structured. Only minor note: `TARGET_MIGRATION = 0.01` is
hardcoded — could become a CLI argument if you ever want to explore different
migration regimes.

### 3.11 `plot_utils.py` (173 lines) — deferred to Phase 2

Will be merged with the bay_area version during cross-project deduplication.
No changes needed now.

### 3.12 `plot_trees.R` (152 lines) — no issues

Clean. Hardcoded colours are acceptable for a plotting script.

---

## 4. Cross-cutting improvements

### 4.1 Create a shared constants module

Model names (`"MASCOT"`, `"MASCOT-DS"`), variant names, colour palettes, and
file naming patterns are scattered across scripts as string literals. A
`constants.py` module would provide a single source of truth. In the nextflow pipeline use this consistent nameing (original = MASCOT, all the datastream variants are subsettings of MASCOT-DS)

### 4.2 Replace debug prints with logging

Several scripts have leftover `print()` statements used during development.
Switch to Python's `logging` module so debug output can be toggled without code
changes.

### 4.3 Standardise CLI patterns

Some scripts use `parse_args()`, others `parse_arguments()`. Some return the
namespace, others call `main()` directly. Standardise to a consistent pattern
(e.g. always `parse_args()` returning `argparse.Namespace`, always
`if __name__ == "__main__": main()`).

---

## 5. Suggested implementation order

| Priority | Task | Testable locally? |
|----------|------|-------------------|
| 1 | Refactor `create_mascot_xml_fixedtree.py`: decompose long functions, deduplicate datastream builders, add `--no-*` flags | Yes — run with sample inputs |
| 2 | Refactor `analyse_posteriors.py`: break up `prepare_skyline_plot_data()` | Yes — run with sample log files |
| 3 | Extract shared XML utils between `create_mascot_xml.py` and `create_mascot_xml_fixedtree.py` | Yes |
| 4 | Clean up `create_birthdeath_simXML.py`: remove debug prints, decompose `main()` | Yes |
| 5 | Clean up `simulate_datastreams.py`: extract generic simulate-and-concat helper | Yes |
| 6 | Clean up `quantify_informationcontent.py`: deduplicate loaders and variant ID | Yes |
| 7 | Remove debug prints from `plot_ess_heatmap.py`, `plot_hpd_validation.py` | Yes |
| 8 | Add CLI to `combine_hpd_validation_ne_by_model.py` for explicit file paths | Yes |
| 9 | Pipeline: drop `storeDir`, add persistent `publishDir` for full chain, remove dead code (clip variants, commented lines) | Pipeline changes — review locally, test on cluster |
| 10 | Pipeline: extract `parseVariant()`, simplify channels, replace empty-file machinery with `--no-*` flags | Pipeline changes — test on cluster |
| 11 | Pipeline: consolidate `COMBINE_LOGS` / `CONCATENATE_HPD_VALIDATION` / `AGGREGATE_ESS` | Pipeline changes — test on cluster |
| 12 | Pipeline: add `ANALYSE_FROM_PUBLISHED` workflow entrypoint for re-analysis after scratch purge | Pipeline changes — test on cluster |
| 13 | Pipeline: add Nextflow processes for unintegrated scripts | Pipeline changes — test on cluster |

Items 1–8 are Python refactoring testable locally. Items 9–13 are pipeline
changes that can be written and reviewed locally but require a cluster run to
verify. Item 12 (the `ANALYSE_FROM_PUBLISHED` entrypoint) should be implemented
after items 9–11 since it depends on the cleaned-up channel structure.
