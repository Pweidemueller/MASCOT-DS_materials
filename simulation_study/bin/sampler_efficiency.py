#!/usr/bin/env python3
"""
Sampler efficiency & inference-quality metrics for the datastream comparison.

Adapted for the *simulation study*: 20 independent simulated datasets, each run
under every datastream configuration with 3 replicate MCMC seeds. Metrics are
computed per (simulation x config x seed) and then summarised across the 20
simulations per config.

Measures whether adding epidemiological datastreams improves MCMC convergence
and inference efficiency, keeping the two confounded effects separate:

  * statistical  -- adding data changes (contracts) the posterior itself;
  * computational -- the geometry changes AND per-step cost rises (more
    likelihood evaluations), so raw samples-to-ESS unfairly flatters the
    data-rich configs by ignoring throughput.

Analyses (all derived from files that already exist; no BEAST re-run):

  #1  Min-ESS across the focused parameter set (log-posterior + both
      migration-rate directions + prevalence knots). Reports the minimum ESS and
      logs the bottleneck parameter. Threshold 100.

  #2  ESS/hour = (ESS/sample) x (samples/hour) in ABSOLUTE units, decomposing
      mixing quality (ESS/sample) from throughput cost (samples/hour). Computed
      for EVERY parameter of interest on the post-burn-in (stationary) trace.

  #3  Per-parameter compute curves, two views (one figure per parameter, in the
      #5 stationarity style -- x = configs, one dot per sim x seed):
        (a) time to ESS=200: wallclock hours, MCMC samples, and relative 95%
            HPD width (HPD/median) at the point each parameter first reaches
            ESS=200. Runs that never reach ESS=200 instead report the
            wallclock/samples/HPD of the FULL run and are drawn as an X
            instead of a dot;
        (b) state at a fixed GLOBAL wallclock budget (default 1 h): ESS, MCMC
            samples, and relative 95% HPD width reached by the budget.

  #5  Time to stationarity (measured burn-in), feeding #1-#3.

Data layout (dataset: simulation study, `2_mascot`):

    <mascot_dir>/<N>_2_simulation/<config>/<seed>_<N>_2_simulation_<config>.log  (trace)
    <mascot_dir>/<N>_2_simulation/<config>/<seed>_<N>_2_simulation_<config>.out  (screen)

for N = 1..20, seed in {410, 430, 450}, config one of the datastream variants.
Trace and screen logs are co-located (no separate logs/ directory).

Model note: this is a 2-deme model (I0/I1, no ghost/background deme). Parameters
are relabelled by biological ROLE (start vs secondary deme, read per simulation
from sim_metadata) because the seeding deme differs across simulations, so
cross-sim pooling must compare start-with-start. Migration uses the RATE
parameter (`f_migrationRatesSkyline.*`), not the migrationEvents count.

See the auto-generated README.md in the output directory for a full explanation
of every output file and column.

Usage (test on a subset):
    conda run -n biopython_env python bin/sampler_efficiency.py \
        --output-dir sandbox/sampler_efficiency --sims 1 2 3

Usage (final):
    conda run -n biopython_env python bin/sampler_efficiency.py \
        --output-dir results_individuallogs/sampler_efficiency
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the existing single-chain ESS so we match ess_summary.csv methodology.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calculate_ess import effective_size  # noqa: E402
import lab_palette as lp  # noqa: E402

# =============================================================================
# CONFIG -- Tier 1: analysis parameters (biological / methodological)
# =============================================================================

ESS_THRESHOLD = 100.0  # "converged" per parameter (#1 min-ESS, #3b at-budget line)

# #3a target ESS for the "time to ESS" study. Runs that never reach this
# within the run instead report the full-run wallclock/samples/HPD, marked
# with an X (rather than a dot) in the plots.
ESS_TIME_TARGET = 200.0

# #3 fixed GLOBAL compute budget (reconstructed wallclock hours), shared by
# every simulation and config.
BUDGET_HOURS = 1.0

# Burn-in is MEASURED per run (see #5 stationarity), then fed into #1-#4.
# BURNIN_FRACTION is only a fallback if a run-level burn-in cannot be found.
BURNIN_FRACTION = 0.10

# #5 stationarity detection.
STATIONARITY_TAIL_FRACTION = 0.50  # last 50% defines the stationarity region
STATIONARITY_CONSECUTIVE = 10  # consecutive in-region samples to declare it

# Datastream configurations to compare. First entry is the reference (all
# datastreams) that the ablations are compared against. Each tuple is
# (config_suffix, display_label). config_suffix names both the config sub-dir
# and the trailing token of the trace/screen log file names.
REFERENCE_KEY = "datastreams"
CONFIGS: list[tuple[str, str]] = [
    ("datastreams", "All DS"),
    ("datastreams_nocasecounts", "No CC"),
    ("datastreams_noseroprevalence", "No SP"),
    ("datastreams_nowastewater", "No WW"),
    ("datastreams_nomascotll", "No phylogeny"),
    ("datastreams_onlytree", "Phylogeny only"),
]

SEEDS = [410, 430, 450]  # replicate MCMC seeds per (sim, config)
SIMS = list(range(1, 21))  # 20 simulated datasets, tagged "<N>_2_simulation"

# Focused scientific parameters. The 2-deme simulation model names its rate and
# knot columns by physical deme (I0/I1, Deme1/Deme2), but WHICH physical deme
# seeded the epidemic ("start") vs was invaded ("secondary") differs per
# simulation (read from sim_metadata). So we relabel every migration/prevalence
# parameter by biological ROLE, so cross-sim pooling compares like with like.
#
# Migration uses the RATE parameter f_migrationRatesSkyline.I<a>_to_I<b> (not
# the migrationEvents count). Prevalence knots are SkylinePrev.Deme<k>.<knot>,
# where Deme<k> maps to physical deme index k-1.
SKYLINEPREV_RE = re.compile(r"^SkylinePrev\.(Deme\d+)\.(\d+)$")
MIGRATION_RATE_FMT = "f_migrationRatesSkyline.I{a}_to_I{b}"

# Canonical (role-based) migration parameter names.
MIG_START_TO_SEC = "mig_rate_start_to_secondary"
MIG_SEC_TO_START = "mig_rate_secondary_to_start"
# Aggregate (summed per posterior sample): total off-diagonal migration rate.
TOTAL_MIG = "mig_rate_total"
# The log-posterior column, treated as a full parameter of interest (its own
# ESS / stationarity / #3 curves). It is a log-density (negative), so like the
# log-prevalence knots its HPD headline is the ABSOLUTE width.
POSTERIOR = "posterior"

# Plot colours per config -- lab_palette (KYBURG_GOLD, UCSF_TEAL, BASEL, BRIDGE,
# HUTCH, RAIN), matching the value_of_information figures.
CONFIG_COLOR = {
    "All DS": lp.KYBURG_GOLD,
    "No CC": lp.UCSF_TEAL,
    "No SP": lp.BASEL,
    "No WW": lp.BRIDGE,
    "No phylogeny": lp.HUTCH,
    "Phylogeny only": lp.RAIN,
}

# =============================================================================
# CONFIG -- Tier 2: machine-specific paths
# =============================================================================

REPO = Path(__file__).resolve().parent.parent
# The mascot per-simulation outputs live under a (doubly-nested) results dir.
MASCOT_DIR = REPO / "results_individuallogs" / "results_individuallogs" / "2_mascot"


# =============================================================================
# Data model
# =============================================================================


@dataclass
class SeedData:
    sim: int  # simulation index, 1..20
    config: str  # config suffix, e.g. "datastreams_nocasecounts"
    label: str  # display label, e.g. "No CC"
    seed: int  # replicate MCMC seed (410/430/450)
    trace: pd.DataFrame  # full trace log (all samples, pre-burnin)
    rate_h_per_msample: float  # representative throughput (last reported rate)
    last_sample: int  # highest MCMC state reached so far
    start_deme: int  # physical deme index (0/1) that seeded this simulation

    @property
    def key(self) -> tuple[int, str, int]:
        return (self.sim, self.config, self.seed)

    @property
    def samples_per_hour(self) -> float:
        return 1e6 / self.rate_h_per_msample

    def elapsed_hours(self, sample: int) -> float:
        """Wallclock hours to reach `sample` under constant throughput."""
        return self.rate_h_per_msample * sample / 1e6


# =============================================================================
# Parsing
# =============================================================================

# Rate is "NhNmNs/Msamples"; the h and m parts are omitted on fast runs
# (e.g. "34m22s/Msamples", "45s/Msamples"), so both are optional.
_RATE_RE = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(\d+)s/Msamples")
_ROW_RE = re.compile(r"^\s*(\d+)\s")


def parse_screen_log(path: Path) -> tuple[list[int], list[float]]:
    """Parse a BEAST .out screen log into (samples, rate_hours_per_Msample).

    The trailing column is a *rate* (time per million samples), a whole-run
    running average, not cumulative elapsed. Rows before it stabilises print
    "--" and are skipped.
    """
    samples: list[int] = []
    rates: list[float] = []
    with open(path) as fh:
        for line in fh:
            rm = _RATE_RE.search(line)
            sm = _ROW_RE.match(line)
            if not (rm and sm):
                continue
            h, m, s = (int(g) if g else 0 for g in rm.groups())
            samples.append(int(sm.group(1)))
            rates.append(h + m / 60.0 + s / 3600.0)
    return samples, rates


def _run_stub(sim: int, config: str, seed: int) -> str:
    """File-name stub shared by trace/screen logs of one (sim, config, seed)."""
    return f"{seed}_{sim}_2_simulation_{config}"


_START_DEME_CACHE: dict[int, int] = {}


def start_deme_for_sim(sim: int) -> int | None:
    """Physical deme index (0/1) that seeded simulation `sim`, from the remaster
    metadata sibling of the mascot dir:
    `1_remaster_sim/<N>_2_simulation_sim_metadata.csv` (deme_type == 'start')."""
    if sim in _START_DEME_CACHE:
        return _START_DEME_CACHE[sim]
    meta = MASCOT_DIR.parent / "1_remaster_sim" / f"{sim}_2_simulation_sim_metadata.csv"
    if not meta.exists():
        print(f"  [warn] missing sim metadata: {meta.name}")
        return None
    md = pd.read_csv(meta)
    row = md[md["deme_type"] == "start"]
    if row.empty:
        print(f"  [warn] no start deme in {meta.name}")
        return None
    val = int(row["deme"].iloc[0])
    _START_DEME_CACHE[sim] = val
    return val


def load_seed_data(sim: int, config: str, label: str, seed: int) -> SeedData | None:
    """Load one sim x config x seed: trace log + throughput from screen log."""
    run_dir = MASCOT_DIR / f"{sim}_2_simulation" / config
    stub = _run_stub(sim, config, seed)
    trace_path = run_dir / f"{stub}.log"
    screen_path = run_dir / f"{stub}.out"

    if not trace_path.exists():
        print(f"  [warn] missing trace log: {trace_path.name}")
        return None
    if not screen_path.exists():
        print(f"  [warn] missing screen log: {screen_path.name}")
        return None

    sdeme = start_deme_for_sim(sim)
    if sdeme is None:
        return None

    trace = pd.read_csv(trace_path, sep="\t", comment="#")
    _, rates = parse_screen_log(screen_path)
    if not rates:
        print(f"  [warn] no throughput rows in {screen_path.name}")
        return None

    return SeedData(
        sim=sim,
        config=config,
        label=label,
        seed=seed,
        trace=trace,
        rate_h_per_msample=rates[-1],  # last = whole-run average, most stable
        last_sample=int(trace["Sample"].iloc[-1]),
        start_deme=sdeme,
    )


# =============================================================================
# Role-based parameter set (start/secondary deme) & parameters of interest
# =============================================================================


def param_members(sd: SeedData) -> dict[str, list[str]]:
    """Map each canonical (role-based) parameter name to the trace column(s)
    that compose it, for `sd`'s start deme.

    * migration RATE directions -> one column each (start->secondary,
      secondary->start), using this sim's start deme;
    * `mig_rate_total` -> both directions summed;
    * prevalence knots -> SkylinePrev.<role>.<knot>, one column each, where the
      physical Deme<k> (deme index k-1) is mapped to start / secondary.
    """
    s, o = sd.start_deme, 1 - sd.start_deme
    start_to_sec = MIGRATION_RATE_FMT.format(a=s, b=o)
    sec_to_start = MIGRATION_RATE_FMT.format(a=o, b=s)
    members: dict[str, list[str]] = {
        POSTERIOR: [POSTERIOR],
        MIG_START_TO_SEC: [start_to_sec],
        MIG_SEC_TO_START: [sec_to_start],
        TOTAL_MIG: [start_to_sec, sec_to_start],
    }
    for c in sd.trace.columns:
        m = SKYLINEPREV_RE.match(c)
        if not m:
            continue
        deme_idx = int(m.group(1)[4:]) - 1  # Deme1 -> physical deme 0
        role = "start" if deme_idx == s else "secondary"
        members[f"SkylinePrev.{role}.{m.group(2)}"] = [c]
    return members


def prev_knots(sd: SeedData) -> list[str]:
    """Role-based prevalence knot names, ordered start-deme knots then
    secondary-deme knots, each ascending by knot index."""
    names = [n for n in param_members(sd) if n.startswith("SkylinePrev.")]

    def key(n: str) -> tuple[int, int]:
        _, role, knot = n.split(".")
        return (0 if role == "start" else 1, int(knot))

    return sorted(names, key=key)


def migration_dirs(sd: SeedData) -> list[str]:
    """Role-based migration rate directions (start->secondary, then reverse)."""
    return [MIG_START_TO_SEC, MIG_SEC_TO_START]


def decomposition_params(sd: SeedData) -> list[str]:
    """Individual focused parameters (#1 bottleneck, #2 decomposition, #3 ESS
    curves): the log-posterior + both migration rate directions + every
    prevalence knot."""
    return [POSTERIOR] + migration_dirs(sd) + prev_knots(sd)


def param_kind(name: str) -> str:
    """HPD-headline kind: 'prevalence'/'posterior' are log-space (absolute width
    headline); 'migration' rates are positive (relative HPD/median headline)."""
    if name == POSTERIOR:
        return "posterior"
    return "prevalence" if name.startswith("SkylinePrev.") else "migration"


def params_of_interest(sd: SeedData) -> list[str]:
    """Ordered parameters of interest for width (#3)/stationarity (#5): the
    log-posterior, the total-migration-rate aggregate, then each migration
    direction, then the prevalence knots."""
    return [POSTERIOR, TOTAL_MIG, *migration_dirs(sd), *prev_knots(sd)]


def pretty_param(name: str) -> str:
    """Human-readable title for a canonical parameter name."""
    labels = {
        POSTERIOR: "posterior (log-density)",
        TOTAL_MIG: "migration rate — total (both directions)",
        MIG_START_TO_SEC: "migration rate — start → secondary",
        MIG_SEC_TO_START: "migration rate — secondary → start",
    }
    if name in labels:
        return labels[name]
    if name.startswith("SkylinePrev."):
        _, role, knot = name.split(".")
        return f"prevalence — {role} deme, knot {knot}"
    return name


def extract_param(
    trace: pd.DataFrame, name: str, members: dict[str, list[str]]
) -> np.ndarray:
    """Per-sample values for a canonical parameter (sums its member columns)."""
    return trace[members[name]].to_numpy(dtype=float).sum(axis=1)


def hpd_widths(samples: np.ndarray, kind: str) -> dict[str, float]:
    """95% HPD (equal-tailed) width, absolute and relative-to-median.

    For migration counts the *relative* width (width / median) is the headline
    metric (matches value_of_information_main); for log-prevalence the
    *absolute* width is already a relative measure in linear space, so that is
    the headline. Returns both plus which one is the headline for this kind.
    """
    lo, hi = np.percentile(samples, [2.5, 97.5])
    med = float(np.median(samples))
    absolute = float(hi - lo)
    relative = absolute / med if med > 0 else np.nan
    headline = absolute if kind == "prevalence" else relative
    return {
        "hpd_width_absolute": absolute,
        "hpd_width_relative": relative,
        "median": med,
        "width_headline": headline,
    }


# =============================================================================
# Across-simulation summary helper
# =============================================================================


def summarize_across_sims(
    df: pd.DataFrame, group_cols: list[str], value_cols: list[str]
) -> pd.DataFrame:
    """Median / IQR / mean / n of each value column, grouped by group_cols.

    The unit summarised over is whatever remains after grouping (typically the
    20 simulations x replicate seeds). NaNs are ignored per column.
    """
    rows = []
    for keys, grp in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        for col in value_cols:
            v = grp[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            row[f"{col}_median"] = float(np.median(v)) if len(v) else np.nan
            row[f"{col}_q25"] = float(np.percentile(v, 25)) if len(v) else np.nan
            row[f"{col}_q75"] = float(np.percentile(v, 75)) if len(v) else np.nan
            row[f"{col}_mean"] = float(np.mean(v)) if len(v) else np.nan
            row[f"{col}_n"] = int(len(v))
        rows.append(row)
    return pd.DataFrame(rows)


def _config_order(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a per-config frame into the canonical CONFIGS order."""
    order = {cfg: i for i, (cfg, _) in enumerate(CONFIGS)}
    return df.assign(_o=df["config"].map(order)).sort_values("_o").drop(columns="_o")


# =============================================================================
# Shared plotting helper (the "stationarity" style: x = configs, one jittered
# dot per (sim, seed), a short horizontal centre bar per config)
# =============================================================================


def _strip_by_config(
    ax,
    df: pd.DataFrame,
    value_col: str,
    agg: str = "median",
    hline: float | None = None,
    seed: int = 0,
    log: bool = False,
    marker_col: str | None = None,
) -> None:
    """Scatter one dot per row (jittered within its config column) + a short
    centre bar (median or mean). df must have a 'config' column.

    If `marker_col` names a boolean column, rows where it is True are drawn as
    an X instead of a dot (e.g. a run that never reached a target ESS, whose
    value here is a full-run fallback rather than the metric at the target).
    """
    rng = np.random.RandomState(seed)
    for xi, (config, label) in enumerate(CONFIGS):
        sub = df[df["config"] == config]
        v = sub[value_col].to_numpy(dtype=float)
        finite = np.isfinite(v)
        if log:
            finite &= v > 0
        if not finite.any():
            continue
        vals = v[finite]
        flagged = (
            sub[marker_col].to_numpy(dtype=bool)[finite]
            if marker_col is not None
            else np.zeros(len(vals), dtype=bool)
        )
        color = CONFIG_COLOR.get(label, "#555")
        jitter = (rng.rand(len(vals)) - 0.5) * 0.28
        normal = ~flagged
        if normal.any():
            ax.scatter(
                xi + jitter[normal],
                vals[normal],
                color=color,
                s=18,
                alpha=0.55,
                zorder=2,
                edgecolor="white",
                linewidth=0.3,
                marker="o",
            )
        if flagged.any():
            ax.scatter(
                xi + jitter[flagged],
                vals[flagged],
                color=color,
                s=32,
                alpha=0.85,
                zorder=3,
                marker="x",
                linewidth=1.4,
            )
        centre = np.median(vals) if agg == "median" else np.mean(vals)
        # median bar drawn ON TOP of the dots: thin, solid black, no halo.
        ax.hlines(
            centre,
            xi - 0.18,
            xi + 0.18,
            color="black",
            lw=1.75,
            zorder=5,
        )
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=0.8)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([lbl for _, lbl in CONFIGS], rotation=25, ha="right", fontsize=8)
    ax.margins(x=0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_per_param_panels(
    df: pd.DataFrame,
    metrics: list[tuple[str, str, float | None]],
    out_dir: Path,
    title_prefix: str,
    fname_prefix: str,
    param_order: list[str] | None = None,
    agg: str = "median",
    marker_col: str | None = None,
) -> None:
    """One figure per parameter, `len(metrics)` subplots side by side, in the
    stationarity style. `metrics` is a list of (value_col, ylabel, hline).

    `marker_col`, if given, names a boolean column in `df`: rows where it is
    True are drawn as an X (e.g. a run that never reached the target ESS and
    is instead showing its full-run value).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names = param_order or list(dict.fromkeys(df["parameter"]))
    for name in names:
        sub = df[df["parameter"] == name]
        if sub.empty:
            continue
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4), squeeze=False)
        for ax, (col, ylabel, hline) in zip(axes.flat, metrics):
            _strip_by_config(ax, sub, col, agg=agg, hline=hline, marker_col=marker_col)
            # ylabel may be a str or a callable(parameter_name) -> str
            ax.set_ylabel(ylabel(name) if callable(ylabel) else ylabel, fontsize=9)
        fig.suptitle(
            f"{title_prefix} — {pretty_param(name)}  (dots: sim×seed)", fontsize=11
        )
        rect = (0, 0, 1, 0.95)
        if marker_col is not None:
            fig.text(
                0.5,
                0.01,
                "✕ = ESS target not reached within the run (full-run value shown)",
                ha="center",
                fontsize=7.5,
                style="italic",
            )
            rect = (0, 0.03, 1, 0.95)
        fig.tight_layout(rect=rect)
        safe = name.replace(".", "_")
        fig.savefig(out_dir / f"{fname_prefix}_{safe}.png", dpi=140)
        fig.savefig(out_dir / f"{fname_prefix}_{safe}.pdf")
        plt.close(fig)


# =============================================================================
# #5 Time to stationarity (measured burn-in)
# =============================================================================


def first_k_consecutive(mask: np.ndarray, k: int) -> int | None:
    """Index of the k-th (last) element of the first run of k consecutive True.

    Returns None if no such run exists.
    """
    if len(mask) < k:
        return None
    rolling = np.convolve(mask.astype(int), np.ones(k, dtype=int), "valid")
    hits = np.where(rolling == k)[0]
    if len(hits) == 0:
        return None
    return int(hits[0] + k - 1)


def analysis_stationarity(seeds: list[SeedData]) -> pd.DataFrame:
    """For each sim x config x seed x parameter of interest:

    * define the stationarity region as the 95% HPD (median, q2.5, q97.5) of the
      LAST `STATIONARITY_TAIL_FRACTION` of the trace;
    * scan from sample 0 for the first `STATIONARITY_CONSECUTIVE` consecutive
      samples that all fall inside that region;
    * report the MCMC sample of the last (k-th) sample of that group and the
      reconstructed wallclock at which it was reached.
    """
    rows = []
    for sd in seeds:
        members = param_members(sd)
        poi = params_of_interest(sd)
        sample_col = sd.trace["Sample"].to_numpy()
        n = len(sd.trace)
        tail_start = int(np.floor(n * (1.0 - STATIONARITY_TAIL_FRACTION)))
        for name in poi:
            series = extract_param(sd.trace, name, members)
            tail = series[tail_start:]
            lo, hi = np.percentile(tail, [2.5, 97.5])
            med = float(np.median(tail))
            in_region = (series >= lo) & (series <= hi)
            idx = first_k_consecutive(in_region, STATIONARITY_CONSECUTIVE)
            if idx is None:
                # extremely unlikely (region is the tail's own 95% HPD);
                # fall back to the start of the tail window.
                idx = tail_start + STATIONARITY_CONSECUTIVE - 1
                idx = min(idx, n - 1)
            samp = int(sample_col[idx])
            rows.append(
                {
                    "sim": sd.sim,
                    "label": sd.label,
                    "config": sd.config,
                    "seed": sd.seed,
                    "parameter": name,
                    "region_lower": float(lo),
                    "region_median": med,
                    "region_upper": float(hi),
                    "stationarity_sample": samp,
                    "stationarity_hours": sd.elapsed_hours(samp),
                }
            )
    return pd.DataFrame(rows)


def derive_burnin(
    seeds: list[SeedData], stationarity: pd.DataFrame
) -> tuple[dict[tuple[int, str, int], int], pd.DataFrame]:
    """Run-level burn-in per (sim, config, seed) = the MAX stationarity sample
    over the parameters of interest (the chain is not stationary until its
    slowest focused parameter is). Returns (burnin_rows, burnin_summary_df)
    where burnin_rows maps (sim, config, seed) -> number of leading trace rows
    to discard.
    """
    burnin_rows: dict[tuple[int, str, int], int] = {}
    summary_rows = []
    for sd in seeds:
        sub = stationarity[
            (stationarity["sim"] == sd.sim)
            & (stationarity["config"] == sd.config)
            & (stationarity["seed"] == sd.seed)
        ]
        j = int(sub["stationarity_sample"].idxmax())
        burnin_sample = int(sub.loc[j, "stationarity_sample"])
        driver = str(sub.loc[j, "parameter"])
        sample_col = sd.trace["Sample"].to_numpy()
        drop = int(np.searchsorted(sample_col, burnin_sample, side="left"))
        # keep a floor of usable post-burn-in samples
        drop = min(drop, len(sd.trace) - STATIONARITY_CONSECUTIVE)
        burnin_rows[sd.key] = drop
        summary_rows.append(
            {
                "sim": sd.sim,
                "label": sd.label,
                "config": sd.config,
                "seed": sd.seed,
                "burnin_sample": burnin_sample,
                "burnin_hours": sd.elapsed_hours(burnin_sample),
                "burnin_rows_discarded": drop,
                "n_rows_total": len(sd.trace),
                "n_rows_postburnin": len(sd.trace) - drop,
                "driving_parameter": driver,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["sim", "config", "seed"])
    return burnin_rows, summary


def plot_stationarity_per_param(stationarity: pd.DataFrame, out_dir: Path) -> None:
    """One figure per parameter of interest: left = samples to stationarity,
    right = hours to stationarity. x = versions; one dot per (sim, seed)
    coloured by version (lab_palette); a short horizontal median bar per version
    (drawn on top of the dots)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in stationarity["parameter"].unique():
        sub = stationarity[stationarity["parameter"] == name]
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, metric, ylabel in [
            (axL, "stationarity_sample", "MCMC samples to stationarity"),
            (axR, "stationarity_hours", "Wallclock hours to stationarity"),
        ]:
            _strip_by_config(ax, sub, metric, agg="median", log=True)
            ax.set_ylabel(f"{ylabel}", fontsize=9)
        fig.suptitle(
            f"Time to stationarity — {pretty_param(name)}  (dots: sim×seed)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        safe = name.replace(".", "_")
        fig.savefig(out_dir / f"stationarity_{safe}.png", dpi=140)
        fig.savefig(out_dir / f"stationarity_{safe}.pdf")
        plt.close(fig)


# =============================================================================
# #1 Min-ESS + bottleneck
# =============================================================================


def analysis_min_ess(
    seeds: list[SeedData], burnin_rows: dict[tuple[int, str, int], int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per sim x config x seed: ESS of every individual parameter of interest.

    ESS is single-chain (calculate_ess.effective_size, Tracer/R-style) on the
    trace after discarding the MEASURED burn-in (#5). The min-ESS bottleneck
    summary is computed over the focused set (log-posterior + both migration
    rate directions + all prevalence knots).
    """
    per_param_rows = []
    summary_rows = []
    for sd in seeds:
        members = param_members(sd)
        dparams = decomposition_params(sd)
        drop = burnin_rows[sd.key]
        post = sd.trace.iloc[drop:].reset_index(drop=True)
        n_post = len(post)
        ess_focal: dict[str, float] = {}
        for p in dparams:
            ess = effective_size(extract_param(post, p, members))
            per_param_rows.append(
                {
                    "sim": sd.sim,
                    "label": sd.label,
                    "config": sd.config,
                    "seed": sd.seed,
                    "parameter": p,
                    "ESS": ess,
                    "n_samples_postburnin": n_post,
                }
            )
            ess_focal[p] = ess
        bottleneck = min(ess_focal, key=ess_focal.get)
        min_ess = ess_focal[bottleneck]
        n_ge = sum(1 for v in ess_focal.values() if v >= ESS_THRESHOLD)
        summary_rows.append(
            {
                "sim": sd.sim,
                "label": sd.label,
                "config": sd.config,
                "seed": sd.seed,
                "n_samples_postburnin": n_post,
                "last_sample": sd.last_sample,
                "min_ESS": min_ess,
                "bottleneck_parameter": bottleneck,
                "n_focused": len(ess_focal),
                "n_focused_ESS_ge_100": n_ge,
                "all_focused_converged": n_ge == len(ess_focal),
            }
        )
    return (
        pd.DataFrame(per_param_rows),
        pd.DataFrame(summary_rows).sort_values(["sim", "config", "seed"]),
    )


def plot_min_ess(min_ess_summary: pd.DataFrame, out_base: Path) -> None:
    """Min focused-ESS distribution per config across (sim, seed)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _strip_by_config(ax, min_ess_summary, "min_ESS", agg="median", seed=1)
    ax.axhline(ESS_THRESHOLD, color="k", ls="--", lw=1, label=f"ESS={ESS_THRESHOLD:g}")
    ax.set_ylabel("min focused-parameter ESS")
    ax.set_title("#1 Min focused-ESS per config (dots: sim×seed; bar: median)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


# =============================================================================
# #2 ESS/hour decomposition -- for EVERY parameter of interest
# =============================================================================


def analysis_decomposition(
    seeds: list[SeedData], ess_per_param: pd.DataFrame
) -> pd.DataFrame:
    """ESS/hour = (ESS/sample) x (samples/hour) for every focused parameter, in
    ABSOLUTE units (not relative to a reference config). ESS is on the
    post-burn-in (stationary) trace, so ESS_per_sample and ESS_per_hour are too.
    """
    spm = {sd.key: sd.samples_per_hour for sd in seeds}
    rate = {sd.key: sd.rate_h_per_msample for sd in seeds}
    df = ess_per_param.copy()
    keys = list(zip(df["sim"], df["config"], df["seed"]))
    df["samples_per_hour"] = [spm[k] for k in keys]
    df["rate_h_per_Msample"] = [rate[k] for k in keys]
    df["ESS_per_sample"] = df["ESS"] / df["n_samples_postburnin"]
    df["ESS_per_hour"] = df["ESS_per_sample"] * df["samples_per_hour"]
    return df.sort_values(["sim", "config", "seed", "parameter"])


def plot_decomposition(decomp: pd.DataFrame, out_base: Path) -> None:
    """Absolute ESS/hour = ESS/sample x samples/hour, per config, pooled over
    (sim, seed, parameter). All configs shown; dots are one per row."""
    metrics = [
        ("ESS_per_hour", "ESS per hour (post-burn-in)"),
        ("ESS_per_sample", "ESS per sample (mixing quality)"),
        ("samples_per_hour", "samples per hour (throughput)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        _strip_by_config(ax, decomp, metric, agg="median", seed=2)
        ax.set_ylabel(ylabel, fontsize=9)
    fig.suptitle(
        "#2 ESS/hour decomposition — absolute (dots: sim×seed×param; bar: median)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


# =============================================================================
# #3 Per-parameter compute curves: time-to-ESS=200 and state at a fixed budget
# =============================================================================


def frontier_curves(
    seeds: list[SeedData], burnin_rows: dict[tuple[int, str, int], int]
) -> pd.DataFrame:
    """Per (sim, config, seed, parameter of interest), along EVERY logged row
    (prefix) after burn-in: single-chain ESS + 95% HPD width (relative &
    absolute) vs reconstructed wallclock.

    The measured burn-in (#5) is discarded up front; at each checkpoint the
    chain is truncated to the prefix and metrics are computed on
    (post-burn-in .. prefix end), so every ESS is on the stationary part.
    Checkpoints with too few post-burn-in samples are skipped. Scanning every
    row (rather than a coarse subsample of prefixes) gives time-to-ESS results
    at the full resolution of the trace's logging interval (`logEvery`),
    instead of quantizing to a handful of widely spaced checkpoints.
    wallclock = rate_h_per_Msample * sample / 1e6.
    """
    rows = []
    for sd in seeds:
        members = param_members(sd)
        poi = params_of_interest(sd)
        sample_col = sd.trace["Sample"].to_numpy()
        n = len(sd.trace)
        drop = burnin_rows[sd.key]
        endpoints = np.arange(drop + STATIONARITY_CONSECUTIVE, n + 1)
        poi_arrays = {name: extract_param(sd.trace, name, members) for name in poi}

        for end in endpoints:
            samp = int(sample_col[end - 1])
            eh = sd.elapsed_hours(samp)
            for name in poi:
                kind = param_kind(name)
                series = poi_arrays[name][drop:end]
                w = hpd_widths(series, kind)
                rows.append(
                    {
                        "sim": sd.sim,
                        "label": sd.label,
                        "config": sd.config,
                        "seed": sd.seed,
                        "parameter": name,
                        "kind": kind,
                        "sample": samp,
                        "elapsed_hours": eh,
                        "ESS": effective_size(series),
                        "hpd_width_relative": w["hpd_width_relative"],
                        "hpd_width_absolute": w["hpd_width_absolute"],
                        "median": w["median"],
                    }
                )
    return pd.DataFrame(rows)


def _headline_width_col(kind: str) -> str:
    """Which HPD column is the headline for a parameter kind: HPD/median
    (relative) for migration rates, absolute for log-space quantities
    (prevalence knots, log-posterior) whose absolute width is already a ratio
    measure in linear space."""
    return (
        "hpd_width_absolute"
        if kind in ("prevalence", "posterior")
        else "hpd_width_relative"
    )


def time_to_ess200(frontier: pd.DataFrame) -> pd.DataFrame:
    """Per (sim, config, seed, parameter): the FIRST checkpoint whose ESS
    reaches ESS_TIME_TARGET, and the wallclock, MCMC sample and headline 95%
    HPD width attained there (HPD/median for migration; absolute for
    prevalence). If a parameter never reaches the target within the run,
    `reached_ess200=False` and the wallclock/samples/HPD instead report the
    FULL run's last checkpoint (`missed_ess200=True` flags these for plotting
    as an X rather than a dot).
    """
    rows = []
    keys = ["sim", "config", "seed", "parameter"]
    for (sim, config, seed, parameter), grp in frontier.groupby(keys):
        grp = grp.sort_values("elapsed_hours")
        kind = grp["kind"].iloc[0]
        wcol = _headline_width_col(kind)
        hit = grp[grp["ESS"] >= ESS_TIME_TARGET]
        reached = len(hit) > 0
        r = hit.iloc[0] if reached else grp.iloc[-1]  # fallback: full-run state
        rows.append(
            {
                "sim": sim,
                "label": grp["label"].iloc[0],
                "config": config,
                "seed": seed,
                "parameter": parameter,
                "kind": kind,
                "reached_ess200": reached,
                "missed_ess200": not reached,
                "hours_to_ess200": float(r["elapsed_hours"]),
                "samples_to_ess200": int(r["sample"]),
                "hpd_at_ess200": float(r[wcol]),
            }
        )
    return pd.DataFrame(rows).sort_values(keys)


def at_budget(frontier: pd.DataFrame, budget_h: float) -> pd.DataFrame:
    """Per (sim, config, seed, parameter): the state at the fixed GLOBAL
    wallclock budget -- the last checkpoint with elapsed_hours <= budget_h
    (i.e. the run's whole post-burn-in chain if it finished before the budget).
    Reports ESS, MCMC sample reached, and headline 95% HPD width there
    (HPD/median for migration; absolute for prevalence)."""
    rows = []
    keys = ["sim", "config", "seed", "parameter"]
    for (sim, config, seed, parameter), grp in frontier.groupby(keys):
        kind = grp["kind"].iloc[0]
        wcol = _headline_width_col(kind)
        within = grp[grp["elapsed_hours"] <= budget_h].sort_values("elapsed_hours")
        r = within.iloc[-1] if len(within) else None
        rows.append(
            {
                "sim": sim,
                "label": grp["label"].iloc[0],
                "config": config,
                "seed": seed,
                "parameter": parameter,
                "kind": kind,
                "budget_hours": budget_h,
                "ESS_at_budget": float(r["ESS"]) if r is not None else np.nan,
                "samples_at_budget": int(r["sample"]) if r is not None else np.nan,
                "hpd_at_budget": float(r[wcol]) if r is not None else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(keys)


# =============================================================================
# Main composite figure (posterior parameter): 2x2 gridspec
# =============================================================================


def _add_panel_labels(
    fig,
    panels: list[tuple[object, str]],
    *,
    x_pad: float = 0.006,
    y_pad: float = 0.005,
    fontsize: float = 13,
) -> None:
    """Place bold panel letters at the top-left corner of each axis.

    Each letter sits just left of the y-axis region (tick labels + y-axis
    title, so the y-axis title reads to the right of the letter) and above any
    axis title. Mirrors the panel-label recipe in
    ``make_composite_figures_voi.py`` / ``make_figure_simstudy.py``.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, lab in panels:
        y_bbox = ax.yaxis.get_tightbbox(renderer)
        x_px = y_bbox.x0 if y_bbox is not None else ax.get_window_extent().x0
        x_fig = inv.transform((x_px, 0))[0] - x_pad
        y_fig = ax.get_position().y1
        if ax.title.get_text():
            title_top = inv.transform((0, ax.title.get_window_extent(renderer).y1))[1]
            y_fig = max(y_fig, title_top)
        fig.text(
            x_fig,
            y_fig + y_pad,
            lab,
            fontsize=fontsize,
            fontweight="bold",
            va="bottom",
            ha="left",
        )


def _n_per_config_labels(df: pd.DataFrame) -> list[str]:
    """Config x-tick labels with a second line giving the row count per config
    (e.g. runs that reached the ESS target, after dropping the rest)."""
    labels = []
    for config, label in CONFIGS:
        n = int((df["config"] == config).sum())
        labels.append(f"{label}\n(n={n})")
    return labels


def plot_main_figure_posterior(
    stationarity: pd.DataFrame, ess200: pd.DataFrame, out_base: Path
) -> None:
    """2x2 main figure, posterior parameter only, no figure title:

    A (0,0) MCMC samples to stationarity   B (0,1) wallclock minutes to stationarity
    C (1,0) wallclock minutes to ESS={ESS_TIME_TARGET}   D (1,1) 95% HPDI at ESS={ESS_TIME_TARGET}

    Panels A/B reuse the #5 stationarity data (hours converted to minutes for
    readability). Panels C/D drop runs that never reached the ESS target
    (rather than marking them with an X); the number of runs actually plotted
    per config is shown as a second line in the x-tick labels.
    """
    sub_stat = stationarity[stationarity["parameter"] == POSTERIOR]
    sub_ess200 = ess200[ess200["parameter"] == POSTERIOR]
    sub_ess200_reached = sub_ess200[~sub_ess200["missed_ess200"]]

    fig = plt.figure(figsize=(9.0, 7.0))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _strip_by_config(ax_a, sub_stat, "stationarity_sample", agg="median", log=True)
    ax_a.set_ylabel("MCMC samples to stationarity", fontsize=9)

    sub_stat = sub_stat.assign(stationarity_minutes=sub_stat["stationarity_hours"] * 60)
    _strip_by_config(ax_b, sub_stat, "stationarity_minutes", agg="median", log=True)
    ax_b.set_ylabel("Wallclock minutes to stationarity", fontsize=9)

    sub_ess200_reached = sub_ess200_reached.assign(
        minutes_to_ess200=sub_ess200_reached["hours_to_ess200"] * 60
    )
    _strip_by_config(ax_c, sub_ess200_reached, "minutes_to_ess200", agg="median")
    ax_c.set_ylabel(f"Wallclock minutes to ESS={ESS_TIME_TARGET:g}", fontsize=9)
    ax_c.set_xticklabels(
        _n_per_config_labels(sub_ess200_reached), rotation=25, ha="right", fontsize=8
    )

    _strip_by_config(ax_d, sub_ess200_reached, "hpd_at_ess200", agg="median")
    ax_d.set_ylabel(f"95% HPDI width at ESS={ESS_TIME_TARGET:g}", fontsize=9)
    ax_d.set_xticklabels(
        _n_per_config_labels(sub_ess200_reached), rotation=25, ha="right", fontsize=8
    )

    _add_panel_labels(fig, [(ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D")])
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=150)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


# =============================================================================
# README
# =============================================================================


def write_readme(out: Path, budget_h: float, n_sims: int) -> None:
    (out / "README.md").write_text(
        README_TEMPLATE.format(
            threshold=ESS_THRESHOLD,
            ess_time_target=f"{ESS_TIME_TARGET:g}",
            reference=REFERENCE_KEY,
            tail_frac=STATIONARITY_TAIL_FRACTION,
            consecutive=STATIONARITY_CONSECUTIVE,
            n_sims=n_sims,
            seeds=", ".join(str(s) for s in SEEDS),
            budget=f"{budget_h:g}",
        )
    )


# =============================================================================
# Driver
# =============================================================================


def main() -> None:
    global BURNIN_FRACTION, MASCOT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "sandbox" / "sampler_efficiency",
        help="Where CSVs, plots and README.md are written (default: sandbox/).",
    )
    ap.add_argument(
        "--mascot-dir",
        type=Path,
        default=MASCOT_DIR,
        help="Directory holding the <N>_2_simulation/<config>/ run folders.",
    )
    ap.add_argument(
        "--sims",
        type=int,
        nargs="+",
        default=SIMS,
        help="Simulation indices to include (default: 1..20). Subset for testing.",
    )
    ap.add_argument(
        "--burnin",
        type=float,
        default=BURNIN_FRACTION,
        help=f"Burn-in fraction fallback (default {BURNIN_FRACTION}).",
    )
    ap.add_argument(
        "--budget-hours",
        type=float,
        default=BUDGET_HOURS,
        help=f"#3 fixed global wallclock budget in hours (default {BUDGET_HOURS}).",
    )
    args = ap.parse_args()
    BURNIN_FRACTION = args.burnin
    MASCOT_DIR = args.mascot_dir

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out}")
    print(f"Mascot dir: {MASCOT_DIR}")
    print(f"Sims: {args.sims}")

    # ---- Load all sim x config x seed ----
    print("\nLoading trace + screen logs...")
    seeds: list[SeedData] = []
    for sim in args.sims:
        for config, label in CONFIGS:
            for seed in SEEDS:
                sd = load_seed_data(sim, config, label, seed)
                if sd is not None:
                    seeds.append(sd)
    if not seeds:
        sys.exit("No data loaded -- check paths.")
    print(
        f"  loaded {len(seeds)} runs "
        f"({len({s.sim for s in seeds})} sims x {len(CONFIGS)} configs x "
        f"{len(SEEDS)} seeds)"
    )

    # ---- #5 stationarity (measured burn-in) -- FIRST, feeds #1-#4 ----
    print("\n#5 Time to stationarity (measured burn-in)...")
    stationarity = analysis_stationarity(seeds)
    stationarity.to_csv(out / "stationarity_by_seed.csv", index=False)
    burnin_rows, burnin_summary = derive_burnin(seeds, stationarity)
    burnin_summary.to_csv(out / "burnin_by_run.csv", index=False)
    burnin_across = _config_order(
        summarize_across_sims(
            burnin_summary,
            ["config", "label"],
            ["burnin_sample", "burnin_hours", "n_rows_postburnin"],
        )
    )
    burnin_across.to_csv(out / "burnin_across_sims.csv", index=False)
    plot_stationarity_per_param(stationarity, out / "stationarity")
    print(
        burnin_across[
            ["label", "burnin_hours_median", "n_rows_postburnin_median"]
        ].to_string(index=False)
    )

    # ---- #1 min-ESS + bottleneck ----
    print("\n#1 Min-ESS across focused scientific parameters...")
    ess_per_param, min_ess_summary = analysis_min_ess(seeds, burnin_rows)
    ess_per_param.to_csv(out / "ess_focused_by_seed.csv", index=False)
    min_ess_summary.to_csv(out / "miness_by_seed.csv", index=False)
    miness_across = _config_order(
        summarize_across_sims(min_ess_summary, ["config", "label"], ["min_ESS"])
    )
    # fraction of (sim, seed) runs fully converged, per config
    conv = (
        min_ess_summary.groupby(["config", "label"])["all_focused_converged"]
        .mean()
        .reset_index(name="frac_runs_all_converged")
    )
    miness_across = _config_order(miness_across.merge(conv, on=["config", "label"]))
    miness_across.to_csv(out / "miness_across_sims.csv", index=False)
    plot_min_ess(min_ess_summary, out / "miness")
    print(
        miness_across[
            [
                "label",
                "min_ESS_median",
                "min_ESS_q25",
                "min_ESS_q75",
                "frac_runs_all_converged",
            ]
        ].to_string(index=False)
    )

    # ---- #2 ESS/hour decomposition (all params) ----
    print("\n#2 ESS/hour = (ESS/sample) x (samples/hour) for all params...")
    decomp = analysis_decomposition(seeds, ess_per_param)
    decomp.to_csv(out / "ess_hour_decomposition.csv", index=False)
    decomp_across = _config_order(
        summarize_across_sims(
            decomp,
            ["config", "label"],
            ["ESS_per_hour", "ESS_per_sample", "samples_per_hour"],
        )
    )
    decomp_across.to_csv(out / "ess_hour_decomposition_across_sims.csv", index=False)
    plot_decomposition(decomp, out / "decomposition")
    print(
        decomp_across[
            [
                "label",
                "ESS_per_hour_median",
                "ESS_per_sample_median",
                "samples_per_hour_median",
            ]
        ].to_string(index=False)
    )

    # ---- #3 per-parameter compute curves ----
    print("\n#3 Per-parameter compute curves (ESS=200 + at-budget)...")
    frontier = frontier_curves(seeds, burnin_rows)
    frontier.to_csv(out / "frontier_curves.csv", index=False)
    poi_order = params_of_interest(seeds[0])

    def hpd_label(suffix: str):
        def _lab(p):
            k = param_kind(p)
            if k == "prevalence":
                return f"95% HPDI width {suffix}"
            if k == "posterior":
                return f"95% HPDI width {suffix}"
            return f"Rel. 95% HPDI width {suffix}"

        return _lab

    # (a) time to ESS=200, per parameter (X = target not reached, full-run
    # value shown instead)
    ess200 = time_to_ess200(frontier)
    ess200.to_csv(out / "ess200_by_seed.csv", index=False)
    ess200_across = _config_order(
        summarize_across_sims(
            ess200,
            ["config", "label", "parameter"],
            ["hours_to_ess200", "samples_to_ess200", "hpd_at_ess200"],
        )
    )
    reached = (
        ess200.groupby(["config", "label", "parameter"])["reached_ess200"]
        .mean()
        .reset_index(name="frac_reached_ess200")
    )
    ess200_across = ess200_across.merge(reached, on=["config", "label", "parameter"])
    ess200_across.to_csv(out / "ess200_across_sims.csv", index=False)
    plot_per_param_panels(
        ess200,
        [
            ("hours_to_ess200", "Wallclock hours to ESS=200", None),
            ("samples_to_ess200", "MCMC samples to ESS=200", None),
            ("hpd_at_ess200", hpd_label("at ESS=200"), None),
        ],
        out / "ess200",
        "#3a Time to ESS=200",
        "ess200",
        param_order=poi_order,
        marker_col="missed_ess200",
    )

    # ---- Main composite figure (posterior parameter, 2x2) ----
    plot_main_figure_posterior(stationarity, ess200, out / "comp_cost_posterior")
    print("  wrote comp_cost_posterior.png/.pdf")

    # (b) state at the fixed global budget, per parameter
    budg = at_budget(frontier, args.budget_hours)
    budg.to_csv(out / "at_budget_by_seed.csv", index=False)
    budg_across = _config_order(
        summarize_across_sims(
            budg,
            ["config", "label", "parameter"],
            ["ESS_at_budget", "samples_at_budget", "hpd_at_budget"],
        )
    )
    budg_across.to_csv(out / "at_budget_across_sims.csv", index=False)
    bh = f"{args.budget_hours:g} h"
    plot_per_param_panels(
        budg,
        [
            ("ESS_at_budget", f"ESS after {bh}", ESS_THRESHOLD),
            ("samples_at_budget", f"MCMC samples after {bh}", None),
            ("hpd_at_budget", hpd_label(f"after {bh}"), None),
        ],
        out / "at_budget",
        f"#3b State after {bh}",
        "at_budget",
        param_order=poi_order,
    )
    print(
        f"  budget = {args.budget_hours:g} h; wrote per-parameter ess200/ and "
        f"at_budget/ figures for {len(poi_order)} parameters."
    )

    # ---- README ----
    write_readme(out, args.budget_hours, len({sd.sim for sd in seeds}))
    print(f"\nDone. All outputs + README.md in {out}")


# =============================================================================
# README template (written into every output directory)
# =============================================================================

README_TEMPLATE = r"""# Sampler efficiency & inference-quality analysis (simulation study)

Generated by `bin/sampler_efficiency.py`. This directory quantifies whether
adding epidemiological datastreams improves MCMC convergence and inference
efficiency, keeping two confounded effects separate:

* **statistical** -- adding data changes (contracts) the posterior itself;
* **computational** -- the posterior geometry changes AND per-step cost rises
  (more likelihood evaluations), so raw *samples*-to-ESS unfairly flatters the
  data-rich configs by ignoring throughput.

Through-line: HPD width = *better answer*; #2 = *at what per-sample and per-hour
sampling cost*; #3 = *how long / how many samples each parameter needs and how
precise it is at a fixed compute budget*.

--------------------------------------------------------------------------------
## Inputs (simulation study, `2_mascot`)

**{n_sims} independent simulated datasets**, each run under every datastream
configuration with **replicate MCMC seeds ({seeds})**. Two sources per run,
aligned by MCMC state (every logger starts at state 0 with the same `logEvery`,
so row k is the same MCMC state in every file -- that is what lets timing from
the screen log join to parameters from the trace log by sample number):

| Source | Path | Provides |
|---|---|---|
| Trace log | `<N>_2_simulation/<config>/<seed>_<N>_2_simulation_<config>.log` | scientific parameters |
| Screen log | `<N>_2_simulation/<config>/<seed>_<N>_2_simulation_<config>.out` | throughput (rate) |

**Configurations compared** (reference first): `All DS` (all datastreams),
`No CC`, `No SP`, `No WW`, `No MLL`, `Tree only`.

Every metric is computed per **(simulation x config x seed)** and then
summarised **across the {n_sims} simulations** per config (`*_across_sims.csv`,
median / IQR / mean).

### Wallclock reconstruction (read this before using any "hours" column)

The screen log's trailing column is a **rate** (`NhNmNs/Msamples`, a whole-run
running average), *not* cumulative elapsed time. We take each run's **last
reported rate** as its representative throughput and reconstruct time under a
**constant-throughput** assumption:

    elapsed_hours(sample) = rate_h_per_Msample * sample / 1e6

This is monotonic and fine for *relative* comparison; absolute hours are
approximate. Throughput varies by run, so all timing is per-run.

### Focused scientific parameter set (labelled by biological ROLE)

This is a 2-deme model (`I0`/`I1`), so every migration direction and every
prevalence knot is a focal parameter (no ghost/background deme). But **which**
physical deme seeded the epidemic (`start`) vs was invaded (`secondary`) differs
per simulation -- read from `1_remaster_sim/<N>_2_simulation_sim_metadata.csv`
(`deme_type`), with physical deme index k mapping to `I<k>` and to `Deme<k+1>`.
Every parameter is therefore relabelled by **role**, so pooling across the 20
simulations compares like with like (all start-deme quantities together):

* **migration** uses the RATE parameter `f_migrationRatesSkyline.*` (not the
  `migrationEvents` count):
  * `mig_rate_start_to_secondary`, `mig_rate_secondary_to_start`;
* **prevalence knots** `SkylinePrev.start.k`, `SkylinePrev.secondary.k`;
* **posterior** -- the log-posterior column, treated as a full parameter (its
  own ESS / stationarity / #3 curves), log-space so its HPD headline is the
  absolute width. It is a bottleneck candidate in #1 like any other parameter.

One aggregate (summed per posterior sample):

* `mig_rate_total` = sum of both migration-rate directions.

| analysis | parameter set |
|---|---|
| #1 min-ESS bottleneck | focused set (posterior + 2 migration rates + prevalence) |
| #5 burn-in aggregation (max) | all parameters of interest |
| #2 decomposition | posterior + migration rates + prevalence (per param) |
| #3 per-parameter curves, #5 stationarity | the above plus the `mig_rate_total` aggregate |

### HPD width metric (kind-aware headline)

95% equal-tailed HPD width = `q97.5 - q2.5`. The #3 HPD subplots use the
**headline** width per parameter kind:

* **migration rates** (positive) -> **relative** width `(q97.5 - q2.5) / median`;
* **prevalence knots** are **log-space** (late-time knots have negative median,
  so HPD/median is undefined) -> **absolute** log-space width, which is already
  a ratio measure in linear space.

Both absolute and relative widths are kept in `frontier_curves.csv`; the derived
`hpd_at_ess200` / `hpd_at_budget` columns hold the headline width (see `kind`).

### ESS is measured on the STATIONARY part of the trace

Burn-in is determined per run by #5 (time to stationarity) and then fed into
#1-#3. The run-level burn-in for a (sim, config, seed) is the **maximum**
stationarity sample over the parameters of interest -- the chain is not treated
as stationary until its slowest focused parameter is (`burnin_by_run.csv`).
Every ESS, ESS/sample and ESS/hour value (#1, #2, #3) is computed *after*
discarding that burn-in, so they describe the stationary chain only.

### Settings for this run

* ESS threshold: **{threshold}** (#1 min-ESS bottleneck, #3b at-budget line)
* #3a time-to-ESS target: **{ess_time_target}** (runs that never reach it report
  full-run values instead, marked with an X)
* reference config: **{reference}** (context only -- #2 is now reported in absolute units)
* stationarity region: last **{tail_frac}** of trace; **{consecutive}** consecutive in-region samples
* #3 fixed wallclock budget: **{budget} h** (global; same for every sim & config)

--------------------------------------------------------------------------------
## Output files

Per-run detail CSVs (one row per sim x config x seed, or x parameter):

* `stationarity_by_seed.csv` -- #5 per parameter of interest.
* `burnin_by_run.csv` -- #5 run-level burn-in + driving parameter.
* `ess_focused_by_seed.csv` -- #1 ESS of every focused parameter (post-burn-in).
* `miness_by_seed.csv` -- #1 min-ESS + bottleneck per run.
* `ess_hour_decomposition.csv` -- #2 absolute ESS/sample, samples/hour, ESS/hour.
* `frontier_curves.csv` -- #3 per-parameter ESS + HPD width at each checkpoint.
* `ess200_by_seed.csv` -- #3a per (sim, config, seed, parameter): wallclock,
  MCMC samples and headline HPD width at first ESS>={ess_time_target}
  (`reached_ess200`). If never reached, these instead report the full-run
  values (`missed_ess200=True`).
* `at_budget_by_seed.csv` -- #3b per (sim, config, seed, parameter): ESS,
  MCMC samples and headline HPD width reached by the {budget} h budget.

Across-simulation summaries (median / IQR / mean; per config, and per config x
parameter for #3):

* `burnin_across_sims.csv`, `miness_across_sims.csv`,
  `ess_hour_decomposition_across_sims.csv`, `ess200_across_sims.csv`,
  `at_budget_across_sims.csv`.

Plots (`.png` + `.pdf`):

* `stationarity/stationarity_<parameter>.*` -- samples & hours to stationarity
  (log scale), x = config, one dot per (sim, seed), bar = median (thin black
  line, drawn on top of the dots).
* `miness.*` -- min focused-ESS per config across (sim, seed).
* `decomposition.*` -- absolute ESS/hour, ESS/sample, samples/hour per config.
* `ess200/ess200_<parameter>.*` -- #3a: per parameter, 3 subplots (hours,
  samples, relative HPD) to reach ESS={ess_time_target}; x = config, one dot
  per (sim, seed) -- an X marks a run that never reached the target, plotting
  its full-run value instead.
* `at_budget/at_budget_<parameter>.*` -- #3b: per parameter, 3 subplots (ESS,
  samples, relative HPD) after {budget} h; x = config, one dot per (sim, seed).

--------------------------------------------------------------------------------
## Reproduce

    conda run -n biopython_env python bin/sampler_efficiency.py \
        --output-dir <this directory>

Options: `--sims`, `--mascot-dir`, `--burnin`, `--budget-hours`.
"""


if __name__ == "__main__":
    main()
