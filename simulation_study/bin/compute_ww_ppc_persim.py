#!/usr/bin/env python3
"""Per-simulation wastewater posterior-predictive coverage.

For every simulation under ``--results_dir`` this script runs a slim wastewater
posterior-predictive check and reduces it to a single number: the fraction of
the true wastewater observations (pooled across both demes) that fall inside the
95 % posterior-predictive interval.

The wastewater observation model is log-normal in both the simulator and the
MASCOT likelihood:

    log(obs) ~ Normal(log(k_ww * I / N), sigma^2)

where k_ww = ``wastewater.scaling``, I = prevalence, N = deme population size,
and sigma = ``wastewater.sigma``. For each of ``--n_samples`` posterior draws s
(sampled at random, without replacement, from the post-burn-in chain) and each
observation time t_i in deme d we draw one replicate

    log y_rep_{s,i} = log(k_ww_s) + log I_s(t_i) - log(N_d) + sigma_s * Z

with Z ~ Normal(0, 1). Prevalence I_s(t_i) is evaluated from the posterior
``SkylinePrev`` spline the same way the BEAST likelihood does (see ``grid_mode``).
The 95 % highest-density interval (HPD) of the replicate draws at each obs time
defines the interval; ``coverage_95`` is the share of true observations inside
their own interval.

This is the per-sim value behind Panel A of the wastewater-sigma supplementary
figure. The posterior sigma columns are also recorded because they are cheap and
likely feed the remaining panels.

Outputs (into ``--output_dir``):
    ww_ppc_persim_summary.csv   one row per sim (Panel A)
    ww_ppc_per_obs.csv          one row per (sim, deme, observation) holding the
                                raw residual ingredients (Panels C and B). The
                                residual and its binning are computed later in
                                the figure script, so bin count and residual
                                definition are adjustable without recompute.

ww_ppc_persim_summary.csv columns:
    simid                  simulation id (e.g. ``20_2``)
    n_obs                  total wastewater observations used, both demes
    coverage_95            fraction of true obs inside the 95 % predictive HPD
    sigma_true             ds_ww_sigma from the simulator's parameters CSV
    sigma_post_median      posterior median of wastewater.sigma (sampled cohort)
    sigma_post_hpd_lower   95 % HPD lower bound of the posterior sigma
    sigma_post_hpd_upper   95 % HPD upper bound of the posterior sigma
    sigma_true_in_hpd      whether sigma_true lies within the 95 % HPD
    n_samples              posterior draws actually used

ww_ppc_per_obs.csv columns:
    simid, deme, t_years   observation identity and forward time (years)
    log_obs                log observed wastewater concentration
    log_mu_true            log(k_ww_true * I_true(t) / N)   [true mean]
    log_mu_post            median over draws of log(k_ww_s * I_s(t) / N) [fitted]
    sigma_true             true ds_ww_sigma
    sigma_post_median      posterior-median sigma
    I_true                 true prevalence count I at the obs time
    I_over_N_true          true prevalence fraction at the obs time
    I_over_N_post          posterior-median prevalence fraction at the obs time
    k_ww_true, k_ww_post   true / posterior-median wastewater scaling (k_ww)
"""

import argparse
import gc
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse loaders and the spline/HPD helpers from the sibling scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_posteriors import (  # noqa: E402
    calculate_hpd,
    get_deme_popsize,
    load_params_csv,
    load_trajectory_file,
    load_wastewater_file,
    read_beast_log,
)
from make_figure_individualsim import (  # noqa: E402
    _build_natural_spline,
    _eval_spline_clamped,
    _snap_to_grid,
    _true_I_at_vec,
)

SIGMA_COL = "wastewater.sigma:SimDataset"


def discover_simulations(results_dir):
    """Return simulation IDs (e.g. ``20_2``) from 1_remaster_sim/*_simulation.traj."""
    remaster = Path(results_dir) / "1_remaster_sim"
    sim_ids = []
    for traj in sorted(remaster.glob("*_simulation.traj")):
        sim_ids.append(traj.stem[: -len("_simulation")])

    def _key(simid):
        head = simid.split("_")[0]
        return (int(head) if head.isdigit() else float("inf"), simid)

    return sorted(sim_ids, key=_key)


def build_sim_paths(results_dir, simid):
    """Return the input paths for one sim, or None if any required file is absent."""
    base = Path(results_dir)
    rm = base / "1_remaster_sim"
    ds = base / "2_mascot" / f"{simid}_simulation" / "datastreams"
    paths = {
        "log": ds / f"{simid}_simulation_datastreams.mascot_logs.combined.log",
        "wastewater": rm / f"{simid}_simulation_wastewater.csv",
        "trajectory": rm / f"{simid}_simulation.traj",
        "params": rm / f"{simid}_simulation_parameters.csv",
    }
    if not all(p.exists() for p in paths.values()):
        return None
    return paths


def _apply_burnin(df, burnin):
    """Drop the first ``burnin`` fraction of samples using the ``Sample`` column."""
    if burnin and burnin > 0 and "Sample" in df.columns:
        return df[df["Sample"] > df["Sample"].max() * burnin]
    return df


def _prevalence_columns(log_df, deme):
    """Return the SkylinePrev columns for ``deme`` (0-based), sorted by knot index.

    BEAST names Deme1 <-> deme 0, so ``deme`` maps to ``SkylinePrev.Deme{deme+1}.``.
    """
    prefix = f"SkylinePrev.Deme{deme + 1}."
    cols = [c for c in log_df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda c: int(c.split(".")[-1]))


def _alpha_true_for_deme(params, deme):
    """True wastewater scaling (ds_ww_scaling) for a deme, or None."""
    if params is None or params.empty:
        return None
    row = params[
        (params["parameter"] == "ds_ww_scaling")
        & (params["deme"].astype(str) == str(deme))
    ]
    if row.empty:
        return None
    return float(row["value"].iloc[0])


def ppc_coverage_for_sim(paths, burnin, n_samples, seed, grid_mode):
    """Compute the per-sim PPC coverage row and per-observation residual rows.

    Returns ``(summary_row, per_obs_rows)`` or ``None`` if the sim is unusable.
    ``per_obs_rows`` is a list of dicts (one per wastewater observation) holding
    the raw ingredients for the residual/binning done later in the figure script.
    """
    log_df, rateshifts, gridpointshifts = read_beast_log(
        str(paths["log"]), read_rateshifts=True
    )
    if log_df is None or log_df.empty:
        return None
    if rateshifts is None or gridpointshifts is None:
        return None
    if SIGMA_COL not in log_df.columns:
        return None

    log_df = _apply_burnin(log_df, burnin)
    # Random draws (without replacement) from the post-burn-in posterior, rather
    # than strided thinning, so each of the n_samples draws is a plain random
    # sample of the posterior. Seeded for reproducibility.
    if len(log_df) > n_samples:
        log_thin = log_df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    else:
        log_thin = log_df.reset_index(drop=True).copy()
    n_draws = len(log_thin)
    if n_draws < 2:
        return None

    ww = load_wastewater_file(str(paths["wastewater"]))
    traj = load_trajectory_file(str(paths["trajectory"]))
    params = load_params_csv(str(paths["params"]))
    if ww is None or ww.empty or traj is None or traj.empty:
        return None

    deme_popsizes = get_deme_popsize(traj)
    traj_I = traj[traj["population"] == "I"]

    sigma_true = None
    if params is not None and not params.empty:
        sig_row = params[params["parameter"] == "ds_ww_sigma"]
        if not sig_row.empty:
            sigma_true = float(sig_row["value"].iloc[0])

    sigma_samples = log_thin[SIGMA_COL].to_numpy(dtype=float)
    sig_lo, sig_hi, sig_med = calculate_hpd(sigma_samples, alpha=0.05)

    rateshifts = np.asarray(rateshifts, dtype=float)
    grid_asc = np.sort(np.asarray(gridpointshifts, dtype=float))
    t_first = float(rateshifts.min())
    t_last = float(rateshifts.max())
    max_rateshift = t_last

    rng = np.random.default_rng(seed)

    in95_all = []
    per_obs_rows = []
    for deme in sorted(int(d) for d in ww["Deme"].unique()):
        if deme not in deme_popsizes:
            continue
        N = float(deme_popsizes[deme])

        prev_cols = _prevalence_columns(log_df, deme)
        if len(prev_cols) != rateshifts.size:
            continue
        alpha_col = f"wastewater.scaling.Deme{deme + 1}:SimDataset"
        if alpha_col not in log_thin.columns:
            continue

        obs_d = ww[ww["Deme"] == deme].sort_values("t_wastewater_fromsimstart")
        if obs_d.empty:
            continue
        t_obs_fwd = obs_d["t_wastewater_fromsimstart"].to_numpy(dtype=float)
        y_obs = obs_d["wastewater"].to_numpy(dtype=float)
        log_obs = np.log(np.clip(y_obs, 1e-300, None))
        t_obs_back = max_rateshift - t_obs_fwd
        n_obs_d = len(obs_d)

        alpha_samples = log_thin[alpha_col].to_numpy(dtype=float)
        alpha_post = float(np.median(alpha_samples))
        knot_samples = log_thin[prev_cols].to_numpy(dtype=float)

        log_I = np.empty((n_draws, n_obs_d), dtype=float)
        for s in range(n_draws):
            spline = _build_natural_spline(rateshifts, knot_samples[s])
            if grid_mode == "snap":
                t_eval_back, _ = _snap_to_grid(t_obs_back, grid_asc)
                log_I[s, :] = _eval_spline_clamped(
                    spline, t_eval_back, t_first, t_last
                )
            else:  # interpolate
                grid_vals = _eval_spline_clamped(spline, grid_asc, t_first, t_last)
                log_I[s, :] = np.interp(t_obs_back, grid_asc, grid_vals)

        log_mu = np.log(alpha_samples)[:, None] + log_I - np.log(N)
        eps = rng.standard_normal((n_draws, n_obs_d)) * sigma_samples[:, None]
        log_y_rep = log_mu + eps

        # Posterior point summaries per obs for the residual (median over draws).
        log_mu_post = np.median(log_mu, axis=0)
        I_over_N_post = np.exp(np.median(log_I, axis=0)) / N

        # True-mean ingredients (true alpha, true step-I(t), true sigma).
        alpha_true = _alpha_true_for_deme(params, deme)
        traj_d = traj_I[traj_I["Deme"] == deme].sort_values("t")
        if not traj_d.empty and alpha_true is not None:
            t_evt = traj_d["t"].to_numpy(dtype=float)
            I_evt = traj_d["value"].to_numpy(dtype=float)
            I_true = _true_I_at_vec(t_obs_fwd, t_evt, I_evt).astype(float)
            I_over_N_true = I_true / N
            with np.errstate(divide="ignore"):
                log_mu_true = np.log(alpha_true * I_true / N)
        else:
            I_true = np.full(n_obs_d, np.nan)
            I_over_N_true = np.full(n_obs_d, np.nan)
            log_mu_true = np.full(n_obs_d, np.nan)

        for j in range(n_obs_d):
            lo, hi, _ = calculate_hpd(log_y_rep[:, j], alpha=0.05)
            in95_all.append(bool(lo <= log_obs[j] <= hi))
            per_obs_rows.append(
                {
                    "simid": None,  # filled by caller
                    "deme": deme,
                    "t_years": float(t_obs_fwd[j]),
                    "log_obs": float(log_obs[j]),
                    "log_mu_true": float(log_mu_true[j]),
                    "log_mu_post": float(log_mu_post[j]),
                    "sigma_true": (
                        float(sigma_true) if sigma_true is not None else np.nan
                    ),
                    "sigma_post_median": float(sig_med),
                    "I_true": float(I_true[j]),
                    "I_over_N_true": float(I_over_N_true[j]),
                    "I_over_N_post": float(I_over_N_post[j]),
                    "k_ww_true": (
                        float(alpha_true) if alpha_true is not None else np.nan
                    ),
                    "k_ww_post": alpha_post,
                }
            )

    n_obs = len(in95_all)
    if n_obs == 0:
        return None

    summary_row = {
        "simid": None,  # filled by caller
        "n_obs": n_obs,
        "coverage_95": float(np.mean(in95_all)),
        "sigma_true": sigma_true,
        "sigma_post_median": float(sig_med),
        "sigma_post_hpd_lower": float(sig_lo),
        "sigma_post_hpd_upper": float(sig_hi),
        "sigma_true_in_hpd": (
            bool(sig_lo <= sigma_true <= sig_hi)
            if sigma_true is not None
            else None
        ),
        "n_samples": int(n_draws),
    }
    return summary_row, per_obs_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results_dir",
        default="simulation_study/results",
        help="Root results dir with 1_remaster_sim/ and 2_mascot/.",
    )
    parser.add_argument(
        "--output_dir",
        default="sandbox/ww_ppc",
        help="Where to write ww_ppc_persim_summary.csv.",
    )
    parser.add_argument("--burnin", type=float, default=0.1)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Posterior draws (random, no replacement) used for the PPC (default 1000).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--grid_mode",
        choices=["snap", "interpolate"],
        default="interpolate",
        help=(
            "How to evaluate prevalence at each obs time. 'interpolate' linearly "
            "interpolates between bracketing grid-point spline values (current "
            "Java likelihood); 'snap' uses the nearest grid point. Default: "
            "interpolate."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N sims."
    )
    parser.add_argument(
        "--only_simids", nargs="*", default=None, help="Process only these SIMIDs."
    )
    cli = parser.parse_args()

    out_dir = Path(cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_ids = discover_simulations(cli.results_dir)
    if cli.only_simids:
        wanted = set(cli.only_simids)
        sim_ids = [s for s in sim_ids if s in wanted]
    if cli.limit:
        sim_ids = sim_ids[: cli.limit]

    print(f"Processing {len(sim_ids)} simulation(s). Output -> {out_dir}")

    rows = []
    per_obs_all = []
    failures = []
    for simid in sim_ids:
        paths = build_sim_paths(cli.results_dir, simid)
        if paths is None:
            print(f"[skip] {simid}: missing one or more required files.")
            failures.append((simid, "missing files"))
            continue
        print(f"[run]  {simid}", flush=True)
        try:
            result = ppc_coverage_for_sim(
                paths,
                burnin=cli.burnin,
                n_samples=cli.n_samples,
                seed=cli.seed,
                grid_mode=cli.grid_mode,
            )
            if result is None:
                print(f"[warn] {simid}: no usable PPC coverage.")
                failures.append((simid, "no coverage"))
            else:
                summary_row, per_obs_rows = result
                summary_row["simid"] = simid
                rows.append(summary_row)
                for r in per_obs_rows:
                    r["simid"] = simid
                per_obs_all.extend(per_obs_rows)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {simid}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures.append((simid, f"{type(exc).__name__}: {exc}"))
        finally:
            gc.collect()

    if not rows:
        print("No simulations produced usable coverage.")
        return

    df_all = pd.DataFrame(rows)
    summary_csv = out_dir / "ww_ppc_persim_summary.csv"
    df_all.to_csv(summary_csv, index=False)

    per_obs_csv = out_dir / "ww_ppc_per_obs.csv"
    pd.DataFrame(per_obs_all).to_csv(per_obs_csv, index=False)

    print("\n--- Summary ---")
    print(f"sims summarised : {len(df_all)}")
    print(f"observations    : {len(per_obs_all)}")
    print(f"median coverage : {df_all['coverage_95'].median():.3f}")
    print(f"written         : {summary_csv}")
    print(f"                : {per_obs_csv}")
    if failures:
        print(f"failures/skips  : {len(failures)}")
        for simid, reason in failures:
            print(f"  {simid}: {reason}")


if __name__ == "__main__":
    main()
