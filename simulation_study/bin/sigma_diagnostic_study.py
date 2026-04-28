#!/usr/bin/env python3
"""
Aggregate per-simulation σ diagnostics for the wastewater observation model.

Loops over every simulation under ``--results_dir`` (default
``simulation_study/results``), loads the same inputs
``make_figure_individualsim.py`` uses, computes posterior-mean residuals
via ``compute_wastewater_residuals``, and records per sim:

    n                       — number of finite wastewater residuals
    sigma_true              — ds_ww_sigma from the simulator's params CSV
    sigma_tilde             — sqrt( Σ r² / n ) using posterior-mean residuals
                              (MLE of σ given posterior-mean μ̂)
    sigma_post_median       — posterior median of wastewater.sigma:SimDataset
    sigma_post_hpd_lower    — 95% HPD lower bound
    sigma_post_hpd_upper    — 95% HPD upper bound
    hpd_width               — sigma_post_hpd_upper − sigma_post_hpd_lower
    hpd_width_over_sigma_tilde
    sigma_post_median_minus_sigma_tilde
    sigma_true_in_HPD       — is sigma_true within the 95% HPD

Outputs (into ``--output_dir``, default ``sandbox/sigma_diagnostic``):
    all_sims_sigma_summary.csv       — one row per sim (all sims)
    problem_sims_sigma_summary.csv   — subset where sigma_true is outside the
                                        95% HPD, truncated to ``--max_problem_sims``
                                        (sorted by most-informative first)
    sigma_median_vs_sigma_tilde.png  — scatter of σ̂_post_median vs σ̃,
                                        with y=x and prior median ≈ 0.50 refs
    sigma_hpd_width_vs_sigma_tilde.png — HPD width vs σ̃ with 0.49·σ̃ reference
                                          (flat-log-σ n=40 expectation)
"""

import argparse
import gc
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse functions from the sibling scripts in this bin/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_posteriors import prepare_skyline_plot_data  # noqa: E402
from make_figure_individualsim import compute_wastewater_residuals  # noqa: E402


REQUIRED_FILE_ATTRS = (
    "log_file_original",
    "log_file_datastream",
    "case_counts_file",
    "seroprevalence_file",
    "wastewater_file",
    "cumulative_incidence_deme1",
    "cumulative_incidence_deme2",
    "nedynamics_deme1",
    "nedynamics_deme2",
    "trajectory_file",
    "params_csv",
    "deme_switches_csv",
)


def discover_simulations(results_dir):
    """Return sorted simulation IDs (e.g. ``20_2``) from 1_remaster_sim/*.traj."""
    remaster = Path(results_dir) / "1_remaster_sim"
    sim_ids = []
    for traj in sorted(remaster.glob("*_simulation.traj")):
        simid = traj.stem[: -len("_simulation")]
        sim_ids.append(simid)
    return sim_ids


def build_args_namespace(results_dir, simid, probe_dir, burnin=0.0):
    """Build a SimpleNamespace matching analyse_posteriors' parse_arguments output.

    ``probe_dir`` is a directory where ``prepare_skyline_plot_data`` is free to
    write sidecar validation CSVs via ``out_prefix``; it must already exist.
    Returns None if any required input file is missing.
    """
    base = Path(results_dir)
    rm = base / "1_remaster_sim"
    ds = base / "2_mascot" / f"{simid}_simulation" / "datastreams"
    og = base / "2_mascot" / f"{simid}_simulation" / "original"

    args = SimpleNamespace(
        log_file_original=str(
            og / f"{simid}_simulation_original.mascot_logs.combined.log"
        ),
        log_file_datastream=str(
            ds / f"{simid}_simulation_datastreams.mascot_logs.combined.log"
        ),
        case_counts_file=str(rm / f"{simid}_simulation_casecounts.csv"),
        seroprevalence_file=str(rm / f"{simid}_simulation_seroprevalence.csv"),
        wastewater_file=str(rm / f"{simid}_simulation_wastewater.csv"),
        cumulative_incidence_deme1=str(
            ds / f"{simid}_simulation_datastreams.cumulativeIncidence.Deme1.combined.log"
        ),
        cumulative_incidence_deme2=str(
            ds / f"{simid}_simulation_datastreams.cumulativeIncidence.Deme2.combined.log"
        ),
        nedynamics_deme1=str(
            ds / f"{simid}_simulation_datastreams.NeDynamics.Deme1.combined.log"
        ),
        nedynamics_deme2=str(
            ds / f"{simid}_simulation_datastreams.NeDynamics.Deme2.combined.log"
        ),
        trajectory_file=str(rm / f"{simid}_simulation.traj"),
        params_csv=str(rm / f"{simid}_simulation_parameters.csv"),
        deme_switches_csv=str(
            rm / f"{simid}_simulation_deme_switches_groundtruth.csv"
        ),
        burnin=burnin,
        # prepare_skyline_plot_data writes HPD validation CSVs using out_prefix
        # as a path prefix. Keep them inside the user's chosen output dir so
        # nothing lands in canonical results folders.
        out_prefix=str(Path(probe_dir) / f"{simid}"),
    )
    for attr in REQUIRED_FILE_ATTRS:
        if not Path(getattr(args, attr)).exists():
            return None
    return args


def summarise_sim(simid, df_resid, sigma_true, sigma_post):
    """Return a one-row dict with σ diagnostics for a single sim, or None."""
    if df_resid is None or df_resid.empty:
        return None
    if sigma_true is None or sigma_post is None:
        return None

    res = df_resid["res_inf"].to_numpy()
    res = res[np.isfinite(res)]
    n = int(res.size)
    if n == 0:
        return None

    sigma_tilde = float(np.sqrt(np.sum(res ** 2) / n))
    hpd_lo = float(sigma_post["hpd_lower"])
    hpd_hi = float(sigma_post["hpd_upper"])
    width = hpd_hi - hpd_lo
    median = float(sigma_post["median"])
    in_hpd = bool(hpd_lo <= sigma_true <= hpd_hi)

    expected_factor = _expected_hpd_width_factor(n)
    expected_width = expected_factor * sigma_tilde

    return {
        "simid": simid,
        "n": n,
        "sigma_true": float(sigma_true),
        "sigma_tilde": sigma_tilde,
        "sigma_post_median": median,
        "sigma_post_hpd_lower": hpd_lo,
        "sigma_post_hpd_upper": hpd_hi,
        "hpd_width": width,
        "hpd_width_over_sigma_tilde": (
            width / sigma_tilde if sigma_tilde > 0 else float("nan")
        ),
        "expected_hpd_width_factor": expected_factor,
        "expected_hpd_width": expected_width,
        "hpd_width_over_expected": (
            width / expected_width if expected_width > 0 else float("nan")
        ),
        "sigma_post_median_minus_sigma_tilde": median - sigma_tilde,
        "sigma_true_in_HPD": in_hpd,
    }


PRIOR_M_LOG = -0.7    # mean of log(sigma) under LogNormal prior
PRIOR_S_LOG = 0.3     # sd of log(sigma) under LogNormal prior
PRIOR_MEDIAN = float(np.exp(PRIOR_M_LOG))  # ~0.497 on linear sigma


def _expected_hpd_width_factor(n_obs):
    """Expected 95% HPD width on sigma as a multiple of sigma_median.

    Uses the asymptotic Normal approximation on log sigma:
        prior precision on log sigma  = 1 / PRIOR_S_LOG^2
        likelihood Fisher info at MLE = 2n
        posterior sd(log sigma)       ~ 1 / sqrt(prior_prec + 2n)
        width on sigma                = sigma_median * (e^{+z*sd} - e^{-z*sd})
    """
    prior_prec = 1.0 / (PRIOR_S_LOG ** 2)
    post_prec = prior_prec + 2.0 * n_obs
    post_sd_log = 1.0 / np.sqrt(post_prec)
    z = 1.959963984540054  # 97.5% standard normal
    return float(np.exp(z * post_sd_log) - np.exp(-z * post_sd_log))


def plot_median_vs_tilde(df_all, output_path):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ok = df_all[df_all["sigma_true_in_HPD"]]
    bad = df_all[~df_all["sigma_true_in_HPD"]]
    ax.scatter(
        ok["sigma_tilde"], ok["sigma_post_median"],
        s=22, alpha=0.7, color="tab:blue",
        label=fr"$\sigma_{{true}}$ in 95% HPD (n={len(ok)})",
    )
    ax.scatter(
        bad["sigma_tilde"], bad["sigma_post_median"],
        s=30, alpha=0.85, color="tab:red",
        label=fr"$\sigma_{{true}}$ outside HPD (n={len(bad)})",
    )
    vals = pd.concat([df_all["sigma_tilde"], df_all["sigma_post_median"]])
    lo = float(vals.min()) * 0.9 if len(vals) else 0.0
    hi = float(vals.max()) * 1.1 if len(vals) else 1.0
    ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1.0, label="y = x")
    ax.set_xlabel(
        r"$\tilde{\sigma}$ (log-scale SD) = RMS of posterior-mean residuals"
    )
    ax.set_ylabel(r"$\hat{\sigma}_{post}$ median (log-scale SD)")
    ax.set_title(
        "Posterior location vs empirical residual scale\n"
        "both axes in log-scale-SD units"
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_hpd_width_vs_tilde(df_all, output_path):
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ok = df_all[df_all["sigma_true_in_HPD"]]
    bad = df_all[~df_all["sigma_true_in_HPD"]]
    ax.scatter(
        ok["sigma_tilde"], ok["hpd_width"],
        s=22, alpha=0.7, color="tab:blue",
        label=fr"$\sigma_{{true}}$ in 95% HPD (n={len(ok)})",
    )
    ax.scatter(
        bad["sigma_tilde"], bad["hpd_width"],
        s=30, alpha=0.85, color="tab:red",
        label=fr"$\sigma_{{true}}$ outside HPD (n={len(bad)})",
    )

    xmax = float(df_all["sigma_tilde"].max()) * 1.05 if len(df_all) else 1.0
    xs = np.linspace(0.0, xmax, 100)

    # Expected HPD width under the actual prior LogNormal(M, S^2) and n residuals,
    # asymptotic Normal on log sigma:
    #   posterior precision on log sigma ~ 1/S^2 + 2n
    #   width on sigma ~ sigma_median * (e^{+z/sqrt(prec)} - e^{-z/sqrt(prec)})
    # Single reference line uses the median n across sims.
    n_ref = int(df_all["n"].median()) if len(df_all) else 40
    factor_actual = _expected_hpd_width_factor(n_ref)
    ax.plot(
        xs, factor_actual * xs, color="tab:purple", ls="-", lw=1.2,
        label=(
            fr"expected width under prior "
            fr"LogNormal($M={PRIOR_M_LOG}$, $S={PRIOR_S_LOG}$), "
            fr"n={n_ref} (median across sims): "
            fr"$\approx {factor_actual:.2f}\,\tilde{{\sigma}}$"
        ),
    )

    ax.set_xlabel(
        r"$\tilde{\sigma}$ (log-scale SD) = RMS of posterior-mean residuals"
    )
    ax.set_ylabel("95% HPD width on σ (log-scale-SD units)")
    ax.set_title(
        "Posterior HPD width vs empirical residual scale\n"
        "both axes in log-scale-SD units"
    )
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sigma_ratio_vs_n(df_all, output_path):
    """Scatter σ̂_post / σ_true vs n, with theoretical overfit curves overlaid.

    Under the classical smoothing-spline overfit argument the MLE satisfies
        E[σ̂²]  ≈  ((n − df_eff) / n) · σ_true²
    => σ̂/σ_true ≈ sqrt(1 − df_eff / n).

    Reference curves drawn for df_eff ∈ {2, 5, 11, 22} (11 = one knot per deme,
    22 = both demes). If ensemble points track one of these curves, that df_eff
    is an empirical estimate of the spline's effective degrees of freedom used
    up fitting the wastewater.
    """
    n = df_all["n"].to_numpy(dtype=float)
    sigma_true = df_all["sigma_true"].to_numpy(dtype=float)
    ratio_med = df_all["sigma_post_median"].to_numpy(dtype=float) / sigma_true
    ratio_lo = df_all["sigma_post_hpd_lower"].to_numpy(dtype=float) / sigma_true
    ratio_hi = df_all["sigma_post_hpd_upper"].to_numpy(dtype=float) / sigma_true
    covered = df_all["sigma_true_in_HPD"].to_numpy(dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for mask, colour, label in [
        (covered, "tab:blue", fr"$\sigma_{{true}}$ in 95% HPD (n={int(covered.sum())})"),
        (~covered, "tab:red", fr"$\sigma_{{true}}$ outside HPD (n={int((~covered).sum())})"),
    ]:
        if mask.any():
            ax.errorbar(
                n[mask], ratio_med[mask],
                yerr=[ratio_med[mask] - ratio_lo[mask],
                      ratio_hi[mask] - ratio_med[mask]],
                fmt="o", ms=4, alpha=0.75, color=colour,
                elinewidth=0.6, capsize=2, label=label,
            )

    n_ref = np.linspace(max(1.0, n.min() * 0.9), n.max() * 1.05, 200)
    for df_eff, colour, ls in [
        (2, "tab:green", ":"),
        (5, "tab:orange", "-."),
        (11, "tab:purple", "--"),
        (22, "black", "--"),
    ]:
        expected = np.sqrt(np.clip(1.0 - df_eff / n_ref, 0, None))
        ax.plot(n_ref, expected, color=colour, ls=ls, lw=1.0,
                label=fr"overfit: $\sqrt{{1 - {df_eff}/n}}$")

    ax.axhline(1.0, color="grey", lw=0.6)

    ax.set_xlabel(r"$n$ = total wastewater observations")
    ax.set_ylabel(r"$\hat{\sigma}_{post,median} / \sigma_{true}$")
    ax.set_title(
        "Posterior σ recovery vs data volume\n"
        r"systematic shortfall below 1 is consistent with spline overfit"
    )
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_linearised_overfit(df_all, output_path):
    """Linearised view: (1 − σ̂²/σ_true²) vs 1/n with a fitted line through origin.

    Under E[σ̂²] ≈ ((n − df_eff)/n)·σ_true²:
        y := 1 − σ̂²/σ_true²  ≈  df_eff · (1/n)
    so the slope of a least-squares line through the origin is an empirical
    estimate of df_eff. If the cloud scatters around zero with no slope, the
    overfit story is wrong. A positive slope localises df_eff.
    """
    n = df_all["n"].to_numpy(dtype=float)
    sigma_true = df_all["sigma_true"].to_numpy(dtype=float)
    ratio_med = df_all["sigma_post_median"].to_numpy(dtype=float) / sigma_true
    y = 1.0 - ratio_med ** 2
    x = 1.0 / n
    covered = df_all["sigma_true_in_HPD"].to_numpy(dtype=bool)

    # Least-squares slope through origin: slope = Σxy / Σx²
    denom = float(np.sum(x ** 2))
    slope = float(np.sum(x * y) / denom) if denom > 0 else float("nan")
    # Rough 95% CI on the slope via residual variance
    resid = y - slope * x
    resid_var = float(np.sum(resid ** 2) / max(1, x.size - 1))
    slope_se = float(np.sqrt(resid_var / denom)) if denom > 0 else float("nan")
    slope_lo = slope - 1.96 * slope_se
    slope_hi = slope + 1.96 * slope_se

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for mask, colour, label in [
        (covered, "tab:blue", fr"$\sigma_{{true}}$ in 95% HPD (n={int(covered.sum())})"),
        (~covered, "tab:red", fr"$\sigma_{{true}}$ outside HPD (n={int((~covered).sum())})"),
    ]:
        if mask.any():
            ax.scatter(x[mask], y[mask], s=28, alpha=0.8, color=colour, label=label)

    xs = np.linspace(0.0, float(x.max()) * 1.05, 100)
    ax.plot(xs, slope * xs, color="black", lw=1.2,
            label=(fr"fit through origin: slope $= df_{{eff}}$ "
                   fr"$= {slope:.2f}$ (95% CI [{slope_lo:.2f}, {slope_hi:.2f}])"))
    for df_eff, colour, ls in [
        (11, "tab:purple", "--"),
        (22, "grey", "--"),
    ]:
        ax.plot(xs, df_eff * xs, color=colour, ls=ls, lw=0.9,
                label=fr"$df_{{eff}}={df_eff}$ reference")
    ax.axhline(0.0, color="grey", lw=0.5)

    ax.set_xlabel(r"$1/n$")
    ax.set_ylabel(r"$1 - (\hat{\sigma}_{post,median} / \sigma_{true})^2$")
    ax.set_title(
        "Linearised overfit test\n"
        r"slope $=$ empirical $df_{eff}$ the spline uses up fitting wastewater"
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return {"df_eff_slope": slope, "df_eff_slope_ci_lo": slope_lo,
            "df_eff_slope_ci_hi": slope_hi, "n_sims": int(x.size)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results_dir", default="simulation_study/results",
        help="Root results dir with 1_remaster_sim/ and 2_mascot/.",
    )
    parser.add_argument(
        "--output_dir", default="sandbox/sigma_diagnostic",
        help="Where to write CSVs and PNGs.",
    )
    parser.add_argument("--burnin", type=float, default=0.0)
    parser.add_argument(
        "--max_problem_sims", type=int, default=10,
        help="Cap for problem_sims_sigma_summary.csv (sorted by smallest HPD "
             "width relative to σ̃, so the most tightly-constrained misses come first).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N sims (smoke test).",
    )
    parser.add_argument(
        "--only_simids", nargs="*", default=None,
        help="If given, process only these SIMIDs.",
    )
    cli = parser.parse_args()

    out_dir = Path(cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_dir = out_dir / "_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)

    sim_ids = discover_simulations(cli.results_dir)
    if cli.only_simids:
        wanted = set(cli.only_simids)
        sim_ids = [s for s in sim_ids if s in wanted]
    if cli.limit:
        sim_ids = sim_ids[: cli.limit]

    print(f"Processing {len(sim_ids)} simulation(s). Output -> {out_dir}")

    rows = []
    failures = []
    for simid in sim_ids:
        sim_args = build_args_namespace(
            cli.results_dir, simid, probe_dir, burnin=cli.burnin
        )
        if sim_args is None:
            print(f"[skip] {simid}: missing one or more required files.")
            failures.append((simid, "missing files"))
            continue
        print(f"[run]  {simid}", flush=True)
        try:
            data = prepare_skyline_plot_data(sim_args)
            df_resid, sigma_true, sigma_post = compute_wastewater_residuals(data)
            row = summarise_sim(simid, df_resid, sigma_true, sigma_post)
            if row is None:
                print(f"[warn] {simid}: no usable residuals or sigma info.")
                failures.append((simid, "no residuals / sigma"))
            else:
                rows.append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {simid}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures.append((simid, f"{type(exc).__name__}: {exc}"))
        finally:
            # Large trajectories + BEAST logs stay in RAM otherwise.
            try:
                del data  # noqa: F821
            except NameError:
                pass
            gc.collect()

    if not rows:
        print("No simulations produced usable diagnostics.")
        return

    df_all = pd.DataFrame(rows).sort_values("simid").reset_index(drop=True)
    df_all.to_csv(out_dir / "all_sims_sigma_summary.csv", index=False)

    problems = df_all[~df_all["sigma_true_in_HPD"]].copy()
    # Smallest width / σ̃ first — the sims whose narrow HPD most tightly
    # excludes the truth are the most informative failures.
    problems.sort_values("hpd_width_over_sigma_tilde", inplace=True)
    problems.head(cli.max_problem_sims).to_csv(
        out_dir / "problem_sims_sigma_summary.csv", index=False
    )

    plot_median_vs_tilde(df_all, out_dir / "sigma_median_vs_sigma_tilde.png")
    plot_hpd_width_vs_tilde(df_all, out_dir / "sigma_hpd_width_vs_sigma_tilde.png")
    plot_sigma_ratio_vs_n(df_all, out_dir / "sigma_ratio_vs_n.png")
    linear_fit = plot_linearised_overfit(
        df_all, out_dir / "sigma_linearised_overfit.png"
    )
    print(
        "linearised overfit fit: "
        f"df_eff ≈ {linear_fit['df_eff_slope']:.2f} "
        f"(95% CI [{linear_fit['df_eff_slope_ci_lo']:.2f}, "
        f"{linear_fit['df_eff_slope_ci_hi']:.2f}], "
        f"{linear_fit['n_sims']} sims)"
    )

    print("\n--- Summary ---")
    print(f"sims summarised      : {len(df_all)}")
    print(f"sigma_true outside HPD: {len(problems)}")
    print(f"written               : {out_dir}/all_sims_sigma_summary.csv")
    print(f"                      : {out_dir}/problem_sims_sigma_summary.csv")
    print(f"                      : {out_dir}/sigma_median_vs_sigma_tilde.png")
    print(f"                      : {out_dir}/sigma_hpd_width_vs_sigma_tilde.png")
    if failures:
        print(f"failures/skips        : {len(failures)}")
        for simid, reason in failures:
            print(f"  {simid}: {reason}")


if __name__ == "__main__":
    main()
