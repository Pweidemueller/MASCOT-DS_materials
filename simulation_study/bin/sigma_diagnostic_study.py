#!/usr/bin/env python3
"""
Aggregate per-simulation σ diagnostics for the wastewater observation model.

For every simulation under ``--results_dir`` this script:
  1. Loads the same inputs ``make_figure_individualsim.py`` uses.
  2. Computes per-observation wastewater residuals via
     ``compute_wastewater_residuals``.
  3. Reduces each sim to a single row of σ diagnostics.
  4. Writes one master CSV plus four diagnostic figures.

Outputs (into ``--output_dir``):
    all_sims_sigma_summary.csv          one row per sim — input for every plot
    sigma_median_vs_sigma_tilde.png     posterior σ median vs empirical σ̃
    sigma_post_vs_sigma_tilde_true.png  posterior σ (median + HPD whiskers)
                                        vs σ̃_true — does the posterior track
                                        the empirical scale on the true mean?
    sigma_hpd_width_vs_sigma_tilde.png  posterior σ HPD width vs σ̃
    sigma_ratio_vs_n.png                σ̂_post,median / σ_true vs n
    sigma_tilde_true_vs_n.png           σ̃_true / σ_true vs n with χ² sampling
                                        bands — quantifies how much of any
                                        apparent σ shortfall is just finite-
                                        sample noise on a known mean.

Definitions of the per-sim quantities written to the CSV:

    n                  number of finite wastewater residuals across both demes.
    sigma_true         ds_ww_sigma from the simulator's parameters CSV. The
                       data-generating log-scale SD on the wastewater
                       likelihood, log(obs) ~ Normal(log(α·I/N), σ²).
    sigma_tilde        sqrt( Σ res_inf² / n ).
                       res_inf = log(obs) − log(μ_post), where μ_post uses the
                       posterior-mean α and the posterior-median I trajectory.
                       This is the MLE of σ if you treated the inferred mean
                       as a fixed truth — i.e. how far the observations land
                       from the model's inferred mean curve. Often referred
                       to as the "RMS of posterior-mean residuals":
                           sqrt(mean(r²))
                       (no centring, because the model already centres each
                       likelihood term at log(μ_post)).
    sigma_tilde_true   sqrt( Σ res_sim² / n ).
                       res_sim = log(obs) − log(μ_true), using the true α and
                       the simulator's stored I trajectory. Under the
                       simulator, res_sim ~ N(0, σ_true²) i.i.d., so
                           n · (σ̃_true / σ_true)²  ~  χ²(n)
                       and σ̃_true is chi-distributed around σ_true with mean
                           E[σ̃_true] = σ_true · sqrt(2/n) · Γ((n+1)/2)/Γ(n/2)
                       which is < σ_true at finite n. This quantifies the
                       finite-sample downward bias of the empirical SD even
                       when the mean is known exactly.
    sigma_post_median  posterior median of wastewater.sigma:SimDataset.
    sigma_post_hpd_lower / _upper   95 % HPD bounds of σ_post.
    hpd_width          σ_post_hpd_upper − σ_post_hpd_lower.
    sigma_true_in_HPD  whether σ_true lies within the 95 % HPD.
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
from scipy.special import gammaln
from scipy.stats import chi2

# Reuse functions from the sibling scripts in this bin/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_posteriors import prepare_skyline_plot_data  # noqa: E402
from make_figure_individualsim import compute_wastewater_residuals  # noqa: E402


# Sigma prior on the wastewater observation model: σ ~ LogNormal(M, S²) on the
# linear scale (M, S parametrise the mean / sd of log σ).
PRIOR_M_LOG = -0.7
PRIOR_S_LOG = 0.3
PRIOR_MEDIAN = float(np.exp(PRIOR_M_LOG))  # ≈ 0.497


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


# --------------------------------------------------------------------------- #
# Per-sim plumbing                                                            #
# --------------------------------------------------------------------------- #


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

    ``probe_dir`` is a scratch directory where ``prepare_skyline_plot_data`` can
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
        out_prefix=str(Path(probe_dir) / f"{simid}"),
    )
    for attr in REQUIRED_FILE_ATTRS:
        if not Path(getattr(args, attr)).exists():
            return None
    return args


def summarise_sim(simid, df_resid, sigma_true, sigma_post):
    """Reduce one sim's per-observation residuals to a single diagnostic row."""
    if df_resid is None or df_resid.empty:
        return None
    if sigma_true is None or sigma_post is None:
        return None

    res_inf = df_resid["res_inf"].to_numpy()
    res_sim = df_resid["res_sim"].to_numpy()
    keep = np.isfinite(res_inf) & np.isfinite(res_sim)
    res_inf = res_inf[keep]
    res_sim = res_sim[keep]
    n = int(res_inf.size)
    if n == 0:
        return None

    # Empirical scales of the residuals. Both are RMS values (uncentred): the
    # likelihood is centred at log(μ) so the residuals already have an assumed
    # mean of zero, and dividing by n (not n-1) matches the σ MLE under that
    # assumption.
    sigma_tilde = float(np.sqrt(np.sum(res_inf ** 2) / n))
    sigma_tilde_true = float(np.sqrt(np.sum(res_sim ** 2) / n))

    hpd_lo = float(sigma_post["hpd_lower"])
    hpd_hi = float(sigma_post["hpd_upper"])
    median = float(sigma_post["median"])

    return {
        "simid": simid,
        "n": n,
        "sigma_true": float(sigma_true),
        "sigma_tilde": sigma_tilde,
        "sigma_tilde_true": sigma_tilde_true,
        "sigma_post_median": median,
        "sigma_post_hpd_lower": hpd_lo,
        "sigma_post_hpd_upper": hpd_hi,
        "hpd_width": hpd_hi - hpd_lo,
        "sigma_true_in_HPD": bool(hpd_lo <= sigma_true <= hpd_hi),
    }


# --------------------------------------------------------------------------- #
# Helpers for the prior-based reference line on the HPD-width plot            #
# --------------------------------------------------------------------------- #


def _expected_hpd_width_factor(n_obs):
    """Expected 95% HPD width on σ as a multiple of σ_median.

    Asymptotic Normal approximation on log σ:
        prior precision on log σ          = 1 / S²
        likelihood Fisher info at the MLE = 2n
        posterior sd(log σ)               ≈ 1 / sqrt(prior_prec + 2n)
        width on σ                        = σ_med · (e^{+z·sd} − e^{−z·sd})
    """
    prior_prec = 1.0 / (PRIOR_S_LOG ** 2)
    post_prec = prior_prec + 2.0 * n_obs
    post_sd_log = 1.0 / np.sqrt(post_prec)
    z = 1.959963984540054  # 97.5 % standard normal
    return float(np.exp(z * post_sd_log) - np.exp(-z * post_sd_log))


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #


def plot_median_vs_tilde(df_all, output_path):
    """σ̂_post,median vs σ̃ — does the posterior locate where the residuals say it should?"""
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


def plot_post_vs_tilde_true(df_all, output_path):
    """σ̂_post (median + 95 % HPD whiskers) vs σ̃_true.

    Same axes as sigma_median_vs_sigma_tilde but with the x-axis swapped to the
    *true-mean* empirical scale σ̃_true (computed from res_sim, which uses the
    simulator's known prevalence and α). If posteriors track σ̃_true rather
    than σ_true, that's evidence the inference is anchored on the realised
    residual scale — and σ̃_true's chi-distributed downward bias at finite n
    propagates straight through to σ̂_post.
    """
    fig, ax = plt.subplots(figsize=(6.8, 6.5))
    covered = df_all["sigma_true_in_HPD"].to_numpy(dtype=bool)
    x = df_all["sigma_tilde_true"].to_numpy(dtype=float)
    y = df_all["sigma_post_median"].to_numpy(dtype=float)
    y_lo = df_all["sigma_post_hpd_lower"].to_numpy(dtype=float)
    y_hi = df_all["sigma_post_hpd_upper"].to_numpy(dtype=float)

    for mask, colour, label in [
        (covered, "tab:blue", fr"$\sigma_{{true}}$ in 95% HPD (n={int(covered.sum())})"),
        (~covered, "tab:red", fr"$\sigma_{{true}}$ outside HPD (n={int((~covered).sum())})"),
    ]:
        if mask.any():
            ax.errorbar(
                x[mask], y[mask],
                yerr=[y[mask] - y_lo[mask], y_hi[mask] - y[mask]],
                fmt="o", ms=4, alpha=0.8, color=colour,
                elinewidth=0.6, capsize=2, label=label,
            )

    vals = np.concatenate([x, y, y_lo, y_hi])
    vals = vals[np.isfinite(vals)]
    lo = float(vals.min()) * 0.9 if vals.size else 0.0
    hi = float(vals.max()) * 1.1 if vals.size else 1.0
    ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1.0, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(
        r"$\tilde{\sigma}_{true}$ (log-scale SD) "
        r"= RMS of true-mean residuals"
    )
    ax.set_ylabel(r"$\hat{\sigma}_{post}$ (median, 95% HPD whiskers)")
    ax.set_title(
        "Posterior σ vs empirical residual scale on the *true* mean\n"
        "y = x means posterior tracks σ̃_true, not σ_true"
    )
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_hpd_width_vs_tilde(df_all, output_path):
    """HPD width vs σ̃ with a single prior-aware reference line at the median n."""
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
    n_ref = int(df_all["n"].median()) if len(df_all) else 40
    factor = _expected_hpd_width_factor(n_ref)
    ax.plot(
        xs, factor * xs, color="tab:purple", ls="-", lw=1.2,
        label=(
            fr"expected width under prior "
            fr"LogNormal($M={PRIOR_M_LOG}$, $S={PRIOR_S_LOG}$), "
            fr"n={n_ref} (median across sims): "
            fr"$\approx {factor:.2f}\,\tilde{{\sigma}}$"
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
    """σ̂_post,median / σ_true vs n with 95 % HPD error bars, coloured by HPD coverage.

    Pure descriptive view: how does posterior σ recovery scale with the number
    of wastewater observations that informed the fit? No theoretical curves —
    those belong on the σ̃_true plot, where the sampling distribution is known.
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

    ax.axhline(1.0, color="grey", lw=0.6)
    ax.set_xlabel(r"$n$ = total wastewater observations")
    ax.set_ylabel(r"$\hat{\sigma}_{post,median} / \sigma_{true}$")
    ax.set_title("Posterior σ recovery vs data volume")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _chi_mean_factor(n):
    """E[σ̃_true / σ_true] for n observations against a known mean.

    σ̃_true² · n / σ_true² ~ χ²(n)  ⇒  σ̃_true / σ_true ~ Chi(n) / sqrt(n)
    E[Chi(n)] = sqrt(2) · Γ((n+1)/2) / Γ(n/2), evaluated stably via gammaln.
    """
    n = np.asarray(n, dtype=float)
    log_factor = 0.5 * np.log(2.0 / n) + gammaln((n + 1) / 2) - gammaln(n / 2)
    return np.exp(log_factor)


def plot_sigma_tilde_true_vs_n(df_all, output_path):
    """σ̃_true / σ_true vs n with χ²-based sampling bands.

    Question: even with the mean known exactly, do the realised wastewater
    observations look like they came from a *narrower* log-normal than the
    true σ_true used to generate them?

    σ̃_true = sqrt(Σ res_sim² / n), where res_sim = log(obs) − log(α_true·I_true/N).
    Under the simulator's data-generating process,
        n · (σ̃_true / σ_true)²  ~  χ²(n)
    so the ratio's sampling distribution depends only on n, with no nuisance
    parameters. The shaded band shows the analytic 2.5–97.5 % envelope and the
    dashed line is the mean E[σ̃_true / σ_true]. Points falling below the band
    are unusually low even relative to what χ² alone allows.

    Most points falling near or below E[…] is the "rarely far from the median"
    story: with finite n on a known mean the empirical SD is downward-biased,
    so observations look like they come from a narrower distribution than the
    σ that actually generated them — and any inference that anchors on the
    empirical residual scale will inherit this bias.
    """
    n = df_all["n"].to_numpy(dtype=float)
    ratio = df_all["sigma_tilde_true"].to_numpy(dtype=float) / df_all[
        "sigma_true"
    ].to_numpy(dtype=float)
    covered = df_all["sigma_true_in_HPD"].to_numpy(dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for mask, colour, label in [
        (covered, "tab:blue", fr"$\sigma_{{true}}$ in 95% HPD (n={int(covered.sum())})"),
        (~covered, "tab:red", fr"$\sigma_{{true}}$ outside HPD (n={int((~covered).sum())})"),
    ]:
        if mask.any():
            ax.scatter(n[mask], ratio[mask], s=26, alpha=0.8, color=colour, label=label)

    n_min = max(2.0, float(np.min(n)) - 1)
    n_max = float(np.max(n)) + 1
    n_grid = np.linspace(n_min, n_max, 400)
    band_lo = np.sqrt(chi2.ppf(0.025, n_grid) / n_grid)
    band_hi = np.sqrt(chi2.ppf(0.975, n_grid) / n_grid)
    band_mean = _chi_mean_factor(n_grid)

    ax.fill_between(
        n_grid, band_lo, band_hi, color="grey", alpha=0.18,
        label=r"χ² 95% sampling band given known mean",
    )
    ax.plot(
        n_grid, band_mean, color="black", ls="--", lw=1.0,
        label=r"$E[\tilde{\sigma}_{true}/\sigma_{true}]$ (chi mean)",
    )
    ax.axhline(1.0, color="grey", lw=0.6)

    frac_below_mean = float(np.mean(ratio < _chi_mean_factor(n)))
    frac_below_band = float(np.mean(ratio < np.sqrt(chi2.ppf(0.025, n) / n)))

    ax.set_xlabel(r"$n$ = total wastewater observations")
    ax.set_ylabel(r"$\tilde{\sigma}_{true} / \sigma_{true}$")
    ax.set_title(
        "Empirical residual SD given the *true* known prevalence vs n\n"
        f"share of sims below chi-mean: {frac_below_mean:.0%}, "
        f"below 2.5% band: {frac_below_band:.0%}"
    )
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results_dir", required=True,
        help="Root results dir with 1_remaster_sim/ and 2_mascot/.",
    )
    parser.add_argument(
        "--output_dir", default="sandbox/sigma_diagnostic",
        help="Where to write CSVs and PNGs.",
    )
    parser.add_argument("--burnin", type=float, default=0.0)
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
            try:
                del data  # noqa: F821
            except NameError:
                pass
            gc.collect()

    if not rows:
        print("No simulations produced usable diagnostics.")
        return

    df_all = pd.DataFrame(rows).sort_values("simid").reset_index(drop=True)

    summary_csv = out_dir / "all_sims_sigma_summary.csv"
    df_all.to_csv(summary_csv, index=False)

    plot_median_vs_tilde(df_all, out_dir / "sigma_median_vs_sigma_tilde.png")
    plot_post_vs_tilde_true(df_all, out_dir / "sigma_post_vs_sigma_tilde_true.png")
    plot_hpd_width_vs_tilde(df_all, out_dir / "sigma_hpd_width_vs_sigma_tilde.png")
    plot_sigma_ratio_vs_n(df_all, out_dir / "sigma_ratio_vs_n.png")
    plot_sigma_tilde_true_vs_n(df_all, out_dir / "sigma_tilde_true_vs_n.png")

    print("\n--- Summary ---")
    print(f"sims summarised       : {len(df_all)}")
    print(f"sigma_true outside HPD: {int((~df_all['sigma_true_in_HPD']).sum())}")
    print(f"written               : {summary_csv}")
    print(f"                      : {out_dir}/sigma_median_vs_sigma_tilde.png")
    print(f"                      : {out_dir}/sigma_post_vs_sigma_tilde_true.png")
    print(f"                      : {out_dir}/sigma_hpd_width_vs_sigma_tilde.png")
    print(f"                      : {out_dir}/sigma_ratio_vs_n.png")
    print(f"                      : {out_dir}/sigma_tilde_true_vs_n.png")
    if failures:
        print(f"failures/skips        : {len(failures)}")
        for simid, reason in failures:
            print(f"  {simid}: {reason}")


if __name__ == "__main__":
    main()
