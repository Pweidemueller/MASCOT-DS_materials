#!/usr/bin/env python3
"""Supplementary figure: wastewater sigma over-estimation.

Three panels:

  A  Posterior-predictive coverage. One dot per simulation (1-D jitter strip),
     y = coverage_95 (fraction of true wastewater obs inside the 95 % posterior
     predictive interval). Median-across-sims line + 0.95 nominal line.

  B  Zoom on one simulation (default 7_2), early time points, start deme.
     Top: log predicted mean concentration log(k_ww * I / N) for the inferred
     (posterior-median) and expected (true) parameters, with the log wastewater
     observations as black dots. Bottom (shared x): the three standardized-
     residual variants of those observations vs time.

  C  Residual RMS (in true-sigma units) vs true prevalence I/N. Per prevalence
     bin, RMS of the log-residual / sigma_true around the inferred mean (rises
     at low prevalence) and around the true mean (flat ~1). Scaling by the true
     noise keeps the sigma inflation visible; reference lines mark sigma_true
     (=1) and the mean posterior sigma_post/sigma_true.

A separate investigative plot (ww_sigma_mean_misfit.png) shows RMS(Delta /
sigma_true) vs true prevalence, Delta = ln mu_true - ln mu_post.

Both panels B and C read only the per-obs CSV from ``compute_ww_ppc_persim.py``
(Panel A reads the per-sim summary CSV).

Usage:
    make_figure_ww_sigma_supp.py \
        --summary_csv <ww_ppc_persim_summary.csv> \
        --per_obs_csv <ww_ppc_per_obs.csv> \
        --output_dir <dir> [--panel_b_simid 7_2] [--panel_b_max_days 15] \
        [--panel_b_deme N]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constants import COLORS  # noqa: E402
from make_figure_simstudy import add_panel_label  # noqa: E402
from plot_utils import (  # noqa: E402
    FONTSIZES_LIST,
    configure_pdf_fonts,
    save_figure_png_and_pdf,
    save_plot_data_csv,
    set_axis_fontsizes,
)

INFERRED_COLOR = COLORS[3]  # orange -- inferred mean & sigma
TRUE_COLOR = COLORS[0]  # light blue -- true (expected) mean & sigma
NOMINAL = 0.95
Z95 = 1.959963984540054  # central 95 % of a standard normal
DAYS_PER_YEAR = 365.0

# The standardized-residual variants shared by Panels B and C:
#   z = (ln obs - ln mu) / sigma
# distinguished by which mean (mu) and which sigma go in.
RESIDUAL_VARIANTS = [
    ("z_inf", INFERRED_COLOR, "Inferred"),  # inferred mean, inferred sigma
    ("z_true", TRUE_COLOR, "Expected"),  # true mean, true sigma
]


# --------------------------------------------------------------------------- #
# Panel A                                                                     #
# --------------------------------------------------------------------------- #


def plot_panel_a_coverage(ax, df, *, seed=0):
    """1-D jitter strip of per-sim coverage_95 with median + 0.95 lines.

    No legend/title; the median and nominal reference lines span only the jitter
    width rather than the whole axis.
    """
    cov = df["coverage_95"].to_numpy(dtype=float)
    cov = cov[np.isfinite(cov)]
    jitter_hw = 0.18
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-jitter_hw, jitter_hw, size=cov.size)

    ax.scatter(
        jitter, cov, s=22, alpha=0.7, color=INFERRED_COLOR, edgecolor="none", zorder=3
    )
    line_hw = jitter_hw + 0.06  # slightly wider than the jitter cloud
    median_cov = float(np.median(cov))
    ax.hlines(median_cov, -line_hw, line_hw, color="0.15", lw=1.4, zorder=4)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylabel("Fraction of true obs in 95% predictive interval")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    set_axis_fontsizes(ax, FONTSIZES_LIST)


# --------------------------------------------------------------------------- #
# Panel C                                                                     #
# --------------------------------------------------------------------------- #


def _binned_rms(x_log10, r, edges):
    """Per-bin RMS of r over log10(x) bins.

    Returns (centers, rms) with centers on the linear (un-logged) scale and NaN
    for bins holding < 2 finite values.
    """
    idx = np.digitize(x_log10, edges) - 1
    n_bins = len(edges) - 1
    centers = 10 ** (0.5 * (edges[:-1] + edges[1:]))
    rms = np.full(n_bins, np.nan)
    for b in range(n_bins):
        vals = r[idx == b]
        vals = vals[np.isfinite(vals)]
        if vals.size >= 2:
            rms[b] = float(np.sqrt(np.mean(vals**2)))
    return centers, rms


def _residual_frame(df_obs):
    """Add the three standardized-residual variants to a per-obs frame copy.

    z = (ln obs - ln mu) / sigma, with:
      z_inf   inferred mean (log_mu_post), inferred sigma (sigma_post_median)
      z_true  true mean (log_mu_true), true sigma (sigma_true)
    """
    d = df_obs.copy()
    d["z_inf"] = (d["log_obs"] - d["log_mu_post"]) / d["sigma_post_median"]
    d["z_true"] = (d["log_obs"] - d["log_mu_true"]) / d["sigma_true"]
    return d


def _true_prevalence_bins(df_obs, n_bins):
    """Filter to positive true prevalence and return (d, x_log10, edges)."""
    d = df_obs.copy()
    x = d["I_over_N_true"].to_numpy(dtype=float)
    keep = np.isfinite(x) & (x > 0)
    d = d[keep]
    x_log10 = np.log10(x[keep])
    edges = np.linspace(x_log10.min(), x_log10.max(), n_bins + 1)
    return d, x_log10, edges


def plot_panel_c_residual_vs_prevalence(ax, df_obs):
    """Residual RMS (in units of the true sigma) vs true prevalence count I.

    Observations are pooled across sims into decade bins of the true prevalence
    count I ([1,10), [10,100), ...). Per bin, the RMS of (obs_ww - mu)/sigma_true
    for mu = the inferred mean (log_mu_post) and the true mean (log_mu_true).
    Scaling by the *true* (known) noise -- never sigma_post -- keeps the sigma
    inflation visible:

      Expected (true mean):    RMS ~ 1, flat  (data noise, prevalence-independent)
      Inferred (inferred mean): RMS > 1 at low prevalence  (mean misfit)
    """
    d = df_obs.copy()
    I = d["I_true"].to_numpy(dtype=float)
    keep = np.isfinite(I) & (I >= 1)
    d = d[keep]
    x_log10 = np.log10(I[keep])
    # Decade bins on the count I: [1,10), [10,100), ...
    k_min = int(np.floor(x_log10.min()))
    k_max = int(np.ceil(x_log10.max()))
    edges = np.arange(k_min, k_max + 1, dtype=float)
    n_bins = len(edges) - 1
    positions = np.arange(n_bins, dtype=float)

    sig_true = d["sigma_true"].to_numpy(dtype=float)
    r_inf = (d["log_obs"] - d["log_mu_post"]).to_numpy(dtype=float) / sig_true
    r_true = (d["log_obs"] - d["log_mu_true"]).to_numpy(dtype=float) / sig_true

    for r, color, label in [
        (r_inf, INFERRED_COLOR, "Inferred"),
        (r_true, TRUE_COLOR, "Expected"),
    ]:
        _, rms = _binned_rms(x_log10, r, edges)
        ax.plot(
            positions,
            rms,
            color=color,
            lw=1.8,
            marker="o",
            ms=3,
            zorder=3,
            label=label,
        )

    # True noise floor at 1 (grey line, no legend entry).
    ax.axhline(1.0, color="grey", lw=0.8, zorder=1)

    # Bin labels [1,10), [10,100), ...; the last bin closes on the actual max I
    # (not the next power of ten) and uses ']' to show it is inclusive.
    max_I = int(np.nanmax(I[keep]))
    labels = []
    for i in range(n_bins):
        lo = int(10 ** edges[i])
        if i == n_bins - 1:
            labels.append(f"[{lo}, {max_I}]")
        else:
            labels.append(f"[{lo}, {int(10 ** edges[i + 1])})")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.set_xlabel("True prevalence  I")
    ax.set_ylabel(r"RMS$\left[(\mathrm{obs}_{ww}-\mu)/\sigma_{\mathrm{true}}\right]$")
    ax.legend(fontsize=FONTSIZES_LIST[2], loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    set_axis_fontsizes(ax, FONTSIZES_LIST)


# --------------------------------------------------------------------------- #
# Panel B                                                                     #
# --------------------------------------------------------------------------- #


def _pick_panel_b_deme(d, max_days):
    """Start deme = the one with the most observations in the zoom window."""
    win = d[d["t_years"] * DAYS_PER_YEAR <= max_days]
    src = win if not win.empty else d
    return int(src["deme"].value_counts().idxmax())


def plot_panel_b_zoom(ax_top, ax_bot, df_obs, *, simid, max_days, deme=None):
    """Panel B: single-sim early-phase zoom, entirely from the per-obs CSV.

    Top -- left y-axis: mu = log(k_ww * I / N) for the inferred (log_mu_post) and
    expected/true (log_mu_true) mean curves, with the log wastewater
    observations as black dots; right y-axis: the same on the linear wastewater
    concentration scale (exp of the left axis).

    Bottom (shared x): the residual (obs_ww - mu)/sigma_true for the inferred
    (orange) and expected (blue) means, over a grey standard-normal 95 % band.
    """
    d = df_obs[df_obs["simid"].astype(str) == str(simid)]
    if d.empty:
        _stub_panel(ax_top, f"B ({simid})")
        _stub_panel(ax_bot, "")
        return
    if deme is None:
        deme = _pick_panel_b_deme(d, max_days)
    d = d[d["deme"] == deme].sort_values("t_years")
    t_days = d["t_years"].to_numpy(dtype=float) * DAYS_PER_YEAR
    d = d[t_days <= max_days]
    t_days = t_days[t_days <= max_days]

    # Top: mean curves + observations in log-concentration space.
    ax_top.plot(
        t_days,
        d["log_mu_true"].to_numpy(),
        color=TRUE_COLOR,
        lw=1.8,
        label="Expected",
        zorder=2,
    )
    ax_top.plot(
        t_days,
        d["log_mu_post"].to_numpy(),
        color=INFERRED_COLOR,
        lw=1.8,
        label="Inferred",
        zorder=3,
    )
    ax_top.scatter(
        t_days,
        d["log_obs"].to_numpy(),
        s=16,
        color="black",
        zorder=4,
        label="Wastewater",
    )
    ax_top.set_ylabel(r"$\mu = \log(k_{ww}\cdot I/N)$")
    ax_top.set_xlim(0, max_days)
    ax_top.spines["top"].set_visible(False)
    ax_top.legend(fontsize=FONTSIZES_LIST[2], loc="upper left", frameon=False)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Right axis: the wastewater observations on the same ln scale (identity
    # mirror of the left), rather than an exp-transformed linear-concentration
    # axis.
    sec = ax_top.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
    sec.set_ylabel(r"$\log(\mathrm{Wastewater})$", rotation=270, labelpad=14)
    sec.tick_params(labelsize=FONTSIZES_LIST[2])

    # Bottom: residuals standardized by the TRUE sigma (same scaling as Panel C).
    sig_true = d["sigma_true"].to_numpy(dtype=float)
    r_inf = (d["log_obs"] - d["log_mu_post"]).to_numpy(dtype=float) / sig_true
    r_true = (d["log_obs"] - d["log_mu_true"]).to_numpy(dtype=float) / sig_true
    ax_bot.axhspan(-Z95, Z95, color="grey", alpha=0.15, zorder=0)
    ax_bot.axhline(0.0, color="grey", lw=0.8, zorder=1)
    ax_bot.scatter(t_days, r_true, s=18, color=TRUE_COLOR, edgecolor="none", zorder=3)
    ax_bot.scatter(
        t_days, r_inf, s=18, color=INFERRED_COLOR, edgecolor="none", zorder=4
    )
    ax_bot.set_xlim(0 - 0.4, max_days + 0.4)
    ax_bot.set_xlabel("Time (days)")
    ax_bot.set_ylabel(r"$(\mathrm{obs}_{ww}-\mu)/\sigma_{\mathrm{true}}$")
    ax_bot.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    set_axis_fontsizes(ax_top, FONTSIZES_LIST)
    set_axis_fontsizes(ax_bot, FONTSIZES_LIST)


# --------------------------------------------------------------------------- #
# Layout                                                                      #
# --------------------------------------------------------------------------- #


def _stub_panel(ax, label):
    if label:
        ax.text(
            0.5,
            0.5,
            f"Panel {label}\n(unavailable)",
            ha="center",
            va="center",
            fontsize=FONTSIZES_LIST[1],
            color="0.5",
            transform=ax.transAxes,
        )
    ax.set_xticks([])
    ax.set_yticks([])


def make_figure(
    df_summary,
    df_obs,
    output_dir,
    *,
    panel_b_simid,
    panel_b_max_days,
    panel_b_deme,
):
    configure_pdf_fonts()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.0, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], wspace=0.5, hspace=0.12)

    ax_a = fig.add_subplot(gs[:, 0])
    plot_panel_a_coverage(ax_a, df_summary)
    add_panel_label(ax_a, "A")

    ax_b_top = fig.add_subplot(gs[0, 1])
    ax_b_bot = fig.add_subplot(gs[1, 1], sharex=ax_b_top)
    plot_panel_b_zoom(
        ax_b_top,
        ax_b_bot,
        df_obs,
        simid=panel_b_simid,
        max_days=panel_b_max_days,
        deme=panel_b_deme,
    )
    add_panel_label(ax_b_top, "B")

    ax_c = fig.add_subplot(gs[:, 2])
    plot_panel_c_residual_vs_prevalence(ax_c, df_obs)
    add_panel_label(ax_c, "C")

    output_png = output_dir / "ww_sigma_supp_figure.png"
    save_figure_png_and_pdf(output_png)
    save_plot_data_csv(df_summary, output_png, suffix="panelA_coverage")
    save_plot_data_csv(_residual_frame(df_obs), output_png, suffix="panelC_residuals")
    plt.close(fig)
    print(f"Figure written to {output_png} (+ .pdf, + _data.csv)")


def plot_extra_mean_misfit(df_obs, output_dir, *, n_bins=20):
    """Standalone investigative plot: RMS(Delta / sigma_true) vs true prevalence.

    Delta = ln mu_true - ln mu_post is the pure mean-model discrepancy (no
    observation noise). Its RMS in sigma_true units isolates the lack-of-fit
    that inflates sigma, and should grow toward low prevalence.
    """
    configure_pdf_fonts()
    output_dir = Path(output_dir)
    d, x_log10, edges = _true_prevalence_bins(df_obs, n_bins)
    delta = (d["log_mu_true"] - d["log_mu_post"]).to_numpy(dtype=float)
    delta_over = delta / d["sigma_true"].to_numpy(dtype=float)
    centers, rms = _binned_rms(x_log10, delta_over, edges)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(centers, rms, color=COLORS[4], lw=1.8, marker="o", ms=3)
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("True prevalence  I/N")
    ax.set_ylabel(r"RMS($\Delta/\sigma_{true}$),  $\Delta=\ln\mu_{true}-\ln\mu_{post}$")
    ax.set_title("Mean-model misfit vs prevalence")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    set_axis_fontsizes(ax, FONTSIZES_LIST)
    fig.tight_layout()
    output_png = output_dir / "ww_sigma_mean_misfit.png"
    save_figure_png_and_pdf(output_png)
    plt.close(fig)
    print(f"Extra plot written to {output_png} (+ .pdf)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--summary_csv", required=True, help="ww_ppc_persim_summary.csv"
    )
    parser.add_argument("--per_obs_csv", required=True, help="ww_ppc_per_obs.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--panel_b_simid", default="7_2")
    parser.add_argument("--panel_b_max_days", type=float, default=20.0)
    parser.add_argument(
        "--panel_b_deme",
        type=int,
        default=None,
        help="Deme to show in Panel B (default: the start deme, auto-detected).",
    )
    cli = parser.parse_args()

    df_summary = pd.read_csv(cli.summary_csv)
    df_obs = pd.read_csv(cli.per_obs_csv)
    if "coverage_95" not in df_summary.columns:
        raise ValueError(f"{cli.summary_csv} has no 'coverage_95' column.")
    make_figure(
        df_summary,
        df_obs,
        cli.output_dir,
        panel_b_simid=cli.panel_b_simid,
        panel_b_max_days=cli.panel_b_max_days,
        panel_b_deme=cli.panel_b_deme,
    )
    plot_extra_mean_misfit(df_obs, cli.output_dir)


if __name__ == "__main__":
    main()
