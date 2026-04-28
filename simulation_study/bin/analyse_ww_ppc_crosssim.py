#!/usr/bin/env python3
"""
Aggregate wastewater posterior predictive check summaries across simulations.

Each per-sim PPC directory is expected to contain the two CSVs that
``plot_wastewater_ppc`` writes:

    ww_ppc_summary_scalars.csv   one row per scope: per_deme and pooled
    ww_ppc_per_obs.csv           one row per observation (optional here,
                                  used only for the pooled residuals-vs-time
                                  scatter)

Usage
-----
    analyse_ww_ppc_crosssim.py \
        --ppc_dirs sandbox/*/ww_ppc \
        --output_dir sandbox/ww_ppc_crosssim

The sim identifier for each directory defaults to the parent directory name
(``sandbox/20_2_simulation/ww_ppc`` → ``20_2_simulation``). Pass
``--sim_id_from parent`` (default) or ``--sim_id_from self`` to switch to the
directory's own name.

Outputs (all PNG + PDF for plots):
  * ``crosssim_scalars_raw.csv``       concatenated per-sim per-scope rows
  * ``crosssim_scalars_summary.csv``   across-sim mean + 95% CI per metric per role
  * ``crosssim_scalars_stripplot``     strip plot of physical scalars per role
  * ``crosssim_sbc_pvalues``           SBC histogram of Bayesian p-values per role
  * ``crosssim_std_resid_vs_time``     pooled standardized residuals vs time
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_utils import configure_pdf_fonts, save_figure_png_and_pdf


# Physical scalars shown as strip plots with a target reference line.
PHYSICAL_SCALARS = [
    ("mean_pit", 0.5, "mean PIT"),
    ("var_pit", 1.0 / 12.0, "var PIT"),
    ("mean_std_resid", 0.0, "mean std resid"),
    ("sd_std_resid", 1.0, "sd std resid"),
    ("cov_50", 0.5, "50% PPC coverage"),
    ("cov_95", 0.95, "95% PPC coverage"),
    ("ks_pit", None, "KS(PIT, U(0,1))"),
]

# Bayesian p-values checked via SBC uniformity across sims.
SBC_PVALUES = ["p_T_mean", "p_T_sd", "p_T_min", "p_T_max"]

ROLE_ORDER = ["start", "secondary", "pooled"]
ROLE_COLOUR = {
    "start": "tab:blue",
    "secondary": "tab:orange",
    "pooled": "tab:purple",
}


def expand_dirs(patterns):
    """Expand shell globs and dedupe while preserving sorted order."""
    dirs = []
    for p in patterns:
        if any(ch in p for ch in "*?["):
            matches = glob.glob(p)
            if not matches:
                print(f"[warn] no matches for pattern: {p}")
            dirs.extend(matches)
        else:
            dirs.append(p)
    return sorted({str(Path(d)) for d in dirs})


def sim_id_from_dir(path, mode):
    p = Path(path)
    if mode == "self":
        return p.name
    # Default: parent directory name.
    return p.parent.name


def load_scalars(dirs, sim_id_mode):
    """Concatenate per-sim scalar CSVs and add a sim_id column."""
    frames = []
    for d in dirs:
        p = Path(d) / "ww_ppc_summary_scalars.csv"
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        df = pd.read_csv(p)
        df.insert(0, "sim_id", sim_id_from_dir(d, sim_id_mode))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_per_obs(dirs, sim_id_mode):
    frames = []
    for d in dirs:
        p = Path(d) / "ww_ppc_per_obs.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.insert(0, "sim_id", sim_id_from_dir(d, sim_id_mode))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def across_sim_summary(raw):
    """Per (role, metric): n_sims, mean, SE, 95% t-CI, target, pass flag."""
    metric_targets = {name: tgt for name, tgt, _ in PHYSICAL_SCALARS}
    # SBC p-values: the *mean* across sims has target 0.5 under uniformity.
    for p in SBC_PVALUES:
        metric_targets[p] = 0.5
    rows = []
    for role, role_df in raw.groupby("role"):
        for metric, target in metric_targets.items():
            if metric not in role_df.columns:
                continue
            vals = role_df[metric].dropna().to_numpy()
            n = vals.size
            mean = float(np.mean(vals)) if n else float("nan")
            if n > 1:
                se = float(np.std(vals, ddof=1) / np.sqrt(n))
                half = 1.96 * se
                ci_lo, ci_hi = mean - half, mean + half
            else:
                se = ci_lo = ci_hi = float("nan")
            if target is None or not np.isfinite(ci_lo):
                within = ""
            else:
                within = bool(ci_lo <= target <= ci_hi)
            rows.append(
                {
                    "role": role,
                    "metric": metric,
                    "n_sims": n,
                    "mean": mean,
                    "se_mean": se,
                    "ci_lo_95": ci_lo,
                    "ci_hi_95": ci_hi,
                    "target": target if target is not None else float("nan"),
                    "target_within_ci": within,
                }
            )
    return pd.DataFrame(rows)


def jitter(n, width=0.18, rng=None):
    rng = rng or np.random.default_rng(0)
    return (rng.random(n) - 0.5) * 2 * width


def plot_scalars_stripplot(raw, output_dir):
    """One subplot per physical scalar; x = role with jittered dots per sim."""
    roles_present = [r for r in ROLE_ORDER if r in set(raw["role"].unique())]
    if not roles_present:
        print("No roles found; skipping strip plot.")
        return
    n_metrics = len(PHYSICAL_SCALARS)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    rng = np.random.default_rng(0)
    for idx, (metric, target, nice) in enumerate(PHYSICAL_SCALARS):
        ax = axes[idx // n_cols, idx % n_cols]
        for ri, role in enumerate(roles_present):
            vals = raw.loc[raw["role"] == role, metric].dropna().to_numpy()
            if vals.size == 0:
                continue
            xs = ri + jitter(vals.size, rng=rng)
            ax.scatter(
                xs,
                vals,
                s=22,
                color=ROLE_COLOUR.get(role, "grey"),
                alpha=0.75,
                edgecolor="black",
                linewidth=0.3,
            )
            # overlay median as a short horizontal bar
            med = float(np.median(vals))
            ax.plot([ri - 0.25, ri + 0.25], [med, med], color="black", lw=1.5, zorder=3)
        if target is not None:
            ax.axhline(
                target, color="red", ls="--", lw=1.0, label=f"target = {target:.3g}"
            )
            ax.legend(fontsize=7, loc="best")
        ax.set_xticks(range(len(roles_present)))
        ax.set_xticklabels(roles_present)
        ax.set_ylabel(nice)
        ax.set_title(nice)
        ax.grid(True, axis="y", alpha=0.2)
    # Hide any unused subplots.
    for idx in range(n_metrics, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig.suptitle("Per-sim PPC scalars across simulations", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure_png_and_pdf(str(output_dir / "crosssim_scalars_stripplot.png"))
    plt.close(fig)


def plot_sbc_pvalues(raw, output_dir):
    """SBC-style histogram: Bayesian p-values across sims should be U(0,1)."""
    roles_present = [r for r in ROLE_ORDER if r in set(raw["role"].unique())]
    if not roles_present:
        return
    n_rows = len(roles_present)
    n_cols = len(SBC_PVALUES)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )
    for ri, role in enumerate(roles_present):
        role_df = raw[raw["role"] == role]
        for ci, pname in enumerate(SBC_PVALUES):
            ax = axes[ri, ci]
            vals = role_df[pname].dropna().to_numpy()
            if vals.size == 0:
                ax.set_title(f"{role}: {pname} (no data)")
                continue
            n_bins = min(20, max(5, vals.size // 3))
            expected = vals.size / n_bins
            ax.hist(
                vals,
                bins=n_bins,
                range=(0, 1),
                color=ROLE_COLOUR.get(role, "grey"),
                alpha=0.6,
                edgecolor="black",
                linewidth=0.4,
            )
            ax.axhline(
                expected,
                color="red",
                ls="--",
                lw=1,
                label=f"expected count (n/{n_bins})",
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel(pname)
            if ci == 0:
                ax.set_ylabel(f"{role}\ncount", fontsize=9)
            else:
                ax.set_ylabel("count")
            if ri == 0:
                ax.set_title(pname, fontsize=10)
            ax.legend(fontsize=6, loc="best")
    fig.suptitle(
        "Simulation-based calibration: Bayesian p-values across sims "
        "(flat => calibrated; U-shape => over-confident posterior; "
        "dome => under-confident; tilt => systematic bias)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure_png_and_pdf(str(output_dir / "crosssim_sbc_pvalues.png"))
    plt.close(fig)


def plot_std_resid_vs_time(per_obs, output_dir, starting_deme_col="deme"):
    """Cross-sim standardized residuals vs time.

    Three stacked panels sharing the time axis:
      * Row 0: pooled scatter, coloured by role (both roles overlaid).
      * Row 1: per-day boxplot of std residuals for the ``start`` role.
      * Row 2: per-day boxplot of std residuals for the ``secondary`` role.

    Boxplots bin observations by rounded day (nearest integer) so each box
    summarises residuals across all sims for that day. Days with <2 finite
    residuals still get a single-dot box drawn at the right position.

    Roles are inferred from per-sim scalars where possible; absent that, we
    fall back to labelling by deme index.
    """
    if per_obs.empty or "std_resid" not in per_obs.columns:
        return
    if "t_obs_days" not in per_obs.columns:
        return
    if "role" in per_obs.columns:
        df = per_obs
    else:
        df = per_obs.copy()
        df["role"] = df[starting_deme_col].astype(str)

    roles_present = sorted(
        df["role"].dropna().unique(),
        key=lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else 99,
    )

    box_roles = ["start", "secondary"]
    t_all = df["t_obs_days"].to_numpy()
    x_lo = float(np.floor(t_all.min() - 0.5)) if t_all.size else 0.0
    x_hi = float(np.ceil(t_all.max() + 0.5)) if t_all.size else 1.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # --- Row 0: pooled scatter ---
    ax = axes[0]
    ax.axhline(0, color="grey", lw=0.5)
    for y in (-2, -1, 1, 2):
        ax.axhline(y, color="grey", lw=0.3, ls=":")
    for role in roles_present:
        sub = df[df["role"] == role]
        ax.scatter(
            sub["t_obs_days"],
            sub["std_resid"],
            s=10,
            color=ROLE_COLOUR.get(role, "grey"),
            alpha=0.45,
            label=f"{role} (n={len(sub)})",
        )
    ax.set_ylabel(
        r"std resid $(\log y_{obs} - \mathrm{med}\,\log\mu_{rep})/\mathrm{med}\,\sigma$"
    )
    ax.set_title("Pooled standardized residuals vs time, across simulations")
    ax.legend(fontsize=8, loc="best")

    # --- Rows 1 & 2: per-day boxplots per role ---
    # Boxplots are restricted to days where every sim contributing to this
    # role has at least one observation, so the boxes are directly
    # comparable (identical sim cohort in each box).
    for row_idx, role in enumerate(box_roles, start=1):
        ax = axes[row_idx]
        ax.axhline(0, color="grey", lw=0.5)
        for y in (-2, -1, 1, 2):
            ax.axhline(y, color="grey", lw=0.3, ls=":")
        ax.set_ylabel(f"{role}\nstd resid", fontsize=9)
        sub = df[df["role"] == role]
        if sub.empty:
            ax.set_title(f"{role} — std residuals per day (no data)")
            continue
        if "sim_id" not in sub.columns:
            ax.set_title(f"{role} — std residuals per day (sim_id missing)")
            continue
        day = np.round(sub["t_obs_days"].to_numpy()).astype(int)
        z = sub["std_resid"].to_numpy()
        sims = sub["sim_id"].to_numpy()
        total_sims = int(sub["sim_id"].nunique())

        unique_days = np.sort(np.unique(day))
        data, positions, kept_sims = [], [], []
        for d in unique_days:
            mask = (day == d) & np.isfinite(z)
            if mask.sum() == 0:
                continue
            sims_today = set(sims[mask])
            if len(sims_today) < total_sims:
                continue
            data.append(z[mask])
            positions.append(int(d))
            kept_sims.append(sims[mask])
        if not data:
            ax.set_title(
                f"{role} — std residuals per day "
                f"(no days with obs from all {total_sims} sims)"
            )
            continue

        colour = ROLE_COLOUR.get(role, "grey")
        ax.boxplot(
            data,
            positions=positions,
            widths=0.7,
            showfliers=True,
            patch_artist=True,
            boxprops=dict(facecolor=colour, alpha=0.35, edgecolor=colour),
            whiskerprops=dict(color=colour),
            capprops=dict(color=colour),
            medianprops=dict(color="black", linewidth=1.2),
            flierprops=dict(
                marker="o",
                markersize=2.5,
                markerfacecolor=colour,
                markeredgecolor=colour,
                alpha=0.5,
            ),
        )

        # Mean of per-sim means (restricted to the days shown), matching the
        # semantics of ``mean_std_resid`` in the strip plot.
        all_vals = np.concatenate(data)
        all_sims = np.concatenate(kept_sims)
        per_sim_mean = pd.Series(all_vals).groupby(all_sims).mean()
        mean_line = float(per_sim_mean.mean())
        ax.axhline(
            mean_line,
            color="red",
            ls="--",
            lw=1.1,
            label=f"mean of per-sim means = {mean_line:.3g}",
        )
        ax.legend(fontsize=7, loc="best")

        n_total = int(all_vals.size)
        ax.set_title(
            f"{role} — std residuals per day "
            f"({len(positions)} days with obs from all {total_sims} sims, "
            f"n={n_total} obs)"
        )

    axes[-1].set_xlabel("Days from simulation start")
    axes[-1].set_xlim(x_lo, x_hi)
    # Matplotlib's boxplot sets a tick per position (one per day here), which
    # gets crowded. Anchor ticks to round multiples instead.
    span = x_hi - x_lo
    step = 1 if span <= 20 else (5 if span <= 60 else 10)
    axes[-1].xaxis.set_major_locator(MultipleLocator(step))

    fig.tight_layout()
    save_figure_png_and_pdf(str(output_dir / "crosssim_std_resid_vs_time.png"))
    plt.close(fig)


def plot_sigma_decomposition(raw, output_dir):
    """Pooled σ recovery ratios across sims.

    σ is shared across demes within a single simulation, so the per-role
    (start/secondary) views aren't meaningful — they'd just show the same
    σ_post divided by the same σ_true with slightly different v_logmu_obs.
    Only the pooled scope is plotted.

    A single panel with two scatter series vs sim index (sorted by σ_true):

      * σ_post / σ_true           — what the posterior recovers
      * σ_reconstructed / σ_true  — posterior + variance the spline absorbed

    Dashed line at y=1 is the target. If the blue series (reconstructed)
    lands on 1 while red (σ_post alone) sits below, variance reallocation
    to the spline fully explains the σ underestimation.
    """
    needed = [
        "sigma_true",
        "sigma_post_over_true",
        "sigma_reconstructed_over_true",
    ]
    if not all(c in raw.columns for c in needed):
        print("[skip] sigma decomposition columns missing; skipping plot.")
        return

    sub = raw[raw["role"] == "pooled"].copy()
    sub = sub.dropna(subset=needed)
    if sub.empty:
        print("[skip] no pooled rows with sigma decomposition; skipping plot.")
        return

    sub = sub.sort_values(["sigma_true", "sim_id"]).reset_index(drop=True)
    x = np.arange(len(sub))

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
    ax.axhline(1.0, color="red", ls="--", lw=1.0, label="target = 1")

    # 95% HPD whiskers on each ratio, obtained from the sigma HPD endpoints
    # divided by sigma_true.
    sig_true_per = sub["sigma_true"].to_numpy()
    r_post_per = sub["sigma_post_over_true"].to_numpy()
    r_recon_per = sub["sigma_reconstructed_over_true"].to_numpy()
    post_lo_per = sub["sigma_post_hpd_lower"].to_numpy() / sig_true_per
    post_hi_per = sub["sigma_post_hpd_upper"].to_numpy() / sig_true_per
    has_recon_hpd = (
        "sigma_reconstructed_hpd_lower" in sub.columns
        and "sigma_reconstructed_hpd_upper" in sub.columns
    )
    post_yerr_per = np.vstack([
        np.clip(r_post_per - post_lo_per, 0, None),
        np.clip(post_hi_per - r_post_per, 0, None),
    ])
    ax.errorbar(
        x, r_post_per, yerr=post_yerr_per,
        fmt="o", color="tab:red", ms=4,
        markeredgecolor="black", markeredgewidth=0.3,
        elinewidth=0.5, capsize=1.0, alpha=0.75,
        label=r"$\sigma_{post}/\sigma_{true}$ (median, 95% HPD)",
    )
    if has_recon_hpd:
        recon_lo_per = sub["sigma_reconstructed_hpd_lower"].to_numpy() / sig_true_per
        recon_hi_per = sub["sigma_reconstructed_hpd_upper"].to_numpy() / sig_true_per
        recon_yerr_per = np.vstack([
            np.clip(r_recon_per - recon_lo_per, 0, None),
            np.clip(recon_hi_per - r_recon_per, 0, None),
        ])
        ax.errorbar(
            x, r_recon_per, yerr=recon_yerr_per,
            fmt="D", color="tab:blue", ms=4,
            markeredgecolor="black", markeredgewidth=0.3,
            elinewidth=0.5, capsize=1.0, alpha=0.75,
            label=r"$\sigma_{reconstructed}/\sigma_{true}$ (median, 95% HPD)",
        )
    else:
        ax.scatter(
            x, r_recon_per, s=34, marker="D", color="tab:blue",
            alpha=0.85, edgecolor="black", linewidth=0.3,
            label=r"$\sigma_{reconstructed}/\sigma_{true}$",
        )

    ax.set_xlabel(r"Simulation index (sorted by $\sigma_{true}$)")
    ax.set_ylabel(r"ratio to $\sigma_{true}$")
    ax.set_title(f"σ recovery ratios across {len(sub)} sims (pooled)")
    ax.set_xticks([])
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=9, loc="best")

    fig.tight_layout()
    save_figure_png_and_pdf(str(output_dir / "crosssim_sigma_decomposition.png"))
    plt.close(fig)

    # --- Paired ratios summary (one panel, two columns) ---
    # A compact view of the decomposition plot: each sim contributes two
    # dots (σ_post/σ_true, σ_reconstructed/σ_true) connected by a thin
    # grey line. Vertical whiskers are the 95% HPD on each ratio, obtained
    # from the sigma HPD endpoints divided by sigma_true.
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.8))
    ax.axhline(1.0, color="red", ls="--", lw=1.0, label="target = 1", zorder=0)

    sig_true = sub["sigma_true"].to_numpy()
    r_post = sub["sigma_post_over_true"].to_numpy()
    r_recon = sub["sigma_reconstructed_over_true"].to_numpy()
    post_lo = sub["sigma_post_hpd_lower"].to_numpy() / sig_true
    post_hi = sub["sigma_post_hpd_upper"].to_numpy() / sig_true
    post_yerr = np.vstack([np.clip(r_post - post_lo, 0, None),
                           np.clip(post_hi - r_post, 0, None)])
    if has_recon_hpd:
        recon_lo = sub["sigma_reconstructed_hpd_lower"].to_numpy() / sig_true
        recon_hi = sub["sigma_reconstructed_hpd_upper"].to_numpy() / sig_true
        recon_yerr = np.vstack([np.clip(r_recon - recon_lo, 0, None),
                                np.clip(recon_hi - r_recon, 0, None)])
    else:
        print(
            "[note] sigma_reconstructed_hpd_{lower,upper} missing from CSVs; "
            "plotting reconstructed ratios without whiskers. "
            "Re-run per-sim PPC to regenerate with new schema."
        )

    jitter_post = jitter(len(sub), width=0.12, rng=rng)
    jitter_recon = jitter(len(sub), width=0.12, rng=rng)
    x_post = 0.0 + jitter_post
    x_recon = 1.0 + jitter_recon

    # Paired connectors (between medians), drawn first so dots sit on top.
    for xp, xr, yp, yr in zip(x_post, x_recon, r_post, r_recon):
        ax.plot([xp, xr], [yp, yr], color="grey", lw=0.5, alpha=0.4, zorder=1)

    ax.errorbar(
        x_post, r_post, yerr=post_yerr,
        fmt="o", color="tab:red", ms=5,
        markeredgecolor="black", markeredgewidth=0.3,
        elinewidth=0.5, capsize=1.2, alpha=0.7,
        label=r"$\sigma_{post}/\sigma_{true}$ (median, 95% HPD)", zorder=2,
    )
    if has_recon_hpd:
        ax.errorbar(
            x_recon, r_recon, yerr=recon_yerr,
            fmt="D", color="tab:blue", ms=5,
            markeredgecolor="black", markeredgewidth=0.3,
            elinewidth=0.5, capsize=1.2, alpha=0.7,
            label=r"$\sigma_{reconstructed}/\sigma_{true}$ (median, 95% HPD)",
            zorder=2,
        )
    else:
        ax.scatter(
            x_recon, r_recon, s=34, marker="D", color="tab:blue",
            alpha=0.7, edgecolor="black", linewidth=0.3,
            label=r"$\sigma_{reconstructed}/\sigma_{true}$", zorder=2,
        )

    # Median-of-medians bars per column.
    for x0, vals in [(0.0, r_post), (1.0, r_recon)]:
        med = float(np.median(vals))
        ax.plot([x0 - 0.25, x0 + 0.25], [med, med],
                color="black", lw=1.8, zorder=3)

    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([
        r"$\sigma_{post}/\sigma_{true}$",
        r"$\sigma_{reconstructed}/\sigma_{true}$",
    ])
    ax.set_ylabel(r"ratio to $\sigma_{true}$")
    ax.set_title(
        f"σ recovery: posterior vs spline-variance reconstruction "
        f"({len(sub)} sims, pooled)"
    )
    ax.set_xlim(-0.5, 1.5)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    save_figure_png_and_pdf(str(output_dir / "crosssim_sigma_ratios.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate wastewater PPC summaries across simulations."
    )
    parser.add_argument(
        "--ppc_dirs",
        nargs="+",
        required=True,
        help=(
            "One or more per-sim PPC output directories. Shell globs are "
            "supported (shell-expanded or passed quoted)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write concatenated CSVs and plots into.",
    )
    parser.add_argument(
        "--sim_id_from",
        choices=["parent", "self"],
        default="parent",
        help=(
            "Where to take each sim's identifier from: the PPC dir's parent "
            "(default; so sandbox/20_2/ww_ppc → '20_2') or the PPC dir itself."
        ),
    )
    args = parser.parse_args()

    configure_pdf_fonts()

    dirs = expand_dirs(args.ppc_dirs)
    if not dirs:
        print("No PPC directories matched; nothing to do.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_scalars(dirs, args.sim_id_from)
    if raw.empty:
        print("No ww_ppc_summary_scalars.csv files found; exiting.")
        return
    raw.to_csv(output_dir / "crosssim_scalars_raw.csv", index=False)

    summary = across_sim_summary(raw)
    summary.to_csv(output_dir / "crosssim_scalars_summary.csv", index=False)

    plot_scalars_stripplot(raw, output_dir)
    plot_sbc_pvalues(raw, output_dir)
    plot_sigma_decomposition(raw, output_dir)

    per_obs = load_per_obs(dirs, args.sim_id_from)
    if not per_obs.empty:
        # Stamp roles onto per-obs rows from the (sim_id, deme) mapping in raw.
        role_map = raw[raw["scope"] == "per_deme"][
            ["sim_id", "deme", "role"]
        ].drop_duplicates()
        per_obs = per_obs.merge(role_map, on=["sim_id", "deme"], how="left")
        plot_std_resid_vs_time(per_obs, output_dir)

    n_sims = raw["sim_id"].nunique()
    print(f"Processed {n_sims} sim(s). Wrote outputs to {output_dir}.")


if __name__ == "__main__":
    main()
