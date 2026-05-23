#!/usr/bin/env python3
"""
Create publication-ready figures per deme: one PNG and one PDF per (deme, panel).

Each panel type (prevalence, ne, cumincidence) is saved as its own file per deme,
with the title clearly labeling start vs secondary deme. The optional
``*_prevalence_log`` file uses log scale on the prevalence axis only (case counts
and wastewater remain linear). PDFs use
embedded TrueType fonts (pdf.fonttype=42) so text stays editable in Illustrator.
Reuses plotting logic from analyse_posteriors.py.

When ``--intro_figure_out`` is set, also assembles ``final_mascotds_intro_figure``:
a combined figure with a placeholder region (rows 0-1, cols 0-1) for an external
graphic plus one example prevalence panel per deme on row 2 (start at col 0,
secondary at col 1). Placeholder / intro layout is optional so the per-deme
supplementary figures can still be produced on their own.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.stats import kstest, norm

from plot_utils import (
    DEFAULT_FONTSIZES,
    FONTSIZES_LIST,
    configure_pdf_fonts,
    save_figure_png_and_pdf,
)
from analyse_posteriors import (
    calculate_hpd,
    parse_arguments,
    prepare_skyline_plot_data,
    _plot_prevalence_panel,
    _plot_ne_panel,
    _plot_cumincidence_panel,
)


def _deme_label_for_filename(deme: int, starting_deme: int) -> str:
    """Return a short filesystem-safe label for the deme (start_deme or secondary_deme)."""
    return "start_deme" if int(deme) == int(starting_deme) else "secondary_deme"


def _ordered_demes(data):
    """Return [start_deme, secondary_deme] for a two-deme dataset, preserving user order."""
    hpd_original = data["hpd_original"]
    demes = sorted(hpd_original["Deme"].unique())
    if data["starting_deme"] is not None and len(demes) == 2:
        other = [d for d in demes if int(d) != int(data["starting_deme"])]
        if len(other) == 1:
            demes = [int(data["starting_deme"]), int(other[0])]
    return demes


def run_per_deme_figures(data, args, time_unit="days"):
    """
    Build and save one figure per (deme, panel type): prevalence, ne, cumincidence.

    Uses data returned by prepare_skyline_plot_data(args). Saves to
    {args.out_prefix}_deme{N}_{start_deme|secondary_deme}_{prevalence|ne|cumincidence}[_log].png (and .pdf).
    The ``_prevalence_log`` suffix means log-scale prevalence with linear case counts and wastewater.
    """
    time_factor = 365.0 if time_unit == "days" else 1.0
    time_label = "days" if time_unit == "days" else "years"

    hpd_original = data["hpd_original"]
    if hpd_original.empty:
        print("No HPD data; skipping per-deme figures.")
        return

    demes = _ordered_demes(data)

    fontsize_tick = DEFAULT_FONTSIZES["tick_label"]
    hpd_datastream = (
        data["hpd_datastream"] if data["hpd_datastream"] is not None else pd.DataFrame()
    )

    common_kw = {
        "time_factor": time_factor,
        "time_label": time_label,
        "starting_deme": data["starting_deme"],
        "fontsizes": FONTSIZES_LIST,
        "fontsize_tick": fontsize_tick,
    }

    for deme in demes:
        label = _deme_label_for_filename(deme, data["starting_deme"])
        prefix = f"{args.out_prefix}_deme{deme}_{label}"

        # --- Prevalence (linear combined row; separate file with log prevalence only) ---
        fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
        _plot_prevalence_panel(
            axes[0],
            deme,
            hpd_datastream=hpd_datastream,
            trajectory_data=data["trajectory_data"],
            case_counts_data=data["case_counts_data"],
            wastewater_data=data["wastewater_data"],
            validation_data_datastreams_prevalence=data[
                "validation_data_datastreams_prevalence"
            ],
            show_logscale=False,
            show_logscale_prevalence_only=False,
            case_counts_P1=False,
            fig=fig,
            n_demes=1,
            **common_kw,
        )
        if data["max_time"] is not None:
            axes[0].set_xlim(0, data["max_time"] * time_factor)

        fig_log, ax_log = plt.subplots(1, 1, figsize=(5, 2))
        _plot_prevalence_panel(
            ax_log,
            deme,
            hpd_datastream=hpd_datastream,
            trajectory_data=data["trajectory_data"],
            case_counts_data=data["case_counts_data"],
            wastewater_data=data["wastewater_data"],
            validation_data_datastreams_prevalence=data[
                "validation_data_datastreams_prevalence"
            ],
            show_logscale=False,
            show_logscale_prevalence_only=True,
            case_counts_P1=False,
            fig=fig_log,
            n_demes=1,
            **common_kw,
        )
        if data["max_time"] is not None:
            ax_log.set_xlim(0, data["max_time"] * time_factor)
        plt.tight_layout()
        save_figure_png_and_pdf(f"{prefix}_prevalence_log.png")
        plt.close()

        # --- Cumulative incidence (linear only) ---
        ax = axes[1]
        _plot_cumincidence_panel(
            ax,
            deme,
            cumulative_incidence_hpd=data["cumulative_incidence_hpd"],
            trajectory_data=data["trajectory_data"],
            seroprevalence_data=data["seroprevalence_data"],
            validation_data_datastreams_cumIncidence=data[
                "validation_data_datastreams_cumIncidence"
            ],
            deme_popsizes=data["deme_popsizes"],
            show_legend=False,
            pin_band_above_sero_one=True,
            **common_kw,
        )
        ax.set_title(None)

        if data["max_time"] is not None:
            ax.set_xlim(0, data["max_time"] * time_factor)
        plt.tight_layout()
        save_figure_png_and_pdf(f"{prefix}_combprevcuminc.png")
        plt.close()

        # --- Ne (linear and log) ---
        for show_log in (False, True):
            fig, ax = plt.subplots(1, 1, figsize=(4, 2))
            _plot_ne_panel(
                ax,
                deme,
                hpd_original=data["hpd_original"],
                hpd_datastream=hpd_datastream,
                expected_ne_data=data["expected_ne_data"],
                max_time=data["max_time"],
                validation_data_original_ne=data["validation_data_original_ne"],
                validation_data_datastreams_ne=data["validation_data_datastreams_ne"],
                show_logscale=show_log,
                **common_kw,
            )
            if data["max_time"] is not None:
                ax.set_xlim(0, data["max_time"] * time_factor)
            plt.tight_layout()
            suffix = "_log" if show_log else ""
            save_figure_png_and_pdf(f"{prefix}_ne{suffix}.png")
            plt.close()


def plot_final_mascotds_intro_figure(data, output_png, time_unit="days"):
    """
    Assemble ``final_mascotds_intro_figure``.

    Layout (3 rows × 2 cols outer grid):

    * rows 0-1, cols 0-1: placeholder axis (axis off) for an external graphic
    * row 2, col 0: example prevalence panel for the start deme
    * row 2, col 1: example prevalence panel for the secondary deme

    The prevalence panels carry two twin y-axes (case counts and wastewater),
    so row 2 is nested in its own sub-gridspec with wider ``wspace`` to keep
    the right-hand axis labels from colliding with the neighbour panel.

    Saves PNG + PDF at ``output_png``. ``output_png`` must end in ``.png``.
    """
    configure_pdf_fonts()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    time_factor = 365.0 if time_unit == "days" else 1.0
    time_label = "days" if time_unit == "days" else "years"

    hpd_original = data["hpd_original"]
    if hpd_original.empty:
        print("No HPD data; skipping intro figure.")
        return

    demes = _ordered_demes(data)
    role_for_deme = {
        int(demes[0]): "start",
        int(demes[1]) if len(demes) > 1 else int(demes[0]): "secondary",
    }

    hpd_datastream = (
        data["hpd_datastream"] if data["hpd_datastream"] is not None else pd.DataFrame()
    )

    fontsize_tick = DEFAULT_FONTSIZES["tick_label"]
    common_kw = {
        "time_factor": time_factor,
        "time_label": time_label,
        "starting_deme": data["starting_deme"],
        "fontsizes": FONTSIZES_LIST,
        "fontsize_tick": fontsize_tick,
    }
    max_time = data["max_time"]

    fig = plt.figure(figsize=(12, 10))
    outer = fig.add_gridspec(3, 2, hspace=0.45)

    ax_placeholder = fig.add_subplot(outer[0:2, 0:2])
    ax_placeholder.set_axis_off()

    # Nested grid for row 2 — needs a wider wspace because each prevalence
    # panel carries two twin y-axes on the right.
    inner = outer[2, 0:2].subgridspec(1, 2, wspace=0.9)

    for deme in demes:
        role = role_for_deme[int(deme)]
        col_idx = 0 if role == "start" else 1
        show_legend = role == "secondary"

        ax_prev = fig.add_subplot(inner[0, col_idx])
        _plot_prevalence_panel(
            ax_prev,
            deme,
            hpd_datastream=hpd_datastream,
            trajectory_data=data["trajectory_data"],
            case_counts_data=data["case_counts_data"],
            wastewater_data=data["wastewater_data"],
            validation_data_datastreams_prevalence=data[
                "validation_data_datastreams_prevalence"
            ],
            show_logscale=False,
            show_logscale_prevalence_only=False,
            case_counts_P1=False,
            fig=fig,
            n_demes=1,
            show_legend=show_legend,
            **common_kw,
        )
        if max_time is not None:
            ax_prev.set_xlim(0, max_time * time_factor)

    save_figure_png_and_pdf(output_png)
    plt.close(fig)


def _true_I_at(t_query, t_events, I_events):
    """Step-function lookup: I at the last event time <= t_query.

    The remaster trajectory stores each state change as its own row, so the
    I compartment is right-continuous between events. Exact times are used
    (no snapping to a grid, no spline smoothing).
    """
    idx = np.searchsorted(t_events, t_query, side="right") - 1
    idx = np.clip(idx, 0, len(t_events) - 1)
    return I_events[idx]


def compute_wastewater_residuals(data):
    """Build per-observation wastewater residuals for simulation and inference checks.

    For each wastewater observation i at time t_i in deme d, computes:
      * log_obs      = log(conc_i)
      * log_mu_true  = log(alpha_true_d * I_true(t_i, d) / N_d)
                       I_true reconstructed as a right-continuous step
                       function of the stored simulation trajectory.
      * log_mu_post  = log(alpha_post_mean_d * I_post_hat(t_i, d) / N_d)
                       alpha_post_mean taken from the datastream BEAST log
                       (mean after burn-in), I_post_hat linearly interpolated
                       from hpd_datastream's posterior median log-prevalence
                       (exp'd back to linear scale).
      * res_sim      = log_obs - log_mu_true   ~ Normal(0, sigma_true^2)
      * res_inf      = log_obs - log_mu_post   ~ Normal(0, sigma_true^2) if well-fit

    Returns:
        (df_resid, sigma_true, sigma_post) where df_resid has columns
        {deme, t_years, log_obs, log_mu_true, log_mu_post, res_sim, res_inf,
         alpha_true, alpha_post_mean}, sigma_true is a float or None,
        and sigma_post is a dict with keys {median, hpd_lower, hpd_upper}
        summarising the posterior of wastewater.sigma:SimDataset (None if
        unavailable).
    """
    ww = data["wastewater_data"]
    traj = data["trajectory_data"]
    params_df = data["params_df"]
    hpd = data["hpd_datastream"]
    log_ds = data["log_content_datastream"]
    deme_popsizes = data["deme_popsizes"]

    if ww is None or ww.empty:
        print("No wastewater data; cannot compute residuals.")
        return pd.DataFrame(), None, None
    if traj is None or traj.empty:
        print("No trajectory data; cannot compute simulator residuals.")
        return pd.DataFrame(), None, None
    if hpd is None or hpd.empty:
        print("No datastream HPD; cannot compute inference residuals.")
        return pd.DataFrame(), None, None
    if log_ds is None or log_ds.empty:
        print("No datastream BEAST log; cannot compute posterior-mean alpha.")
        return pd.DataFrame(), None, None

    sigma_true = None
    if params_df is not None and not params_df.empty:
        sig_row = params_df[params_df["parameter"] == "ds_ww_sigma"]
        if not sig_row.empty:
            sigma_true = float(sig_row["value"].iloc[0])

    sigma_post = None
    sigma_col = "wastewater.sigma:SimDataset"
    if sigma_col in log_ds.columns:
        sig_samples = (
            pd.to_numeric(log_ds[sigma_col], errors="coerce").dropna().to_numpy()
        )
        if sig_samples.size > 0:
            hpd_lo, hpd_hi, hpd_med = calculate_hpd(sig_samples, alpha=0.05)
            sigma_post = {
                "median": float(hpd_med),
                "hpd_lower": float(hpd_lo),
                "hpd_upper": float(hpd_hi),
            }

    traj_I = traj[traj["population"] == "I"]

    rows = []
    for deme, obs_grp in ww.groupby("Deme"):
        d_int = int(deme)
        if d_int not in deme_popsizes:
            print(f"Skipping deme {d_int}: no popsize in deme_popsizes.")
            continue
        N = float(deme_popsizes[d_int])

        alpha_true = None
        if params_df is not None and not params_df.empty:
            a_row = params_df[
                (params_df["parameter"] == "ds_ww_scaling")
                & (params_df["deme"].astype(str) == str(d_int))
            ]
            if not a_row.empty:
                alpha_true = float(a_row["value"].iloc[0])
        if alpha_true is None:
            print(f"Skipping deme {d_int}: no ds_ww_scaling in params.")
            continue

        # BEAST names Deme1 <-> deme 0, Deme2 <-> deme 1
        beast_col = f"wastewater.scaling.Deme{d_int + 1}:SimDataset"
        if beast_col not in log_ds.columns:
            print(f"Skipping deme {d_int}: column {beast_col} not in BEAST log.")
            continue
        alpha_post_mean = float(log_ds[beast_col].mean())

        traj_d = traj_I[traj_I["Deme"] == d_int].sort_values("t")
        if traj_d.empty:
            print(f"Skipping deme {d_int}: no I trajectory rows.")
            continue
        t_evt = traj_d["t"].to_numpy(dtype=float)
        I_evt = traj_d["value"].to_numpy(dtype=float)

        hpd_d = hpd[hpd["Deme"] == d_int].sort_values("timesincestart")
        if hpd_d.empty:
            print(f"Skipping deme {d_int}: no posterior HPD rows.")
            continue
        t_post = hpd_d["timesincestart"].to_numpy(dtype=float)
        I_post_med = np.exp(hpd_d["logPrevalence"].to_numpy(dtype=float))

        for _, obs in obs_grp.iterrows():
            t_i = float(obs["t_wastewater_fromsimstart"])
            conc = float(obs["wastewater"])
            if not np.isfinite(conc) or conc <= 0:
                continue
            I_true_i = _true_I_at(t_i, t_evt, I_evt)
            if I_true_i <= 0:
                continue
            I_post_i = float(np.interp(t_i, t_post, I_post_med))
            if I_post_i <= 0:
                continue

            log_obs = float(np.log(conc))
            log_mu_true = float(np.log(alpha_true * I_true_i / N))
            log_mu_post = float(np.log(alpha_post_mean * I_post_i / N))

            rows.append(
                {
                    "deme": d_int,
                    "t_years": t_i,
                    "log_obs": log_obs,
                    "log_mu_true": log_mu_true,
                    "log_mu_post": log_mu_post,
                    "res_sim": log_obs - log_mu_true,
                    "res_inf": log_obs - log_mu_post,
                    "alpha_true": alpha_true,
                    "alpha_post_mean": alpha_post_mean,
                }
            )

    return pd.DataFrame(rows), sigma_true, sigma_post


def plot_wastewater_residual_diagnostics(data, output_dir, time_unit="days"):
    """Write a combined wastewater residual diagnostic figure (PNG + PDF) and a summary CSV.

    Combined figure (3x2):
      * (0,0) start deme: log(obs), log(mu_true), log(mu_post) vs time
      * (0,1) start deme: residuals vs time (sim and inf)
      * (1,0) secondary deme: log(obs), log(mu_true), log(mu_post) vs time
      * (1,1) secondary deme: residuals vs time (sim and inf)
      * (2,0) histogram of simulation residuals pooled across both demes with N(0, sigma_true^2) overlay
      * (2,1) histogram of inference residuals pooled across both demes with N(0, sigma_true^2) overlay

    Both demes jointly inform the single shared sigma, so the histogram row pools their residuals.

    Summary CSV per deme: n, sd/var of each residual set, sigma_true, sigma_post.
    """
    configure_pdf_fonts()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_resid, sigma_true, sigma_post = compute_wastewater_residuals(data)
    if df_resid.empty:
        return df_resid

    if sigma_post is not None:
        sigma_post_title = (
            rf"$\hat{{\sigma}}_{{post}}$ median={sigma_post['median']:.3g}, "
            rf"95% HPD=[{sigma_post['hpd_lower']:.3g}, {sigma_post['hpd_upper']:.3g}]"
        )
    else:
        sigma_post_title = None

    time_factor = 365.0 if time_unit == "days" else 1.0
    time_label = "days" if time_unit == "days" else "years"
    starting_deme = data["starting_deme"]

    df_resid.to_csv(output_dir / "ww_residuals_per_obs.csv", index=False)

    demes_in_resid = {int(d) for d in df_resid["deme"].unique()}
    demes_ordered = [int(d) for d in _ordered_demes(data) if int(d) in demes_in_resid]

    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    summary_rows = []

    for row_idx, deme in enumerate(demes_ordered):
        sub = df_resid[df_resid["deme"] == deme].sort_values("t_years")
        role = "start_deme" if int(deme) == int(starting_deme) else "secondary_deme"
        role_nice = "Start deme" if role == "start_deme" else "Secondary deme"
        t_disp = sub["t_years"].to_numpy() * time_factor

        ax = axes[row_idx, 0]
        ax.scatter(
            t_disp, sub["log_obs"], s=14, color="black", alpha=0.7, label="log(obs)"
        )
        ax.scatter(
            t_disp,
            sub["log_mu_true"],
            s=14,
            color="tab:green",
            marker="x",
            alpha=0.8,
            label=r"log $\mu_{true}$",
        )
        ax.scatter(
            t_disp,
            sub["log_mu_post"],
            s=14,
            color="tab:orange",
            marker="+",
            alpha=0.8,
            label=r"log $\hat{\mu}_{post}$",
        )
        ax.set_xlabel(f"Time ({time_label} from simulation start)")
        ax.set_ylabel("log(concentration)")
        ax.set_title(f"Deme {deme} ({role_nice}): observations vs fitted means")
        ax.legend(fontsize=8, loc="best")

        ax = axes[row_idx, 1]
        ax.axhline(0, color="grey", lw=0.5)
        if sigma_true is not None:
            for y in (sigma_true, -sigma_true, 2 * sigma_true, -2 * sigma_true):
                ax.axhline(y, color="grey", lw=0.4, ls=":")
        ax.scatter(
            t_disp,
            sub["res_sim"],
            s=14,
            color="tab:green",
            alpha=0.7,
            label=r"sim: obs $-$ $\mu_{true}$",
        )
        ax.scatter(
            t_disp,
            sub["res_inf"],
            s=14,
            color="tab:orange",
            alpha=0.7,
            label=r"inf: obs $-$ $\hat{\mu}_{post}$",
        )
        ax.set_xlabel(f"Time ({time_label} from simulation start)")
        ax.set_ylabel("residual (log scale)")
        ax.set_title(f"Deme {deme} ({role_nice}): residuals vs time")
        ax.legend(fontsize=8, loc="best")

        vals_sim = sub["res_sim"].to_numpy()
        vals_inf = sub["res_inf"].to_numpy()
        summary_rows.append(
            {
                "deme": int(deme),
                "role": "start" if role == "start_deme" else "secondary",
                "n": int(sub.shape[0]),
                "sd_sim": (
                    float(np.std(vals_sim, ddof=1))
                    if vals_sim.size > 1
                    else float("nan")
                ),
                "var_sim": (
                    float(np.var(vals_sim, ddof=1))
                    if vals_sim.size > 1
                    else float("nan")
                ),
                "mean_sim": float(np.mean(vals_sim)) if vals_sim.size else float("nan"),
                "sd_inf": (
                    float(np.std(vals_inf, ddof=1))
                    if vals_inf.size > 1
                    else float("nan")
                ),
                "var_inf": (
                    float(np.var(vals_inf, ddof=1))
                    if vals_inf.size > 1
                    else float("nan")
                ),
                "mean_inf": float(np.mean(vals_inf)) if vals_inf.size else float("nan"),
                "sigma_true": sigma_true,
                "sigma_post_median": (
                    sigma_post["median"] if sigma_post is not None else float("nan")
                ),
                "sigma_post_hpd_lower": (
                    sigma_post["hpd_lower"] if sigma_post is not None else float("nan")
                ),
                "sigma_post_hpd_upper": (
                    sigma_post["hpd_upper"] if sigma_post is not None else float("nan")
                ),
                "alpha_true": float(sub["alpha_true"].iloc[0]),
                "alpha_post_mean": float(sub["alpha_post_mean"].iloc[0]),
            }
        )

    # Row 2: histograms pooled across both demes — a single sigma is inferred
    # jointly from all wastewater observations, so pooling is the right check.
    for ax_h, col, color, nice in [
        (axes[2, 0], "res_sim", "tab:green", "Simulation"),
        (axes[2, 1], "res_inf", "tab:orange", "Inference"),
    ]:
        vals = df_resid[col].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax_h.set_title(f"{nice} residuals (no data)")
            continue
        ax_h.hist(
            vals,
            bins=max(10, min(40, vals.size // 3 + 5)),
            density=True,
            color=color,
            alpha=0.55,
            edgecolor="black",
            linewidth=0.4,
        )
        if sigma_true is not None and sigma_true > 0:
            span = max(abs(vals.min()), abs(vals.max()), 3 * sigma_true)
            xs = np.linspace(-span, span, 400)
            pdf = (1.0 / (sigma_true * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * (xs / sigma_true) ** 2
            )
            ax_h.plot(
                xs,
                pdf,
                color="black",
                lw=1.0,
                ls="--",
                label=rf"$N(0,\sigma_{{true}}^2)$, $\sigma_{{true}}$={sigma_true:.3g}",
            )
        sd_obs = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
        ax_h.set_xlabel("residual (log scale)")
        ax_h.set_ylabel("density")
        title = (
            f"{nice} residuals (both demes): n={vals.size}, "
            rf"$\hat{{\sigma}}$={sd_obs:.3g}"
        )
        if sigma_post_title is not None:
            title += "\n" + sigma_post_title
        ax_h.set_title(title)
        ax_h.legend(fontsize=8, loc="best")

    fig.suptitle("Wastewater residual diagnostics", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure_png_and_pdf(str(output_dir / "ww_residuals_combined.png"))
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "ww_residuals_summary.csv", index=False)
    return df_resid


def _true_I_at_vec(t_query, t_events, I_events):
    """Vectorised step-function lookup of I at each t in t_query."""
    idx = np.searchsorted(t_events, t_query, side="right") - 1
    idx = np.clip(idx, 0, len(t_events) - 1)
    return I_events[idx]


def _build_natural_spline(knot_times_back, log_knot_vals):
    """Natural cubic spline in backward time — matches Apache Commons Math SplineInterpolator.

    Knot times must be strictly increasing (they already are in ``rateShifts``).
    """
    return make_interp_spline(knot_times_back, log_knot_vals, k=3, bc_type="natural")


def _eval_spline_clamped(spline, t_back, t_first_knot, t_last_knot):
    """Replicate Spline.java: outside [first,last] knot, clamp to the boundary knot."""
    t_clamped = np.clip(t_back, t_first_knot, t_last_knot)
    return spline(t_clamped)


def _snap_to_grid(t_query, grid_times):
    """For each t in t_query, return (t_snap, idx_snap) where idx_snap is the nearest grid index.

    Matches Spline.java getValueAtGridPoint / getPrevalenceAtGridPoint: if the query
    falls inside a segment [grid[k], grid[k+1]] the closer end is returned. If t <= grid[0]
    we return grid[0]; t > grid[-1] returns grid[-1].
    """
    t_query = np.asarray(t_query, dtype=float)
    grid_times = np.asarray(grid_times, dtype=float)
    n = grid_times.size
    left = np.searchsorted(grid_times, t_query, side="right") - 1
    left = np.clip(left, 0, n - 1)
    right = np.clip(left + 1, 0, n - 1)
    dl = np.abs(t_query - grid_times[left])
    dr = np.abs(grid_times[right] - t_query)
    # Tie -> take the left (matches Java: "<= " goes left)
    take_right = dr < dl
    idx = np.where(take_right, right, left)
    # Edge clamp (already handled by clip above)
    return grid_times[idx], idx


def compute_spline_gridpoint_diagnostics(data, early_window_days=10, grid_mode="snap"):
    """Per-observation posterior spline reconstruction + grid-evaluation bias quantities.

    For each wastewater obs at forward time t_obs in deme d, builds one natural cubic
    spline per posterior sample from ``SkylinePrev.Deme{d+1}.{1..K}`` and computes:

      * Ŝ(t_obs)           = logI_spline_obs       (spline at exact obs time)
      * Ŝ(t_eval)          = logI_spline_eval      (spline evaluated the way the
                                                    BEAST likelihood does — see
                                                    ``grid_mode`` below)
      * Δ_grid_spline      = Ŝ(t_obs) - Ŝ(t_eval)
      * Δ_grid_true        = logI_true(t_obs) - logI_true(t_eval)
      * B_infer            = Ŝ(t_obs) - logI_true(t_obs)     (pure inference bias)
      * B_total            = Ŝ(t_eval) - logI_true(t_obs)    (what the likelihood sees)

    ``grid_mode`` controls how prevalence is evaluated at each observation time:

      * ``"snap"`` (default): use the spline value at the nearest grid point.
        Matches the earlier BEAST likelihood.
      * ``"interpolate"``: precompute the spline at every grid point and linearly
        interpolate between the two bracketing grid values.  Matches the newer
        BEAST likelihood.  In this mode ``t_eval == t_obs`` (no time displacement)
        but Ŝ(t_eval) may still differ from Ŝ(t_obs) due to the piecewise-linear
        approximation.

    All spline-based quantities are summarised per obs as posterior median + 95% HPD.
    ``Δ_grid_true`` is sample-free (depends only on ground truth).

    Returns:
        dict with keys:
          - ``per_obs_df``: pd.DataFrame, one row per (deme, obs)
          - ``dense``: {deme: {"t_back_y","t_fwd_d","median","hpd_lo","hpd_hi"}}
            for plotting the full posterior spline band
          - ``knots``: {deme: {"t_fwd_d","median","hpd_lo","hpd_hi","t_back_y"}}
            posterior-summarised knot values in forward time
          - ``rate_shifts_y`` (knot backward times, years)
          - ``grid_shifts_y`` (grid backward times, years)
          - ``max_rateshift_y`` (backward time of simstart = forward_T)
          - ``early_window_days``
          - ``grid_mode``
    """
    if grid_mode not in ("snap", "interpolate"):
        raise ValueError(
            f"grid_mode must be 'snap' or 'interpolate', got {grid_mode!r}"
        )
    log_ds = data["log_content_datastream"]
    traj = data["trajectory_data"]
    ww = data["wastewater_data"]
    rate_shifts = np.asarray(data["rateshifts_datastream"], dtype=float)
    grid_shifts = np.asarray(data["gridpointshifts_datastream"], dtype=float)

    if log_ds is None or log_ds.empty:
        print("No datastream log; cannot run spline gridpoint diagnostics.")
        return None
    if ww is None or ww.empty:
        print("No wastewater data; cannot run spline gridpoint diagnostics.")
        return None
    if traj is None or traj.empty:
        print("No trajectory data; cannot run spline gridpoint diagnostics.")
        return None
    if rate_shifts.size == 0 or grid_shifts.size == 0:
        print("Missing rate/grid shifts; cannot run spline gridpoint diagnostics.")
        return None

    grid_shifts_asc = np.sort(grid_shifts)
    max_rateshift = float(rate_shifts.max())
    t_first_knot = float(rate_shifts.min())
    t_last_knot = float(rate_shifts.max())

    demes = sorted(int(d) for d in ww["Deme"].unique())
    traj_I = traj[traj["population"] == "I"]

    # Dense backward-time grid for the spline band (for plotting)
    t_back_dense = np.linspace(t_first_knot, t_last_knot, 500)

    per_obs_rows = []
    dense_out = {}
    knots_out = {}

    for deme in demes:
        deme_py = int(deme)
        prev_cols = [
            c for c in log_ds.columns if c.startswith(f"SkylinePrev.Deme{deme_py + 1}.")
        ]
        if not prev_cols:
            print(f"Deme {deme_py}: no SkylinePrev columns found; skipping.")
            continue
        # Sort numerically by the trailing index
        prev_cols_sorted = sorted(prev_cols, key=lambda c: int(c.split(".")[-1]))
        if len(prev_cols_sorted) != rate_shifts.size:
            print(
                f"Deme {deme_py}: {len(prev_cols_sorted)} SkylinePrev columns "
                f"!= {rate_shifts.size} rate shifts; skipping."
            )
            continue
        knot_samples = log_ds[prev_cols_sorted].to_numpy(dtype=float)  # (S, K)
        n_samples = knot_samples.shape[0]
        if n_samples == 0:
            print(f"Deme {deme_py}: empty posterior; skipping.")
            continue

        # True I step function for this deme
        traj_d = traj_I[traj_I["Deme"] == deme_py].sort_values("t")
        if traj_d.empty:
            print(f"Deme {deme_py}: no trajectory rows; skipping.")
            continue
        t_evt = traj_d["t"].to_numpy(dtype=float)
        I_evt = traj_d["value"].to_numpy(dtype=float)

        # Wastewater observations for this deme
        obs_d = ww[ww["Deme"] == deme_py].copy()
        if obs_d.empty:
            continue
        t_obs_fwd = obs_d["t_wastewater_fromsimstart"].to_numpy(dtype=float)
        t_obs_back = max_rateshift - t_obs_fwd

        # Evaluation time depends on grid_mode:
        #   snap:        nearest grid point in backward time
        #   interpolate: same as obs time (interpolation happens in value space)
        if grid_mode == "snap":
            t_eval_back, _ = _snap_to_grid(t_obs_back, grid_shifts_asc)
        else:
            t_eval_back = t_obs_back.copy()
        t_eval_fwd = max_rateshift - t_eval_back

        # Pre-allocate per-sample eval matrices (S, n_obs) and (S, n_dense)
        n_obs = t_obs_back.size
        S_obs = np.empty((n_samples, n_obs), dtype=float)
        S_eval = np.empty((n_samples, n_obs), dtype=float)
        S_dense = np.empty((n_samples, t_back_dense.size), dtype=float)
        S_knots = np.empty((n_samples, rate_shifts.size), dtype=float)

        for s in range(n_samples):
            spline = _build_natural_spline(rate_shifts, knot_samples[s])
            S_obs[s, :] = _eval_spline_clamped(
                spline, t_obs_back, t_first_knot, t_last_knot
            )
            if grid_mode == "snap":
                S_eval[s, :] = _eval_spline_clamped(
                    spline, t_eval_back, t_first_knot, t_last_knot
                )
            else:
                grid_vals = _eval_spline_clamped(
                    spline, grid_shifts_asc, t_first_knot, t_last_knot
                )
                S_eval[s, :] = np.interp(t_obs_back, grid_shifts_asc, grid_vals)
            S_dense[s, :] = _eval_spline_clamped(
                spline, t_back_dense, t_first_knot, t_last_knot
            )
            S_knots[s, :] = knot_samples[s]  # at knots the spline passes exactly

        # Ground truth at obs and eval times
        logI_true_obs = np.log(
            np.clip(_true_I_at_vec(t_obs_fwd, t_evt, I_evt), 1e-300, None)
        )
        logI_true_eval = np.log(
            np.clip(_true_I_at_vec(t_eval_fwd, t_evt, I_evt), 1e-300, None)
        )

        # Four derived quantities per sample, per obs
        D_grid_spline = S_obs - S_eval  # (S, n_obs)
        B_infer = S_obs - logI_true_obs[None, :]  # (S, n_obs)
        B_total = S_eval - logI_true_obs[None, :]  # (S, n_obs)
        D_grid_true = logI_true_obs - logI_true_eval  # (n_obs,) — no sample dim

        def _summ(arr_2d_or_1d):
            """Return (median, hpd_lo, hpd_hi) along axis=0 using calculate_hpd."""
            arr = np.asarray(arr_2d_or_1d)
            if arr.ndim == 1:
                return arr, np.full_like(arr, np.nan), np.full_like(arr, np.nan)
            med = np.empty(arr.shape[1])
            lo = np.empty(arr.shape[1])
            hi = np.empty(arr.shape[1])
            for j in range(arr.shape[1]):
                col = arr[:, j]
                col = col[np.isfinite(col)]
                if col.size < 2:
                    med[j] = np.nan
                    lo[j] = np.nan
                    hi[j] = np.nan
                else:
                    lo_j, hi_j, med_j = calculate_hpd(col, alpha=0.05)
                    med[j] = med_j
                    lo[j] = lo_j
                    hi[j] = hi_j
            return med, lo, hi

        S_obs_med, S_obs_lo, S_obs_hi = _summ(S_obs)
        S_eval_med, S_eval_lo, S_eval_hi = _summ(S_eval)
        D_grid_spl_med, D_grid_spl_lo, D_grid_spl_hi = _summ(D_grid_spline)
        B_infer_med, B_infer_lo, B_infer_hi = _summ(B_infer)
        B_total_med, B_total_lo, B_total_hi = _summ(B_total)
        S_dense_med, S_dense_lo, S_dense_hi = _summ(S_dense)
        S_knot_med, S_knot_lo, S_knot_hi = _summ(S_knots)

        for i in range(n_obs):
            per_obs_rows.append(
                {
                    "deme": deme_py,
                    "grid_mode": grid_mode,
                    "t_obs_years": float(t_obs_fwd[i]),
                    "t_obs_days": float(t_obs_fwd[i] * 365.0),
                    "t_eval_years": float(t_eval_fwd[i]),
                    "t_eval_days": float(t_eval_fwd[i] * 365.0),
                    "eval_delta_days": float((t_eval_fwd[i] - t_obs_fwd[i]) * 365.0),
                    "t_obs_back_years": float(t_obs_back[i]),
                    "t_eval_back_years": float(t_eval_back[i]),
                    "logI_spline_obs_med": S_obs_med[i],
                    "logI_spline_obs_lo": S_obs_lo[i],
                    "logI_spline_obs_hi": S_obs_hi[i],
                    "logI_spline_eval_med": S_eval_med[i],
                    "logI_spline_eval_lo": S_eval_lo[i],
                    "logI_spline_eval_hi": S_eval_hi[i],
                    "logI_true_obs": float(logI_true_obs[i]),
                    "logI_true_eval": float(logI_true_eval[i]),
                    "D_grid_spline_med": D_grid_spl_med[i],
                    "D_grid_spline_lo": D_grid_spl_lo[i],
                    "D_grid_spline_hi": D_grid_spl_hi[i],
                    "D_grid_true": float(D_grid_true[i]),
                    "B_infer_med": B_infer_med[i],
                    "B_infer_lo": B_infer_lo[i],
                    "B_infer_hi": B_infer_hi[i],
                    "B_total_med": B_total_med[i],
                    "B_total_lo": B_total_lo[i],
                    "B_total_hi": B_total_hi[i],
                    "wastewater": float(obs_d["wastewater"].iloc[i]),
                }
            )

        t_fwd_dense = max_rateshift - t_back_dense
        dense_out[deme_py] = {
            "t_back_y": t_back_dense,
            "t_fwd_d": t_fwd_dense * 365.0,
            "median": S_dense_med,
            "hpd_lo": S_dense_lo,
            "hpd_hi": S_dense_hi,
        }
        knots_out[deme_py] = {
            "t_back_y": rate_shifts,
            "t_fwd_d": (max_rateshift - rate_shifts) * 365.0,
            "median": S_knot_med,
            "hpd_lo": S_knot_lo,
            "hpd_hi": S_knot_hi,
        }

    per_obs_df = pd.DataFrame(per_obs_rows)
    return {
        "per_obs_df": per_obs_df,
        "dense": dense_out,
        "knots": knots_out,
        "rate_shifts_y": rate_shifts,
        "grid_shifts_y": grid_shifts,
        "max_rateshift_y": max_rateshift,
        "early_window_days": early_window_days,
        "grid_mode": grid_mode,
    }


def _deme_role_title(deme, starting_deme):
    if starting_deme is None:
        return f"Deme {deme}"
    role = "Start deme" if int(deme) == int(starting_deme) else "Secondary deme"
    return f"Deme {deme} ({role})"


def plot_spline_gridpoint_diagnostics(
    data, output_dir, early_window_days=10, grid_mode="snap"
):
    """Produce grid-evaluation bias diagnostic figures + CSVs.

    Figures written (PNG + PDF each):
      * ``spline_grid_headline_earlywindow.png``  — per deme: log prev (left y), log
         wastewater (right y), x = days from simstart over the first
         ``early_window_days``. True I step function, posterior-median spline with
         95% HPD band, knots as filled circles, grid as rug ticks, obs as points
         with dashed connectors showing the grid-evaluation displacement.
      * ``spline_grid_deltas_vs_time.png``       — 2x2 per deme: Δ_grid_spline and
         Δ_grid_true vs time (full span); B_infer and B_total vs time.
      * ``spline_grid_paired_scatter.png``       — per deme: logI_spline(t_eval)
         vs logI_true(t_obs) with y=x; coloured by eval direction.
      * ``spline_grid_histograms.png``           — per deme + pooled: distributions
         of Δ_grid_spline, Δ_grid_true, B_infer, B_total with mean/median marks.

    CSVs:
      * ``spline_grid_per_obs.csv``              — every per-obs quantity
      * ``spline_grid_summary.csv``              — per-deme + early-window summary
    """
    configure_pdf_fonts()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out = compute_spline_gridpoint_diagnostics(
        data, early_window_days=early_window_days, grid_mode=grid_mode
    )
    if out is None:
        return None
    per_obs_df = out["per_obs_df"]
    if per_obs_df.empty:
        return per_obs_df
    dense = out["dense"]
    knots = out["knots"]
    grid_shifts_y = out["grid_shifts_y"]
    max_rateshift_y = out["max_rateshift_y"]
    starting_deme = data["starting_deme"]

    per_obs_df.to_csv(output_dir / "spline_grid_per_obs.csv", index=False)
    mode_label = "snap-to-grid" if grid_mode == "snap" else "grid-interpolation"
    demes_ordered = [int(d) for d in _ordered_demes(data) if int(d) in dense]
    n_demes = len(demes_ordered)
    if n_demes == 0:
        return per_obs_df

    grid_fwd_days = (max_rateshift_y - grid_shifts_y) * 365.0

    # Headline plot is only meaningful for demes with obs inside the early window:
    # a deme that hasn't seeded yet (e.g. secondary deme before the deme switch)
    # has log-prevalence at numerical floor and stretches the y-axis uselessly.
    headline_demes = [
        d
        for d in demes_ordered
        if (
            (per_obs_df["deme"] == d) & (per_obs_df["t_obs_days"] <= early_window_days)
        ).sum()
        > 0
    ]

    # ---------- Headline early-window plot ----------
    if not headline_demes:
        print("No obs within the early window; skipping headline zoom plot.")
    n_headline = max(len(headline_demes), 1)
    fig_h, axes_h = plt.subplots(
        n_headline,
        1,
        figsize=(9, 3.5 * n_headline),
        squeeze=False,
    )
    for row_idx, deme in enumerate(headline_demes):
        ax = axes_h[row_idx, 0]
        sub = per_obs_df[per_obs_df["deme"] == deme]
        sub_early = sub[sub["t_obs_days"] <= early_window_days]

        dd = dense[deme]
        mask_dense = dd["t_fwd_d"] <= early_window_days
        ax.plot(
            dd["t_fwd_d"][mask_dense],
            dd["median"][mask_dense],
            color="tab:blue",
            lw=1.5,
            label="Posterior median spline",
        )
        ax.fill_between(
            dd["t_fwd_d"][mask_dense],
            dd["hpd_lo"][mask_dense],
            dd["hpd_hi"][mask_dense],
            color="tab:blue",
            alpha=0.2,
            label="95% HPD (spline)",
        )

        # Knot markers (posterior-median log prev at each knot)
        kk = knots[deme]
        km = kk["t_fwd_d"] <= early_window_days
        ax.errorbar(
            kk["t_fwd_d"][km],
            kk["median"][km],
            yerr=[
                kk["median"][km] - kk["hpd_lo"][km],
                kk["hpd_hi"][km] - kk["median"][km],
            ],
            fmt="o",
            color="tab:blue",
            mfc="white",
            mec="tab:blue",
            ms=7,
            elinewidth=1.0,
            capsize=2.5,
            label="Knot (posterior)",
        )

        gm = grid_fwd_days <= early_window_days
        # True I step function
        traj = data["trajectory_data"]
        traj_I = traj[(traj["population"] == "I") & (traj["Deme"] == deme)].sort_values(
            "t"
        )
        if not traj_I.empty:
            t_days = traj_I["t"].to_numpy() * 365.0
            I_vals = traj_I["value"].to_numpy()
            mask_t = t_days <= early_window_days + 1.0
            with np.errstate(divide="ignore"):
                logI = np.log(
                    np.clip(I_vals, 0.5, None)
                )  # every I < 1 means true I = 0, but to plot just choose a value below 1. -> log(<1) < 0
            ax.step(
                t_days[mask_t],
                logI[mask_t],
                where="post",
                color="tab:green",
                lw=1.2,
                alpha=0.85,
                label="True log I (step)",
            )

        # Dashed connector from exact spline value to grid-evaluated value
        for _, obs in sub_early.iterrows():
            ax.plot(
                [obs["t_obs_days"], obs["t_eval_days"]],
                [obs["logI_spline_obs_med"], obs["logI_spline_eval_med"]],
                color="grey",
                lw=0.5,
                ls="--",
                alpha=0.8,
            )
            ax.plot(
                obs["t_obs_days"],
                obs["logI_spline_obs_med"],
                marker="o",
                ms=3,
                color="tab:blue",
            )
            ax.plot(
                obs["t_eval_days"],
                obs["logI_spline_eval_med"],
                marker="s",
                ms=3,
                color="tab:red",
                alpha=0.8,
            )

        # Grid rug at bottom
        ymin, ymax = ax.get_ylim()
        rug_y = ymin + 0.02 * (ymax - ymin)
        ax.plot(
            grid_fwd_days[gm],
            np.full(gm.sum(), rug_y),
            "|",
            color="black",
            ms=6,
            mew=0.4,
            alpha=0.4,
            label="Grid points",
        )

        ax.set_xlim(0, early_window_days)
        ax.set_xlabel("Days from simulation start")
        ax.set_ylabel("log prevalence")
        ax.set_title(
            _deme_role_title(deme, starting_deme)
            + f" \u2014 first {early_window_days}d: spline vs truth ({mode_label})"
        )

        # Twin axis: log wastewater
        ax2 = ax.twinx()
        if not sub_early.empty:
            with np.errstate(divide="ignore"):
                log_ww = np.log(
                    np.clip(sub_early["wastewater"].to_numpy(), 1e-300, None)
                )
            ax2.scatter(
                sub_early["t_obs_days"],
                log_ww,
                s=22,
                color="tab:purple",
                marker="x",
                alpha=0.9,
                label="log wastewater obs",
            )
        ax2.set_ylabel("log(wastewater concentration)", color="tab:purple")
        ax2.tick_params(axis="y", colors="tab:purple")

        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")

    if headline_demes:
        fig_h.tight_layout()
        save_figure_png_and_pdf(
            str(output_dir / "spline_grid_headline_earlywindow.png")
        )
    plt.close(fig_h)

    # ---------- Δ and B vs time (full span + early window shading) ----------
    fig_d, axes_d = plt.subplots(n_demes, 2, figsize=(12, 3.5 * n_demes), squeeze=False)
    for row_idx, deme in enumerate(demes_ordered):
        sub = per_obs_df[per_obs_df["deme"] == deme].sort_values("t_obs_days")
        t = sub["t_obs_days"].to_numpy()

        ax = axes_d[row_idx, 0]
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvspan(0, early_window_days, color="gold", alpha=0.08, label="early window")
        ax.errorbar(
            t,
            sub["D_grid_spline_med"],
            yerr=[
                sub["D_grid_spline_med"] - sub["D_grid_spline_lo"],
                sub["D_grid_spline_hi"] - sub["D_grid_spline_med"],
            ],
            fmt="o",
            ms=3,
            color="tab:blue",
            alpha=0.8,
            elinewidth=0.6,
            label=r"$\Delta_{grid,spline}$ = $\hat{S}(t_{obs}) - \hat{S}(t_{eval})$",
        )
        ax.scatter(
            t,
            sub["D_grid_true"],
            s=14,
            marker="x",
            color="tab:green",
            alpha=0.85,
            label=r"$\Delta_{grid,true}$ = $\log I_{true}(t_{obs}) - \log I_{true}(t_{eval})$",
        )
        ax.set_xlabel("Days from simulation start")
        ax.set_ylabel("log-prev difference (grid-eval error)")
        ax.set_title(
            _deme_role_title(deme, starting_deme)
            + f" \u2014 grid deltas ({mode_label})"
        )
        ax.legend(fontsize=7, loc="best")

        ax = axes_d[row_idx, 1]
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvspan(0, early_window_days, color="gold", alpha=0.08, label="early window")
        ax.errorbar(
            t,
            sub["B_infer_med"],
            yerr=[
                sub["B_infer_med"] - sub["B_infer_lo"],
                sub["B_infer_hi"] - sub["B_infer_med"],
            ],
            fmt="o",
            ms=3,
            color="tab:orange",
            alpha=0.75,
            elinewidth=0.6,
            label=r"$B_{infer}$ = $\hat{S}(t_{obs}) - \log I_{true}(t_{obs})$",
        )
        ax.errorbar(
            t,
            sub["B_total_med"],
            yerr=[
                sub["B_total_med"] - sub["B_total_lo"],
                sub["B_total_hi"] - sub["B_total_med"],
            ],
            fmt="s",
            ms=3,
            color="tab:red",
            alpha=0.75,
            elinewidth=0.6,
            label=r"$B_{total}$ = $\hat{S}(t_{eval}) - \log I_{true}(t_{obs})$",
        )
        ax.set_xlabel("Days from simulation start")
        ax.set_ylabel("log-prev bias")
        ax.set_title(
            _deme_role_title(deme, starting_deme)
            + f" \u2014 inference vs total bias ({mode_label})"
        )
        ax.legend(fontsize=7, loc="best")

    fig_d.tight_layout()
    save_figure_png_and_pdf(str(output_dir / "spline_grid_deltas_vs_time.png"))
    plt.close(fig_d)

    # ---------- Paired scatter: Ŝ(t_eval) vs logI_true(t_obs) ----------
    fig_p, axes_p = plt.subplots(1, n_demes, figsize=(5 * n_demes, 4.5), squeeze=False)
    for col_idx, deme in enumerate(demes_ordered):
        ax = axes_p[0, col_idx]
        sub = per_obs_df[per_obs_df["deme"] == deme]
        early_mask = sub["t_obs_days"] <= early_window_days
        eval_dir = np.where(
            sub["eval_delta_days"] > 0,
            "eval forward (later)",
            np.where(sub["eval_delta_days"] < 0, "eval backward (earlier)", "exact"),
        )
        for label, colour, marker in [
            ("eval forward (later)", "tab:red", "^"),
            ("eval backward (earlier)", "tab:blue", "v"),
            ("exact", "black", "o"),
        ]:
            m = eval_dir == label
            if m.sum():
                ax.scatter(
                    sub.loc[m, "logI_true_obs"],
                    sub.loc[m, "logI_spline_eval_med"],
                    s=22,
                    marker=marker,
                    color=colour,
                    alpha=0.75,
                    label=label,
                )
        if early_mask.any():
            ax.scatter(
                sub.loc[early_mask, "logI_true_obs"],
                sub.loc[early_mask, "logI_spline_eval_med"],
                s=70,
                facecolors="none",
                edgecolors="gold",
                lw=1.2,
                label=f"first {early_window_days} d",
            )
        lo = min(sub["logI_true_obs"].min(), sub["logI_spline_eval_med"].min())
        hi = max(sub["logI_true_obs"].max(), sub["logI_spline_eval_med"].max())
        ax.plot([lo, hi], [lo, hi], ls="--", color="grey", lw=0.8, label="y = x")
        ax.set_xlabel(r"$\log I_{true}(t_{obs})$")
        ax.set_ylabel(r"$\hat{S}(t_{eval})$ (posterior median)")
        ax.set_title(
            _deme_role_title(deme, starting_deme) + f" ({mode_label})"
        )
        ax.legend(fontsize=7, loc="best")

    fig_p.tight_layout()
    save_figure_png_and_pdf(str(output_dir / "spline_grid_paired_scatter.png"))
    plt.close(fig_p)

    # ---------- Histograms (per deme, per quantity; full span + early window) ----------
    quantities = [
        ("D_grid_spline_med", r"$\Delta_{grid,spline}$", "tab:blue"),
        ("D_grid_true", r"$\Delta_{grid,true}$", "tab:green"),
        ("B_infer_med", r"$B_{infer}$", "tab:orange"),
        ("B_total_med", r"$B_{total}$", "tab:red"),
    ]
    fig_h2, axes_h2 = plt.subplots(
        n_demes,
        len(quantities),
        figsize=(3.5 * len(quantities), 3.0 * n_demes),
        squeeze=False,
    )
    for row_idx, deme in enumerate(demes_ordered):
        sub = per_obs_df[per_obs_df["deme"] == deme]
        sub_early = sub[sub["t_obs_days"] <= early_window_days]
        for col_idx, (col, label, colour) in enumerate(quantities):
            ax = axes_h2[row_idx, col_idx]
            vals_all = sub[col].to_numpy()
            vals_all = vals_all[np.isfinite(vals_all)]
            vals_early = sub_early[col].to_numpy()
            vals_early = vals_early[np.isfinite(vals_early)]
            if vals_all.size:
                ax.hist(
                    vals_all,
                    bins=max(10, min(30, vals_all.size // 3 + 5)),
                    color=colour,
                    alpha=0.35,
                    edgecolor="black",
                    linewidth=0.4,
                    label="all obs",
                )
                ax.axvline(
                    float(np.mean(vals_all)),
                    color=colour,
                    lw=1.2,
                    label=f"mean={np.mean(vals_all):.3g}",
                )
            if vals_early.size:
                ax.hist(
                    vals_early,
                    bins=max(5, min(20, vals_early.size // 2 + 3)),
                    color="gold",
                    alpha=0.45,
                    edgecolor="black",
                    linewidth=0.4,
                    label=f"first {early_window_days} d",
                )
                ax.axvline(
                    float(np.mean(vals_early)),
                    color="goldenrod",
                    lw=1.2,
                    ls="--",
                    label=f"early mean={np.mean(vals_early):.3g}",
                )
            ax.axvline(0, color="grey", lw=0.5)
            ax.set_xlabel(label)
            ax.set_ylabel("count")
            if row_idx == 0:
                ax.set_title(label)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{_deme_role_title(deme, starting_deme)}\ncount",
                    fontsize=9,
                )
            ax.legend(fontsize=6, loc="best")
    fig_h2.tight_layout()
    save_figure_png_and_pdf(str(output_dir / "spline_grid_histograms.png"))
    plt.close(fig_h2)

    # ---------- Summary CSV ----------
    summary_rows = []
    for deme in demes_ordered:
        for window_label, sub in [
            ("all", per_obs_df[per_obs_df["deme"] == deme]),
            (
                f"first_{early_window_days}d",
                per_obs_df[
                    (per_obs_df["deme"] == deme)
                    & (per_obs_df["t_obs_days"] <= early_window_days)
                ],
            ),
        ]:
            if sub.empty:
                continue
            eval_fwd = (sub["eval_delta_days"] > 0).sum()
            eval_back = (sub["eval_delta_days"] < 0).sum()
            eval_exact = (sub["eval_delta_days"] == 0).sum()

            def _stat(series):
                a = series.dropna().to_numpy()
                if a.size == 0:
                    return (np.nan, np.nan, np.nan, np.nan)
                return (
                    float(np.mean(a)),
                    float(np.median(a)),
                    float(np.std(a, ddof=1)) if a.size > 1 else float("nan"),
                    int(a.size),
                )

            row = {
                "deme": deme,
                "grid_mode": grid_mode,
                "role": (
                    "start"
                    if data["starting_deme"] is not None
                    and int(deme) == int(data["starting_deme"])
                    else "secondary"
                ),
                "window": window_label,
                "n_obs": int(sub.shape[0]),
                "eval_forward": int(eval_fwd),
                "eval_backward": int(eval_back),
                "eval_exact": int(eval_exact),
            }
            for col_name, series in [
                ("D_grid_spline", sub["D_grid_spline_med"]),
                ("D_grid_true", sub["D_grid_true"]),
                ("B_infer", sub["B_infer_med"]),
                ("B_total", sub["B_total_med"]),
            ]:
                mean_, med_, sd_, n_ = _stat(series)
                row[f"{col_name}_mean"] = mean_
                row[f"{col_name}_median"] = med_
                row[f"{col_name}_sd"] = sd_
            summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "spline_grid_summary.csv", index=False
    )
    return per_obs_df


def _thin_to_n(df, n):
    """Stride-sample a DataFrame down to ~n rows (returns the first n strided rows).

    Assumes burn-in is already removed. If len(df) <= n, returns a copy of df.
    """
    N = len(df)
    if N <= n:
        return df.reset_index(drop=True).copy()
    stride = max(1, N // n)
    idx = np.arange(0, N, stride)[:n]
    return df.iloc[idx].reset_index(drop=True)


def compute_wastewater_ppc(data, n_samples=1000, rng_seed=0, grid_mode="snap"):
    """Posterior predictive replicates of the wastewater observations.

    For each of ``n_samples`` thinned posterior draws s, for each obs time t_i
    in deme d, evaluate log prevalence at t_i using the scheme the BEAST
    likelihood used:

      * ``grid_mode="snap"`` (default): use the spline value at the nearest
        grid point (``_snap_to_grid``). Matches the earlier BEAST likelihood.
      * ``grid_mode="interpolate"``: precompute the spline at every grid
        point and linearly interpolate between the two bracketing grid
        values (``np.interp``). Matches the newer BEAST likelihood.

    Then:

      * read alpha_s = wastewater.scaling.Deme{d+1}:SimDataset
      * read sigma_s = wastewater.sigma:SimDataset
      * simulate  log y_rep_{s,i} = log(alpha_s) + log I_s(t_i) - log(N_d) + sigma_s * Z

    Returns a dict with:
      * ``per_obs_df``: per-observation summaries incl. PIT values and std residuals
      * ``log_y_rep_by_deme``: {deme: (S, n_obs) log-replicate matrix}
      * ``log_obs_by_deme``: {deme: (n_obs,) log observations}
      * ``sigma_samples``: (S,) posterior sigma draws
      * ``test_stats_by_deme`` / ``test_stats_pooled``: Bayesian p-values for
        T in {mean, sd, min, max} of log y
      * ``n_samples``: actual number of draws used
    """
    if grid_mode not in ("snap", "interpolate"):
        raise ValueError(
            f"grid_mode must be 'snap' or 'interpolate', got {grid_mode!r}"
        )

    log_ds = data["log_content_datastream"]
    ww = data["wastewater_data"]
    rate_shifts = np.asarray(data["rateshifts_datastream"], dtype=float)
    grid_shifts = np.asarray(data["gridpointshifts_datastream"], dtype=float)
    deme_popsizes = data["deme_popsizes"]

    if log_ds is None or log_ds.empty:
        print("No datastream log; cannot run wastewater PPC.")
        return None
    if ww is None or ww.empty:
        print("No wastewater data; cannot run wastewater PPC.")
        return None
    if rate_shifts.size == 0 or grid_shifts.size == 0:
        print("Missing rate/grid shifts; cannot run wastewater PPC.")
        return None

    sigma_col = "wastewater.sigma:SimDataset"
    if sigma_col not in log_ds.columns:
        print(f"Column {sigma_col} not in BEAST log; cannot run wastewater PPC.")
        return None

    # np.interp requires ascending x; grid_shifts is already ascending but
    # sort defensively in case callers pass a different order.
    grid_shifts_asc = np.sort(np.asarray(grid_shifts, dtype=float))

    log_ds_thin = _thin_to_n(log_ds, n_samples)
    S = len(log_ds_thin)
    sigma_samples = log_ds_thin[sigma_col].to_numpy(dtype=float)

    # Ground-truth sigma, if provided in the simulator parameter CSV. Used by
    # the variance-decomposition diagnostic (v_logmu_obs, sigma_reconstructed).
    params_df = data.get("params_df")
    sigma_true = None
    if params_df is not None and not params_df.empty:
        sig_row = params_df[params_df["parameter"] == "ds_ww_sigma"]
        if not sig_row.empty:
            sigma_true = float(sig_row["value"].iloc[0])

    # Posterior summary of sigma from the thinned draws (matches PPC cohort).
    if sigma_samples.size >= 2:
        hpd_lo, hpd_hi, hpd_med = calculate_hpd(sigma_samples, alpha=0.05)
        sigma_post_summary = {
            "median": float(hpd_med),
            "hpd_lower": float(hpd_lo),
            "hpd_upper": float(hpd_hi),
        }
    else:
        sigma_post_summary = None

    max_rateshift = float(rate_shifts.max())
    t_first = float(rate_shifts.min())
    t_last = float(rate_shifts.max())

    rng = np.random.default_rng(rng_seed)

    demes = sorted(int(d) for d in ww["Deme"].unique())
    per_obs_rows = []
    log_y_rep_by_deme = {}
    log_obs_by_deme = {}
    # Per-obs posterior variance of log μ (log α + log prev − log N) stored
    # per deme as a 1-D array of length n_obs_deme. Used downstream to
    # compute sigma_reconstructed = sqrt(sigma_post² + mean v_logmu_obs).
    v_logmu_per_obs_by_deme = {}

    for deme in demes:
        d_int = int(deme)
        if d_int not in deme_popsizes:
            print(f"Skipping deme {d_int}: no popsize in deme_popsizes.")
            continue
        N = float(deme_popsizes[d_int])

        prev_cols_sorted = sorted(
            [
                c
                for c in log_ds.columns
                if c.startswith(f"SkylinePrev.Deme{d_int + 1}.")
            ],
            key=lambda c: int(c.split(".")[-1]),
        )
        if len(prev_cols_sorted) != rate_shifts.size:
            print(
                f"Skipping deme {d_int}: {len(prev_cols_sorted)} SkylinePrev "
                f"columns != {rate_shifts.size} rate shifts."
            )
            continue

        alpha_col = f"wastewater.scaling.Deme{d_int + 1}:SimDataset"
        if alpha_col not in log_ds.columns:
            print(f"Skipping deme {d_int}: column {alpha_col} not in BEAST log.")
            continue
        alpha_samples = log_ds_thin[alpha_col].to_numpy(dtype=float)
        knot_samples = log_ds_thin[prev_cols_sorted].to_numpy(dtype=float)

        obs_d = ww[ww["Deme"] == d_int].copy().reset_index(drop=True)
        if obs_d.empty:
            continue
        t_obs_fwd = obs_d["t_wastewater_fromsimstart"].to_numpy(dtype=float)
        t_obs_back = max_rateshift - t_obs_fwd

        # Effective evaluation time for the likelihood (shown in the per-obs
        # CSV's t_snap_* columns). Only differs from t_obs when grid_mode
        # snaps to a grid point.
        if grid_mode == "snap":
            t_eval_back, _ = _snap_to_grid(t_obs_back, grid_shifts_asc)
        else:  # interpolate
            t_eval_back = t_obs_back.copy()
        t_eval_fwd = max_rateshift - t_eval_back

        y_obs = obs_d["wastewater"].to_numpy(dtype=float)
        log_obs = np.log(np.clip(y_obs, 1e-300, None))

        n_obs = len(obs_d)
        log_I_at_obs = np.empty((S, n_obs), dtype=float)
        for s in range(S):
            spline = _build_natural_spline(rate_shifts, knot_samples[s])
            if grid_mode == "snap":
                log_I_at_obs[s, :] = _eval_spline_clamped(
                    spline, t_eval_back, t_first, t_last
                )
            else:
                # Precompute the spline at each grid point, then linearly
                # interpolate between the two bracketing grid values at
                # each observation time. Matches the newer BEAST likelihood.
                grid_vals = _eval_spline_clamped(
                    spline, grid_shifts_asc, t_first, t_last
                )
                log_I_at_obs[s, :] = np.interp(t_obs_back, grid_shifts_asc, grid_vals)

        log_mu = (
            np.log(alpha_samples)[:, None] + log_I_at_obs - np.log(N)
        )  # (S, n_obs): log of predictive mean
        eps = rng.standard_normal((S, n_obs)) * sigma_samples[:, None]
        log_y_rep = log_mu + eps  # (S, n_obs): log of predictive draws

        log_y_rep_by_deme[d_int] = log_y_rep
        log_obs_by_deme[d_int] = log_obs
        # Posterior variance of the predictive mean log μ at each obs time;
        # mean(v_logmu) + sigma_post² is the predictive total log-variance.
        if S > 1:
            v_logmu_per_obs_by_deme[d_int] = np.var(log_mu, axis=0, ddof=1)
        else:
            v_logmu_per_obs_by_deme[d_int] = np.zeros(n_obs)

        lm_med = np.median(log_mu, axis=0)
        yr_med = np.median(log_y_rep, axis=0)
        lm_lo = np.empty(n_obs)
        lm_hi = np.empty(n_obs)
        yr_lo95 = np.empty(n_obs)
        yr_hi95 = np.empty(n_obs)
        yr_lo50 = np.empty(n_obs)
        yr_hi50 = np.empty(n_obs)
        for j in range(n_obs):
            lo, hi, _ = calculate_hpd(log_mu[:, j], alpha=0.05)
            lm_lo[j] = lo
            lm_hi[j] = hi
            lo, hi, _ = calculate_hpd(log_y_rep[:, j], alpha=0.05)
            yr_lo95[j] = lo
            yr_hi95[j] = hi
            lo, hi, _ = calculate_hpd(log_y_rep[:, j], alpha=0.50)
            yr_lo50[j] = lo
            yr_hi50[j] = hi

        # Empirical PIT: fraction of replicate draws at or below the obs.
        # A tiny ordering jitter at ties shouldn't bias a continuous lognormal.
        pit = np.mean(log_y_rep <= log_obs[None, :], axis=0)

        # Standardized residual using posterior-median mean and sigma.
        sigma_med = float(np.median(sigma_samples))
        std_resid = (
            (log_obs - lm_med) / sigma_med if sigma_med > 0 else np.full(n_obs, np.nan)
        )

        for j in range(n_obs):
            per_obs_rows.append(
                {
                    "deme": d_int,
                    "t_obs_years": float(t_obs_fwd[j]),
                    "t_obs_days": float(t_obs_fwd[j] * 365.0),
                    "t_snap_years": float(t_eval_fwd[j]),
                    "t_snap_days": float(t_eval_fwd[j] * 365.0),
                    "y_obs": float(y_obs[j]),
                    "log_obs": float(log_obs[j]),
                    "log_mu_rep_med": float(lm_med[j]),
                    "log_mu_rep_lo95": float(lm_lo[j]),
                    "log_mu_rep_hi95": float(lm_hi[j]),
                    "log_y_rep_med": float(yr_med[j]),
                    "log_y_rep_lo95": float(yr_lo95[j]),
                    "log_y_rep_hi95": float(yr_hi95[j]),
                    "log_y_rep_lo50": float(yr_lo50[j]),
                    "log_y_rep_hi50": float(yr_hi50[j]),
                    "pit": float(pit[j]),
                    "std_resid": float(std_resid[j]),
                }
            )

    per_obs_df = pd.DataFrame(per_obs_rows)

    def _test_stats(log_y_rep_mat, log_obs_vec):
        rep = log_y_rep_mat
        obs = log_obs_vec
        has_var = rep.shape[1] > 1
        stats = {
            "mean": (float(np.mean(obs)), np.mean(rep, axis=1)),
            "sd": (
                float(np.std(obs, ddof=1)) if obs.size > 1 else float("nan"),
                (
                    np.std(rep, axis=1, ddof=1)
                    if has_var
                    else np.full(rep.shape[0], np.nan)
                ),
            ),
            "min": (float(np.min(obs)), np.min(rep, axis=1)),
            "max": (float(np.max(obs)), np.max(rep, axis=1)),
        }
        pvals = {}
        for k, (t_obs_scalar, t_rep_arr) in stats.items():
            rep_finite = t_rep_arr[np.isfinite(t_rep_arr)]
            if rep_finite.size == 0 or not np.isfinite(t_obs_scalar):
                pvals[k] = float("nan")
            else:
                pvals[k] = float(np.mean(rep_finite >= t_obs_scalar))
        return stats, pvals

    test_stats_by_deme = {}
    for d_int, mat in log_y_rep_by_deme.items():
        stats, pvals = _test_stats(mat, log_obs_by_deme[d_int])
        test_stats_by_deme[d_int] = {"stats": stats, "pvals": pvals}

    if log_y_rep_by_deme:
        pooled_rep = np.concatenate(list(log_y_rep_by_deme.values()), axis=1)
        pooled_obs = np.concatenate(list(log_obs_by_deme.values()))
        stats, pvals = _test_stats(pooled_rep, pooled_obs)
        test_stats_pooled = {"stats": stats, "pvals": pvals}
    else:
        test_stats_pooled = None

    return {
        "per_obs_df": per_obs_df,
        "log_y_rep_by_deme": log_y_rep_by_deme,
        "log_obs_by_deme": log_obs_by_deme,
        "sigma_samples": sigma_samples,
        "test_stats_by_deme": test_stats_by_deme,
        "test_stats_pooled": test_stats_pooled,
        "n_samples": S,
        "sigma_true": sigma_true,
        "sigma_post_summary": sigma_post_summary,
        "v_logmu_per_obs_by_deme": v_logmu_per_obs_by_deme,
    }


def summarize_wastewater_ppc(out, starting_deme):
    """Collapse a wastewater PPC run to tidy scalar metrics per (deme, pooled).

    Emits one row per scope with the scalars designed for cross-sim aggregation:

      * ``mean_pit``            target 0.5     — location bias
      * ``var_pit``             target 1/12    — dispersion (Uniform variance)
      * ``mean_std_resid``      target 0       — location bias on residual scale
      * ``sd_std_resid``        target 1       — dispersion on residual scale
      * ``cov_50``, ``cov_95``  targets 0.5, 0.95 — empirical coverage
      * ``ks_pit``              KS distance of PIT from Uniform(0,1)
      * ``ks_pit_pvalue``
      * ``p_T_{mean,sd,min,max}`` Bayesian p-values for T in
        {mean, sd, min, max} of log y; each should be U(0,1) across sims
        if the posterior is calibrated (SBC check).

    Variance-decomposition fields also added per row (same value repeated
    across scopes where sigma is shared, v_logmu_obs differs by scope):

      * ``sigma_true``              from simulator params CSV (NaN if missing)
      * ``sigma_post_median``       posterior median of wastewater sigma
      * ``sigma_post_hpd_lower``    95% HPD lower bound
      * ``sigma_post_hpd_upper``    95% HPD upper bound
      * ``v_logmu_obs``             mean_i Var_s( log μ_s(t_i) ), the posterior
        variance of the predictive log-mean at observation times, averaged
        over obs in that scope.
      * ``sigma_reconstructed``     sqrt(sigma_post_median² + v_logmu_obs).
        If the posterior has correctly split total observational noise into
        spline-uncertainty + σ, this should land on σ_true.
      * ``sigma_reconstructed_hpd_lower`` / ``_upper``
        95% HPD for sigma_reconstructed obtained by the monotonic transform
        sqrt(sig² + v_logmu_obs) applied to the sigma_post HPD endpoints.
      * ``sigma_post_over_true``    sigma_post_median / sigma_true
      * ``sigma_reconstructed_over_true`` sigma_reconstructed / sigma_true

    Parameters
    ----------
    out : dict
        Return value of ``compute_wastewater_ppc``.
    starting_deme : int or None
        Used only to assign each row a ``role`` label in
        {"start", "secondary", "pooled"} for cross-sim grouping.

    Returns
    -------
    pd.DataFrame
        Columns: scope, deme, role, n_obs, n_samples, + the scalars above.
    """
    per_obs_df = out["per_obs_df"]
    S = out["n_samples"]
    sigma_true = out.get("sigma_true")
    sigma_post_summary = out.get("sigma_post_summary")
    v_logmu_by_deme = out.get("v_logmu_per_obs_by_deme", {}) or {}

    def _role(deme):
        if starting_deme is None:
            return f"deme_{deme}"
        return "start" if int(deme) == int(starting_deme) else "secondary"

    def _v_logmu_for_scope(scope, deme):
        if scope == "pooled":
            if not v_logmu_by_deme:
                return float("nan")
            all_v = np.concatenate(list(v_logmu_by_deme.values()))
        else:
            arr = v_logmu_by_deme.get(int(deme))
            if arr is None:
                return float("nan")
            all_v = np.asarray(arr)
        all_v = all_v[np.isfinite(all_v)]
        return float(np.mean(all_v)) if all_v.size else float("nan")

    def _row(sub, ts, scope, deme, role):
        pit = sub["pit"].to_numpy()
        zr = sub["std_resid"].to_numpy()
        zr = zr[np.isfinite(zr)]
        in50 = (
            (sub["log_obs"] >= sub["log_y_rep_lo50"])
            & (sub["log_obs"] <= sub["log_y_rep_hi50"])
        ).to_numpy()
        in95 = (
            (sub["log_obs"] >= sub["log_y_rep_lo95"])
            & (sub["log_obs"] <= sub["log_y_rep_hi95"])
        ).to_numpy()
        if pit.size >= 2:
            ks_res = kstest(pit, "uniform")
            ks_stat = float(ks_res.statistic)
            ks_p = float(ks_res.pvalue)
        else:
            ks_stat = float("nan")
            ks_p = float("nan")
        pvals = ts["pvals"]

        sig_med = (
            float(sigma_post_summary["median"])
            if sigma_post_summary is not None
            else float("nan")
        )
        sig_lo = (
            float(sigma_post_summary["hpd_lower"])
            if sigma_post_summary is not None
            else float("nan")
        )
        sig_hi = (
            float(sigma_post_summary["hpd_upper"])
            if sigma_post_summary is not None
            else float("nan")
        )
        sig_true = float(sigma_true) if sigma_true is not None else float("nan")
        v_logmu = _v_logmu_for_scope(scope, deme)
        if np.isfinite(sig_med) and np.isfinite(v_logmu):
            sig_recon = float(np.sqrt(sig_med**2 + v_logmu))
        else:
            sig_recon = float("nan")
        sig_post_over_true = (
            sig_med / sig_true
            if np.isfinite(sig_true) and sig_true > 0
            else float("nan")
        )
        sig_recon_over_true = (
            sig_recon / sig_true
            if np.isfinite(sig_true) and sig_true > 0 and np.isfinite(sig_recon)
            else float("nan")
        )
        # HPD endpoints for sigma_reconstructed follow from the sigma_post
        # HPD by the monotonic transform sqrt(sig² + v_logmu_obs), treating
        # v_logmu_obs as a fixed posterior-summary scalar.
        if np.isfinite(sig_lo) and np.isfinite(v_logmu):
            sig_recon_lo = float(np.sqrt(sig_lo**2 + v_logmu))
        else:
            sig_recon_lo = float("nan")
        if np.isfinite(sig_hi) and np.isfinite(v_logmu):
            sig_recon_hi = float(np.sqrt(sig_hi**2 + v_logmu))
        else:
            sig_recon_hi = float("nan")

        return {
            "scope": scope,
            "deme": int(deme),
            "role": role,
            "n_obs": int(sub.shape[0]),
            "n_samples": int(S),
            "mean_pit": float(np.mean(pit)) if pit.size else float("nan"),
            "var_pit": (float(np.var(pit, ddof=1)) if pit.size > 1 else float("nan")),
            "mean_std_resid": float(np.mean(zr)) if zr.size else float("nan"),
            "sd_std_resid": (
                float(np.std(zr, ddof=1)) if zr.size > 1 else float("nan")
            ),
            "cov_50": float(np.mean(in50)) if in50.size else float("nan"),
            "cov_95": float(np.mean(in95)) if in95.size else float("nan"),
            "ks_pit": ks_stat,
            "ks_pit_pvalue": ks_p,
            "p_T_mean": float(pvals.get("mean", float("nan"))),
            "p_T_sd": float(pvals.get("sd", float("nan"))),
            "p_T_min": float(pvals.get("min", float("nan"))),
            "p_T_max": float(pvals.get("max", float("nan"))),
            "sigma_true": sig_true,
            "sigma_post_median": sig_med,
            "sigma_post_hpd_lower": sig_lo,
            "sigma_post_hpd_upper": sig_hi,
            "v_logmu_obs": v_logmu,
            "sigma_reconstructed": sig_recon,
            "sigma_reconstructed_hpd_lower": sig_recon_lo,
            "sigma_reconstructed_hpd_upper": sig_recon_hi,
            "sigma_post_over_true": sig_post_over_true,
            "sigma_reconstructed_over_true": sig_recon_over_true,
        }

    rows = []
    for deme, ts in out["test_stats_by_deme"].items():
        sub = per_obs_df[per_obs_df["deme"] == deme]
        rows.append(_row(sub, ts, scope="per_deme", deme=deme, role=_role(deme)))
    if out["test_stats_pooled"] is not None:
        rows.append(
            _row(
                per_obs_df,
                out["test_stats_pooled"],
                scope="pooled",
                deme=-1,
                role="pooled",
            )
        )
    return pd.DataFrame(rows)


def plot_wastewater_ppc(data, output_dir, n_samples=1000, rng_seed=0, grid_mode="snap"):
    """Run the wastewater posterior predictive check and write PNG+PDF plots + CSVs.

    Figures (PNG + PDF):
      * ``ww_ppc_ribbon.png``         — per deme: 50% and 95% posterior predictive
        bands and median vs time (days), observed wastewater overlaid.
      * ``ww_ppc_pit_hist.png``       — per deme + pooled: PIT histogram with
        uniform reference line and ±2 SE band. Flat = calibrated; U = under-
        dispersed; dome = overdispersed; tilted = bias.
      * ``ww_ppc_std_resid_vs_time.png`` — per deme: (log y_obs − med log μ_rep) /
        med σ vs time. Trend over time flags systematic bias.
      * ``ww_ppc_test_stats.png``     — per deme and pooled, for T in
        {mean, sd, min, max} of log y: histogram of T(y_rep) across posterior
        draws with T(y_obs) marked and Bayesian p-value in the title.
      * ``ww_ppc_qq.png``             — per deme and pooled: Q–Q plot of
        standardized residuals against N(0, 1).

    CSVs:
      * ``ww_ppc_per_obs.csv``               — per-observation summaries
      * ``ww_ppc_test_stats_summary.csv``    — Bayesian p-values per deme + pooled
    """
    configure_pdf_fonts()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out = compute_wastewater_ppc(
        data,
        n_samples=n_samples,
        rng_seed=rng_seed,
        grid_mode=grid_mode,
    )
    if out is None:
        return None
    per_obs_df = out["per_obs_df"]
    if per_obs_df.empty:
        return per_obs_df

    per_obs_df.to_csv(output_dir / "ww_ppc_per_obs.csv", index=False)

    starting_deme = data["starting_deme"]
    scalars_df = summarize_wastewater_ppc(out, starting_deme)
    scalars_df.to_csv(output_dir / "ww_ppc_summary_scalars.csv", index=False)

    demes_ordered = [
        int(d) for d in _ordered_demes(data) if int(d) in out["log_y_rep_by_deme"]
    ]
    n_demes = len(demes_ordered)
    if n_demes == 0:
        return per_obs_df

    S_used = out["n_samples"]

    # ---------- 1. Ribbon plot per deme ----------
    fig, axes = plt.subplots(n_demes, 1, figsize=(9, 3.5 * n_demes), squeeze=False)
    for i, deme in enumerate(demes_ordered):
        ax = axes[i, 0]
        sub = per_obs_df[per_obs_df["deme"] == deme].sort_values("t_obs_days")
        t = sub["t_obs_days"].to_numpy()
        ax.fill_between(
            t,
            sub["log_y_rep_lo95"],
            sub["log_y_rep_hi95"],
            color="tab:blue",
            alpha=0.18,
            label="95% PPC",
        )
        ax.fill_between(
            t,
            sub["log_y_rep_lo50"],
            sub["log_y_rep_hi50"],
            color="tab:blue",
            alpha=0.35,
            label="50% PPC",
        )
        ax.plot(t, sub["log_y_rep_med"], color="tab:blue", lw=1.2, label="PPC median")
        ax.scatter(t, sub["log_obs"], s=20, color="black", zorder=5, label="observed")
        ax.set_xlabel("Days from simulation start")
        ax.set_ylabel("log(wastewater concentration)")
        ax.set_title(
            f"{_deme_role_title(deme, starting_deme)} — posterior predictive "
            f"(S={S_used})"
        )
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_figure_png_and_pdf(str(output_dir / "ww_ppc_ribbon.png"))
    plt.close(fig)

    # ---------- 2. PIT histogram per deme + pooled ----------
    n_bins = 20
    fig, axes = plt.subplots(
        1,
        n_demes + 1,
        figsize=(4 * (n_demes + 1), 3.5),
        squeeze=False,
    )
    panel_specs = [
        (
            deme,
            per_obs_df[per_obs_df["deme"] == deme]["pit"].to_numpy(),
            _deme_role_title(deme, starting_deme),
            "tab:blue",
        )
        for deme in demes_ordered
    ]
    panel_specs.append((None, per_obs_df["pit"].to_numpy(), "Pooled", "tab:purple"))
    for i, (deme, u, title, colour) in enumerate(panel_specs):
        ax = axes[0, i]
        n = u.size
        expected = n / n_bins
        ax.hist(
            u,
            bins=n_bins,
            range=(0, 1),
            color=colour,
            alpha=0.6,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.axhline(
            expected, color="red", ls="--", lw=1, label=f"expected count (n/{n_bins})"
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"PIT $u_i$")
        ax.set_ylabel("count")
        ax.set_title(f"{title} (n={n})")
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    save_figure_png_and_pdf(str(output_dir / "ww_ppc_pit_hist.png"))
    plt.close(fig)

    # ---------- 3. Standardized residuals vs time ----------
    fig, axes = plt.subplots(n_demes, 1, figsize=(9, 3.0 * n_demes), squeeze=False)
    for i, deme in enumerate(demes_ordered):
        ax = axes[i, 0]
        sub = per_obs_df[per_obs_df["deme"] == deme].sort_values("t_obs_days")
        ax.axhline(0, color="grey", lw=0.5)
        for y in (-2, -1, 1, 2):
            ax.axhline(y, color="grey", lw=0.3, ls=":")
        ax.scatter(
            sub["t_obs_days"], sub["std_resid"], s=20, color="tab:orange", alpha=0.8
        )
        ax.set_xlabel("Days from simulation start")
        ax.set_ylabel(
            r"std resid $(\log y_{obs} - \mathrm{med}\,\log\mu_{rep})/\mathrm{med}\,\sigma$"
        )
        ax.set_title(
            f"{_deme_role_title(deme, starting_deme)} — standardized residuals"
        )
    fig.tight_layout()
    save_figure_png_and_pdf(str(output_dir / "ww_ppc_std_resid_vs_time.png"))
    plt.close(fig)

    # ---------- 4. Test statistics (mean, sd, min, max of log y) ----------
    stat_names = ["mean", "sd", "min", "max"]
    n_rows_ts = n_demes + (1 if out["test_stats_pooled"] is not None else 0)
    fig, axes = plt.subplots(
        n_rows_ts,
        len(stat_names),
        figsize=(3.2 * len(stat_names), 2.8 * n_rows_ts),
        squeeze=False,
    )
    summary_rows = []
    row_specs = [
        (
            deme,
            test_stats_by_deme_entry,
            _deme_role_title(deme, starting_deme),
            "tab:blue",
            "per_deme",
        )
        for deme, test_stats_by_deme_entry in (
            (d, out["test_stats_by_deme"][d]) for d in demes_ordered
        )
    ]
    if out["test_stats_pooled"] is not None:
        row_specs.append(
            (None, out["test_stats_pooled"], "Pooled", "tab:purple", "pooled")
        )
    for row_idx, (deme, ts, title, colour, scope) in enumerate(row_specs):
        stats = ts["stats"]
        pvals = ts["pvals"]
        for col_idx, name in enumerate(stat_names):
            ax = axes[row_idx, col_idx]
            t_obs_val, t_rep_arr = stats[name]
            t_rep_finite = t_rep_arr[np.isfinite(t_rep_arr)]
            if t_rep_finite.size:
                ax.hist(
                    t_rep_finite,
                    bins=30,
                    color=colour,
                    alpha=0.55,
                    edgecolor="black",
                    linewidth=0.4,
                )
            if np.isfinite(t_obs_val):
                ax.axvline(
                    t_obs_val, color="red", lw=1.5, label=f"obs = {t_obs_val:.3g}"
                )
            p = pvals[name]
            p_str = f"{p:.3f}" if np.isfinite(p) else "nan"
            ax.set_xlabel(f"T = {name}(log y)")
            ax.set_ylabel("count")
            ax.set_title(f"{title}\np = {p_str}", fontsize=9)
            ax.legend(fontsize=7, loc="best")
            summary_rows.append(
                {
                    "scope": scope,
                    "deme": int(deme) if deme is not None else -1,
                    "statistic": name,
                    "T_obs": t_obs_val,
                    "T_rep_mean": (
                        float(np.mean(t_rep_finite))
                        if t_rep_finite.size
                        else float("nan")
                    ),
                    "T_rep_sd": (
                        float(np.std(t_rep_finite, ddof=1))
                        if t_rep_finite.size > 1
                        else float("nan")
                    ),
                    "p_value": p,
                    "n_obs": (
                        int((per_obs_df["deme"] == deme).sum())
                        if deme is not None
                        else int(per_obs_df.shape[0])
                    ),
                    "n_samples": S_used,
                }
            )
    fig.tight_layout()
    save_figure_png_and_pdf(str(output_dir / "ww_ppc_test_stats.png"))
    plt.close(fig)

    # ---------- 5. Q–Q plot of standardized residuals vs N(0, 1) ----------
    qq_specs = [
        (
            per_obs_df[per_obs_df["deme"] == deme]["std_resid"].to_numpy(),
            _deme_role_title(deme, starting_deme),
        )
        for deme in demes_ordered
    ]
    qq_specs.append((per_obs_df["std_resid"].to_numpy(), "Pooled"))
    fig, axes = plt.subplots(
        1,
        len(qq_specs),
        figsize=(4 * len(qq_specs), 4),
        squeeze=False,
    )
    for i, (z, title) in enumerate(qq_specs):
        ax = axes[0, i]
        z = z[np.isfinite(z)]
        if z.size > 1:
            z_sorted = np.sort(z)
            theoretical = norm.ppf((np.arange(1, z.size + 1) - 0.5) / z.size)
            ax.scatter(theoretical, z_sorted, s=14, color="tab:orange", alpha=0.85)
            lim_lo = float(min(theoretical[0], z_sorted[0]))
            lim_hi = float(max(theoretical[-1], z_sorted[-1]))
            ax.plot(
                [lim_lo, lim_hi],
                [lim_lo, lim_hi],
                ls="--",
                color="grey",
                lw=0.8,
                label="y = x",
            )
            ax.legend(fontsize=7, loc="best")
        ax.set_xlabel("Theoretical quantile (N(0,1))")
        ax.set_ylabel("Empirical std resid quantile")
        ax.set_title(f"{title} (n={z.size})")
    fig.suptitle("Q–Q of standardized residuals vs N(0, 1)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure_png_and_pdf(str(output_dir / "ww_ppc_qq.png"))
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "ww_ppc_test_stats_summary.csv", index=False
    )
    return per_obs_df


def main():
    """Load data and produce one publication figure per (deme, panel) (PNG + PDF).

    When ``--intro_figure_out`` is given, the combined intro figure is also
    written at that path. ``--skip_per_deme_figures`` suppresses the individual
    per-deme files (useful when only the intro figure is needed).
    """
    # Intro-figure-specific flags are extracted here so analyse_posteriors'
    # parse_arguments (which reads sys.argv directly) doesn't choke on them.
    # ``add_help=False`` keeps ``--help`` routed to the downstream parser, which
    # has the full list of shared options; the intro flags are listed in this
    # module's docstring so ``head make_figure_individualsim.py`` surfaces them.
    local_parser = argparse.ArgumentParser(add_help=False)
    local_parser.add_argument(
        "--intro_figure_out",
        type=str,
        default=None,
        help="If set, also write final_mascotds_intro_figure PNG/PDF at this path.",
    )
    local_parser.add_argument(
        "--skip_per_deme_figures",
        action="store_true",
        help="Suppress per-(deme, panel) file outputs.",
    )
    local_parser.add_argument(
        "--ww_diagnostics_out",
        type=str,
        default=None,
        help=(
            "If set, write wastewater residual diagnostics into this directory: "
            "per-deme PNG+PDF plots, ww_residuals_per_obs.csv, "
            "and ww_residuals_summary.csv."
        ),
    )
    local_parser.add_argument(
        "--spline_gridpoint_diagnostics_out",
        type=str,
        default=None,
        help=(
            "If set, write spline grid-evaluation diagnostics into this "
            "directory: posterior spline vs truth zoom plot, Δ/B vs time "
            "plots, paired scatter, histograms, plus spline_grid_per_obs.csv "
            "and spline_grid_summary.csv.  The evaluation method (snap vs "
            "interpolate) is controlled by --ww_ppc_grid_mode."
        ),
    )
    local_parser.add_argument(
        "--spline_gridpoint_early_window_days",
        type=float,
        default=10.0,
        help="Early-window cutoff in days for spline gridpoint diagnostics (default 10).",
    )
    local_parser.add_argument(
        "--ww_ppc_out",
        type=str,
        default=None,
        help=(
            "If set, write wastewater posterior predictive check outputs into "
            "this directory: ribbon, PIT histogram, standardized residuals vs "
            "time, test-statistic histograms, and Q–Q plot, plus "
            "ww_ppc_per_obs.csv and ww_ppc_test_stats_summary.csv."
        ),
    )
    local_parser.add_argument(
        "--ww_ppc_n_samples",
        type=int,
        default=1000,
        help="Thinned posterior draws to use for wastewater PPC (default 1000).",
    )
    local_parser.add_argument(
        "--ww_ppc_seed",
        type=int,
        default=0,
        help="RNG seed for wastewater PPC replicate draws (default 0).",
    )
    local_parser.add_argument(
        "--ww_ppc_grid_mode",
        choices=["snap", "interpolate"],
        default="snap",
        help=(
            "How to evaluate prevalence at each wastewater observation time.  "
            "Affects both the wastewater PPC (--ww_ppc_out) and the spline "
            "gridpoint diagnostics (--spline_gridpoint_diagnostics_out).  "
            "'snap' uses the value at the nearest grid point (earlier BEAST "
            "likelihood). 'interpolate' linearly interpolates between the two "
            "bracketing grid-point spline values (newer BEAST likelihood). "
            "Must match the likelihood used in the MCMC run whose log is "
            "being analyzed. Default: snap."
        ),
    )
    local_args, remaining = local_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = parse_arguments()

    data = prepare_skyline_plot_data(args)
    if not local_args.skip_per_deme_figures:
        run_per_deme_figures(data, args, time_unit="days")
    if local_args.intro_figure_out is not None:
        plot_final_mascotds_intro_figure(
            data, local_args.intro_figure_out, time_unit="days"
        )
    if local_args.ww_diagnostics_out is not None:
        plot_wastewater_residual_diagnostics(
            data, local_args.ww_diagnostics_out, time_unit="days"
        )
    if local_args.spline_gridpoint_diagnostics_out is not None:
        plot_spline_gridpoint_diagnostics(
            data,
            local_args.spline_gridpoint_diagnostics_out,
            early_window_days=local_args.spline_gridpoint_early_window_days,
            grid_mode=local_args.ww_ppc_grid_mode,
        )
    if local_args.ww_ppc_out is not None:
        plot_wastewater_ppc(
            data,
            local_args.ww_ppc_out,
            n_samples=local_args.ww_ppc_n_samples,
            rng_seed=local_args.ww_ppc_seed,
            grid_mode=local_args.ww_ppc_grid_mode,
        )


if __name__ == "__main__":
    main()
