#!/usr/bin/env python3
"""
Create publication-ready figures per deme: one PNG and one PDF per (deme, panel).

Each panel type (prevalence, ne, cumincidence) is saved as its own file per deme,
with the title clearly labeling start vs secondary deme. The optional
``*_prevalence_log`` file uses log scale on the prevalence axis only (case counts
and wastewater remain linear). PDFs use
embedded TrueType fonts (pdf.fonttype=42) so text stays editable in Illustrator.
Reuses plotting logic from analyse_posteriors.py.
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import (
    DEFAULT_FONTSIZES,
    FONTSIZES_LIST,
    save_figure_png_and_pdf,
)
from analyse_posteriors import (
    parse_arguments,
    prepare_skyline_plot_data,
    _plot_prevalence_panel,
    _plot_ne_panel,
    _plot_cumincidence_panel,
)


def _deme_label_for_filename(deme: int, starting_deme: int) -> str:
    """Return a short filesystem-safe label for the deme (start_deme or secondary_deme)."""
    return (
        "start_deme"
        if int(deme) == int(starting_deme)
        else "secondary_deme"
    )


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

    demes = sorted(hpd_original["Deme"].unique())
    if data["starting_deme"] is not None and len(demes) == 2:
        other = [d for d in demes if int(d) != int(data["starting_deme"])]
        if len(other) == 1:
            demes = [int(data["starting_deme"]), int(other[0])]

    fontsize_tick = DEFAULT_FONTSIZES["tick_label"]
    hpd_datastream = (
        data["hpd_datastream"]
        if data["hpd_datastream"] is not None
        else pd.DataFrame()
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


def main():
    """Load data and produce one publication figure per (deme, panel) (PNG + PDF)."""
    parser = argparse.ArgumentParser(
        description="Create publication-ready per-deme figures (prevalence, ne, cumincidence) as separate files."
    )
    # Reuse the same arguments as analyse_posteriors so we can call prepare_skyline_plot_data
    args = parse_arguments()

    data = prepare_skyline_plot_data(args)
    run_per_deme_figures(data, args, time_unit="days")


if __name__ == "__main__":
    main()
