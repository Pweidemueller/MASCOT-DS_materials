#!/usr/bin/env python3
"""
Script to create HPD validation plots for cumulative incidence, ne, prevalence, and parameters.
"""

import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from constants import COLORBLINDFR, COLORS, MODEL_MASCOT, MODEL_MASCOT_DS
from plot_utils import save_figure_png_and_pdf, set_axis_fontsizes

logger = logging.getLogger(__name__)

fontsizes = [18, 15, 12]

colorblindfr = COLORBLINDFR
colors = COLORS


def assign_gridpoint_indices(df):
    """
    Assign gridpoint indices based on unique timesincestart values.
    Each unique timesincestart gets an index starting at 0, per simulation.
    """
    df = df.copy()
    df["gridpoint_index"] = None

    # Group by Simulation and assign indices per simulation
    for sim, group in df.groupby("Simulation"):
        unique_times = sorted(group["timesincestart"].unique())
        time_to_index = {time: idx for idx, time in enumerate(unique_times)}
        df.loc[df["Simulation"] == sim, "gridpoint_index"] = df.loc[
            df["Simulation"] == sim, "timesincestart"
        ].map(time_to_index)

    # Get the maximum number of gridpoints across all simulations
    max_gridpoints = df.groupby("Simulation")["gridpoint_index"].max().max() + 1

    return df, max_gridpoints


def create_hpd_plot(
    df, title, x_label, output_path, is_parameters=False, simulations_with_asterisk=None
):
    """
    Create a 3-subplot HPD validation plot.

    Parameters:
    -----------
    df : DataFrame
        Data with inHPD, Simulation, and either gridpoint_index or Parameter
    title : str
        Plot title
    x_label : str
        Label for x-axis (either "Gridpoint Index" or "Parameter")
    output_path : str
        Path to save the plot
    is_parameters : bool
        Whether this is the parameters file (uses Parameter column instead of gridpoint_index)
    simulations_with_asterisk : set, optional
        Set of simulation names that should have an asterisk appended to their y-axis label
    """
    if simulations_with_asterisk is None:
        simulations_with_asterisk = set()

    # Prepare data
    if is_parameters:
        # For parameters, use Parameter as x-axis
        df = df.copy()
        if "Model" in df.columns:
            df = df[df["Model"] == MODEL_MASCOT_DS]
        df["x_value"] = df["Parameter"]
        unique_x = df["Parameter"].unique()
        x_to_index = {x: idx for idx, x in enumerate(unique_x)}
        df["x_index"] = df["x_value"].map(x_to_index)
        x_col = "x_index"
        x_labels = unique_x
    else:
        # For data files, use gridpoint_index
        df = df.copy()
        x_col = "gridpoint_index"
        x_labels = sorted(df[x_col].unique())

    # Get unique simulations and calculate percentages for sorting
    unique_sims = df["Simulation"].unique()

    # Calculate percentage for each simulation to sort by
    sim_percentages_dict = {}
    for sim in unique_sims:
        subset = df[df["Simulation"] == sim]
        if len(subset) > 0:
            percentage = (subset["inHPD"] == 1).sum() / len(subset) * 100
        else:
            percentage = 0
        sim_percentages_dict[sim] = percentage

    # Sort simulations by percentage (descending - highest first)
    unique_sims = sorted(
        unique_sims, key=lambda x: sim_percentages_dict[x], reverse=True
    )
    sim_to_index = {sim: idx for idx, sim in enumerate(unique_sims)}
    df["sim_index"] = df["Simulation"].map(sim_to_index)

    # Create figure with custom layout
    width = np.ceil((len(x_labels) / 8 + 5)).astype(int)
    height = 8
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[6, 1],
        height_ratios=[1, 3],
        hspace=0.05,
        wspace=0.05,
    )

    # Main plot (bottom left, takes up 3x3 of the grid)
    ax_main = fig.add_subplot(gs[1, 0])

    n_x = len(x_labels)
    n_sims = len(unique_sims)

    # Vectorized approach: extract coordinates as arrays
    x_coords = df[x_col].values
    y_coords = df["sim_index"].values
    in_hpd = df["inHPD"].values

    # Create masks for inHPD == 1 and inHPD == 0
    mask_in_hpd = in_hpd == 1
    mask_not_in_hpd = in_hpd == 0

    # Calculate marker size to fill the space appropriately
    if n_x > 0 and n_sims > 0:
        # Calculate size based on figure dimensions to make rectangles fill the space
        marker_size = max(500 / n_x, 500 / n_sims)
    else:
        marker_size = 10

    # Plot scatter with square markers - two separate plots for better performance
    # Orange (colors[3]) for inHPD == 1
    if mask_in_hpd.any():
        ax_main.scatter(
            x_coords[mask_in_hpd],
            y_coords[mask_in_hpd],
            c=colors[3],
            s=marker_size,
            marker="s",
            edgecolors="none",
        )
    # Grey for inHPD == 0
    if mask_not_in_hpd.any():
        ax_main.scatter(
            x_coords[mask_not_in_hpd],
            y_coords[mask_not_in_hpd],
            c="grey",
            s=marker_size,
            marker="s",
            edgecolors="none",
        )

    # Set axis limits to properly display all points
    if n_x > 0:
        ax_main.set_xlim(-0.5, n_x - 0.5)
    if n_sims > 0:
        ax_main.set_ylim(-0.5, n_sims - 0.5)

    # Set labels and ticks
    ax_main.set_xlabel(x_label, fontsize=fontsizes[1])
    ax_main.set_ylabel("Simulation", fontsize=fontsizes[1])

    # Set x-axis ticks
    if is_parameters:
        # For parameters, show all parameter names
        ax_main.set_xticks(range(len(x_labels)))
        ax_main.set_xticklabels(
            x_labels, rotation=45, ha="right", fontsize=fontsizes[2]
        )
    else:
        # For gridpoints, show indices
        ax_main.set_xticks(range(n_x))
        ax_main.set_xticklabels(x_labels, fontsize=fontsizes[2], rotation=90)

    # Set y-axis ticks
    ax_main.set_yticks(range(n_sims))
    # Add asterisks to simulations that started in this deme
    y_tick_labels = [
        f"{sim}*" if sim in simulations_with_asterisk else sim for sim in unique_sims
    ]
    ax_main.set_yticklabels(y_tick_labels, fontsize=fontsizes[2])
    set_axis_fontsizes(ax_main, fontsizes)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    # Top subplot: percentage of simulations where inHPD=1 for each gridpoint/parameter
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)

    # Calculate percentage for each x value
    x_percentages = []
    for x_val in x_labels:
        if is_parameters:
            subset = df[df["x_value"] == x_val]
        else:
            subset = df[df[x_col] == x_val]
        if len(subset) > 0:
            percentage = (subset["inHPD"] == 1).sum() / len(subset) * 100
        else:
            percentage = 0
        x_percentages.append(percentage)

    x_indices = range(len(x_labels))
    ax_top.bar(x_indices, x_percentages, color=colors[2], alpha=1.0)
    ax_top.axhline(y=95, color="black", linestyle="--", linewidth=2)
    ax_top.set_ylabel("% in HPD", fontsize=fontsizes[1])
    ax_top.set_ylim(0, 100)
    ax_top.grid(axis="y", alpha=0.3)
    set_axis_fontsizes(ax_top, fontsizes)
    ax_top.set_title(title, fontsize=fontsizes[0], fontweight="bold")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Right subplot: percentage of gridpoints/parameters where inHPD=1 for each simulation
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Use pre-calculated percentages (already sorted by percentage)
    sim_percentages = [sim_percentages_dict[sim] for sim in unique_sims]

    sim_indices = range(len(unique_sims))
    ax_right.barh(sim_indices, sim_percentages, color=colors[2], alpha=1.0)
    ax_right.axvline(x=95, color="black", linestyle="--", linewidth=2)
    ax_right.set_xlabel("% in HPD", fontsize=fontsizes[1])
    ax_right.set_xlim(0, 100)
    ax_right.grid(axis="x", alpha=0.3)
    set_axis_fontsizes(ax_right, fontsizes)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    # Save figure
    save_figure_png_and_pdf(output_path)
    plt.close()


def create_empty_plot(output_path, title):
    """
    Create an empty plot with a message when data is missing.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(
        0.5,
        0.5,
        "No cumulative incidence data",
        ha="center",
        va="center",
        fontsize=fontsizes[1],
    )
    ax.set_title(title, fontsize=fontsizes[0], fontweight="bold")
    ax.set_axis_off()
    save_figure_png_and_pdf(output_path)
    plt.close()


def plot_hpd_whisker(ax, x_pos, hpd_lower, hpd_upper, color, cap_width=0.2):
    """
    Plot HPD bounds as a whisker with horizontal caps.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x_pos : float
        X position for the whisker
    hpd_lower : float
        Lower HPD bound
    hpd_upper : float
        Upper HPD bound
    color : str
        Color for the whisker
    cap_width : float
        Width of the horizontal caps (default: 0.2)
    """
    # Vertical line for whisker
    ax.plot([x_pos, x_pos], [hpd_lower, hpd_upper], color=color, linewidth=3)
    # Horizontal caps at ends
    ax.plot(
        [x_pos - cap_width, x_pos + cap_width],
        [hpd_lower, hpd_lower],
        color=color,
        linewidth=3,
        alpha=1.0,
    )
    ax.plot(
        [x_pos - cap_width, x_pos + cap_width],
        [hpd_upper, hpd_upper],
        color=color,
        linewidth=3,
        alpha=1.0,
    )


def plot_median_point(ax, x_pos, median, color, label=None, s=100):
    """
    Plot median as a dot.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x_pos : float
        X position for the median point
    median : float
        Median value
    color : str
        Color for the point
    label : str, optional
        Label for legend
    s : int
        Size of the point (default: 100)
    """
    ax.scatter(x_pos, median, color=color, s=s, zorder=3, label=label)


def setup_parameter_plot_axes(
    ax, param, x_tick_positions, x_tick_labels, true_value=None
):
    """
    Set up axes labels, title, ticks, and grid for parameter plots.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to configure
    param : str
        Parameter name for title
    x_tick_positions : list
        Positions for x-axis ticks
    x_tick_labels : list
        Labels for x-axis ticks
    true_value : float, optional
        If provided, plot as horizontal line
    """
    # Set labels and title
    ax.set_xlabel("Simulation", fontsize=fontsizes[1])
    ax.set_ylabel("Value", fontsize=fontsizes[1])
    ax.set_title(f"Parameter: {param}", fontsize=fontsizes[0], fontweight="bold")

    # Set x-axis ticks
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right", fontsize=fontsizes[2])

    # Plot true value as horizontal line if provided
    if true_value is not None:
        ax.axhline(
            y=true_value,
            color=colors[0],
            linewidth=2,
            label=f"True value ({true_value})",
        )

    # Add grid
    ax.grid(axis="y", alpha=0.3)
    set_axis_fontsizes(ax, fontsizes)


def save_parameter_plot(fig, output_dir, param, prefix="all_hpd_validation_parameter"):
    """
    Save parameter plot to file.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    output_dir : Path
        Directory to save the plot
    param : str
        Parameter name (will be sanitized for filename)
    prefix : str
        Prefix for the filename (default: "all_hpd_validation_parameter")
    """
    plt.tight_layout()
    safe_param_name = param.replace(":", "_").replace(".", "_").replace("/", "_")
    output_path = output_dir / f"{prefix}_{safe_param_name}.png"
    save_figure_png_and_pdf(output_path)
    plt.close()


def create_parameter_median_plots(
    df_params, output_dir, prefix="all_hpd_validation_parameter"
):
    """
    Create individual plots for each parameter showing median, HPD bounds, and true value.

    Parameters:
    -----------
    df_params : DataFrame
        Data with Parameter, Simulation, median, hpd_lower, hpd_upper, true_value columns
    output_dir : Path
        Directory to save the plots
    """
    # Get unique parameters
    unique_params = df_params["Parameter"].unique()

    for param in unique_params:
        # Filter data for this parameter
        param_data = df_params[df_params["Parameter"] == param].copy()

        # Check if true value exists for this parameter (column present and not all NaN)
        has_true_value = (
            "true_value" in param_data.columns
            and not param_data["true_value"].isna().all()
        )
        true_value = param_data["true_value"].iloc[0] if has_true_value else None
        if has_true_value and pd.isna(true_value):
            has_true_value = False
            true_value = None

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        if not has_true_value:
            # Parameter was not estimated: empty subplot with message
            ax.set_title(f"Parameter: {param}", fontsize=fontsizes[0], fontweight="bold")
            ax.text(
                0.5,
                0.5,
                "parameter was not estimated",
                transform=ax.transAxes,
                fontsize=fontsizes[1],
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
        else:
            # Sort simulations by median value
            param_data = param_data.sort_values("median")

            # Get number of simulations
            n_sims = len(param_data)
            sim_indices = range(n_sims)

            # Extract values
            medians = param_data["median"].values
            hpd_lower = param_data["hpd_lower"].values
            hpd_upper = param_data["hpd_upper"].values
            simulations = param_data["Simulation"].values

            # Plot whiskers (HPD bounds) and medians
            for idx, lower, upper, median in zip(
                sim_indices, hpd_lower, hpd_upper, medians
            ):
                plot_hpd_whisker(ax, idx, lower, upper, colors[3])
                plot_median_point(
                    ax, idx, median, colors[3], label="Median" if idx == 0 else None
                )

            # Set up axes
            setup_parameter_plot_axes(ax, param, sim_indices, simulations, true_value)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Save figure
        save_parameter_plot(fig, output_dir, param, prefix=prefix)


def create_migration_rates_median_plots(
    df_migration, output_dir, prefix="all_hpd_validation_migration_rates"
):
    """
    Create a single plot for all migration rate parameters showing median, HPD bounds for both models.
    Grouped by simulation first, then by parameter. For each simulation-parameter combination,
    shows both MASCOT and MASCOT-DS side by side.

    Parameters:
    -----------
    df_migration : DataFrame
        Data with Parameter, Model, Simulation, median, hpd_lower, hpd_upper columns
    output_dir : Path
        Directory to save the plots
    """
    # Get unique parameters and simulations
    unique_params = sorted(df_migration["Parameter"].unique())
    unique_sims = sorted(df_migration["Simulation"].unique())

    # Colors: MASCOT uses colors[4], MASCOT-DS uses colors[3]
    model_colors = {MODEL_MASCOT: colors[4], MODEL_MASCOT_DS: colors[3]}
    # Position offset for models within the same simulation-parameter combination
    model_offset = 0.2
    # Gap between simulation groups
    sim_gap = 1.0

    # Create figure
    fig, ax = plt.subplots(
        figsize=(max(12, len(unique_sims) * len(unique_params) * 0.6), 8)
    )

    x_tick_positions = []
    x_tick_labels = []
    true_value_label_added = False
    current_x_pos = 0

    # Store data for n_deme_transitions bar plot
    n_transitions_positions = []
    n_transitions_values = []

    # Plot data grouped by simulation first, then by parameter
    for sim in unique_sims:
        sim_data = df_migration[df_migration["Simulation"] == sim]

        # Plot each parameter for this simulation
        for param in unique_params:
            param_data = sim_data[sim_data["Parameter"] == param]

            if len(param_data) > 0:
                center_pos = current_x_pos

                # Plot each model
                for model in [MODEL_MASCOT, MODEL_MASCOT_DS]:
                    model_data = param_data[param_data["Model"] == model]

                    if len(model_data) > 0:
                        # Calculate x position (MASCOT on left, MASCOT-DS on right)
                        x_pos = (
                            center_pos - model_offset
                            if model == MODEL_MASCOT
                            else center_pos + model_offset
                        )

                        # Get values
                        median = model_data["median"].iloc[0]
                        hpd_lower = model_data["hpd_lower"].iloc[0]
                        hpd_upper = model_data["hpd_upper"].iloc[0]
                        color = model_colors[model]

                        # Plot whisker and median using helper functions
                        plot_hpd_whisker(
                            ax, x_pos, hpd_lower, hpd_upper, color, cap_width=0.15
                        )
                        plot_median_point(
                            ax,
                            x_pos,
                            median,
                            color,
                            label=model if current_x_pos == 0 else None,
                        )

                # Plot true value / expected value as horizontal line spanning both models
                if (
                    "true_value" in param_data.columns
                    and not param_data["true_value"].isna().all()
                ):
                    true_value = param_data["true_value"].iloc[0]
                    x_left = center_pos - model_offset - 0.1
                    x_right = center_pos + model_offset + 0.1
                    label = None
                    if not true_value_label_added:
                        label = "True value"
                        true_value_label_added = True
                    ax.plot(
                        [x_left, x_right],
                        [true_value, true_value],
                        color=colors[0],
                        linewidth=2,
                        label=label,
                        zorder=10,
                    )

                # Store n_deme_transitions data for bar plot
                if "n_deme_transitions" in param_data.columns:
                    n_transitions = param_data["n_deme_transitions"].iloc[0]
                    n_transitions_positions.append(center_pos)
                    n_transitions_values.append(n_transitions)

                # Store tick position and label for this simulation-parameter combination
                x_tick_positions.append(center_pos)
                x_tick_labels.append(f"{sim}_{param}")

                # Move to next position (1 unit per simulation-parameter combination)
                current_x_pos += 1

        # Add gap after all parameters for this simulation
        current_x_pos += sim_gap

    # Create second y-axis for n_deme_transitions bar plot
    if len(n_transitions_positions) > 0:
        ax2 = ax.twinx()
        # Plot bars with grey color and low zorder
        bar_width = 0.3
        ax2.bar(
            n_transitions_positions,
            n_transitions_values,
            width=bar_width,
            color="grey",
            alpha=0.5,
            zorder=1,
        )
        ax2.set_ylabel("n_deme_transitions", fontsize=fontsizes[1])
        set_axis_fontsizes(ax2, fontsizes)
        ax2.spines["top"].set_visible(False)

    # Set up axes
    ax.set_xlabel("Simulation_Parameter", fontsize=fontsizes[1])
    ax.set_ylabel("Value", fontsize=fontsizes[1])
    ax.set_title(
        "Migration Rates - All Parameters", fontsize=fontsizes[0], fontweight="bold"
    )

    # Set x-axis ticks
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right", fontsize=fontsizes[2])

    # Add grid
    ax.grid(axis="y", alpha=0.3)
    set_axis_fontsizes(ax, fontsizes)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Align zero points of both y-axes if ax2 exists
    if len(n_transitions_positions) > 0:
        # Get current y-limits of both axes
        ylim_ax = ax.get_ylim()
        ylim_ax2 = ax2.get_ylim()

        # Calculate the position of zero in ax (as a fraction of the range, 0 to 1)
        # If zero is within the range, calculate its position
        if ylim_ax[0] <= 0 <= ylim_ax[1]:
            zero_pos_ax = (0 - ylim_ax[0]) / (ylim_ax[1] - ylim_ax[0])
        elif ylim_ax[0] > 0:
            # Zero is below the range
            zero_pos_ax = 0.0
        else:
            # Zero is above the range
            zero_pos_ax = 1.0

        # Preserve ax2's range but shift it so zero aligns at the same position
        range_ax2 = ylim_ax2[1] - ylim_ax2[0]

        # Calculate new limits: zero should be at position zero_pos_ax in ax2's range
        new_bottom = -zero_pos_ax * range_ax2
        new_top = (1 - zero_pos_ax) * range_ax2
        ax2.set_ylim(new_bottom, new_top)

    # Add legend
    ax.legend(fontsize=fontsizes[2])

    # Save figure
    plt.tight_layout()
    output_path = output_dir / f"{prefix}_all_params.png"
    save_figure_png_and_pdf(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create HPD validation plots for cumulative incidence, ne, prevalence, and parameters"
    )
    parser.add_argument(
        "--cumulative_incidence",
        type=str,
        required=True,
        help="Path to cumulative incidence CSV file",
    )
    parser.add_argument("--ne", type=str, required=True, help="Path to ne CSV file")
    parser.add_argument(
        "--prevalence", type=str, required=True, help="Path to prevalence CSV file"
    )
    parser.add_argument(
        "--parameters", type=str, required=True, help="Path to parameters CSV file"
    )
    parser.add_argument(
        "--migration_rates",
        type=str,
        required=True,
        help="Path to migration rates CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for plots (default: current directory)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Variant name used to prefix all output files (all_hpd_validation_{variant}*)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = str(output_dir / f"all_hpd_validation_{args.variant}")

    # Process prevalence
    logger.info("Processing prevalence...")
    df_prev = pd.read_csv(args.prevalence)
    df_prev, n_gridpoints_prev = assign_gridpoint_indices(df_prev)

    # Determine which simulations have outbreaks starting in each deme
    # Outbreak started in a deme if expectedlogPrev == 0.0 at timesincestart == 0.0
    outbreaks_by_deme = {}
    for deme in df_prev["Deme"].unique():
        # Filter for timesincestart == 0.0 and this deme
        t0_data = df_prev[
            (df_prev["timesincestart"] == 0.0) & (df_prev["Deme"] == deme)
        ]
        # Find simulations where expectedlogPrev == 0.0 (not NaN)
        outbreak_sims = t0_data[
            (t0_data["expectedlogPrev"] == 0.0) & (t0_data["expectedlogPrev"].notna())
        ]["Simulation"].unique()
        outbreaks_by_deme[deme] = set(outbreak_sims)

    for deme, df in df_prev.groupby("Deme"):
        create_hpd_plot(
            df,
            f"Prevalence - Deme {deme}",
            "Gridpoint Index",
            output_prefix + f"_prevalence_deme{deme}.png",
            is_parameters=False,
            simulations_with_asterisk=outbreaks_by_deme.get(deme, set()),
        )

    # Process cumulative incidence
    logger.info("Processing cumulative incidence...")
    ci_path = Path(args.cumulative_incidence)
    if ci_path.exists() and ci_path.stat().st_size == 0:
        create_empty_plot(
            Path(output_prefix + "_cumulative_incidence_empty.png"),
            "Cumulative Incidence - No data",
        )
    else:
        df_ci = pd.read_csv(args.cumulative_incidence)
        df_ci, n_gridpoints_ci = assign_gridpoint_indices(df_ci)
        for deme, df in df_ci.groupby("Deme"):
            create_hpd_plot(
                df,
                f"Cumulative Incidence - Deme {deme}",
                "Gridpoint Index",
                output_prefix + f"_cumulative_incidence_deme{deme}.png",
                is_parameters=False,
                simulations_with_asterisk=outbreaks_by_deme.get(deme, set()),
            )

    # Process ne
    logger.info("Processing ne...")
    df_ne = pd.read_csv(args.ne)
    df_ne, n_gridpoints_ne = assign_gridpoint_indices(df_ne)
    for deme, df in df_ne.groupby("Deme"):
        create_hpd_plot(
            df,
            f"Ne - Deme {deme}",
            "Gridpoint Index",
            output_prefix + f"_ne_deme{deme}.png",
            is_parameters=False,
            simulations_with_asterisk=outbreaks_by_deme.get(deme, set()),
        )

    # Process parameters
    logger.info("Processing parameters...")
    df_params = pd.read_csv(args.parameters)
    create_hpd_plot(
        df_params,
        "Parameters",
        "Parameter",
        output_prefix + "_parameters.png",
        is_parameters=True,
    )

    # Process migration rates
    logger.info("Processing migration rates...")
    df_migration_rates = pd.read_csv(args.migration_rates)
    create_hpd_plot(
        df_migration_rates,
        "Migration Rates",
        "Parameter",
        output_prefix + "_migration_rates.png",
        is_parameters=True,
    )

    # Create individual parameter plots with medians and HPD bounds
    logger.info("Creating individual parameter plots...")
    create_parameter_median_plots(
        df_params, output_dir, prefix=f"all_hpd_validation_{args.variant}_parameter"
    )

    # Create individual migration rates plots with medians and HPD bounds for both models
    logger.info("Creating individual migration rates plots...")
    create_migration_rates_median_plots(
        df_migration_rates,
        output_dir,
        prefix=f"all_hpd_validation_{args.variant}_migration_rates",
    )

    logger.info("All plots created successfully!")


if __name__ == "__main__":
    main()
