#!/usr/bin/env python3
"""
Script to quantify information content by comparing HPD widths across different datastream configurations.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

from constants import (
    COLORBLINDFR,
    COLORS,
    LEAVE_ONE_OUT_VARIANTS,
    MODEL_MASCOT,
    MODEL_MASCOT_DS,
    VARIANT_COLORS,
    VARIANT_LABELS,
    VARIANT_SUFFIXES,
)
from plot_utils import (
    DEFAULT_FONTSIZES,
    beautify_plot,
)

VARIANT_ORDER = list(LEAVE_ONE_OUT_VARIANTS)
_VARIANT_SUFFIXES = VARIANT_SUFFIXES


def load_data(filepath):
    """Load CSV file and calculate HPD width."""
    df = pd.read_csv(filepath)

    # Calculate HPD width (hpd_upper - hpd_lower)
    # Handle empty/missing values
    df["hpd_width"] = df["hpd_upper"] - df["hpd_lower"]

    # Convert inHPD to boolean (handle both 1/0 and 1.0/0.0)
    df["inHPD"] = df["inHPD"].astype(float).fillna(0).astype(bool)

    # Filter out rows where HPD width cannot be calculated (NaN or empty)
    df = df.dropna(subset=["hpd_width"])

    return df


def identify_variant(filename, all_keyword="all_params_datastreams"):
    """Identify the variant type from a filename.

    Args:
        filename: The filename to inspect.
        all_keyword: Substring that identifies the "all datastreams" variant.
            Defaults to ``"all_params_datastreams"``; pass
            ``"all_migration_rates"`` for migration-rate files, or
            ``"all_prevalence_datastreams"`` for prevalence files.

    Returns:
        Variant string (e.g. ``"nocasecounts"``, ``"all"``) or ``None``.
    """
    filename_lower = filename.lower()
    if "nocasecounts" in filename_lower:
        return "nocasecounts"
    elif "nowastewater" in filename_lower:
        return "nowastewater"
    elif "noseroprevalence" in filename_lower:
        return "noseroprevalence"
    elif all_keyword in filename_lower and "no" not in filename_lower:
        return "all"
    return None


def extract_simulation_name(sim_string):
    """Extract base simulation name (remove variant suffix and _datastreams)."""
    for suffix in _VARIANT_SUFFIXES:
        if sim_string.endswith(suffix):
            return sim_string[: -len(suffix)]
    return sim_string


def extract_starting_deme(prevalence_file):
    """
    Extract starting deme for each simulation from prevalence file.

    Returns a dictionary mapping simulation name to starting deme (as integer).
    The starting deme is identified as the deme where expectedlogPrev == 0.0
    at timesincestart == 0.0.
    """
    df_prev = pd.read_csv(prevalence_file)

    # Dictionary to store simulation -> starting deme mapping
    starting_deme_by_sim = {}

    # For each simulation, find the starting deme
    for sim in df_prev["Simulation"].unique():
        # Filter for timesincestart == 0.0 and this simulation
        t0_data = df_prev[
            (df_prev["timesincestart"] == 0.0) & (df_prev["Simulation"] == sim)
        ]

        # Find deme where expectedlogPrev == 0.0 (not NaN)
        outbreak_deme = t0_data[
            (t0_data["expectedlogPrev"] == 0.0) & (t0_data["expectedlogPrev"].notna())
        ]

        if not outbreak_deme.empty:
            # Get the deme value (should be unique for each simulation at t0)
            deme_value = outbreak_deme["Deme"].iloc[0]
            starting_deme_by_sim[sim] = int(deme_value)

    return starting_deme_by_sim


def find_all_prevalence_file(prevalence_files):
    """
    Find the 'all' prevalence file (not a leave-one-out variant).

    Args:
        prevalence_files: List of prevalence file paths

    Returns:
        Path to the 'all' prevalence file, or None if not found
    """
    if not prevalence_files:
        return None

    for filepath in prevalence_files:
        filename = Path(filepath).name.lower()
        if "all_prevalence_datastreams" in filename and "no" not in filename:
            return filepath
    return None


def _load_and_tag_files(filepaths, all_keyword, loader=load_data, label="data"):
    """Load CSVs, tag each with its variant and base simulation name.

    This is the generic loop shared by all ``load_all_*_files`` functions.

    Args:
        filepaths: Iterable of CSV file paths.
        all_keyword: Substring passed to ``identify_variant`` to detect the
            "all datastreams" variant (e.g. ``"all_params_datastreams"``).
        loader: Callable that reads a single CSV and returns a DataFrame.
            Defaults to ``load_data`` (adds ``hpd_width``).
        label: Human-readable name for error messages.

    Returns:
        List of DataFrames, each with ``variant`` and ``simulation_base`` columns.

    Raises:
        ValueError: If no files could be loaded.
    """
    all_data = []
    for filepath in filepaths:
        variant = identify_variant(Path(filepath).name, all_keyword=all_keyword)
        if variant is None:
            print(f"Warning: Could not identify variant for {filepath}, skipping")
            continue

        df = loader(filepath)
        df["variant"] = variant
        df["simulation_base"] = df["Simulation"].apply(extract_simulation_name)
        all_data.append(df)

    if not all_data:
        raise ValueError(f"Error: No valid {label} files loaded")

    return all_data


def load_all_data_files(filepaths):
    """Load and combine all HPD-parameter data files.

    Returns:
        Combined DataFrame with variant and simulation_base columns.
    """
    return pd.concat(
        _load_and_tag_files(filepaths, "all_params_datastreams", label="data"),
        ignore_index=True,
    )


def load_all_migration_rates_files(filepaths):
    """Load and combine all migration rates files.

    Returns:
        Combined DataFrame with variant, simulation_base, and Model columns.
    """
    frames = _load_and_tag_files(
        filepaths, "all_migration_rates", label="migration rates"
    )
    combined_df = pd.concat(frames, ignore_index=True)

    # All variants have MASCOT model entries, which should be identical across variants
    # Set MASCOT model rows to have variant="MASCOT" and deduplicate
    combined_df.loc[combined_df["Model"] == MODEL_MASCOT, "variant"] = MODEL_MASCOT

    # Drop duplicate MASCOT rows (they're identical across variant files)
    cols_interest = [
        i for i in combined_df.columns if i not in ["variant", "Simulation"]
    ]
    num_cols = (
        combined_df[cols_interest].select_dtypes(include=[np.number]).columns.tolist()
    )
    non_num_cols = [col for col in cols_interest if col not in num_cols]

    tmp_df = combined_df.copy()
    tolerance_decimals = 6
    for col in num_cols:
        tmp_df[col] = tmp_df[col].round(tolerance_decimals)

    combined_df = tmp_df.drop_duplicates(subset=non_num_cols + num_cols, keep="first")

    return combined_df


def load_all_prevalence_files(filepaths):
    """Load and combine all prevalence files.

    Returns:
        Combined DataFrame with variant and simulation_base columns.
    """
    return pd.concat(
        _load_and_tag_files(
            filepaths,
            "all_prevalence_datastreams",
            loader=pd.read_csv,
            label="prevalence",
        ),
        ignore_index=True,
    )


def calculate_prevalence_rmse(prevalence_df):
    """
    Calculate RMSE between logPrevalence and expectedlogPrevP1 for each simulation/deme/variant.

    Args:
        prevalence_df: DataFrame with prevalence data

    Returns:
        DataFrame with columns: simulation_base, Deme, variant, rmse
    """
    results = []

    for (sim_base, deme, variant), group in prevalence_df.groupby(
        ["simulation_base", "Deme", "variant"]
    ):
        # Filter out rows where either logPrevalence or expectedlogPrevP1 is NaN
        valid_data = group[
            group["logPrevalence"].notna() & group["expectedlogPrevP1"].notna()
        ]

        if len(valid_data) == 0:
            continue

        # Calculate squared errors
        squared_errors = (
            valid_data["logPrevalence"] - valid_data["expectedlogPrevP1"]
        ) ** 2

        # Calculate RMSE
        rmse = np.sqrt(squared_errors.mean())

        results.append(
            {
                "simulation_base": sim_base,
                "Deme": deme,
                "variant": variant,
                "rmse": rmse,
            }
        )

    return pd.DataFrame(results)


def calculate_prevalence_coverage(prevalence_df):
    """
    Calculate coverage as number of timepoints where inHPD is True for each simulation/deme/variant.

    Args:
        prevalence_df: DataFrame with prevalence data

    Returns:
        DataFrame with columns: simulation_base, Deme, variant, coverage, total_timepoints
    """
    results = []

    for (sim_base, deme, variant), group in prevalence_df.groupby(
        ["simulation_base", "Deme", "variant"]
    ):
        # Convert inHPD to boolean if needed
        in_hpd = group["inHPD"].astype(float).fillna(0).astype(bool)

        # Count timepoints where inHPD is True
        coverage = in_hpd.sum()
        total_timepoints = len(group)

        results.append(
            {
                "simulation_base": sim_base,
                "Deme": deme,
                "variant": variant,
                "coverage": coverage,
                "total_timepoints": total_timepoints,
            }
        )

    return pd.DataFrame(results)


def determine_migration_direction(df, starting_deme_by_sim):
    """
    Determine migration direction for each row based on starting deme.

    Adds a 'migration_direction' column:
    - 'start_to_other': migration from starting deme to other deme
    - 'other_to_start': migration from other deme to starting deme

    Args:
        df: DataFrame with migration rates data (must have Parameter and Simulation columns)
        starting_deme_by_sim: Dictionary mapping simulation name to starting deme

    Returns:
        DataFrame with added 'migration_direction' column
    """
    df = df.copy()
    df["migration_direction"] = None

    for idx, row in df.iterrows():
        sim_base = extract_simulation_name(row["Simulation"])
        param = row["Parameter"]

        if sim_base not in starting_deme_by_sim:
            continue

        starting_deme = starting_deme_by_sim[sim_base]

        # Extract source and target demes from parameter name
        # Format: f_migrationRatesSkyline.I0_to_I1 or f_migrationRatesSkyline.I1_to_I0
        if "I0_to_I1" in param:
            source_deme = 0
            target_deme = 1
        elif "I1_to_I0" in param:
            source_deme = 1
            target_deme = 0
        else:
            continue

        # Determine direction based on starting deme
        if source_deme == starting_deme:
            df.at[idx, "migration_direction"] = "start_to_other"
        elif target_deme == starting_deme:
            df.at[idx, "migration_direction"] = "other_to_start"

    return df


def order_simulations_by_deme(all_simulations, starting_deme_by_sim):
    """
    Order simulations by starting deme (sorted by deme number), then alphabetically.

    Args:
        all_simulations: List of all simulation names
        starting_deme_by_sim: Dictionary mapping simulation name to starting deme

    Returns:
        Ordered list of simulation names
    """
    if not starting_deme_by_sim:
        print(
            "Warning: No starting deme information available, using alphabetical order"
        )
        return sorted(all_simulations)

    # Group simulations by starting deme (dictionary: deme -> list of simulations)
    sims_by_deme = {}
    no_deme_sims = []

    for sim in all_simulations:
        # Extract base simulation name to match with prevalence file
        base_sim = extract_simulation_name(sim)
        if base_sim in starting_deme_by_sim:
            deme = starting_deme_by_sim[base_sim]
            if deme not in sims_by_deme:
                sims_by_deme[deme] = []
            sims_by_deme[deme].append(sim)
        else:
            no_deme_sims.append(sim)

    # Sort each group alphabetically
    for deme in sims_by_deme:
        sims_by_deme[deme].sort()
    no_deme_sims.sort()

    # Combine: sorted by deme number (0, 1, 2, ...), then unknown
    sorted_demes = sorted(sims_by_deme.keys())
    simulations_ordered = []
    for deme in sorted_demes:
        simulations_ordered.extend(sims_by_deme[deme])
    simulations_ordered.extend(no_deme_sims)

    return simulations_ordered


def create_simulation_labels(simulations_ordered, starting_deme_by_sim):
    """
    Create labels for simulations with starting deme info.

    Args:
        simulations_ordered: Ordered list of simulation names
        starting_deme_by_sim: Dictionary mapping simulation name to starting deme

    Returns:
        List of simulation labels
    """
    simulation_labels = []
    simulation_labels_nodeme = []
    for sim in simulations_ordered:
        base_sim = extract_simulation_name(sim)
        if base_sim in starting_deme_by_sim:
            deme = starting_deme_by_sim[base_sim]
            sim_str = base_sim.replace("_simulation", "")
            simulation_labels.append(f"Deme {deme} start - {sim_str}")
            simulation_labels_nodeme.append(sim_str)
        else:
            simulation_labels.append(base_sim)
            simulation_labels_nodeme.append(base_sim)
    return simulation_labels, simulation_labels_nodeme


def calculate_hpd_differences(
    param_data, simulations_ordered, variant_order, all_data_param
):
    """
    Calculate HPD width differences for each variant and simulation.

    Args:
        param_data: DataFrame filtered for a specific parameter
        simulations_ordered: Ordered list of simulation names
        variant_order: List of variant names to process
        all_data_param: DataFrame with "all" variant data for baseline

    Returns:
        Dictionary mapping variant to tuple of (widths, sim_indices, face_colors, edge_colors)
        where sim_indices are the indices in simulations_ordered for each width
    """
    variant_data_dict = {}

    for variant_idx, variant in enumerate(variant_order):
        variant_data = param_data[param_data["variant"] == variant]

        if variant_data.empty:
            continue

        # Prepare bar data for each simulation
        widths = []
        edge_colors = []
        face_colors = []
        sim_indices = []

        for sim_idx, sim in enumerate(simulations_ordered):
            sim_variant_data = variant_data[variant_data["simulation_base"] == sim]
            sim_all_data = all_data_param[all_data_param["simulation_base"] == sim]

            # Calculate difference: variant HPD width - all datastreams HPD width
            if not sim_variant_data.empty and not sim_all_data.empty:
                variant_row = sim_variant_data.iloc[0]
                all_row = sim_all_data.iloc[0]

                variant_hpd_width = variant_row["hpd_width"]
                all_hpd_width = all_row["hpd_width"]
                in_hpd = variant_row["inHPD"]

                # Only add bar if both HPD widths are valid
                if pd.notna(variant_hpd_width) and pd.notna(all_hpd_width):
                    hpd_diff = (variant_hpd_width - all_hpd_width) / all_hpd_width
                    widths.append(hpd_diff)

                    # Set colors based on inHPD
                    if in_hpd:
                        face_colors.append(VARIANT_COLORS[variant])
                        edge_colors.append(VARIANT_COLORS[variant])
                    else:
                        face_colors.append("none")
                        edge_colors.append(VARIANT_COLORS[variant])

                    # Store simulation index for y-position calculation
                    sim_indices.append(sim_idx)

        if widths:
            variant_data_dict[variant] = (widths, sim_indices, face_colors, edge_colors)

    return variant_data_dict


def calculate_migration_hpd_differences(
    migration_data,
    simulations_ordered,
    variant_order,
    all_data_migration,
    variant_colors,
):
    """
    Calculate HPD width differences for migration rates variants.

    Args:
        migration_data: DataFrame filtered for a specific migration direction
        simulations_ordered: Ordered list of simulation names
        variant_order: List of variant names to process (including "MASCOT")
        all_data_migration: DataFrame with "all" variant data for baseline (MASCOT-DS)
        variant_colors: Dictionary mapping variant names to colors

    Returns:
        Dictionary mapping variant to tuple of (widths, sim_indices, face_colors, edge_colors)
        where sim_indices are the indices in simulations_ordered for each width
    """
    variant_data_dict = {}

    # Process all variants uniformly (including MASCOT)
    for variant_idx, variant in enumerate(variant_order):
        variant_data = migration_data[(migration_data["variant"] == variant)]

        if variant_data.empty:
            continue

        widths = []
        edge_colors = []
        face_colors = []
        sim_indices = []

        for sim_idx, sim in enumerate(simulations_ordered):
            sim_variant_data = variant_data[variant_data["simulation_base"] == sim]
            sim_all_data = all_data_migration[
                (all_data_migration["simulation_base"] == sim)
                & (all_data_migration["Model"] == MODEL_MASCOT_DS)
            ]

            if not sim_variant_data.empty and not sim_all_data.empty:
                variant_row = sim_variant_data.iloc[0]
                all_row = sim_all_data.iloc[0]

                variant_hpd_width = variant_row["hpd_width"]
                all_hpd_width = all_row["hpd_width"]
                in_hpd = variant_row["inHPD"]

                if pd.notna(variant_hpd_width) and pd.notna(all_hpd_width):
                    hpd_diff = (variant_hpd_width - all_hpd_width) / all_hpd_width
                    widths.append(hpd_diff)

                    variant_color = variant_colors.get(variant, "grey")
                    if in_hpd:
                        face_colors.append(variant_color)
                        edge_colors.append(variant_color)
                    else:
                        face_colors.append("none")
                        edge_colors.append(variant_color)

                    # Store simulation index for y-position calculation
                    sim_indices.append(sim_idx)

        if widths:
            variant_data_dict[variant] = (widths, sim_indices, face_colors, edge_colors)

    return variant_data_dict


def plot_bars(ax, variant_data_dict, variant_order, n_simulations, bar_width=0.25):
    """
    Plot horizontal bars for HPD width differences.

    Args:
        ax: Matplotlib axis object
        variant_data_dict: Dictionary mapping variant to (widths, sim_indices, face_colors, edge_colors)
        variant_order: List of variant names in order (for positioning)
        n_simulations: Number of simulations (for y-position calculation)
        bar_width: Width of bars for grouping
    """
    y_positions = np.arange(n_simulations)
    spacebetweenbars = bar_width * 0.4

    for variant_idx, variant in enumerate(variant_order):
        if variant not in variant_data_dict:
            continue

        widths, sim_indices, face_colors, edge_colors = variant_data_dict[variant]

        # Calculate y positions with offset for grouping
        y_pos = []
        for sim_idx in sim_indices:
            offset = (variant_idx - len(variant_order) / 2 + 0.5) * (
                bar_width + spacebetweenbars
            )
            y_pos.append(y_positions[sim_idx] + offset)

        ax.barh(
            y_pos,
            widths,
            height=bar_width,
            color=face_colors,
            edgecolor=edge_colors,
            linewidth=1.0,
        )


def add_scatter_markers(ax, all_data, simulations_ordered, model_filter=None):
    """
    Add scatter markers for inHPD status of "all" datastreams version.

    Args:
        ax: Matplotlib axis object
        all_data: DataFrame with "all" variant data for baseline
        simulations_ordered: Ordered list of simulation names
        model_filter: Optional dict with Model column filter (e.g., {"Model": "MASCOT-DS"})
    """
    y_positions = np.arange(len(simulations_ordered))
    scatter_y = []
    scatter_colors = []

    for sim_idx, sim in enumerate(simulations_ordered):
        sim_all_data = all_data[all_data["simulation_base"] == sim]

        # Apply model filter if provided
        if model_filter:
            for key, value in model_filter.items():
                sim_all_data = sim_all_data[sim_all_data[key] == value]

        if not sim_all_data.empty:
            all_row = sim_all_data.iloc[0]
            in_hpd = all_row["inHPD"]

            if in_hpd:
                scatter_colors.append(VARIANT_COLORS["all"])
            else:
                scatter_colors.append("grey")

            scatter_y.append(y_positions[sim_idx])

    # Position scatter markers
    if scatter_y:
        ax.relim()
        ax.autoscale()
        xlim = ax.get_xlim()
        subplot_width = xlim[1] - xlim[0]
        offset = subplot_width * 0.05
        scatter_x_pos = max(abs(xlim[0]), abs(xlim[1])) + offset
        scatter_x = [scatter_x_pos] * len(scatter_y)

        ax.scatter(
            scatter_x,
            scatter_y,
            marker="s",
            color=scatter_colors,
            s=30,
            zorder=10,
        )


def configure_comparison_axis(ax, title, show_xlabel=False):
    """
    Configure axis for comparison plot.

    Args:
        ax: Matplotlib axis object
        title: Title for the subplot
        show_xlabel: If True, show x-axis label
    """
    if show_xlabel:
        ax.set_xlabel(
            "HPD width difference\n(variant - all)/all",
            fontsize=DEFAULT_FONTSIZES["axis_label"],
        )
    ax.set_title(title, fontsize=DEFAULT_FONTSIZES["title"])
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    beautify_plot(ax)


def plot_parameter_comparison(
    ax,
    param,
    param_data,
    simulations_ordered,
    all_data_param,
    parameter_legible_names=None,
    bar_width=0.2,
    show_xlabel=False,
):
    """
    Plot HPD width differences for a single parameter.

    Args:
        ax: Matplotlib axis object
        param: Parameter name
        param_data: DataFrame filtered for this parameter
        simulations_ordered: Ordered list of simulation names
        all_data_param: DataFrame with "all" variant data for baseline
        parameter_legible_names: Optional dictionary mapping parameter names to legible names
        bar_width: Width of bars for grouping
        show_xlabel: If True, show x-axis label
    """
    # Calculate HPD differences for all variants
    variant_data_dict = calculate_hpd_differences(
        param_data, simulations_ordered, VARIANT_ORDER, all_data_param
    )

    # Plot bars
    plot_bars(ax, variant_data_dict, VARIANT_ORDER, len(simulations_ordered), bar_width)

    # Add scatter markers
    add_scatter_markers(ax, all_data_param, simulations_ordered)

    # Configure axis - use legible name if available
    param_title = param
    if parameter_legible_names:
        param_title = parameter_legible_names.get(param, param)
    configure_comparison_axis(ax, param_title, show_xlabel)


def create_legend(ax, variant_order, variant_colors, variant_labels):
    """
    Create legend for comparison plot.

    Args:
        ax: Matplotlib axis object
        variant_order: List of variant names in order
        variant_colors: Dictionary mapping variant names to colors
        variant_labels: Dictionary mapping variant names to labels
    """
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=variant_colors.get(variant, "grey"))
        for variant in variant_order
    ]
    legend_labels = [variant_labels.get(variant, variant) for variant in variant_order]
    ax.legend(
        legend_elements,
        legend_labels,
        fontsize=DEFAULT_FONTSIZES["legend"],
        loc="upper right",
    )


def setup_y_axis(axes, simulations_ordered, simulation_labels, invert=False):
    """
    Setup y-axis labels for comparison plot.

    Args:
        axes: Array of matplotlib axis objects
        simulations_ordered: Ordered list of simulation names
        simulation_labels: List of simulation labels for y-axis
        invert: If True, invert y-axis
    """
    if invert:
        for ax in axes:
            ax.invert_yaxis()

    y_positions = np.arange(len(simulations_ordered))
    axes[0].set_ylabel("Simulation", fontsize=DEFAULT_FONTSIZES["axis_label"])
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(simulation_labels, fontsize=DEFAULT_FONTSIZES["tick_label"])


def create_information_content_plot(
    combined_df,
    parameters,
    simulations_ordered,
    simulation_labels,
    parameter_legible_names=None,
):
    """
    Create the main information content comparison plot.

    Args:
        combined_df: Combined DataFrame with all data
        parameters: List of parameter names
        simulations_ordered: Ordered list of simulation names
        simulation_labels: List of simulation labels for y-axis
        parameter_legible_names: Optional dictionary mapping parameter names to legible names

    Returns:
        Tuple of (figure, axes)
    """
    # Create figure with subplots (share y-axis but NOT x-axis)
    n_params = len(parameters)
    n_rows = (n_params + 3) // 4
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, sharey=True, figsize=(8, n_rows * 3))
    axes = axes.flatten()

    # Set default font size
    plt.rcParams["font.size"] = DEFAULT_FONTSIZES["axis_label"]

    # Determine which plots are in the bottom row
    bottom_row_start = (n_rows - 1) * n_cols

    # Plot each parameter
    for param_idx, param in enumerate(parameters):
        ax = axes[param_idx]

        # Get data for this parameter
        param_data = combined_df[combined_df["Parameter"] == param]

        # Get "all" datastreams data for this parameter (baseline)
        all_data_param = param_data[param_data["variant"] == "all"]

        # Only show xlabel for bottom row plots
        show_xlabel = param_idx >= bottom_row_start

        # Plot parameter comparison
        plot_parameter_comparison(
            ax,
            param,
            param_data,
            simulations_ordered,
            all_data_param,
            parameter_legible_names=parameter_legible_names,
            show_xlabel=show_xlabel,
        )

    # Add legend
    create_legend(axes[0], VARIANT_ORDER, VARIANT_COLORS, VARIANT_LABELS)

    # Setup y-axis
    setup_y_axis(axes, simulations_ordered, simulation_labels, invert=False)

    plt.tight_layout()

    return fig, axes


def plot_migration_comparison(
    ax,
    migration_direction,
    migration_data,
    simulations_ordered,
    all_data_migration,
    variant_colors,
    bar_width=0.15,
    show_xlabel=False,
):
    """
    Plot HPD width differences for migration rates in a specific direction.

    Args:
        ax: Matplotlib axis object
        migration_direction: Either "start_to_other" or "other_to_start"
        migration_data: DataFrame filtered for this migration direction
        simulations_ordered: Ordered list of simulation names
        all_data_migration: DataFrame with "all" variant data for baseline
        variant_colors: Dictionary mapping variant names to colors
        bar_width: Width of bars for grouping
        show_xlabel: If True, show x-axis label
    """
    # Filter for this migration direction
    direction_data = migration_data[
        migration_data["migration_direction"] == migration_direction
    ]
    all_data_direction = all_data_migration[
        all_data_migration["migration_direction"] == migration_direction
    ]

    # Calculate HPD differences for all variants (including MASCOT)
    variant_order = [MODEL_MASCOT] + VARIANT_ORDER
    variant_data_dict = calculate_migration_hpd_differences(
        direction_data,
        simulations_ordered,
        variant_order,
        all_data_direction,
        variant_colors,
    )

    # Plot bars
    plot_bars(ax, variant_data_dict, variant_order, len(simulations_ordered), bar_width)

    # Add scatter markers (filter by Model==MASCOT-DS for migration rates)
    add_scatter_markers(
        ax, all_data_direction, simulations_ordered, {"Model": MODEL_MASCOT_DS}
    )

    # Configure axis
    direction_label = (
        "Start -> Other"
        if migration_direction == "start_to_other"
        else "Other -> Start"
    )
    configure_comparison_axis(ax, direction_label, show_xlabel)


def create_migration_rates_plot(
    combined_migration_df, simulations_ordered, simulation_labels, variant_colors
):
    """
    Create the migration rates information content comparison plot.

    Args:
        combined_migration_df: Combined DataFrame with all migration rates data
        simulations_ordered: Ordered list of simulation names
        simulation_labels: List of simulation labels for y-axis
        variant_colors: Dictionary mapping variant names to colors

    Returns:
        Tuple of (figure, axes)
    """
    # Create figure with 2 subplots (one for each migration direction)
    fig, axes = plt.subplots(
        1, 2, sharey=True, figsize=(6, len(simulations_ordered) * 0.4)
    )

    plt.rcParams["font.size"] = DEFAULT_FONTSIZES["axis_label"]

    # Get "all" datastreams data for baseline
    all_data_migration = combined_migration_df[
        combined_migration_df["variant"] == "all"
    ]

    # Plot each migration direction
    directions = ["start_to_other", "other_to_start"]
    for dir_idx, direction in enumerate(directions):
        ax = axes[dir_idx]
        show_xlabel = True

        plot_migration_comparison(
            ax,
            direction,
            combined_migration_df,
            simulations_ordered,
            all_data_migration,
            variant_colors,
            show_xlabel=show_xlabel,
        )

    # Add legend
    variant_order = [MODEL_MASCOT] + VARIANT_ORDER
    migration_variant_labels = {MODEL_MASCOT: MODEL_MASCOT, **VARIANT_LABELS}
    create_legend(axes[0], variant_order, variant_colors, migration_variant_labels)

    # Setup y-axis
    setup_y_axis(axes, simulations_ordered, simulation_labels, invert=False)

    plt.tight_layout()

    return fig, axes


def plot_prevalence_bars(
    ax,
    data_df,
    simulations_ordered,
    deme_type,
    variant_order,
    variant_colors,
    value_column,
    starting_deme_by_sim,
    bar_width=0.1,
):
    """
    Plot horizontal bars for prevalence RMSE or coverage.

    Args:
        ax: Matplotlib axis object
        data_df: DataFrame with rmse or coverage data
        simulations_ordered: Ordered list of simulation names
        deme_type: Either "starting" or "other" to indicate which deme to plot
        variant_order: List of variant names in order
        variant_colors: Dictionary mapping variant names to colors
        value_column: Name of the column to plot (e.g., "rmse" or "coverage")
        starting_deme_by_sim: Dictionary mapping simulation name to starting deme
        bar_width: Width of bars for grouping
    """
    y_positions = np.arange(len(simulations_ordered))
    spacebetweenbars = bar_width * 0.4

    for variant_idx, variant in enumerate(variant_order):
        variant_data = data_df[data_df["variant"] == variant]

        if variant_data.empty:
            continue

        # Prepare bar data for each simulation
        values = []
        y_pos = []
        face_colors = []

        for sim_idx, sim in enumerate(simulations_ordered):
            # Determine which deme to use for this simulation
            base_sim = extract_simulation_name(sim)
            if base_sim in starting_deme_by_sim:
                starting_deme = starting_deme_by_sim[base_sim]
                other_deme = 1 - starting_deme

                if deme_type == "starting":
                    target_deme = starting_deme
                else:
                    target_deme = other_deme
            else:
                # If we don't have starting deme info, skip this simulation
                continue

            # Filter for this simulation and target deme
            sim_variant_data = variant_data[
                (variant_data["simulation_base"] == sim)
                & (variant_data["Deme"] == target_deme)
            ]

            if not sim_variant_data.empty:
                value = sim_variant_data.iloc[0][value_column]
                values.append(value)

                offset = (variant_idx - len(variant_order) / 2 + 0.5) * (
                    bar_width + spacebetweenbars
                )
                y_pos.append(y_positions[sim_idx] + offset)

                variant_color = variant_colors.get(variant, "grey")
                face_colors.append(variant_color)

        if values:
            ax.barh(
                y_pos,
                values,
                height=bar_width,
                color=face_colors,
                edgecolor=face_colors,
                linewidth=1.0,
            )


def create_prevalence_rmse_plot(
    rmse_df,
    simulations_ordered,
    simulation_labels,
    variant_colors,
    starting_deme_by_sim,
):
    """
    Create the prevalence RMSE barplot.

    Args:
        rmse_df: DataFrame with RMSE data (columns: simulation_base, Deme, variant, rmse)
        simulations_ordered: Ordered list of simulation names
        simulation_labels: List of simulation labels for y-axis
        variant_colors: Dictionary mapping variant names to colors
        starting_deme_by_sim: Dictionary mapping simulation name to starting deme

    Returns:
        Tuple of (figure, axes)
    """
    # Create figure with 2 subplots (one for starting deme, one for other deme)
    fig, axes = plt.subplots(
        1, 2, sharey=True, sharex=True, figsize=(6, len(simulations_ordered) * 0.4)
    )

    plt.rcParams["font.size"] = DEFAULT_FONTSIZES["axis_label"]

    # Variant order: all, then leave-one-out variants (no MASCOT)
    variant_order = ["all"] + VARIANT_ORDER

    # Plot starting deme and other deme
    for deme_idx, deme_type in enumerate(["starting", "other"]):
        ax = axes[deme_idx]

        plot_prevalence_bars(
            ax,
            rmse_df,
            simulations_ordered,
            deme_type,
            variant_order,
            variant_colors,
            "rmse",
            starting_deme_by_sim,
        )

        # Configure axis
        if deme_type == "starting":
            ax.set_title("Outbreak starting deme", fontsize=DEFAULT_FONTSIZES["title"])
        else:
            ax.set_title("Outbreak other deme", fontsize=DEFAULT_FONTSIZES["title"])
        ax.set_xlabel("RMSE", fontsize=DEFAULT_FONTSIZES["axis_label"])
        ax.grid(True, alpha=0.3, axis="x")
        beautify_plot(ax)

    # Add legend
    prevalence_variant_labels = {"all": "All", **VARIANT_LABELS}
    create_legend(axes[0], variant_order, variant_colors, prevalence_variant_labels)

    # Setup y-axis
    setup_y_axis(axes, simulations_ordered, simulation_labels, invert=False)

    plt.tight_layout()

    return fig, axes


def create_prevalence_coverage_plot(
    coverage_df,
    simulations_ordered,
    simulation_labels,
    variant_colors,
    starting_deme_by_sim,
):
    """
    Create the prevalence coverage barplot.

    Args:
        coverage_df: DataFrame with coverage data (columns: simulation_base, Deme, variant, coverage, total_timepoints)
        simulations_ordered: Ordered list of simulation names
        simulation_labels: List of simulation labels for y-axis
        variant_colors: Dictionary mapping variant names to colors
        starting_deme_by_sim: Dictionary mapping simulation name to starting deme

    Returns:
        Tuple of (figure, axes)
    """
    # Create figure with 2 subplots (one for starting deme, one for other deme)
    fig, axes = plt.subplots(
        1, 2, sharey=True, sharex=True, figsize=(6, len(simulations_ordered) * 0.4)
    )

    plt.rcParams["font.size"] = DEFAULT_FONTSIZES["axis_label"]

    # Variant order: all, then leave-one-out variants (no MASCOT)
    variant_order = ["all"] + VARIANT_ORDER

    # Plot starting deme and other deme
    for deme_idx, deme_type in enumerate(["starting", "other"]):
        ax = axes[deme_idx]

        plot_prevalence_bars(
            ax,
            coverage_df,
            simulations_ordered,
            deme_type,
            variant_order,
            variant_colors,
            "coverage",
            starting_deme_by_sim,
        )

        # Configure axis
        if deme_type == "starting":
            ax.set_title("Outbreak starting deme", fontsize=DEFAULT_FONTSIZES["title"])
        else:
            ax.set_title("Outbreak other deme", fontsize=DEFAULT_FONTSIZES["title"])
        ax.set_xlabel(
            "Coverage (number of timepoints)", fontsize=DEFAULT_FONTSIZES["axis_label"]
        )
        ax.grid(True, alpha=0.3, axis="x")
        beautify_plot(ax)

    # Add legend
    prevalence_variant_labels = {"all": "All", **VARIANT_LABELS}
    create_legend(axes[0], variant_order, variant_colors, prevalence_variant_labels)

    # Setup y-axis
    setup_y_axis(axes, simulations_ordered, simulation_labels, invert=False)

    plt.tight_layout()

    return fig, axes


def select_equally_spaced_timepoints(prevalence_df, n_timepoints=10):
    """
    Select n equally spaced timepoints for each simulation/deme/variant combination.

    Args:
        prevalence_df: DataFrame with prevalence data (must have timesincestart, Deme, Simulation, variant columns)
        n_timepoints: Number of timepoints to select (default: 10)

    Returns:
        DataFrame with selected timepoints, including timepoint_index column
    """
    selected_data = []

    for (sim_base, deme, variant), group in prevalence_df.groupby(
        ["simulation_base", "Deme", "variant"]
    ):
        # Sort by timesincestart
        group_sorted = group.sort_values("timesincestart")

        # Get unique timepoints
        unique_times = group_sorted["timesincestart"].unique()

        if len(unique_times) == 0:
            continue

        # Select n equally spaced timepoints
        if len(unique_times) <= n_timepoints:
            # If we have fewer or equal timepoints, use all of them
            selected_indices = np.arange(len(unique_times))
        else:
            # Select equally spaced indices
            selected_indices = np.linspace(
                0, len(unique_times) - 1, n_timepoints, dtype=int
            )

        selected_times = unique_times[selected_indices]

        # Get rows for selected timepoints
        for timepoint_idx, time in enumerate(selected_times):
            time_rows = group_sorted[group_sorted["timesincestart"] == time]
            if not time_rows.empty:
                # Take first row if multiple rows have same time
                row = time_rows.iloc[0].copy()
                row["timepoint_index"] = timepoint_idx
                selected_data.append(row)

    if not selected_data:
        return pd.DataFrame()

    return pd.DataFrame(selected_data)


def calculate_prevalence_hpd_width(prevalence_df):
    """
    Calculate HPD width for prevalence data (using hpd_upperP1 and hpd_lowerP1).

    Args:
        prevalence_df: DataFrame with prevalence data

    Returns:
        DataFrame with added hpd_width column
    """
    df = prevalence_df.copy()

    # Calculate HPD width if columns exist
    if (
        "logPrevalence_hpd_upper" in df.columns
        and "logPrevalence_hpd_lower" in df.columns
    ):
        df["hpd_width"] = df["logPrevalence_hpd_upper"] - df["logPrevalence_hpd_lower"]
    else:
        # If columns don't exist, set hpd_width to NaN
        df["hpd_width"] = np.nan

    if "inHPD" not in df.columns:
        df["inHPD"] = False

    return df


def calculate_timepoint_hpd_differences(
    timepoint_data, simulations_ordered, variant_order, all_data_timepoint
):
    """
    Calculate HPD width differences for a specific timepoint (similar to calculate_hpd_differences).

    Args:
        timepoint_data: DataFrame filtered for a specific timepoint and deme
        simulations_ordered: Ordered list of simulation names
        variant_order: List of variant names to process
        all_data_timepoint: DataFrame with "all" variant data for baseline

    Returns:
        Dictionary mapping variant to tuple of (widths, sim_indices, face_colors, edge_colors)
    """
    variant_data_dict = {}

    for variant_idx, variant in enumerate(variant_order):
        variant_data = timepoint_data[timepoint_data["variant"] == variant]

        if variant_data.empty:
            continue

        widths = []
        edge_colors = []
        face_colors = []
        sim_indices = []

        for sim_idx, sim in enumerate(simulations_ordered):
            sim_variant_data = variant_data[variant_data["simulation_base"] == sim]
            sim_all_data = all_data_timepoint[
                all_data_timepoint["simulation_base"] == sim
            ]

            if not sim_variant_data.empty and not sim_all_data.empty:
                variant_row = sim_variant_data.iloc[0]
                all_row = sim_all_data.iloc[0]

                variant_hpd_width = variant_row["hpd_width"]
                all_hpd_width = all_row["hpd_width"]
                in_hpd = variant_row["inHPD"]

                if pd.notna(variant_hpd_width) and pd.notna(all_hpd_width):
                    hpd_diff = (variant_hpd_width - all_hpd_width) / all_hpd_width
                    widths.append(hpd_diff)

                    if in_hpd:
                        face_colors.append(VARIANT_COLORS[variant])
                        edge_colors.append(VARIANT_COLORS[variant])
                    else:
                        face_colors.append("none")
                        edge_colors.append(VARIANT_COLORS[variant])

                    sim_indices.append(sim_idx)

        if widths:
            variant_data_dict[variant] = (widths, sim_indices, face_colors, edge_colors)

    return variant_data_dict


def plot_timepoint_comparison(
    ax,
    timepoint_idx,
    deme,
    timepoint_data,
    simulations_ordered,
    all_data_timepoint,
    bar_width=0.2,
    show_xlabel=False,
):
    """
    Plot HPD width differences for a single timepoint (similar to plot_parameter_comparison).

    Args:
        ax: Matplotlib axis object
        timepoint_idx: Index of the timepoint (0-9)
        deme: Deme number (0 or 1) - used for reference but not in title
        timepoint_data: DataFrame filtered for this timepoint and deme
        simulations_ordered: Ordered list of simulation names
        all_data_timepoint: DataFrame with "all" variant data for baseline
        bar_width: Width of bars for grouping
        show_xlabel: If True, show x-axis label
    """
    # Calculate HPD differences for all variants
    variant_data_dict = calculate_timepoint_hpd_differences(
        timepoint_data, simulations_ordered, VARIANT_ORDER, all_data_timepoint
    )

    # Plot bars
    plot_bars(ax, variant_data_dict, VARIANT_ORDER, len(simulations_ordered), bar_width)

    # Add scatter markers
    add_scatter_markers(ax, all_data_timepoint, simulations_ordered)

    # Configure axis
    title = f"Timepoint {timepoint_idx + 1}"
    configure_comparison_axis(ax, title, show_xlabel)


def create_prevalence_timepoints_plot(
    prevalence_df,
    simulations_ordered,
    simulation_labels,
    starting_deme_by_sim,
    n_timepoints=10,
    invert=False,
):
    """
    Create the prevalence timepoints information content comparison plot.

    Args:
        prevalence_df: DataFrame with prevalence data
        simulations_ordered: Ordered list of simulation names
        simulation_labels: List of simulation labels for y-axis
        starting_deme_by_sim: Dictionary mapping simulation name to starting deme
        n_timepoints: Number of timepoints to plot (default: 10)

    Returns:
        Tuple of (figure, axes)
    """
    # Calculate HPD width for prevalence data
    prevalence_df = calculate_prevalence_hpd_width(prevalence_df)

    # Select equally spaced timepoints
    timepoint_df = select_equally_spaced_timepoints(prevalence_df, n_timepoints)

    if timepoint_df.empty:
        raise ValueError("No timepoint data available after selection")

    # Create figure with 2 rows and n_timepoints columns
    fig, axes = plt.subplots(
        2, n_timepoints, sharey=True, figsize=(n_timepoints * 1.5, 6)
    )
    axes = axes.flatten()

    plt.rcParams["font.size"] = DEFAULT_FONTSIZES["axis_label"]

    # Determine starting and other deme for each simulation
    # Row 0: starting deme, Row 1: other deme
    for row_idx, target_deme_type in enumerate(["starting", "other"]):
        for timepoint_idx in range(n_timepoints):
            ax_idx = row_idx * n_timepoints + timepoint_idx
            ax = axes[ax_idx]

            # Filter for this timepoint
            timepoint_data_all = timepoint_df[
                timepoint_df["timepoint_index"] == timepoint_idx
            ]

            # Determine which deme to plot for this row
            filtered_data = []
            target_deme = None
            for sim_base in simulations_ordered:
                base_sim = extract_simulation_name(sim_base)
                if base_sim in starting_deme_by_sim:
                    starting_deme = starting_deme_by_sim[base_sim]
                    other_deme = 1 - starting_deme

                    if target_deme_type == "starting":
                        target_deme = starting_deme
                    else:
                        target_deme = other_deme

                    # Filter for this simulation and deme
                    sim_deme_data = timepoint_data_all[
                        (timepoint_data_all["simulation_base"] == sim_base)
                        & (timepoint_data_all["Deme"] == target_deme)
                    ]
                    filtered_data.append(sim_deme_data)

            if filtered_data:
                timepoint_data = pd.concat(filtered_data, ignore_index=True)
            else:
                timepoint_data = pd.DataFrame()

            # Get "all" datastreams data for baseline
            all_data_timepoint = timepoint_data[timepoint_data["variant"] == "all"]

            # Only show xlabel for bottom row plots
            show_xlabel = row_idx == 1

            # Plot timepoint comparison
            plot_timepoint_comparison(
                ax,
                timepoint_idx,
                target_deme if target_deme is not None else 0,
                timepoint_data,
                simulations_ordered,
                all_data_timepoint,
                show_xlabel=show_xlabel,
            )

    # Add legend to first subplot
    create_legend(axes[0], VARIANT_ORDER, VARIANT_COLORS, VARIANT_LABELS)

    # Setup y-axis - first invert if requested (before setting ticks)
    # Note: invert=False by default, but can be enabled if needed
    if invert:
        for ax in axes:
            ax.invert_yaxis()

    y_positions = np.arange(len(simulations_ordered))

    # Setup y-axis for leftmost subplots only
    for row_idx in range(2):
        leftmost_ax = axes[row_idx * n_timepoints]
        leftmost_ax.set_yticks(y_positions)
        leftmost_ax.set_yticklabels(
            simulation_labels, fontsize=DEFAULT_FONTSIZES["tick_label"]
        )

        # Add row label
        if row_idx == 0:
            leftmost_ax.set_ylabel(
                "Outbreak starting deme", fontsize=DEFAULT_FONTSIZES["axis_label"]
            )
        else:
            leftmost_ax.set_ylabel(
                "Outbreak other deme", fontsize=DEFAULT_FONTSIZES["axis_label"]
            )

    plt.tight_layout()

    return fig, axes


def save_plots(fig, output_base):
    """
    Save plots as PDF and PNG.

    Args:
        fig: Matplotlib figure object
        output_base: Base filename (without extension)
    """
    fig.savefig(f"{output_base}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    print(f"Saved plots to {output_base}.pdf and {output_base}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Quantify information content by comparing HPD widths across datastream configurations"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="CSV files: first should be all datastreams, then leave-one-out files",
    )
    parser.add_argument(
        "--prevalence-files",
        nargs="+",
        help="CSV files for prevalence data (all_prevalence_* files)",
    )
    parser.add_argument(
        "--migration-rates-files",
        nargs="+",
        help="CSV files for migration rates data (all_migration_rates* files)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="information_content",
        help="Output filename prefix (default: information_content)",
    )

    args = parser.parse_args()

    # Extract starting deme dictionary from prevalence file
    starting_deme_by_sim = {}
    if args.prevalence_files:
        all_prevalence_file = find_all_prevalence_file(args.prevalence_files)
        if all_prevalence_file:
            starting_deme_by_sim = extract_starting_deme(all_prevalence_file)
            print(
                f"Extracted starting deme for {len(starting_deme_by_sim)} simulations"
            )
        else:
            print(
                "Warning: Could not find all_prevalence_datastreams_hpd_validation.csv file"
            )

    # Load and combine all data files
    combined_df = load_all_data_files(args.files)

    # Get unique parameters and simulations
    parameters = [
        "caseCounts.scaling.Deme1:SimDataset",
        "caseCounts.scaling.Deme2:SimDataset",
        "wastewater.scaling.Deme1:SimDataset",
        "wastewater.scaling.Deme2:SimDataset",
        "seroprevalence.scaling.Deme1:SimDataset",
        "seroprevalence.scaling.Deme2:SimDataset",
        "caseCounts.dispersion:SimDataset",
        "wastewater.sigma:SimDataset",
    ]

    # Dictionary to translate parameter names to more legible names
    parameter_legible_names = {
        "caseCounts.scaling.Deme1:SimDataset": "CC scaling Deme1",
        "caseCounts.scaling.Deme2:SimDataset": "CC scaling Deme2",
        "caseCounts.dispersion:SimDataset": "CC dispersion",
        "wastewater.scaling.Deme1:SimDataset": "WW scaling Deme1",
        "wastewater.scaling.Deme2:SimDataset": "WW scaling Deme2",
        "wastewater.sigma:SimDataset": "WW sigma",
        "seroprevalence.scaling.Deme1:SimDataset": "SP scaling Deme1",
        "seroprevalence.scaling.Deme2:SimDataset": "SP scaling Deme2",
    }
    all_simulations = sorted(combined_df["simulation_base"].unique())

    # Order simulations by starting deme
    simulations_ordered = order_simulations_by_deme(
        all_simulations, starting_deme_by_sim
    )

    # Create simulation labels
    simulation_labels, simulation_labels_nodeme = create_simulation_labels(
        simulations_ordered, starting_deme_by_sim
    )

    # Create and save parameter plot
    fig, axes = create_information_content_plot(
        combined_df,
        parameters,
        simulations_ordered,
        simulation_labels,
        parameter_legible_names,
    )

    save_plots(fig, f"{args.output}_parameters")

    # Handle migration rates if provided
    if args.migration_rates_files:
        # Load migration rates files
        combined_migration_df = load_all_migration_rates_files(
            args.migration_rates_files
        )

        # Add migration direction column based on starting deme
        combined_migration_df = determine_migration_direction(
            combined_migration_df, starting_deme_by_sim
        )

        # Filter out rows where migration direction could not be determined
        combined_migration_df = combined_migration_df[
            combined_migration_df["migration_direction"].notna()
        ]

        # Create variant colors dictionary (include MASCOT)
        variant_colors = VARIANT_COLORS.copy()
        variant_colors[MODEL_MASCOT] = COLORS[4] if len(COLORS) > 4 else "#cc79a7"

        # Create and save migration rates plot
        fig_migration, axes_migration = create_migration_rates_plot(
            combined_migration_df,
            simulations_ordered,
            simulation_labels,
            variant_colors,
        )

        migration_output = f"{args.output}_migration_rates"
        save_plots(fig_migration, migration_output)

    # Handle prevalence files if provided
    if args.prevalence_files:
        # Load prevalence files
        combined_prevalence_df = load_all_prevalence_files(args.prevalence_files)

        # Calculate RMSE for each simulation/deme/variant
        rmse_df = calculate_prevalence_rmse(combined_prevalence_df)

        # Calculate coverage for each simulation/deme/variant
        coverage_df = calculate_prevalence_coverage(combined_prevalence_df)

        # Create variant colors dictionary (no MASCOT for prevalence)
        variant_colors = VARIANT_COLORS.copy()

        # Create and save RMSE plot
        fig_rmse, axes_rmse = create_prevalence_rmse_plot(
            rmse_df,
            simulations_ordered,
            simulation_labels_nodeme,
            variant_colors,
            starting_deme_by_sim,
        )

        rmse_output = f"{args.output}_prevalence_rmse"
        save_plots(fig_rmse, rmse_output)

        # Create and save coverage plot
        fig_coverage, axes_coverage = create_prevalence_coverage_plot(
            coverage_df,
            simulations_ordered,
            simulation_labels_nodeme,
            variant_colors,
            starting_deme_by_sim,
        )

        coverage_output = f"{args.output}_prevalence_coverage"
        save_plots(fig_coverage, coverage_output)

        # Create and save timepoints plot
        fig_timepoints, axes_timepoints = create_prevalence_timepoints_plot(
            combined_prevalence_df,
            simulations_ordered,
            simulation_labels_nodeme,
            starting_deme_by_sim,
            n_timepoints=10,
        )

        timepoints_output = f"{args.output}_prevalence_timepoints"
        save_plots(fig_timepoints, timepoints_output)


if __name__ == "__main__":
    main()
