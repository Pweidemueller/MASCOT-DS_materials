#!/usr/bin/env python3
"""
Plot ESS values as dot heatmaps for mascot_original and mascot_datastreams.

Creates two subplots showing ESS values as a dot heatmap where:
- y-axis: simulations
- x-axis: parameters (posterior and prior first, then alphabetical)
- Dot size: log scale of ESS (same range for both subplots)
- Color: black X if ESS < 200, otherwise colorbar from light hue at 200 to target color
"""

import argparse
import logging
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap
from constants import COLORBLINDFR
from plot_utils import save_figure_png_and_pdf, set_axis_fontsizes

logger = logging.getLogger(__name__)

# Configure matplotlib to save PDFs with editable text (not paths)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

fontsizes = [18, 15, 12]

colorblindfr = COLORBLINDFR


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot ESS values as dot heatmaps for mascot models"
    )
    parser.add_argument(
        "--ess_summary",
        type=str,
        required=True,
        help="Path to ess_summary.csv file",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        required=False,
        default="ess_heatmap",
        help="Prefix for output files (default: ess_heatmap)",
    )
    return parser.parse_args()


def extract_simulation_name(run_name):
    """
    Extract simulation name from beast_run_name.

    Examples:
    - "1_2_simulation_original.mascot_logs" -> "1_2_simulation"
    - "sim_5_datastreams.mascot_logs" -> "sim_5"
    """
    # Remove .mascot_logs suffix if present
    name = run_name.replace(".mascot_logs", "")

    # Remove _original or _datastreams suffix
    if name.endswith("_original"):
        name = name[:-9]  # Remove "_original"
    elif name.endswith("_datastreams"):
        name = name[:-12]  # Remove "_datastreams"

    return name


def filter_inf_parameters(df):
    """
    Remove parameters where all ESS values are inf across all simulations.

    Args:
        df: DataFrame with columns: simulation, parameter, ESS

    Returns:
        filtered DataFrame with all-inf parameters removed
    """
    # Group by parameter and check if all ESS values are inf
    param_groups = df.groupby("parameter")["ESS"]

    # Find parameters where all values are inf
    parameters_to_remove = []
    for param_name, param_data in param_groups:
        # Drop NA values and check if all remaining values are inf
        non_na_values = param_data.dropna()
        if len(non_na_values) > 0:
            all_inf = all(np.isinf(val) for val in non_na_values)
            if all_inf:
                parameters_to_remove.append(param_name)

    # Filter out these parameters
    if parameters_to_remove:
        df_filtered = df[~df["parameter"].isin(parameters_to_remove)].copy()
        return df_filtered

    return df.copy()


def natural_sort_key(text):
    """
    Generate a sort key for natural sorting that handles numeric parts correctly.

    Example: "SkylineNe.Deme1.1.10" will sort after "SkylineNe.Deme1.1.9"

    Args:
        text: string to generate sort key for

    Returns:
        tuple of alternating (text, int) values for proper sorting
    """
    # Split into alternating text and numeric parts
    parts = re.split(r"(\d+)", text)
    # Convert numeric parts to int, keep text parts as strings
    key = []
    for part in parts:
        if part:  # Skip empty strings
            if part.isdigit():
                key.append(int(part))  # Numeric part as int
            else:
                key.append(part.lower())  # Text part as lowercase string
    return tuple(key)


def order_parameters(parameters):
    """
    Order parameters: posterior and prior first, then natural sort.

    Args:
        parameters: list of parameter names

    Returns:
        ordered list of parameters
    """
    params_list = list(parameters)

    # Find posterior and prior
    priority_params = []
    other_params = []

    for param in params_list:
        param_lower = param.lower()
        if param_lower == "posterior" or param_lower == "prior":
            priority_params.append(param)
        else:
            other_params.append(param)

    # Sort priority params: posterior first, then prior
    priority_params.sort(key=lambda x: (x.lower() != "posterior", x.lower()))

    # Sort other params using natural sort (handles numeric parts correctly)
    other_params.sort(key=natural_sort_key)

    return priority_params + other_params


def filter_all_inf_parameters(df, params):
    """Return parameters that have at least one finite ESS value."""
    filtered = []
    for param in params:
        param_data = df[df["parameter"] == param]["ESS"].dropna()
        if len(param_data) > 0 and not all(np.isinf(val) for val in param_data):
            filtered.append(param)
    return filtered


def create_colorbar_colormap(min_val, max_val, target_color):
    """
    Create a colormap from light hue at min_val to target_color at max_val.

    Args:
        min_val: minimum ESS value (should be 200)
        max_val: maximum ESS value
        target_color: hex color for maximum value

    Returns:
        tuple: (color_func, cmap) where color_func maps values to colors and cmap is the matplotlib colormap
    """
    # Convert hex to RGB
    target_rgb = tuple(int(target_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))

    # Create light version (mix with white - 80% white, 20% target color)
    light_rgb = tuple(0.8 + 0.2 * c for c in target_rgb)

    # Create colormap
    colors = [light_rgb, target_rgb]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    def color_func(val):
        if val < min_val:
            return None
        # Handle edge case where val equals max_val exactly
        if val >= max_val:
            # Return target color directly for max_val and above
            return target_color
        # Normalize to [0, 1)
        normalized = (val - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
        rgba = cmap(normalized)
        # Convert to hex
        return matplotlib.colors.rgb2hex(rgba)

    return color_func, cmap


def plot_ess_heatmap(
    df,
    model_type,
    ax,
    size_min,
    size_max,
    color_func,
    target_color,
    cmap,
    min_ess,
    max_ess,
):
    """
    Plot ESS values as dot heatmap.

    Args:
        df: DataFrame with columns: simulation, parameter, ESS
        model_type: "original" or "datastreams"
        ax: matplotlib axis
        size_min: minimum log ESS for size scaling
        size_max: maximum log ESS for size scaling
        color_func: function to map ESS to color
        target_color: target color for colorbar
        cmap: matplotlib colormap for colorbar
        min_ess: minimum ESS value for colorbar (200)
        max_ess: maximum ESS value for colorbar
    """
    # Get unique simulations and parameters
    simulations = sorted(df["simulation"].unique())
    parameters = order_parameters(df["parameter"].unique())

    # Create pivot table
    pivot = df.pivot_table(
        index="simulation", columns="parameter", values="ESS", aggfunc="first"
    )

    # Reorder columns according to parameter order
    pivot = pivot[parameters]

    # Reorder rows according to simulation order
    pivot = pivot.reindex(simulations)

    # Filter out parameters where all ESS values are inf
    parameters_to_keep = []
    for param in parameters:
        param_values = pivot[param].dropna()
        if len(param_values) == 0:
            # Skip if all values are NA
            continue
        # Check if all non-NA values are inf
        all_inf = all(np.isinf(val) for val in param_values)
        if not all_inf:
            parameters_to_keep.append(param)

    # Update parameters list and pivot table
    parameters = parameters_to_keep
    if len(parameters) == 0:
        # No parameters to plot
        ax.text(
            0.5,
            0.5,
            "No parameters to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    pivot = pivot[parameters]

    # Prepare data for scatter plot
    x_coords = []
    y_coords = []
    sizes = []
    colors_list = []
    markers = []

    for sim_idx, sim in enumerate(simulations):
        for param_idx, param in enumerate(parameters):
            ess = pivot.loc[sim, param]

            if pd.isna(ess):
                continue

            # Handle inf values
            is_inf = np.isinf(ess) or ess == np.inf

            x_coords.append(param_idx)
            y_coords.append(sim_idx)

            # Size: log scale of ESS
            if is_inf:
                # For inf, use maximum size
                normalized_size = 1.0
            else:
                log_ess = np.log(max(ess, 1))  # Avoid log(0)
                # Normalize to [0, 1] based on global min/max
                normalized_size = (log_ess - size_min) / (size_max - size_min)
                normalized_size = max(0, min(1, normalized_size))  # Clamp
            # Scale to reasonable dot size (e.g., 20 to 200)
            sizes.append(10 + normalized_size * 140)

            # Color: black X if ESS < 200, otherwise use colorbar
            if is_inf or ess >= 200:
                if is_inf:
                    # For inf, use maximum color (target color)
                    colors_list.append(target_color)
                else:
                    colors_list.append(color_func(ess))
                markers.append("o")
            else:
                colors_list.append("black")
                markers.append("x")

    # Plot points - separate circles and X markers for better control
    x_coords_circle = []
    y_coords_circle = []
    sizes_circle = []
    colors_circle = []

    x_coords_x = []
    y_coords_x = []
    sizes_x = []

    for x, y, size, color, marker in zip(
        x_coords, y_coords, sizes, colors_list, markers
    ):
        if marker == "x":
            x_coords_x.append(x)
            y_coords_x.append(y)
            sizes_x.append(size)
        else:
            x_coords_circle.append(x)
            y_coords_circle.append(y)
            sizes_circle.append(size)
            colors_circle.append(color)

    # Plot circles
    if x_coords_circle:
        ax.scatter(
            x_coords_circle,
            y_coords_circle,
            s=sizes_circle,
            c=colors_circle,
            marker="o",
            edgecolors="none",
            alpha=0.7,
        )

    # Plot X markers (black)
    if x_coords_x:
        ax.scatter(
            x_coords_x,
            y_coords_x,
            s=sizes_x,
            c="black",
            marker="x",
            linewidths=2,
            alpha=0.8,
        )

    # Set labels
    ax.set_xticks(range(len(parameters)))
    ax.set_xticklabels(parameters, rotation=45, ha="right")
    ax.set_yticks(range(len(simulations)))
    ax.set_yticklabels(simulations)

    ax.set_xlabel("Parameter", fontsize=fontsizes[1])
    ax.set_ylabel("Simulation", fontsize=fontsizes[1])
    ax.set_title(f"mascot_{model_type}", fontsize=fontsizes[0])

    set_axis_fontsizes(ax, fontsizes)

    # Set limits
    ax.set_xlim(-0.5, len(parameters) - 0.5)
    ax.set_ylim(-0.5, len(simulations) - 0.5)

    # Invert y-axis so first simulation is at top
    ax.invert_yaxis()

    # Add colorbar on the right side
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=min_ess, vmax=max_ess)
    )
    sm.set_array([])  # Empty array, we just want the colormap
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("ESS", fontsize=fontsizes[1])
    cbar.ax.tick_params(labelsize=fontsizes[2])


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_arguments()

    # Read ESS summary CSV
    ess_df = pd.read_csv(args.ess_summary)

    # Ensure ESS column is numeric (handles "inf" strings)
    ess_df["ESS"] = pd.to_numeric(ess_df["ESS"], errors="coerce")

    # Filter for mascot_original and "other" model (any non-original)
    # Format in CSV: "*_original.mascot_logs" and one other variant (e.g. datastreams, datastreams_nocasecounts, ...)
    mask_original = ess_df["beast_run_name"].str.contains(
        "_original.mascot_logs", case=False, na=False
    )
    mask_datastreams = ~mask_original

    df_original = ess_df[mask_original].copy()
    df_datastreams = ess_df[mask_datastreams].copy()

    if len(df_original) == 0:
        raise ValueError("No rows found for mascot_original")
    if len(df_datastreams) == 0:
        raise ValueError("No rows found for mascot_datastreams")

    # Extract simulation names
    df_original["simulation"] = df_original["beast_run_name"].apply(
        extract_simulation_name
    )
    df_datastreams["simulation"] = df_datastreams["beast_run_name"].apply(
        extract_simulation_name
    )

    # Rename column for consistency
    df_original = df_original.rename(columns={"parameter_name": "parameter"})
    df_datastreams = df_datastreams.rename(columns={"parameter_name": "parameter"})

    # Remove parameters where all ESS values are inf (before calculating global min/max)
    df_original = filter_inf_parameters(df_original)
    df_datastreams = filter_inf_parameters(df_datastreams)

    # Calculate global min/max for log ESS (for consistent size scaling)
    all_ess = pd.concat([df_original["ESS"], df_datastreams["ESS"]])
    # Convert "inf" strings to np.inf if needed, then filter out inf and non-positive values
    all_ess = all_ess.replace([np.inf, -np.inf], np.nan)
    all_ess = all_ess[all_ess > 0]  # Remove zeros/negatives/nan
    log_ess_min = np.log(all_ess.min())
    log_ess_max = np.log(all_ess.max())

    logger.debug("log_ess_min: %s, log_ess_max: %s", log_ess_min, log_ess_max)

    # Calculate max ESS for each model (for colorbar)
    # If max is inf, use the maximum finite value instead
    max_ess_original = df_original["ESS"].replace([np.inf, -np.inf], np.nan).max()
    max_ess_datastreams = df_datastreams["ESS"].replace([np.inf, -np.inf], np.nan).max()

    # If all values are inf, use a default large value
    if pd.isna(all_ess.max()):
        max_ess = 10000
    else:
        max_ess = all_ess.max()

    # Create color functions and colormaps
    # For values >= 200, use colorbar from light hue at 200 to target color
    color_func_original, cmap_original = create_colorbar_colormap(
        200, max_ess, colorblindfr["main"][4]  # "#cc79a7" for original
    )
    color_func_datastreams, cmap_datastreams = create_colorbar_colormap(
        200, max_ess, colorblindfr["main"][3]  # "#e69d00" for datastreams
    )

    # Get parameter counts before plotting to determine subplot widths
    # We need to know how many parameters each subplot will have
    params_original = order_parameters(df_original["parameter"].unique())
    params_datastreams = order_parameters(df_datastreams["parameter"].unique())

    # Filter out all-inf parameters
    params_original_filtered = filter_all_inf_parameters(df_original, params_original)
    params_datastreams_filtered = filter_all_inf_parameters(df_datastreams, params_datastreams)

    n_params_original = len(params_original_filtered)
    n_params_datastreams = len(params_datastreams_filtered)

    # Calculate relative widths based on parameter counts
    # Width should be proportional to number of parameters to keep spacing similar
    if n_params_original == 0 and n_params_datastreams == 0:
        width_ratios = [1, 1]
    elif n_params_original == 0:
        width_ratios = [0.1, 1]
    elif n_params_datastreams == 0:
        width_ratios = [1, 0.1]
    else:
        # Proportional to number of parameters
        width_ratios = [n_params_original, n_params_datastreams]

    # Create figure with two subplots with adjusted widths
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": width_ratios}
    )

    # Plot both subplots
    plot_ess_heatmap(
        df_original,
        "original",
        ax1,
        log_ess_min,
        log_ess_max,
        color_func_original,
        colorblindfr["main"][4],
        cmap_original,
        200,
        max_ess,
    )

    plot_ess_heatmap(
        df_datastreams,
        "datastreams",
        ax2,
        log_ess_min,
        log_ess_max,
        color_func_datastreams,
        colorblindfr["main"][3],
        cmap_datastreams,
        200,
        max_ess,
    )

    plt.tight_layout()

    # Save figure
    output_file = f"{args.out_prefix}.png"
    save_figure_png_and_pdf(output_file)

    logger.info(
        "ESS heatmap plot saved to %s.png and %s.pdf", args.out_prefix, args.out_prefix
    )


if __name__ == "__main__":
    main()
