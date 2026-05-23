#!/usr/bin/env python3
"""
Script to process BEAST2 log files by skipping comment lines.
"""

import argparse
import logging
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import scipy as scp
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)
from plot_utils import (
    COLORBLINDFR,
    COLORS,
    DEFAULT_FONTSIZES,
    beautify_plot,
    get_outbreak_start_deme,
    save_figure_png_and_pdf,
    set_axis_fontsizes,
)

# Local font sizes (copy of DEFAULT_FONTSIZES so user can change in script)
fontsizes = dict(DEFAULT_FONTSIZES)

# Posterior columns for datastream auxiliary parameters (KDE / HPD tables).
DATASTREAM_POSTERIOR_PARAMETERS = (
    "caseCounts.dispersion:SimDataset",
    "caseCounts.scaling.Deme1:SimDataset",
    "caseCounts.scaling.Deme2:SimDataset",
    # "seroprevalence.scaling.Deme1:SimDataset",  # fixed to 1.0; not estimated
    # "seroprevalence.scaling.Deme2:SimDataset",  # fixed to 1.0; not estimated
    "wastewater.scaling.Deme1:SimDataset",
    "wastewater.scaling.Deme2:SimDataset",
    "wastewater.sigma:SimDataset",
)


def _fontsizes_list():
    """List [title, axis_label, tick_label] for set_axis_fontsizes, from current fontsizes."""
    return [fontsizes["title"], fontsizes["axis_label"], fontsizes["tick_label"]]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process BEAST2 log files by skipping comment lines."
    )
    parser.add_argument(
        "--log_file_original",
        type=str,
        required=True,
        help="Path to the BEAST2 log file to analyse for default model",
    )
    parser.add_argument(
        "--log_file_datastream",
        type=str,
        required=True,
        help="Path to the BEAST2 log file to analyse for datastream model",
    )
    parser.add_argument(
        "--trajectory_file",
        type=str,
        required=False,
        help="Path to the trajectory file to load as a table",
    )
    parser.add_argument(
        "--case_counts_file",
        type=str,
        required=False,
        help="Path to the case counts file to load as a table",
    )
    parser.add_argument(
        "--seroprevalence_file",
        type=str,
        required=False,
        help="Path to the seroprevalence file to load as a table",
    )
    parser.add_argument(
        "--wastewater_file",
        type=str,
        required=False,
        help="Path to the wastewater file to load as a table",
    )
    parser.add_argument(
        "--params_csv",
        type=str,
        required=False,
        help="Optional path to the parameters CSV exported by create_birthdeath_simXML.py",
    )
    parser.add_argument(
        "--deme_switches_csv",
        type=str,
        required=False,
        help="Optional path to the deme switches CSV with columns: Deme_parent, Deme_child, switch",
    )
    parser.add_argument(
        "--out_prefix", type=str, required=False, help="Prefix for output files"
    )
    parser.add_argument(
        "--cumulative_incidence_deme1",
        type=str,
        required=False,
        help="Path to cumulative incidence log file for deme 1",
    )
    parser.add_argument(
        "--cumulative_incidence_deme2",
        type=str,
        required=False,
        help="Path to cumulative incidence log file for deme 2",
    )
    parser.add_argument(
        "--nedynamics_deme1",
        type=str,
        required=False,
        help="Path to NeDynamics log file for deme 1",
    )
    parser.add_argument(
        "--nedynamics_deme2",
        type=str,
        required=False,
        help="Path to NeDynamics log file for deme 2",
    )
    parser.add_argument(
        "--burnin",
        type=float,
        required=False,
        default=0.1,
        help="Burnin percentage",
    )
    return parser.parse_args()


def extract_mostrecent_sample_t(trajectory_df):
    """
    Extract the most recent sample time from a trajectory file.
    """
    # Filter for the "sample" population
    sample_times = trajectory_df.loc[
        (trajectory_df.population == "sample") & (trajectory_df["index"] == 0)
    ].sort_values("t")
    sample_values = sample_times["value"].values
    sample_t = sample_times["t"].values

    # Find last index where value changes compared to the next — i.e., after which it remains constant
    # We'll search for the last change in the "value" sequence
    value_diff = np.diff(sample_values)
    changed_indices = np.where(value_diff != 0)[0]
    if len(changed_indices) > 0:
        last_change_idx = (
            changed_indices[-1] + 1
        )  # +1 because diff is one element shorter
        mostrecent_sample_t = sample_t[last_change_idx]
    return mostrecent_sample_t


def read_beast_log(file_path, read_rateshifts=False):
    """
    Read a BEAST2 log file, skipping comment lines.

    Args:
        file_path (str): Path to the log file
        read_rateshifts (bool): If True, also extract rateshifts from comment lines
        read_gridpointshifts (bool): If True, also extract gridpoint shifts from comment lines
    Returns:
        pd.DataFrame or tuple: DataFrame of log data. If read_rateshifts is True,
            returns a tuple (DataFrame, list of rateshifts as floats)
            If read_gridpointshifts is True, returns a tuple (DataFrame, list of gridpoint shifts as floats)
    """
    with open(file_path, "r", encoding="utf-8") as file:
        log_content = [line.strip() for line in file if not line.startswith("#")]
    data = [line.split("\t") for line in log_content]
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.apply(pd.to_numeric, errors="coerce")

    if not read_rateshifts:
        return df

    # Read the file again to extract rateshifts from comment lines
    rateshifts = None
    gridpointshifts = None
    detected_rateshifts = False
    detected_gridpointshifts = False
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("#") and 'id="SkygrowthRateShifts"' in line:
                # Extract content between <rateShifts...> and </rateShifts>
                start_tag = "<rateShifts"
                end_tag = "</rateShifts>"
                start_idx = line.find(start_tag)
                if start_idx != -1:
                    # Find the closing > of the opening tag
                    tag_end_idx = line.find(">", start_idx)
                    if tag_end_idx != -1:
                        # Find the closing tag
                        end_idx = line.find(end_tag, tag_end_idx)
                        if end_idx != -1:
                            # Extract the content between > and </rateShifts>
                            rateshifts_str = line[tag_end_idx + 1 : end_idx].strip()
                            # Split by whitespace and convert to floats
                            rateshifts = [float(x) for x in rateshifts_str.split()]
                            detected_rateshifts = True
            if line.startswith("#") and 'id="SplineGridRateShifts"' in line:
                # Extract content between <rateShifts...> and </rateShifts>
                start_tag = "<gridRateShifts"
                end_tag = "</gridRateShifts>"
                start_idx = line.find(start_tag)
                if start_idx != -1:
                    # Find the closing > of the opening tag
                    tag_end_idx = line.find(">", start_idx)
                    if tag_end_idx != -1:
                        # Find the closing tag
                        end_idx = line.find(end_tag, tag_end_idx)
                        if end_idx != -1:
                            # Extract the content between > and </rateShifts>
                            gridpointshifts_str = line[
                                tag_end_idx + 1 : end_idx
                            ].strip()
                            # Split by whitespace and convert to floats
                            gridpointshifts = [
                                float(x) for x in gridpointshifts_str.split()
                            ]
                            detected_gridpointshifts = True
            if detected_rateshifts and detected_gridpointshifts:
                break

    rateshifts = np.array(rateshifts)
    gridpointshifts = np.array(gridpointshifts)
    return df, rateshifts, gridpointshifts


def create_cumulative_incidence_long(df):

    wide_data = []

    for sample_idx, row in df.iterrows():
        sample = row["Sample"]

        # Group data by gridpoint
        gridpoint_data = {}

        for col_name, value in row.items():
            if col_name == "Sample":
                continue

            param_type, gridpoint = parse_column_name(col_name)
            if param_type is not None:
                if gridpoint not in gridpoint_data:
                    gridpoint_data[gridpoint] = {
                        "Sample": sample,
                        "gridpoint": gridpoint,
                    }
                gridpoint_data[gridpoint][param_type] = value

        # Add each gridpoint as a separate row
        for gridpoint, data in gridpoint_data.items():
            wide_data.append(data)

    return pd.DataFrame(wide_data)


# Transform Ne_dynamics_deme1 to wide format with Sample, gridpoint, and parameter columns
def parse_column_name(col_name):
    """Parse column names to extract parameter type and gridpoint"""

    if col_name.startswith("Ne_"):
        return "logNe", float(col_name.split("_")[1])
    elif col_name.startswith("I"):
        return "logPrevalence", float(col_name.split("_")[1])
    elif col_name.startswith("transmissionRate_"):
        return "transmissionRate", float(col_name.split("_")[1])
    elif col_name.startswith("cumulativeIncidence_"):
        return "cumulativeIncidence", float(col_name.split("_")[1])
    elif col_name.startswith("propSeropositive_"):
        return "propSeropositive", float(col_name.split("_")[1])
    else:
        return None, None


def create_Ne_dynamics_long(Ne_dynamics):
    # Create wide format data with Sample, gridpoint, logPrevalence, transmissionRate, logNe columns
    wide_data = []

    for sample_idx, row in Ne_dynamics.iterrows():
        sample = row["Sample"]

        # Group data by gridpoint
        gridpoint_data = {}

        for col_name, value in row.items():
            if col_name == "Sample":
                continue

            param_type, gridpoint = parse_column_name(col_name)
            if param_type is not None:
                if gridpoint not in gridpoint_data:
                    gridpoint_data[gridpoint] = {
                        "Sample": sample,
                        "gridpoint": gridpoint,
                    }
                gridpoint_data[gridpoint][param_type] = value

        # Add each gridpoint as a separate row
        for gridpoint, data in gridpoint_data.items():
            wide_data.append(data)

    return pd.DataFrame(wide_data)


def load_trajectory_file(file_path):
    """
    Load a trajectory file as a pandas DataFrame.

    Args:
        file_path (str): Path to the trajectory file

    Returns:
        pd.DataFrame: DataFrame with columns Sample, t, population, index, value
    """
    try:
        # Read the trajectory file as tab-separated values
        df = pd.read_csv(file_path, sep="\t")

        # Convert numeric columns to appropriate types
        numeric_columns = ["Sample", "t", "index", "value"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Deme"] = df["index"]

        logger.info("Successfully loaded trajectory file: %s", file_path)

        return df

    except FileNotFoundError:
        logger.error("Trajectory file not found: %s", file_path)
        return pd.DataFrame()
    except Exception as e:
        logger.error("Error loading trajectory file %s: %s", file_path, e)
        return pd.DataFrame()


def load_case_counts_file(file_path):
    """
    Load a case counts file as a pandas DataFrame.

    Args:
        file_path (str): Path to the case counts file

    Returns:
        pd.DataFrame: DataFrame with columns case_counts, t_case_counts_fromsimstart, t_case_counts_frommostrecentsample, index
    """
    try:
        # Read the case counts file as comma-separated values
        df = pd.read_csv(file_path)

        # Convert numeric columns to appropriate types
        numeric_columns = [
            "case_counts",
            "t_case_counts_fromsimstart",
            "t_case_counts_frommostrecentsample",
            "index",
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Deme"] = df["index"]

        logger.info("Successfully loaded case counts file: %s", file_path)
        logger.debug("Shape: %s", df.shape)
        logger.debug("Columns: %s", list(df.columns))

        return df

    except FileNotFoundError:
        logger.error("Case counts file not found: %s", file_path)
        return pd.DataFrame()
    except Exception as e:
        logger.error("Error loading case counts file %s: %s", file_path, e)
        return pd.DataFrame()


def load_wastewater_file(file_path):
    """
    Load a wastewater file as a pandas DataFrame.

    Args:
        file_path (str): Path to the wastewater file

    Returns:
        pd.DataFrame: DataFrame with columns wastewater, t_wastewater_fromsimstart, t_wastewater_frommostrecentsample, index
    """
    try:
        # Read the wastewater file as comma-separated values
        df = pd.read_csv(file_path)

        # Convert numeric columns to appropriate types
        numeric_columns = [
            "wastewater",
            "t_wastewater_fromsimstart",
            "t_wastewater_frommostrecentsample",
            "index",
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Deme"] = df["index"]

        logger.info("Successfully loaded wastewater file: %s", file_path)
        logger.debug("Shape: %s", df.shape)
        logger.debug("Columns: %s", list(df.columns))

        return df

    except FileNotFoundError:
        logger.error("Case counts file not found: %s", file_path)
        return pd.DataFrame()
    except Exception as e:
        logger.error("Error loading case counts file %s: %s", file_path, e)
        return pd.DataFrame()


def load_seroprevalence_file(file_path):
    """
    Load a seroprevalence file as a pandas DataFrame.

    Args:
        file_path (str): Path to the seroprevalence file

    Returns:
        pd.DataFrame: DataFrame with columns seroprevalence, seroprevalence_numpeopletested, seroprevalence_numpeoplewithantibodies, t_seroprevalence_fromsimstart, t_seroprevalence_frommostrecentsample, index
    """
    try:
        # Read the seroprevalence file as comma-separated values
        df = pd.read_csv(file_path)

        # Convert numeric columns to appropriate types
        numeric_columns = [
            "seroprevalence",
            "seroprevalence_numpeopletested",
            "seroprevalence_numpeoplewithantibodies",
            "t_seroprevalence_fromsimstart",
            "t_seroprevalence_frommostrecentsample",
            "index",
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Deme"] = df["index"]

        logger.info("Successfully loaded seroprevalence file: %s", file_path)
        logger.debug("Shape: %s", df.shape)
        logger.debug("Columns: %s", list(df.columns))

        return df

    except FileNotFoundError:
        logger.error("Seroprevalence file not found: %s", file_path)
        return pd.DataFrame()
    except Exception as e:
        logger.error("Error loading seroprevalence file %s: %s", file_path, e)
        return pd.DataFrame()


def get_timesincestart(
    df,
    index_column,
    relative_rateshifts=True,
    rateshifts=None,
    mostrecent_sample_t=None,
    subtract_one=True,
):
    """
    Get the timesincestart values for the dataframe.
    """
    if relative_rateshifts:
        max_index = df[index_column].max()
        min_index = df[index_column].min()
        timesinceroot = ((max_index - df[index_column]) / (max_index - min_index)) * df[
            "Tree.height"
        ]
        timesincestart = timesinceroot + mostrecent_sample_t - df["Tree.height"]
        return timesincestart
    else:
        max_rateshift = rateshifts.max()
        indices = (
            df[index_column].values - 1 if subtract_one else df[index_column].values
        )
        indices = indices.astype(int)
        timesincestart = max_rateshift - rateshifts[indices]
        return timesincestart


def transform_skyline_columns(
    df, relative_rateshifts=True, rateshifts=None, mostrecent_sample_t=None
):
    """
    Transform wide-format SkylineNe columns to long-format.

    Args:
        df (pd.DataFrame): Input dataframe with SkylineNe columns

    Returns:
        pd.DataFrame: Long-format dataframe with columns:
            - logNe: The original values
            - Deme: The deme name (e.g., 'Anseriformes')
            - SkylineNe_index: The index number from the column name
            - timesinceroot: Constant value 1 (as specified)
    """
    # Find all columns that start with 'SkylineNe.'
    skyline_cols = [
        col for col in df.columns if col.startswith(("SkylineNe.", "SkylinePrev."))
    ]
    # Create a list to store the melted dataframes
    melted_dfs = []

    for col in skyline_cols:
        # Split column name into parts
        parts = col.split(".")
        if len(parts) >= 3:  # Ensure we have at least 3 parts
            # parts[1] has form Deme<number>, only extract the number part
            # Here demes start at 1 so we need to subtract 1 from the number
            deme = int(parts[1].replace("Deme", "")) - 1
            idx = parts[2]

            # Create a temporary dataframe for this column
            temp_df = df[["Sample", "Tree.height", col]].copy()
            temp_df = temp_df.rename(columns={col: "logNe"})
            temp_df["Deme"] = deme
            temp_df["SkylineNe_index"] = int(idx) if idx.isdigit() else idx
            temp_df["timesincestart"] = 1  # As specified

            melted_dfs.append(temp_df)

    # Combine all dataframes if we found any matching columns
    if melted_dfs:
        df = pd.concat(melted_dfs, ignore_index=True)
        # this is only valid if no measurements younger than the most recent sample were used
        df["timesincestart"] = get_timesincestart(
            df, "SkylineNe_index", relative_rateshifts, rateshifts, mostrecent_sample_t
        )
        return df
    else:
        return pd.DataFrame(
            columns=[
                "logNe",
                "Deme",
                "SkylineNe_index",
                "timesincestart",
            ]
        )


def load_params_csv(file_path: str) -> pd.DataFrame:
    """Load the parameters CSV (parameter, deme, value) as a DataFrame.
    Returns an empty DataFrame if not found or invalid.
    """
    if not file_path:
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        required_cols = {"parameter", "deme", "value"}
        if not required_cols.issubset(set(df.columns)):
            logger.error(
                "Params CSV missing required columns %s: %s", required_cols, file_path
            )
            return pd.DataFrame()
        df["Deme"] = df["deme"]
        return df
    except FileNotFoundError:
        logger.error("Parameters CSV not found: %s", file_path)
        return pd.DataFrame()
    except Exception as e:
        logger.error("Error loading parameters CSV %s: %s", file_path, e)
        return pd.DataFrame()


def compute_expected_ne_from_trajectory(
    trajectory_data: pd.DataFrame, params_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute expected Ne per deme over time using Volz (2012): Ne = I / (2 * beta * S).

    Handles time-varying betas. Extracts beta values from params_df where parameter names
    follow the pattern 'beta_<time>' (e.g., beta_0, beta_0.055555, etc.). Each beta is used
    from its time until the next beta time.

    Uses within-deme beta (i->i) from params_df rows where parameter starts with 'beta_'
    and deme like 'i->i'.

    Returns a DataFrame with columns: index (deme id), t (years), Ne_expected.
    """
    if trajectory_data is None or trajectory_data.empty:
        return pd.DataFrame()
    if params_df is None or params_df.empty:
        return pd.DataFrame()

    # Extract all beta parameters that were scaled by the population size
    beta_rows = params_df[
        params_df["parameter"].str.startswith("beta", na=False)
        & params_df["parameter"].str.endswith("Nscaled", na=False)
    ].copy()

    if beta_rows.empty:
        logger.warning(
            "No beta entries found in params CSV; cannot compute expected Ne."
        )
        return pd.DataFrame()

    # Parse time from parameter names and extract within-deme betas
    # Structure: parameter like "beta_0" or "beta_0.055555", deme like "0->0" or "1->1"
    time_beta_map = {}  # {time: {deme_idx: beta_value}}

    for _, row in beta_rows.iterrows():
        param_name = str(row["parameter"])
        deme_str = str(row["Deme"])

        # Extract time from parameter name (e.g., "beta_0" -> 0, "beta_0.055555" -> 0.055555)
        # Extract time string between "beta_" and either end or first "_" (for "_Nscaled" or other suffix)
        time_part = param_name[5:]  # Remove "beta_" prefix
        # Remove trailing suffixes like "_Nscaled" if present
        if "_" in time_part:
            time_str = time_part.split("_")[0]
        else:
            time_str = time_part
        time_val = float(time_str)

        left, right = deme_str.split("->")
        i = int(left.strip())
        j = int(right.strip())

        if i == j:
            if time_val not in time_beta_map:
                time_beta_map[time_val] = {}
            time_beta_map[time_val][i] = float(row["value"])

    if not time_beta_map:
        logger.warning(
            "No within-deme beta entries (i->i) found in params CSV; cannot compute expected Ne."
        )
        return pd.DataFrame()

    # Sort time points to create intervals
    time_points = sorted(time_beta_map.keys())

    # Prepare S and I time series per deme
    df_S = trajectory_data[trajectory_data["population"] == "S"].copy()
    df_I = trajectory_data[trajectory_data["population"] == "I"].copy()
    if df_S.empty or df_I.empty:
        logger.warning("Missing S or I in trajectory data; cannot compute expected Ne.")
        return pd.DataFrame()

    # Merge S and I by deme index and time t
    df_S = df_S.rename(columns={"value": "S"})[["index", "t", "S"]]
    df_I = df_I.rename(columns={"value": "I"})[["index", "t", "I"]]
    merged = pd.merge(df_S, df_I, on=["index", "t"], how="inner")

    # Function to find the appropriate beta for a given time and deme
    def get_beta_for_time(t_val, deme_idx):
        """Find which beta interval a time point belongs to."""
        # Find the largest time point <= t_val
        beta_time = None
        for tp in reversed(time_points):
            if tp <= t_val:
                beta_time = tp
                break

        if beta_time is None:
            # Time is before first beta, use first beta
            beta_time = time_points[0]

        deme_betas = time_beta_map.get(beta_time, {})
        return deme_betas.get(deme_idx, np.nan)

    # Compute Ne for each row using time-appropriate beta
    def row_ne(row):
        idx = int(row["index"])
        t_val = float(row["t"])
        beta = get_beta_for_time(t_val, idx)

        if beta is None or np.isnan(beta) or row["S"] <= 0:
            return np.nan
        # Volz (2012) formula, expects beta to be scaled by N_i
        return float(row["I"]) / (2.0 * beta * float(row["S"]))

    merged["Ne_expected"] = merged.apply(row_ne, axis=1)
    merged = merged.dropna(subset=["Ne_expected"])
    merged["Deme"] = merged["index"]
    return merged[["index", "t", "Ne_expected", "Deme"]]


def calculate_hpd(data, alpha=0.05):
    """
    Calculate highest posterior density (HPD) interval for given alpha.

    Args:
        data (np.array): 1D array of MCMC samples
        alpha (float): Desired exclusion probability (default 0.05 for 95% HPD)

    Returns:
        tuple: (lower_bound, upper_bound) of HPD interval
    """
    data = np.sort(data)
    n = len(data)
    m = int((1 - alpha) * n)  # Number of points in the interval

    if m < 1:
        return (data[0], data[-1])

    intervals = data[m:] - data[: n - m]
    min_idx = np.argmin(intervals)
    hpd_min = data[min_idx]
    hpd_max = data[min_idx + m]

    median = np.median(data)
    return (hpd_min, hpd_max, median)


def linear_interpolation(x: np.array, y: np.array, x_new: np.array) -> np.array:
    return np.interp(x=x_new, xp=x, fp=y)


def cubic_normal_spline_interpolation(
    x: np.array, y: np.array, x_new: np.array
) -> np.array:
    spline = scp.interpolate.make_interp_spline(x, y, k=3, bc_type="natural")
    return spline(x_new)


def interpolate_skyline(skyline_long, time_points, interpolation_method="linear"):
    """
    Interpolate Ne values to a predefined grid of time points for each sample and deme.

    Args:
        skyline_long (pd.DataFrame): Long-format dataframe with columns:
            - Sample: Sample identifier
            - Deme: Deme identifier
            - timesinceroot: Time points (in years)
            - logNe: Ne values
        time_points (np.array): Array of time points to interpolate to (in years)
        interpolation_method (str): Interpolation method ('linear' or 'cubic_normal_spline')

    Returns:
        hpd_intervals: DataFrame with HPD intervals for each deme and time point
            - Deme: Deme identifier
            - timesincestart: Time point (in years)
            - logNe: Median Ne value
            - logNe_hpd_lower: Lower bound of HPD interval
            - logNe_hpd_upper: Upper bound of HPD interval
    """
    # Check if the input dataframe is empty
    if skyline_long.empty:
        return pd.DataFrame()

    results = []

    # Group by sample and deme
    grouped = skyline_long.groupby(["Sample", "Deme"])

    for (sample, deme), group in grouped:
        # Sort by time for proper interpolation
        group = group.sort_values("timesincestart")
        times = group["timesincestart"].values
        logNes = group["logNe"].values

        # Create interpolation function
        if len(times) > 1:
            # Linear interpolation
            if interpolation_method == "linear":
                logNes_grid = linear_interpolation(x=times, y=logNes, x_new=time_points)
            elif interpolation_method == "cubic_normal_spline":
                logNes_grid = cubic_normal_spline_interpolation(
                    x=times, y=logNes, x_new=time_points
                )
            else:
                raise ValueError(
                    f"Invalid interpolation method: {interpolation_method}"
                )
            # Create result dataframe
            result = pd.DataFrame(
                {
                    "Sample": sample,
                    "Deme": deme,
                    "timesincestart": time_points,
                    "Ne_interpolated": logNes_grid,
                }
            )
            results.append(result)

    if not results:
        return pd.DataFrame()

    interpolated_df = pd.concat(results, ignore_index=True)

    # Calculate HPD intervals for each deme and time point
    hpd_intervals = []
    for (deme, time), group in interpolated_df.groupby(["Deme", "timesincestart"]):
        logNe_hpd_lower, logNe_hpd_upper, logNe_median = calculate_hpd(
            group["Ne_interpolated"].values
        )
        hpd_intervals.append(
            {
                "Deme": deme,
                "timesincestart": time,
                "logNe": logNe_median,
                "logNe_hpd_lower": logNe_hpd_lower,
                "logNe_hpd_upper": logNe_hpd_upper,
            }
        )

    hpd_df = pd.DataFrame(hpd_intervals) if hpd_intervals else pd.DataFrame()

    return hpd_df


def process_nedynamics_log(file_path, deme_index, tree_height, burnin, rateshifts):
    """
    Process NeDynamics log file and convert to HPD format matching interpolate_skyline output.

    Args:
        file_path (str): Path to NeDynamics log file
        deme_index (int): Deme index (0 for deme1, 1 for deme2)
        tree_height (pd.DataFrame): DataFrame with columns:
            - Sample: Sample identifier
            - Tree.height: Tree height (in years)
        burnin (float): Burnin percentage
        rateshifts (np.array): Rate shift times (in years)

    Returns:
        pd.DataFrame: DataFrame with columns matching interpolate_skyline output:
            - Deme: Deme identifier
            - timesincestart: Time points (in years)
            - logNe: Mean Ne value
            - logNe_hpd_lower: Lower bound of HPD interval
            - logNe_hpd_upper: Upper bound of HPD interval
            - logPrevalence: Mean prevalence value (if available)
            - logPrevalence_hpd_lower: Lower bound of prevalence HPD interval (if available)
            - logPrevalence_hpd_upper: Upper bound of prevalence HPD interval (if available)
    """
    if not file_path:
        return pd.DataFrame()

    # Read the log file
    nedynamics_df = read_beast_log(file_path)

    if nedynamics_df.empty:
        return pd.DataFrame()

    # Convert to long format
    nedynamics_long = create_Ne_dynamics_long(nedynamics_df)

    nedynamics_long = nedynamics_long[
        nedynamics_long["Sample"] > nedynamics_long["Sample"].max() * burnin
    ]

    nedynamics_long = nedynamics_long.merge(tree_height, on="Sample", how="left")

    if nedynamics_long.empty:
        return pd.DataFrame()

    nedynamics_long["timesincestart"] = get_timesincestart(
        nedynamics_long,
        "gridpoint",
        relative_rateshifts=False,
        rateshifts=rateshifts,
        mostrecent_sample_t=None,
        subtract_one=False,
    )

    nedynamics_long["Deme"] = int(deme_index)

    n_gridpoints = nedynamics_long["gridpoint"].nunique()

    ne_hpd = get_hpd_intervals(nedynamics_long, n_gridpoints, "logNe")
    prev_hpd = get_hpd_intervals(nedynamics_long, n_gridpoints, "logPrevalence")

    hpd_df = pd.merge(ne_hpd, prev_hpd, on=["Deme", "timesincestart"], how="left")

    return hpd_df


def process_cumulative_incidence_log(file_path, deme_index, tree_height, rateshifts):
    """
    Process cumulative incidence log file and convert to long format with forward time.

    Args:
        file_path (str): Path to cumulative incidence log file
        deme_index (int): Deme index (0 for deme1, 1 for deme2)
        tree_height (pd.DataFrame): DataFrame with columns:
            - Sample: Sample identifier
            - Tree.height: Tree height (in years)
        rateshifts (np.array): Rate shift times (in years)
    Returns:
        pd.DataFrame: DataFrame with columns:
            - Deme: Deme identifier
            - timesincestart: Time from simulation start (in years)
            - cumulativeIncidence: Median cumulative incidence value
            - cumulativeIncidence_hpd_lower: Lower bound of HPD interval
            - cumulativeIncidence_hpd_upper: Upper bound of HPD interval
    """
    if not file_path:
        return pd.DataFrame()

    # Check if file exists and is not empty
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return pd.DataFrame()

    # Read the log file
    cuminc_df = read_beast_log(file_path)

    if cuminc_df.empty:
        return pd.DataFrame()

    # Convert to long format
    cuminc_long = create_cumulative_incidence_long(cuminc_df)

    if cuminc_long.empty:
        return pd.DataFrame()

    # Get Tree.height (root age relative to most recent sample time)
    cuminc_long = cuminc_long.merge(tree_height, on="Sample", how="left")

    # Convert gridpoints to forward time
    cuminc_long["timesincestart"] = get_timesincestart(
        cuminc_long,
        "gridpoint",
        relative_rateshifts=False,
        rateshifts=rateshifts,
        mostrecent_sample_t=None,
        subtract_one=False,
    )

    cuminc_long["Deme"] = int(deme_index)

    n_gridpoints = cuminc_long["gridpoint"].nunique()

    cuminc_hpd = get_hpd_intervals(cuminc_long, n_gridpoints, "cumulativeIncidence")

    return cuminc_hpd


def get_hpd_intervals(df, n_gridpoints, column_name):

    # Create a fixed grid from min to max time since root
    min_time = df["timesincestart"].min()
    max_time = df["timesincestart"].max()
    time_grid = np.linspace(min_time, max_time, n_gridpoints)

    # Assign each sample's time point to the nearest grid point
    def assign_to_grid(time_val):
        """Find the closest grid point for a given time value."""
        idx = np.argmin(np.abs(time_grid - time_val))
        return time_grid[idx]

    df["timesinceroot_grid"] = df["timesincestart"].apply(assign_to_grid)

    # Calculate HPD intervals for each deme and grid time point
    hpd_intervals = []
    for (deme, grid_time), group in df.groupby(["Deme", "timesinceroot_grid"]):
        hpd_lower, hpd_upper, median = calculate_hpd(group[column_name].values)
        hpd_intervals.append(
            {
                "Deme": deme,
                "timesincestart": grid_time,
                column_name: median,
                f"{column_name}_hpd_lower": hpd_lower,
                f"{column_name}_hpd_upper": hpd_upper,
            }
        )

    hpd_df = pd.DataFrame(hpd_intervals) if hpd_intervals else pd.DataFrame()

    return hpd_df


def get_datastream_true_values_from_params(params_df):
    """
    Build a dict of BEAST/MASCOT parameter name -> true value from params_df
    (e.g. from 1_2_simulation_parameters.csv with columns parameter, deme, value).
    Deme in CSV: 0, 1 correspond to Deme1, Deme2 in BEAST names.
    """
    if params_df is None or params_df.empty:
        return {}
    true_values = {}
    # Global (no deme): ds_cc_dispersion, ds_ww_sigma
    for param_name, beast_name in [
        ("ds_cc_dispersion", "caseCounts.dispersion:SimDataset"),
        ("ds_ww_sigma", "wastewater.sigma:SimDataset"),
    ]:
        row = params_df[(params_df["parameter"] == param_name)]
        if not row.empty:
            true_values[beast_name] = float(row["value"].iloc[0])
    # Per-deme: ds_cc_scaling, ds_ww_scaling -> Deme1 (deme 0), Deme2 (deme 1)
    # ds_sp_scaling is fixed to 1.0 and not estimated; excluded from true-value lookup.
    for param_name, beast_suffix in [
        ("ds_cc_scaling", "caseCounts.scaling.{}:SimDataset"),
        ("ds_ww_scaling", "wastewater.scaling.{}:SimDataset"),
    ]:
        for deme_idx, deme_label in enumerate(["Deme1", "Deme2"]):
            beast_name = beast_suffix.format(deme_label)
            row = params_df[
                (params_df["parameter"] == param_name)
                & (params_df["deme"].astype(str) == str(deme_idx))
            ]
            if not row.empty:
                true_values[beast_name] = float(row["value"].iloc[0])
    return true_values


def build_datastream_params_hpd_dataframe(df, params_df=None):
    """
    One row per listed parameter: HPD bounds, median, and inHPD vs params_df truth when present.

    Rows are still emitted for missing or empty posterior columns (inHPD None / NaN as before).
    """
    true_values = get_datastream_true_values_from_params(params_df)
    hpd_results = []
    for parameter in DATASTREAM_POSTERIOR_PARAMETERS:
        if parameter in df.columns:
            param_values = df[parameter].dropna().values
            if len(param_values) > 0:
                hpd_lower, hpd_upper, median = calculate_hpd(param_values, alpha=0.05)
                true_value = true_values.get(parameter)
                in_hpd = (
                    1
                    if (true_value is not None and hpd_lower <= true_value <= hpd_upper)
                    else (0 if true_value is not None else None)
                )
                hpd_results.append(
                    {
                        "Parameter": parameter,
                        "inHPD": in_hpd,
                        "true_value": true_value,
                        "hpd_lower": hpd_lower,
                        "hpd_upper": hpd_upper,
                        "median": median,
                    }
                )
            else:
                hpd_results.append({"Parameter": parameter, "inHPD": None})
        else:
            hpd_results.append({"Parameter": parameter, "inHPD": None})
    return pd.DataFrame(hpd_results)


def save_datastream_params_hpd_validation_csv(df, params_df, out_prefix):
    """
    Write ``{out_prefix}_datastreams_hpd_validation_params.csv``.

    ``out_prefix`` is the run/stem prefix (same as ``output_file`` with
    ``_datastream_paramsestimates.png`` removed).
    """
    print("HHHHAAAALLLO)")
    hpd_table = build_datastream_params_hpd_dataframe(df, params_df=params_df)
    print(hpd_table.columns, hpd_table.empty)
    hpd_table["Simulation"] = out_prefix
    table_output = f"{out_prefix}_datastreams_hpd_validation_params.csv"
    print(hpd_table.head())
    hpd_table.to_csv(table_output, index=False)
    logger.info("HPD validation table saved to %s", table_output)
    print("BLAAAAAAHHHHHHAHAHAHAH")


def plot_datastream_params(df, output_file=None, params_df=None):
    """
    Plot the parameters of the datastream MASCOT model.
    True values for vertical lines are taken from params_df when provided.

    For HPD validation CSV without figures, use
    ``save_datastream_params_hpd_validation_csv``.
    """
    if df.empty:
        logger.warning("No data to plot.")
        return

    parameters = DATASTREAM_POSTERIOR_PARAMETERS

    true_values = get_datastream_true_values_from_params(params_df)

    fig, ax = plt.subplots(4, 2, figsize=(5, 8))
    ax = ax.flatten()

    for i, parameter in enumerate(parameters):
        if parameter in df.columns:
            param_values = df[parameter].dropna().values
            if len(param_values) > 0:

                x_min, x_max = param_values.min(), param_values.max()
                x_grid = np.linspace(x_min, x_max, 200)
                kde = gaussian_kde(param_values)
                y_grid = kde(x_grid)

                ax[i].fill_between(x_grid, y_grid, color=COLORS[3], alpha=0.35)
                ax[i].plot(x_grid, y_grid, color=COLORS[3], lw=3)
                set_axis_fontsizes(
                    ax[i], _fontsizes_list(), xlabel=parameter, ylabel="Density"
                )
                if parameter in true_values:
                    ax[i].axvline(
                        true_values[parameter],
                        color=COLORS[0],
                        linestyle="solid",
                        lw=2,
                    )
            else:
                ax[i].set_visible(False)
        else:
            ax[i].set_visible(False)

    plt.tight_layout()

    if output_file:
        save_figure_png_and_pdf(output_file)
    else:
        plt.show()


def _prepare_migration_rates_context(
    df_original,
    df_datastream,
    params_df=None,
    deme_switches_df=None,
    starting_deme=None,
):
    """
    Shared setup: BEAST parameter names, expected values, transition counts, and
    HPD records (with None placeholders aligned to plot positions).
    """
    pair_to_param = {
        (0, 1): "f_migrationRatesSkyline.I0_to_I1",
        (1, 0): "f_migrationRatesSkyline.I1_to_I0",
    }

    if starting_deme is None:
        index_pairs = [(0, 1), (1, 0)]
    else:
        other_deme = 1 - int(starting_deme)
        index_pairs = [
            (int(starting_deme), other_deme),
            (other_deme, int(starting_deme)),
        ]

    params = [pair_to_param[(i_from, i_to)] for (i_from, i_to) in index_pairs]

    def migration_label(from_idx: int, to_idx: int) -> str:
        if starting_deme is None:
            return f"Deme {from_idx}->{to_idx}"
        from_label = get_deme_display_label(from_idx, starting_deme)
        to_label = get_deme_display_label(to_idx, starting_deme)
        return f"{from_label} -> {to_label}"

    param_labels = [
        migration_label(*index_pairs[0]),
        migration_label(*index_pairs[1]),
    ]

    expected_keys = {
        "f_migrationRatesSkyline.I0_to_I1": ("beta_0", "0->1"),
        "f_migrationRatesSkyline.I1_to_I0": ("beta_0", "1->0"),
    }

    expected_values = {}
    if params_df is not None and not params_df.empty:
        for param, (param_name, deme_direction) in expected_keys.items():
            row = params_df[
                (params_df["parameter"] == param_name)
                & (params_df["deme"] == deme_direction)
            ]
            if not row.empty:
                expected_values[param] = row["value"].iloc[0]

    transition_counts = {}
    if deme_switches_df is not None and not deme_switches_df.empty:
        switches_df = deme_switches_df[deme_switches_df["switch"] == "yes"]
        unique_pairs = switches_df[["Deme_parent", "Deme_child"]].drop_duplicates()
        for _, row in unique_pairs.iterrows():
            parent_deme = int(row["Deme_parent"])
            child_deme = int(row["Deme_child"])
            param_name = f"f_migrationRatesSkyline.I{parent_deme}_to_I{child_deme}"
            count = switches_df[
                (switches_df["Deme_parent"] == parent_deme)
                & (switches_df["Deme_child"] == child_deme)
            ]["count"].values[0]
            transition_counts[param_name] = count

    positions = [0.75, 1.25, 1.75, 2.25]
    box_colors = []
    hpd_results = []

    for param in params:
        original_values = df_original[param].dropna().values
        box_colors.append(COLORS[4])

        if len(original_values) > 0:
            hpd_lower, hpd_upper, median = calculate_hpd(original_values, alpha=0.05)
            hpd_result = {
                "Parameter": param,
                "Model": "MASCOT",
                "true_value": expected_values.get(param, None),
                "hpd_lower": hpd_lower,
                "hpd_upper": hpd_upper,
                "median": median,
            }
            if param in transition_counts:
                hpd_result["n_deme_transitions"] = transition_counts[param]
            hpd_results.append(hpd_result)
        else:
            hpd_results.append(None)

        datastream_values = df_datastream[param].dropna().values
        box_colors.append(COLORS[3])

        if len(datastream_values) > 0:
            hpd_lower, hpd_upper, median = calculate_hpd(datastream_values, alpha=0.05)
            hpd_result = {
                "Parameter": param,
                "Model": "MASCOT-DS",
                "true_value": expected_values.get(param, None),
                "hpd_lower": hpd_lower,
                "hpd_upper": hpd_upper,
                "median": median,
            }
            if param in transition_counts:
                hpd_result["n_deme_transitions"] = transition_counts[param]
            hpd_results.append(hpd_result)
        else:
            hpd_results.append(None)

    return {
        "params": params,
        "param_labels": param_labels,
        "expected_values": expected_values,
        "hpd_results": hpd_results,
        "positions": positions,
        "box_colors": box_colors,
    }


def build_migration_rates_hpd_dataframe(
    df_original,
    df_datastream,
    params_df=None,
    deme_switches_df=None,
    starting_deme=None,
):
    """
    Table of migration-rate HPDs for MASCOT vs MASCOT-DS (one row per model x parameter).
    """
    if df_original.empty or df_datastream.empty:
        return pd.DataFrame()

    ctx = _prepare_migration_rates_context(
        df_original,
        df_datastream,
        params_df=params_df,
        deme_switches_df=deme_switches_df,
        starting_deme=starting_deme,
    )
    rows = [r for r in ctx["hpd_results"] if r is not None]
    if not rows:
        return pd.DataFrame()
    hpd_table = pd.DataFrame(rows)
    hpd_table["inHPD"] = np.where(
        (hpd_table["true_value"] >= hpd_table["hpd_lower"])
        & (hpd_table["true_value"] <= hpd_table["hpd_upper"]),
        1,
        0,
    )
    return hpd_table


def save_migration_rates_hpd_csv(
    df_original,
    df_datastream,
    out_prefix,
    params_df=None,
    deme_switches_df=None,
    starting_deme=None,
):
    """
    Write ``{out_prefix}_hpd_validation_migration_rates.csv``.

    ``out_prefix`` matches the stem used with ``_migration_rates.png`` removed.
    """
    if df_original.empty or df_datastream.empty:
        logger.warning("No migration-rate data to save.")
        return
    hpd_table = build_migration_rates_hpd_dataframe(
        df_original,
        df_datastream,
        params_df=params_df,
        deme_switches_df=deme_switches_df,
        starting_deme=starting_deme,
    )
    if hpd_table.empty:
        logger.warning("No migration-rate HPD rows to save.")
        return
    hpd_table["Simulation"] = out_prefix
    table_output = f"{out_prefix}_hpd_validation_migration_rates.csv"
    hpd_table.to_csv(table_output, index=False)
    logger.info("HPD validation table saved to %s", table_output)


def plot_migration_rates(
    df_original,
    df_datastream,
    params_df=None,
    deme_switches_df=None,
    output_file=None,
    starting_deme=None,
):
    """
    Plot the migration rates of the datastream MASCOT model.
    Expected values from params_df when provided.

    For HPD CSV without figures, use ``save_migration_rates_hpd_csv``.
    """
    if df_original.empty or df_datastream.empty:
        logger.warning("No data to plot.")
        return

    ctx = _prepare_migration_rates_context(
        df_original,
        df_datastream,
        params_df=params_df,
        deme_switches_df=deme_switches_df,
        starting_deme=starting_deme,
    )
    params = ctx["params"]
    param_labels = ctx["param_labels"]
    expected_values = ctx["expected_values"]
    hpd_results = ctx["hpd_results"]
    positions = ctx["positions"]
    box_colors = ctx["box_colors"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Plot dots and whiskers instead of boxplots
    for pos, color, hpd_data in zip(positions, box_colors, hpd_results):
        if hpd_data is None:
            continue

        median = hpd_data["median"]
        hpd_lower = hpd_data["hpd_lower"]
        hpd_upper = hpd_data["hpd_upper"]

        # Plot whisker (vertical line from hpd_lower to hpd_upper)
        ax.plot(
            [pos, pos],
            [hpd_lower, hpd_upper],
            color=color,
            linewidth=1,
            alpha=0.8,
        )

        # Plot horizontal caps at whisker ends
        cap_width = 0.1
        ax.plot(
            [pos - cap_width, pos + cap_width],
            [hpd_lower, hpd_lower],
            color=color,
            linewidth=1,
            alpha=0.8,
        )
        ax.plot(
            [pos - cap_width, pos + cap_width],
            [hpd_upper, hpd_upper],
            color=color,
            linewidth=1,
            alpha=0.8,
        )

        # Plot dot at median
        ax.scatter(
            pos,
            median,
            color=color,
            s=150,
            zorder=3,
            edgecolors="white",
            linewidth=1,
        )

    # Add horizontal lines for expected values
    if expected_values:
        for i, param in enumerate(params):
            if param in expected_values:
                expected_val = expected_values[param]
                # Draw line across both boxes for this parameter
                # Boxes are at positions [0.75, 1.25] for first param, [1.75, 2.25] for second
                if i == 0:
                    x_start, x_end = 0.5, 1.5
                else:
                    x_start, x_end = 1.5, 2.5
                ax.plot(
                    [x_start, x_end],
                    [expected_val, expected_val],
                    color=COLORS[0],
                    linestyle="solid",
                    linewidth=1,
                    label="Expected" if i == 0 else "",
                    zorder=10,
                )

    # Create legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS[4],
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1,
            label="MASCOT",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS[3],
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1,
            label="MASCOT-DS",
        ),
    ]
    if expected_values:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=COLORS[0],
                linestyle="solid",
                linewidth=1,
                label="Expected",
            )
        )
    ax.legend(handles=legend_elements, fontsize=fontsizes["legend"])

    # Set x-axis labels
    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels(param_labels)

    # Set axis labels and font sizes
    set_axis_fontsizes(
        ax, _fontsizes_list(), xlabel="Migration rate", ylabel="Forward migration rate"
    )

    plt.tight_layout()

    if output_file:
        save_figure_png_and_pdf(output_file)
    else:
        plt.show()


def transform_value(value, show_logscale, is_log_space=False):
    """
    Transform value between log and linear space based on show_logscale flag.

    Args:
        value: Value to transform (can be scalar or array-like)
        show_logscale: If True, keep/return in log space; if False, convert to linear
        is_log_space: If True, value is already in log space; if False, value is in linear space

    Returns:
        Transformed value
    """
    if show_logscale:
        if is_log_space:
            return value
        else:
            return np.log(value)
    else:
        if is_log_space:
            return np.exp(value)
        else:
            return value


def plot_hpd_interval(
    ax,
    times,
    hpd_lower,
    hpd_upper,
    median_values,
    color,
    label,
    show_logscale,
    is_log_space=True,
    time_factor=1.0,
    alpha=0.2,
    linewidth=2,
    zorder_interval=3,
    zorder_line=4,
):
    """
    Plot HPD interval with fill_between and median line.

    Args:
        ax: Matplotlib axis to plot on
        times: Time values (will be multiplied by time_factor)
        hpd_lower: Lower bound of HPD interval (in log or linear space)
        hpd_upper: Upper bound of HPD interval (in log or linear space)
        median_values: Median/mean values to plot as line (in log or linear space)
        color: Color for the plot
        label: Label for the line
        show_logscale: If True, values are in log space; if False, convert to linear
        is_log_space: If True, input values are in log space
        time_factor: Factor to multiply times by (for unit conversion)
        alpha: Transparency for fill_between
        linewidth: Width of median line
        zorder_interval: Z-order for HPD interval
        zorder_line: Z-order for median line
    """
    times_plot = times * time_factor

    # Transform HPD bounds
    hpd_lower_transformed = transform_value(hpd_lower, show_logscale, is_log_space)
    hpd_upper_transformed = transform_value(hpd_upper, show_logscale, is_log_space)

    # Plot HPD interval
    ax.fill_between(
        times_plot,
        hpd_lower_transformed,
        hpd_upper_transformed,
        color=color,
        alpha=alpha,
        zorder=zorder_interval,
    )

    # Plot median line
    median_transformed = transform_value(median_values, show_logscale, is_log_space)
    ax.plot(
        times_plot,
        median_transformed,
        color=color,
        linewidth=linewidth,
        label=label,
        alpha=0.6,
        zorder=zorder_line,
    )


def plot_hpd_validation_scatter(
    ax,
    validation_data,
    deme,
    time_factor,
    in_hpd_column="inHPD",
    in_hpd_color="#e69d00",
    offset_fraction=0.1,
    marker_size=10,
    scatter_y_override=None,
):
    """
    Plot scatter points above the plot to indicate HPD validation results.

    Args:
        ax: Matplotlib axis to plot on
        validation_data: DataFrame with validation results (must have columns: Deme, timesincestart, and in_hpd_column)
        deme: Deme ID to filter by
        time_factor: Factor to multiply times by (for unit conversion)
        in_hpd_column: Column name for inHPD indicator (default "inHPD", can be "inHPDP1")
        in_hpd_color: Color for points that are in HPD (default orange)
        offset_fraction: Fraction of y-range to offset scatter above max value (default 0.1)
        marker_size: Size of scatter markers (default 30)

    Returns:
        None (modifies ax in place)
    """
    if validation_data is None or validation_data.empty:
        return

    deme_validation = validation_data[validation_data["Deme"] == deme].sort_values(
        "timesincestart"
    )

    if deme_validation.empty:
        return

    # Calculate scatter position based on primary axis
    ylim = ax.get_ylim()
    max_val = ylim[1]
    y_range = ylim[1] - ylim[0]
    if scatter_y_override is not None:
        scatter_y = scatter_y_override
    else:
        scatter_y = max_val + offset_fraction * y_range

    # Get data
    times_plot = deme_validation["timesincestart"] * time_factor
    in_hpd = deme_validation[in_hpd_column].values

    # Plot points that are in HPD
    in_hpd_mask = in_hpd == 1
    if in_hpd_mask.any():
        ax.scatter(
            times_plot[in_hpd_mask],
            np.full(np.sum(in_hpd_mask), scatter_y),
            color=in_hpd_color,
            marker="s",
            s=marker_size,
            zorder=10,
            edgecolors="none",
        )

    # Plot points that are not in HPD
    not_in_hpd_mask = in_hpd == 0
    if not_in_hpd_mask.any():
        ax.scatter(
            times_plot[not_in_hpd_mask],
            np.full(np.sum(not_in_hpd_mask), scatter_y),
            color="grey",
            marker="s",
            s=marker_size,
            zorder=10,
            edgecolors="none",
        )

    # Calculate and display coverage statistics
    total_points = len(deme_validation)
    in_hpd_count = int(in_hpd.sum())
    coverage = (in_hpd_count / total_points) * 100 if total_points > 0 else 0

    # Add text annotation
    # Interpret x as a relative position (axes fraction) and y as absolute data value
    ax.text(
        ax.get_xlim()[1] - 0.06 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
        scatter_y + 0.02 * y_range,
        # f"{in_hpd_count}/{total_points} = {coverage:.1f}%",
        f"{coverage:.1f}%",
        fontsize=fontsizes["tick_label"],
        verticalalignment="bottom",
        horizontalalignment="right",
        zorder=11,
    )

    # Adjust ylim to ensure scatter is visible on primary axis
    current_ylim = ax.get_ylim()
    if scatter_y > current_ylim[1]:
        ax.set_ylim(current_ylim[0], scatter_y * 1.05)


def setup_twin_axis_for_secondary_data(
    ax,
    times,
    values,
    ylabel,
    color="dimgrey",
    label=None,
    marker="o",
    markersize=2,
    linewidth=1,
    fontsize=None,
    fontsize_tick=None,
):
    """
    Set up a twin axis and plot secondary data (e.g., case counts, seroprevalence).

    Args:
        ax: Main matplotlib axis
        times: Time values to plot
        values: Values to plot on secondary axis
        ylabel: Label for secondary y-axis
        color: Color for the plot (default "dimgrey")
        label: Label for the line (optional)
        marker: Marker style (default "o")
        markersize: Size of markers (default 4)
        linewidth: Width of line (default 2)
        fontsize: Font size for axis label (default from fontsizes["axis_label"])
        fontsize_tick: Font size for tick labels (default from fontsizes["tick_label"])

    Returns:
        ax2: The twin axis object
    """
    if fontsize is None:
        fontsize = fontsizes["axis_label"]
    if fontsize_tick is None:
        fontsize_tick = fontsizes["tick_label"]
    ax2 = ax.twinx()
    ax2.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)

    ax2.plot(
        times,
        values,
        color=color,
        linewidth=linewidth,
        label=label,
        linestyle="solid",
        zorder=2,
        marker=marker,
        markersize=markersize,
    )
    # Tick fontsize only here; y-label is set below with rotation=270 and extra
    # labelpad. Rotation 270 vs. the default 90 flips which way the text extends
    # from the anchor, so the default labelpad often pulls the label over the ticks.
    fontsizes_local = [fontsize, fontsize, fontsize_tick]
    set_axis_fontsizes(ax2, fontsizes_local)
    labelpad = max(6.0, fontsizes_local[1] * 0.8)
    ax2.set_ylabel(
        ylabel,
        fontsize=fontsizes_local[1],
        color="black",
        rotation=270,
        labelpad=labelpad,
        va="center",
    )
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.spines["top"].set_visible(False)

    return ax2


def _plot_prevalence_panel(
    ax,
    deme,
    *,
    hpd_datastream,
    trajectory_data,
    case_counts_data,
    wastewater_data,
    validation_data_datastreams_prevalence,
    time_factor,
    time_label,
    show_logscale=False,
    show_logscale_prevalence_only=False,
    case_counts_P1=False,
    diff_treeroot_start=None,
    starting_deme=None,
    fig=None,
    n_demes=1,
    fontsizes=None,
    fontsize_tick=None,
    show_legend=True,
):
    """
    Draw the prevalence panel (subplot 1) for a single deme on the given axis.

    Used by plot_skyline_ne and make_figure_individualsim. Caller is responsible
    for figure layout and saving.

    If show_logscale_prevalence_only is True, inferred/expected prevalence are
    shown in log space while case counts and wastewater stay on a linear scale.
    When show_logscale is True and show_logscale_prevalence_only is False, all
    series use the same log/linear choice as before.
    """
    if fontsizes is None:
        fontsizes = _fontsizes_list()
    if fontsize_tick is None:
        fontsize_tick = fontsizes[2]

    prev_log = show_logscale or show_logscale_prevalence_only
    aux_log = show_logscale and not show_logscale_prevalence_only

    ax2 = None
    ax3 = None
    if case_counts_data is not None and not case_counts_data.empty:
        deme_case_counts = case_counts_data[
            case_counts_data["Deme"] == deme
        ].sort_values("t_case_counts_fromsimstart")

        if not deme_case_counts.empty:
            casecount_label = "Log Case counts" if aux_log else "Case counts"
            time_plot = deme_case_counts["t_case_counts_fromsimstart"] * time_factor
            case_counts_vals = deme_case_counts["case_counts"].values
            if case_counts_P1:
                case_counts_vals = case_counts_vals + 1
                casecount_label = casecount_label + " + 1"
            case_counts_vals = transform_value(
                case_counts_vals, aux_log, is_log_space=False
            )
            ax2 = ax.twinx()
            ax2.set_zorder(1)
            ax.set_zorder(3)
            ax.patch.set_visible(False)
            if len(time_plot) > 1:
                bar_width = min(
                    np.diff(time_plot).min() * 0.8,
                    (time_plot.max() - time_plot.min()) / len(time_plot) * 0.8,
                )
            else:
                bar_width = (
                    (time_plot.max() - time_plot.min()) * 0.1
                    if time_plot.max() > time_plot.min()
                    else 0.1
                )
            ax2.bar(
                time_plot,
                case_counts_vals,
                width=bar_width,
                color="silver",
                label=casecount_label,
                zorder=1,
            )
            labelpad = max(6.0, fontsizes[1] * 0.8)
            ax2.set_ylabel(
                casecount_label,
                fontsize=fontsizes[1],
                color="black",
                rotation=270,
                labelpad=labelpad,
                va="center",
            )
            ax2.tick_params(axis="y", labelsize=fontsize_tick, colors="black")
            ax2.spines["right"].set_color("black")
            ax2.spines["top"].set_visible(False)
            ylim = ax2.get_ylim()
            yrange = ylim[1] - ylim[0]
            ax2.set_ylim(ylim[0], ylim[1] + 0.07 * yrange)

    if wastewater_data is not None and not wastewater_data.empty:
        deme_wastewater = wastewater_data[wastewater_data["Deme"] == deme].sort_values(
            "t_wastewater_fromsimstart"
        )

        if not deme_wastewater.empty:
            wastewater_label = "Log Wastewater" if aux_log else "Wastewater"
            time_plot_wastewater = (
                deme_wastewater["t_wastewater_fromsimstart"] * time_factor
            )
            wastewater_vals = transform_value(
                deme_wastewater["wastewater"].values,
                aux_log,
                is_log_space=False,
            )
            if ax2 is not None:
                ax3 = ax.twinx()
                ax3.set_zorder(2)
                ax3.patch.set_visible(False)
                # Outward offset must clear ax2's tick labels + rotated y-axis
                # label. That space depends on font sizes, not the subplot width:
                #   ~3 char widths of tick labels (char ≈ 0.55 * fontsize pt)
                # + labelpad between ticks and axis label
                # + font height of the rotated axis label (~ 1.1 * fontsize pt)
                # + small buffer so the two labels don't crowd.
                label_fontsize = fontsizes[1]
                labelpad_est = max(6.0, label_fontsize)
                offset_points = (
                    3.0 * fontsize_tick + labelpad_est + label_fontsize * 1.1 + 6.0
                )
                ax3.spines["right"].set_position(("outward", offset_points))
            else:
                ax3 = ax.twinx()
                ax3.set_zorder(1)
                ax.set_zorder(3)
                ax.patch.set_visible(False)
            ax3.plot(
                time_plot_wastewater,
                wastewater_vals,
                color="dimgrey",
                linewidth=1,
                label=wastewater_label,
                linestyle="solid",
                zorder=2,
                marker="o",
                markersize=2,
            )
            labelpad = max(6.0, fontsizes[1] * 0.8)
            ax3.set_ylabel(
                wastewater_label,
                fontsize=fontsizes[1],
                color="black",
                rotation=270,
                labelpad=labelpad,
                va="center",
            )
            ax3.tick_params(axis="y", labelsize=fontsize_tick, colors="black")
            ax3.spines["right"].set_color("black")
            ax3.spines["top"].set_visible(False)
            ylim = ax3.get_ylim()
            yrange = ylim[1] - ylim[0]
            ax3.set_ylim(ylim[0], ylim[1] + 0.07 * yrange)

    if not hpd_datastream.empty:
        deme_datastream = hpd_datastream[hpd_datastream["Deme"] == deme].sort_values(
            "timesincestart"
        )
        if (
            not deme_datastream.empty
            and "logPrevalence_hpd_lower" in deme_datastream.columns
        ):
            plot_hpd_interval(
                ax,
                deme_datastream["timesincestart"],
                deme_datastream["logPrevalence_hpd_lower"],
                deme_datastream["logPrevalence_hpd_upper"],
                deme_datastream["logPrevalence"],
                COLORS[3],
                "MASCOT-DS",
                prev_log,
                is_log_space=True,
                time_factor=time_factor,
                alpha=0.2,
                linewidth=2,
                zorder_interval=3,
                zorder_line=5,
            )

    if trajectory_data is not None and not trajectory_data.empty:
        infected_data = trajectory_data[trajectory_data["population"] == "I"]
        deme_infected = infected_data[infected_data["Deme"] == deme].sort_values("t")
        if not deme_infected.empty:
            prevalence_vals = transform_value(
                deme_infected["value"], prev_log, is_log_space=False
            )
            ax.plot(
                deme_infected["t"] * time_factor,
                prevalence_vals,
                color=COLORS[0],
                linewidth=2,
                label="Expected",
                alpha=0.6,
                zorder=4,
            )

    plot_hpd_validation_scatter(
        ax,
        validation_data_datastreams_prevalence,
        deme,
        time_factor,
        in_hpd_column="inHPD",
        in_hpd_color=COLORS[3],
        offset_fraction=0.05,
        marker_size=10,
    )

    if diff_treeroot_start is not None:
        ax.axvline(
            x=diff_treeroot_start * time_factor,
            color="black",
            linestyle="--",
            linewidth=1,
        )

    margin_frac = 0.01
    ax_ylim = ax.get_ylim()
    ax_lo, ax_hi = ax_ylim[0], ax_ylim[1]
    ax_range = ax_hi - ax_lo
    if ax_lo < 0:
        ax_ymin = ax_lo - margin_frac * ax_range
        ax_ymax = ax_hi
    elif ax_lo <= 0 <= ax_hi:
        ax_ymin = 0 - margin_frac * (ax_hi - 0)
        ax_ymax = ax_hi
    else:
        ax_ymin = ax_lo - margin_frac * ax_range
        ax_ymax = ax_hi
    ax.set_ylim(ax_ymin, ax_ymax)

    if ax_ymin < 0 < ax_ymax:
        fraction_0 = (0 - ax_ymin) / (ax_ymax - ax_ymin)
    else:
        fraction_0 = None

    def set_secondary_ylim_align_zero(sec_ax, r_lo, r_hi):
        r_range = r_hi - r_lo
        if fraction_0 is not None and r_lo <= 0 <= r_hi:
            r_ymax = r_hi
            r_ymin = 0 - fraction_0 * (r_ymax - 0) / (1 - fraction_0)
            if r_ymin <= r_lo:
                sec_ax.set_ylim(r_ymin, r_ymax)
                return
            r_ymin = r_lo - margin_frac * r_range
            r_ymax = r_ymin + (0 - r_ymin) / fraction_0
            sec_ax.set_ylim(r_ymin, max(r_ymax, r_hi))
        else:
            sec_ax.set_ylim(r_lo - margin_frac * r_range, r_hi)

    if ax2 is not None:
        r2_ylim = ax2.get_ylim()
        set_secondary_ylim_align_zero(ax2, r2_ylim[0], r2_ylim[1])
    if ax3 is not None:
        r3_ylim = ax3.get_ylim()
        set_secondary_ylim_align_zero(ax3, r3_ylim[0], r3_ylim[1])

    ylabel = "Log Prevalence" if prev_log else "Prevalence"
    set_axis_fontsizes(ax, fontsizes, ylabel=ylabel)
    if starting_deme is not None:
        title = get_deme_display_label(deme, starting_deme)
    else:
        title = f"Deme {deme}"
    ax.set_title(title, fontsize=fontsizes[0], pad=10)
    ax.spines["top"].set_visible(False)

    if show_legend:
        legend_handles = [
            Line2D(
                [],
                [],
                color=COLORS[0],
                linewidth=2,
                alpha=0.6,
                label="Expected Prev.",
            ),
            Line2D(
                [],
                [],
                color=COLORS[3],
                linewidth=2,
                alpha=0.6,
                label="Inferred Prev.",
            ),
            Line2D(
                [],
                [],
                color="dimgrey",
                linewidth=1,
                linestyle="solid",
                marker="o",
                markersize=2,
                label="WW",
            ),
            Patch(
                facecolor="silver",
                edgecolor="silver",
                label="CC",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.0, 0.90),
            fontsize=fontsize_tick,
            frameon=False,
        )


def _plot_ne_panel(
    ax,
    deme,
    *,
    hpd_original,
    hpd_datastream,
    expected_ne_data,
    max_time,
    validation_data_original_ne,
    validation_data_datastreams_ne,
    time_factor,
    time_label,
    show_logscale=False,
    diff_treeroot_start=None,
    starting_deme=None,
    fontsizes=None,
    fontsize_tick=None,
):
    """
    Draw the Ne panel (subplot 2) for a single deme on the given axis.
    """
    if fontsizes is None:
        fontsizes = _fontsizes_list()
    if fontsize_tick is None:
        fontsize_tick = fontsizes[2]

    if not hpd_original.empty:
        deme_original = hpd_original[hpd_original["Deme"] == deme].sort_values(
            "timesincestart"
        )
        if not deme_original.empty and "logNe_hpd_lower" in deme_original.columns:
            plot_hpd_interval(
                ax,
                deme_original["timesincestart"],
                deme_original["logNe_hpd_lower"],
                deme_original["logNe_hpd_upper"],
                deme_original["logNe"],
                COLORS[4],
                "MASCOT",
                show_logscale,
                is_log_space=True,
                time_factor=time_factor,
                alpha=0.2,
                linewidth=2,
                zorder_interval=3,
                zorder_line=4,
            )

    if not hpd_datastream.empty:
        deme_datastream = hpd_datastream[hpd_datastream["Deme"] == deme].sort_values(
            "timesincestart"
        )
        if not deme_datastream.empty and "logNe_hpd_lower" in deme_datastream.columns:
            plot_hpd_interval(
                ax,
                deme_datastream["timesincestart"],
                deme_datastream["logNe_hpd_lower"],
                deme_datastream["logNe_hpd_upper"],
                deme_datastream["logNe"],
                COLORS[3],
                "MASCOT-DS",
                show_logscale,
                is_log_space=True,
                time_factor=time_factor,
                alpha=0.2,
                linewidth=2,
                zorder_interval=3,
                zorder_line=4,
            )

    if expected_ne_data is not None and not expected_ne_data.empty:
        deme_expected = expected_ne_data[
            (expected_ne_data["Deme"] == deme) & (expected_ne_data["t"] <= max_time)
        ].sort_values("t")
        if not deme_expected.empty:
            ne_expected_vals = transform_value(
                deme_expected["Ne_expected"], show_logscale, is_log_space=False
            )
            ax.plot(
                deme_expected["t"] * time_factor,
                ne_expected_vals,
                color=COLORS[0],
                linewidth=2,
                label="Expected",
                alpha=0.8,
                zorder=2,
            )

    plot_hpd_validation_scatter(
        ax,
        validation_data_original_ne,
        deme,
        time_factor,
        in_hpd_column="inHPD",
        in_hpd_color=COLORS[4],
        offset_fraction=0.05,
        marker_size=10,
    )
    plot_hpd_validation_scatter(
        ax,
        validation_data_datastreams_ne,
        deme,
        time_factor,
        in_hpd_column="inHPD",
        in_hpd_color=COLORS[3],
        offset_fraction=0.08,
        marker_size=10,
    )

    ylabel = "Log Ne" if show_logscale else "Ne"
    if diff_treeroot_start is not None:
        ax.axvline(
            x=diff_treeroot_start * time_factor,
            color="black",
            linestyle="--",
            linewidth=1,
        )
    set_axis_fontsizes(ax, fontsizes, ylabel=ylabel)
    if starting_deme is not None:
        title = get_deme_display_label(deme, starting_deme)
    else:
        title = f"Deme {deme}"
    ax.set_title(title, fontsize=fontsizes[0], pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    loc = "lower center" if show_logscale else "center left"
    ax.legend(fontsize=fontsize_tick, loc=loc, frameon=False)


def _plot_cumincidence_panel(
    ax,
    deme,
    *,
    cumulative_incidence_hpd,
    trajectory_data,
    seroprevalence_data,
    validation_data_datastreams_cumIncidence,
    time_factor,
    time_label,
    diff_treeroot_start=None,
    starting_deme=None,
    deme_popsizes=None,
    fontsizes=None,
    fontsize_tick=None,
    show_legend=True,
    pin_band_above_sero_one=False,
):
    """
    Draw the cumulative incidence panel (subplot 3) for a single deme on the given axis.
    """
    if fontsizes is None:
        fontsizes = _fontsizes_list()
    if fontsize_tick is None:
        fontsize_tick = fontsizes[2]

    ax2 = None
    if cumulative_incidence_hpd is not None and not cumulative_incidence_hpd.empty:
        deme_cuminc = cumulative_incidence_hpd[
            cumulative_incidence_hpd["Deme"] == deme
        ].sort_values("timesincestart")
        if (
            not deme_cuminc.empty
            and "cumulativeIncidence_hpd_lower" in deme_cuminc.columns
        ):
            plot_hpd_interval(
                ax,
                deme_cuminc["timesincestart"],
                deme_cuminc["cumulativeIncidence_hpd_lower"],
                deme_cuminc["cumulativeIncidence_hpd_upper"],
                deme_cuminc["cumulativeIncidence"],
                COLORS[3],
                "Inferred",
                show_logscale=False,
                is_log_space=False,
                time_factor=time_factor,
                alpha=0.2,
                linewidth=2,
                zorder_interval=3,
                zorder_line=5,
            )

    if trajectory_data is not None and not trajectory_data.empty:
        infected_data = trajectory_data[
            trajectory_data["population"] == "NewInfectCount"
        ]
        deme_infected = infected_data[infected_data["Deme"] == deme].sort_values("t")
        if not deme_infected.empty:
            cuminc_vals = transform_value(
                deme_infected["value"], show_logscale=False, is_log_space=False
            )
            ax.plot(
                deme_infected["t"] * time_factor,
                cuminc_vals,
                color=COLORS[0],
                linewidth=2,
                label="Expected",
                alpha=0.8,
                zorder=4,
            )

    if seroprevalence_data is not None and not seroprevalence_data.empty:
        deme_seroprevalence = seroprevalence_data[
            seroprevalence_data["Deme"] == deme
        ].sort_values("t_seroprevalence_fromsimstart")
        if not deme_seroprevalence.empty:
            time_plot = (
                deme_seroprevalence["t_seroprevalence_fromsimstart"] * time_factor
            )
            ax2 = setup_twin_axis_for_secondary_data(
                ax,
                time_plot,
                deme_seroprevalence["seroprevalence"],
                "Seroprevalence",
                color="dimgrey",
                label="Seroprevalence",
                marker="o",
                markersize=4,
                linewidth=1,
                fontsize=fontsizes[1],
                fontsize_tick=fontsize_tick,
            )

    scatter_y_override = None
    if pin_band_above_sero_one and deme_popsizes is not None and deme in deme_popsizes:
        scatter_y_override = float(deme_popsizes[deme]) * 1.04

    plot_hpd_validation_scatter(
        ax,
        validation_data_datastreams_cumIncidence,
        deme,
        time_factor,
        in_hpd_column="inHPD",
        in_hpd_color=COLORS[3],
        offset_fraction=0.05,
        marker_size=10,
        scatter_y_override=scatter_y_override,
    )

    margin_frac = 0.05
    if deme_popsizes is not None and deme in deme_popsizes:
        popsize = float(deme_popsizes[deme])
        left_min_data, left_max_data = 0.0, popsize * 1.15
    else:
        left_min_data, left_max_data = ax.get_ylim()
    left_range = left_max_data - left_min_data
    left_ymin = left_min_data - margin_frac * left_range
    ax.set_ylim(left_ymin, left_max_data)

    if ax2 is not None:
        if deme_popsizes is not None and deme in deme_popsizes:
            popsize = float(deme_popsizes[deme])
            if popsize > 0:
                k = left_max_data / popsize
                right_min_data, right_max_data = 0.0, k
            else:
                right_min_data, right_max_data = 0.0, 1.0
        else:
            right_min_data, right_max_data = 0.0, 1.15
        right_range = right_max_data - right_min_data
        right_ymin = right_min_data - margin_frac * right_range
        ax2.set_ylim(right_ymin, right_max_data)
        ax2.set_yticks(np.arange(0.0, 1.01, 0.2))

    ylabel = "Cumulative incidence"
    if diff_treeroot_start is not None:
        ax.axvline(
            x=diff_treeroot_start * time_factor,
            color="black",
            linestyle="--",
            linewidth=1,
        )
    set_axis_fontsizes(ax, fontsizes, xlabel=f"Time ({time_label})", ylabel=ylabel)
    if starting_deme is not None:
        title = get_deme_display_label(deme, starting_deme)
    else:
        title = f"Deme {deme}"
    ax.set_title(title, fontsize=fontsizes[0], pad=10)
    ax.spines["top"].set_visible(False)

    if show_legend:
        legend_handles = [
            Line2D(
                [],
                [],
                color=COLORS[0],
                linewidth=2,
                alpha=0.8,
                label="Expected Cum. inc.",
            ),
            Line2D(
                [],
                [],
                color=COLORS[3],
                linewidth=2,
                alpha=0.6,
                label="Inferred Cum. inc.",
            ),
        ]
        if ax2 is not None:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    color="dimgrey",
                    linewidth=1,
                    linestyle="solid",
                    marker="o",
                    markersize=4,
                    label="SP",
                )
            )
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.0, 0.90),
            fontsize=fontsize_tick,
            frameon=False,
        )


def plot_skyline_ne(
    hpd_original,
    hpd_datastream,
    output_file=None,
    max_time=None,
    trajectory_data=None,
    case_counts_data=None,
    seroprevalence_data=None,
    wastewater_data=None,
    expected_ne_data=None,
    cumulative_incidence_hpd=None,
    validation_data_datastreams_prevalence=None,
    validation_data_datastreams_ne=None,
    validation_data_datastreams_cumIncidence=None,
    validation_data_original_ne=None,
    show_logscale=False,
    time_unit="years",
    diff_treeroot_start=None,
    case_counts_P1=False,
    starting_deme=None,
    deme_popsizes=None,
):
    """
    Plot 3 subplots:
    1. Prevalence estimates from Mascot datastreams, true prevalence from traj file, and case counts on right axis (forward time)
    2. Ne estimates from MASCOT original and expected Nes
    3. Cumulative incidence from Mascot datastreams, true cumulative incidence from traj file, and seroprevalence on right axis

    Args:
        hpd_original (pd.DataFrame): HPD intervals for default MASCOT model (times in years)
        hpd_datastream (pd.DataFrame): HPD intervals for datastream MASCOT model (times in years)
        output_file (str, optional): Path to save the plot
        max_time (float, optional): Maximum time for interpolation (in years)
        trajectory_data (pd.DataFrame, optional): Trajectory data with columns Sample, t (in years), population, index, value
        case_counts_data (pd.DataFrame, optional): Case counts data (times in years)
        seroprevalence_data (pd.DataFrame, optional): Seroprevalence data (times in years)
        wastewater_data (pd.DataFrame, optional): Wastewater data (times in years)
        expected_ne_data (pd.DataFrame, optional): Expected Ne data (times in years)
        cumulative_incidence_hpd (pd.DataFrame, optional): Cumulative incidence data from log files (times in years)
        validation_data_datastreams_prevalence (pd.DataFrame, optional): Validation results for prevalence with columns Deme, timesincestart, inHPD
        validation_data_datastreams_ne (pd.DataFrame, optional): Validation results for Ne with columns Deme, timesincestart, inHPD
        validation_data_original_ne (pd.DataFrame, optional): Validation results for Ne with columns Deme, timesincestart, inHPD
        show_logscale (bool, optional): If True, plot in log space. If False, plot in linear space. Default False.
        time_unit (str, optional): Time unit for plotting ("years" or "days"). Default "years".
        diff_treeroot_start (float, optional): Time difference between tree root and start of simulation (in years). Default None.
        case_counts_P1 (bool, optional): If True, add +1 to case counts. Default False.
        starting_deme (int, optional): Starting deme for the plot. Default None.
        deme_popsizes (dict, optional): Dictionary of deme populations sizes. Default None.
    """
    if hpd_original.empty:
        logger.warning("No data to plot.")
        return

    # Determine time conversion factor for plotting
    time_factor = 365.0 if time_unit == "days" else 1.0
    time_label = "days" if time_unit == "days" else "years"

    # Get unique demes and ensure the start deme is always the first column
    demes = sorted(hpd_original["Deme"].unique())
    if starting_deme is not None and len(demes) == 2:
        other_deme = [d for d in demes if int(d) != int(starting_deme)]
        if len(other_deme) == 1:
            demes = [int(starting_deme), int(other_deme[0])]
    n_demes = len(demes)

    # Create figure with 3 rows, n_demes columns
    fig, axes = plt.subplots(3, n_demes, figsize=(4 * n_demes, 7), sharex="col")
    if n_demes == 1:
        axes = axes.reshape(3, 1)

    fontsize_tick = fontsizes["tick_label"]

    # Subplot 1: Prevalence and Case Counts (forward time)
    for i, deme in enumerate(demes):
        _plot_prevalence_panel(
            axes[0, i],
            deme,
            hpd_datastream=hpd_datastream,
            trajectory_data=trajectory_data,
            case_counts_data=case_counts_data,
            wastewater_data=wastewater_data,
            validation_data_datastreams_prevalence=validation_data_datastreams_prevalence,
            time_factor=time_factor,
            time_label=time_label,
            show_logscale=show_logscale,
            case_counts_P1=case_counts_P1,
            diff_treeroot_start=diff_treeroot_start,
            starting_deme=starting_deme,
            fig=fig,
            n_demes=n_demes,
            fontsizes=_fontsizes_list(),
            fontsize_tick=fontsize_tick,
        )

    # Subplot 2: Ne Estimates
    for i, deme in enumerate(demes):
        _plot_ne_panel(
            axes[1, i],
            deme,
            hpd_original=hpd_original,
            hpd_datastream=hpd_datastream,
            expected_ne_data=expected_ne_data,
            max_time=max_time,
            validation_data_original_ne=validation_data_original_ne,
            validation_data_datastreams_ne=validation_data_datastreams_ne,
            time_factor=time_factor,
            time_label=time_label,
            show_logscale=show_logscale,
            diff_treeroot_start=diff_treeroot_start,
            starting_deme=starting_deme,
            fontsizes=_fontsizes_list(),
            fontsize_tick=fontsize_tick,
        )

    # Subplot 3: Cumulative Incidence and Seroprevalence
    for i, deme in enumerate(demes):
        _plot_cumincidence_panel(
            axes[2, i],
            deme,
            cumulative_incidence_hpd=cumulative_incidence_hpd,
            trajectory_data=trajectory_data,
            seroprevalence_data=seroprevalence_data,
            validation_data_datastreams_cumIncidence=validation_data_datastreams_cumIncidence,
            time_factor=time_factor,
            time_label=time_label,
            diff_treeroot_start=diff_treeroot_start,
            starting_deme=starting_deme,
            deme_popsizes=deme_popsizes,
            fontsizes=_fontsizes_list(),
            fontsize_tick=fontsize_tick,
        )

    # Restrict the x-axis of each subplot to (0, max_time)
    if max_time is not None:
        for i, ax_row in enumerate(axes):
            # Handle both case of axes being 2D array (3, n_demes) or reshaped (3, 1)
            if isinstance(ax_row, np.ndarray):
                for ax in ax_row:
                    ax.set_xlim(0, max_time * time_factor)
            else:
                ax_row.set_xlim(0, max_time * time_factor)

    # Increase horizontal spacing between subplots to accommodate axis labels
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()

    # Save or show the plot
    if output_file:
        save_figure_png_and_pdf(output_file)
    else:
        plt.show()

    # Close the plot to free memory
    plt.close()


def plot_trajectory_compartments(
    trajectory_data, output_file=None, time_unit="years", starting_deme=None
):
    """
    Plot S, I, and Rh+Rs compartments for each deme in a 1-row, 2-column layout.

    Args:
        trajectory_data (pd.DataFrame): Trajectory data with columns Sample, t (in years), population, index, value
        output_file (str, optional): Path to save the plot. If None, display the plot.
        time_unit (str, optional): Time unit for plotting ("years" or "days"). Default "years".
    """
    if trajectory_data is None or trajectory_data.empty:
        logger.warning("No trajectory data to plot.")
        return

    # Determine time conversion factor for plotting
    time_factor = 365.0 if time_unit == "days" else 1.0
    time_label = "days" if time_unit == "days" else "years"

    # Get unique demes
    demes = sorted(trajectory_data["Deme"].unique())
    n_demes = len(demes)

    # Define colors for compartments
    compartment_colors = {
        "S": "#808080",  # Grey
        "I": COLORS[2],  # Blue (from existing colorblind-friendly palette)
        "Rh_Rs": COLORBLINDFR["additional"][1],
    }

    # Create figure with n_demes rows, 1 column
    fig, axes = plt.subplots(
        n_demes, 1, figsize=(2.5 * n_demes, 3), sharex=True, sharey=True
    )
    if n_demes == 1:
        axes = [axes]  # Ensure axes is always a list for consistent handling

    # Plot each deme in its own subplot
    for i, deme in enumerate(demes):
        ax = axes[i]

        # Filter data for current deme
        deme_data = trajectory_data[trajectory_data["Deme"] == deme].sort_values("t")

        # Get data for each compartment
        s_data = deme_data[deme_data["population"] == "S"]
        i_data = deme_data[deme_data["population"] == "I"]
        rh_data = deme_data[deme_data["population"] == "Rh"]
        rs_data = deme_data[deme_data["population"] == "Rs"]

        # Plot S compartment
        if not s_data.empty:
            ax.plot(
                s_data["t"] * time_factor,
                s_data["value"],
                color=compartment_colors["S"],
                linewidth=2,
                label="S",
            )

        # Plot I compartment
        if not i_data.empty:
            ax.plot(
                i_data["t"] * time_factor,
                i_data["value"],
                color=compartment_colors["I"],
                linewidth=2,
                label="I",
            )

        # Combine Rh and Rs data for plotting
        if not rh_data.empty and not rs_data.empty:
            # Merge Rh and Rs data by time point
            rh_rs_combined = rh_data.merge(rs_data, on="t", suffixes=("_rh", "_rs"))
            rh_rs_combined["combined_value"] = (
                rh_rs_combined["value_rh"] + rh_rs_combined["value_rs"]
            )

            ax.plot(
                rh_rs_combined["t"] * time_factor,
                rh_rs_combined["combined_value"],
                color=compartment_colors["Rh_Rs"],
                linewidth=2,
                label="R",
            )
        elif not rh_data.empty:
            # Only Rh data available
            ax.plot(
                rh_data["t"] * time_factor,
                rh_data["value"],
                color=compartment_colors["Rh_Rs"],
                linewidth=2,
                label="Rh",
            )
        elif not rs_data.empty:
            # Only Rs data available
            ax.plot(
                rs_data["t"] * time_factor,
                rs_data["value"],
                color=compartment_colors["Rh_Rs"],
                linewidth=2,
                label="Rs",
            )

        # Customize subplot
        if starting_deme is not None:
            title = get_deme_display_label(deme, starting_deme)
        else:
            title = f"Deme {deme}"
        ax.set_title(title, fontsize=fontsizes["title"], pad=10)

        # Set xlabel for all subplots
        if i == n_demes - 1:
            set_axis_fontsizes(
                ax,
                _fontsizes_list(),
                xlabel=f"Time ({time_label})",
                ylabel="Population Count",
            )
            ax.legend(fontsize=fontsizes["legend"], frameon=False)
        else:
            set_axis_fontsizes(ax, _fontsizes_list(), ylabel="Population Count")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save or show the plot
    if output_file:
        save_figure_png_and_pdf(output_file)
    else:
        plt.show()

    # Close the plot to free memory
    plt.close()


def transform_logNE_toNE(df):
    df["logNe"] = np.exp(df["logNe"].astype(float))
    df["logNe_hpd_lower"] = np.exp(df["logNe_hpd_lower"].astype(float))
    df["logNe_hpd_upper"] = np.exp(df["logNe_hpd_upper"].astype(float))
    return df


def interpolate_trajectory(trajectory_data, times, population="I"):
    """
    Interpolate prevalence trajectory data to a predefined grid of time points.

    The trajectory data contains time points when something changes. Between two
    time points, the conditions remain the same as the previous time point.
    This function creates a dataframe with population values for every time
    point in times and every deme.

    If a timepoint exists in infected_data, it copies that info. If a timepoint
    is between two timepoints in infected_data, it uses the info from the
    previous timepoint (forward-fill behavior).

    Args:
        trajectory_data: DataFrame with columns Sample, t, population, index, value, Deme
        times: Array or list of time points to interpolate to
        population: Population to interpolate (default "I")
    Returns:
        pd.DataFrame: DataFrame with columns Deme, t, population values
    """
    if trajectory_data is None or trajectory_data.empty:
        return pd.DataFrame()

    infected_data = trajectory_data[trajectory_data["population"] == population].copy()
    if infected_data.empty:
        return pd.DataFrame()

    # Get all unique demes
    demes = infected_data["Deme"].unique()

    # Create a grid of all time points and demes
    time_deme_grid = []
    for t in times:
        for deme in demes:
            time_deme_grid.append({"t": t, "Deme": deme})

    grid_df = pd.DataFrame(time_deme_grid)

    # Prepare infected data: keep only needed columns and sort by Deme and t
    # merge_asof requires the right DataFrame to be sorted by the merge key
    infected_prep = (
        infected_data[["Deme", "t", "value"]].sort_values(["t", "Deme"]).copy()
    )

    # Sort grid by t and Deme for merge_asof
    grid_df = grid_df.sort_values(["t", "Deme"])

    # Use merge_asof with direction='backward' to match each time point to the
    # most recent time point in infected_data that is <= the target time
    # This implements the forward-fill behavior (last observation carried forward)
    merged = pd.merge_asof(
        grid_df,
        infected_prep,
        left_on="t",
        right_on="t",
        by="Deme",
        direction="backward",
        suffixes=("", "_infected"),
    )

    # Rename value column to I for clarity
    merged = merged.rename(columns={"value": population})

    # Select and reorder columns
    result_df = merged[["Deme", "t", population]].copy()

    return result_df


def interpolate_expected_ne_at_times(
    trajectory_data: pd.DataFrame,
    params_df: pd.DataFrame,
    times: np.array,
) -> pd.DataFrame:
    """
    Interpolate expected Ne values at specific time points using trajectory data and parameters.

    Similar to interpolate_trajectory, but computes expected Ne using Volz (2012) formula:
    Ne = I / (2 * beta * S).

    The trajectory data contains time points when something changes. Between two time points,
    the conditions remain the same as the previous time point. This function:
    1. Gets I and S values at the target time points (using forward-fill behavior)
    2. Determines the appropriate beta for each time point based on time-varying betas
    3. Computes Ne using the formula

    Args:
        trajectory_data: DataFrame with columns Sample, t, population, index, value, Deme
        params_df: DataFrame with parameter values (must contain beta parameters)
        times: Array of time points to interpolate to

    Returns:
        pd.DataFrame: DataFrame with columns Deme, t, Ne_expected
    """
    if trajectory_data is None or trajectory_data.empty:
        return pd.DataFrame()
    if params_df is None or params_df.empty:
        return pd.DataFrame()

    # Extract all beta parameters that were scaled by the population size
    beta_rows = params_df[
        params_df["parameter"].str.startswith("beta", na=False)
        & params_df["parameter"].str.endswith("Nscaled", na=False)
    ].copy()

    if beta_rows.empty:
        logger.warning(
            "No beta entries found in params CSV; cannot compute expected Ne."
        )
        return pd.DataFrame()

    # Parse time from parameter names and extract within-deme betas
    # Structure: parameter like "beta_0" or "beta_0.055555", deme like "0->0" or "1->1"
    time_beta_map = {}  # {time: {deme_idx: beta_value}}

    for _, row in beta_rows.iterrows():
        param_name = str(row["parameter"])
        deme_str = str(row["Deme"])

        # Extract time from parameter name (e.g., "beta_0" -> 0, "beta_0.055555" -> 0.055555)
        # Extract time string between "beta_" and either end or first "_" (for "_Nscaled" or other suffix)
        time_part = param_name[5:]  # Remove "beta_" prefix
        # Remove trailing suffixes like "_Nscaled" if present
        if "_" in time_part:
            time_str = time_part.split("_")[0]
        else:
            time_str = time_part
        time_val = float(time_str)

        left, right = deme_str.split("->")
        i = int(left.strip())
        j = int(right.strip())

        if i == j:
            if time_val not in time_beta_map:
                time_beta_map[time_val] = {}
            time_beta_map[time_val][i] = float(row["value"])

    if not time_beta_map:
        logger.warning(
            "No within-deme beta entries (i->i) found in params CSV; cannot compute expected Ne."
        )
        return pd.DataFrame()

    # Sort time points to create intervals
    time_points = sorted(time_beta_map.keys())

    # Prepare S and I time series per deme
    df_S = trajectory_data[trajectory_data["population"] == "S"].copy()
    df_I = trajectory_data[trajectory_data["population"] == "I"].copy()
    if df_S.empty or df_I.empty:
        logger.warning("Missing S or I in trajectory data; cannot compute expected Ne.")
        return pd.DataFrame()

    # Get all unique demes
    demes = df_S["Deme"].unique()

    # Create a grid of all time points and demes
    time_deme_grid = []
    for t in times:
        for deme in demes:
            time_deme_grid.append({"t": t, "Deme": deme})

    grid_df = pd.DataFrame(time_deme_grid)

    # Prepare S data: keep only needed columns and sort by Deme and t
    df_S_prep = df_S[["Deme", "t", "value"]].sort_values(["t", "Deme"]).copy()
    df_S_prep = df_S_prep.rename(columns={"value": "S"})

    # Prepare I data: keep only needed columns and sort by Deme and t
    df_I_prep = df_I[["Deme", "t", "value"]].sort_values(["t", "Deme"]).copy()
    df_I_prep = df_I_prep.rename(columns={"value": "I"})

    # Sort grid by t and Deme for merge_asof
    grid_df = grid_df.sort_values(["t", "Deme"])

    # Use merge_asof to get S and I values at target time points (forward-fill behavior)
    merged = pd.merge_asof(
        grid_df,
        df_S_prep,
        left_on="t",
        right_on="t",
        by="Deme",
        direction="backward",
    )

    merged = pd.merge_asof(
        merged,
        df_I_prep,
        left_on="t",
        right_on="t",
        by="Deme",
        direction="backward",
        suffixes=("", "_I"),
    )

    # Function to find the appropriate beta for a given time and deme
    def get_beta_for_time(t_val, deme_idx):
        """Find which beta interval a time point belongs to."""
        # Find the largest time point <= t_val
        beta_time = None
        for tp in reversed(time_points):
            if tp <= t_val:
                beta_time = tp
                break

        if beta_time is None:
            # Time is before first beta, use first beta
            beta_time = time_points[0]

        deme_betas = time_beta_map.get(beta_time, {})
        return deme_betas.get(deme_idx, np.nan)

    # Compute Ne for each row using time-appropriate beta
    def row_ne(row):
        idx = int(row["Deme"])
        t_val = float(row["t"])
        beta = get_beta_for_time(t_val, idx)

        if beta is None or np.isnan(beta) or row["S"] <= 0:
            return np.nan
        # Volz (2012) formula, expects beta to be scaled by N_i
        return float(row["I"]) / (2.0 * beta * float(row["S"]))

    merged["Ne_expected"] = merged.apply(row_ne, axis=1)
    merged = merged.dropna(subset=["Ne_expected"])

    # Select and reorder columns
    result_df = merged[["Deme", "t", "Ne_expected"]].copy()

    return result_df


def validate_hpd_intervals(
    hpd_intervals: pd.DataFrame,
    trajectory_data: pd.DataFrame,
    out_prefix: str,
    validation_type: str = "prevalence",
    params_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Validate HPD intervals by comparing predicted values with actual trajectory data.

    Supports two validation types:
    1. "prevalence": Compares log prevalence from trajectory with HPD intervals for logPrevalence
    2. "ne": Compares log expected Ne from trajectory with HPD intervals for logNe

    For each time point in hpd_intervals, matches timesincestart to the closest
    time point in trajectory_data, calculates the expected value, and checks if it
    falls within the HPD interval.

    Args:
        hpd_intervals: DataFrame with columns Deme, timesincestart, and either:
            - For prevalence: logPrevalence, logPrevalence_hpd_lower, logPrevalence_hpd_upper
            - For ne: logNe, logNe_hpd_lower, logNe_hpd_upper
        trajectory_data: DataFrame with columns Sample, t, population, index, value, Deme
        out_prefix: Prefix for output CSV file
        validation_type: Type of validation ("prevalence" or "ne")
        params_df: DataFrame with parameter values (required for "ne" validation)

    Returns:
        pd.DataFrame: Validation results with columns depending on validation_type:
            For prevalence:
                Deme, timesincestart, matched_t, logPrevalence, logPrevalence_hpd_lower,
                logPrevalence_hpd_upper, expectedlogPrev, expectedlogPrevP1, inHPD, inHPDP1
            For ne:
                Deme, timesincestart, matched_t, logNe, logNe_hpd_lower,
                logNe_hpd_upper, expectedlogNe, inHPD
        Returns empty DataFrame if validation cannot be performed.
    """
    if hpd_intervals is None or hpd_intervals.empty:
        return pd.DataFrame()

    if trajectory_data is None or trajectory_data.empty:
        logger.warning("trajectory_data is not available for HPD validation")
        return pd.DataFrame()

    basename = out_prefix.replace("_original", "").replace("_datastreams", "")

    if validation_type == "prevalence":
        # Check if logPrevalence columns exist
        required_cols = [
            "logPrevalence",
            "logPrevalence_hpd_lower",
            "logPrevalence_hpd_upper",
        ]
        if not all(col in hpd_intervals.columns for col in required_cols):
            logger.warning("logPrevalence columns not found in hpd_intervals")
            return pd.DataFrame()

        # Filter and prepare infected data from trajectory
        times = hpd_intervals["timesincestart"].unique()
        infected_data = interpolate_trajectory(trajectory_data, times)
        if infected_data.empty:
            return pd.DataFrame()

        # Prepare trajectory data: keep only needed columns and sort by Deme and t
        trajectory_prep = infected_data[["Deme", "t", "I"]].sort_values(["t", "Deme"])

        # Prepare HPD intervals: sort by Deme and timesincestart for merge
        hpd_prep = hpd_intervals[
            [
                "Deme",
                "timesincestart",
                "logPrevalence",
                "logPrevalence_hpd_lower",
                "logPrevalence_hpd_upper",
            ]
        ].sort_values(["timesincestart", "Deme"])

        # Use merge to match each timesincestart to the nearest t for each deme
        merged = pd.merge(
            hpd_prep,
            trajectory_prep,
            left_on=["timesincestart", "Deme"],
            right_on=["t", "Deme"],
        )

        # Check if merge was successful
        if merged.empty:
            logger.warning("No matching trajectory data found for HPD validation")
            return pd.DataFrame()

        # Calculate log prevalence vectorized
        # Handle cases where value is NaN (no match) or <= 0 (log undefined)
        merged["expectedlogPrev"] = np.where(
            (merged["I"].notna()) & (merged["I"] > 0),
            np.log(merged["I"]),
            np.nan,
        )
        merged["expectedlogPrevP1"] = np.where(
            (merged["I"].notna()),
            np.log(merged["I"] + 1),
            np.nan,
        )

        # Check if within HPD interval vectorized
        merged["inHPD"] = np.where(
            (merged["expectedlogPrev"] >= merged["logPrevalence_hpd_lower"])
            & (merged["expectedlogPrev"] <= merged["logPrevalence_hpd_upper"]),
            1,
            0,
        )
        merged["inHPDP1"] = np.where(
            (merged["expectedlogPrevP1"] >= merged["logPrevalence_hpd_lower"])
            & (merged["expectedlogPrevP1"] <= merged["logPrevalence_hpd_upper"]),
            1,
            0,
        )

        # Set inHPD to 0 where log is undefined (value <= 0)
        merged.loc[merged["expectedlogPrev"].isna(), "inHPD"] = 0
        merged.loc[merged["expectedlogPrevP1"].isna(), "inHPDP1"] = 0
        # Rename matched_t column for clarity
        merged = merged.rename(columns={"t": "matched_t"})

        # Select and reorder columns for output
        hpd_validation_df = merged[
            [
                "Deme",
                "timesincestart",
                "matched_t",
                "logPrevalence",
                "logPrevalence_hpd_lower",
                "logPrevalence_hpd_upper",
                "expectedlogPrev",
                "expectedlogPrevP1",
                "inHPD",
                "inHPDP1",
            ]
        ].copy()

        # Save results to CSV
        validation_output = f"{out_prefix}_hpd_validation_prevalence.csv"
        hpd_validation_df["Simulation"] = basename
        hpd_validation_df.to_csv(validation_output, index=False)
        logger.info(
            "HPD validation results (prevalence) saved to %s", validation_output
        )

        # Print summary statistics
        if "inHPD" in hpd_validation_df.columns:
            total_points = len(hpd_validation_df)
            in_hpd_count = hpd_validation_df["inHPD"].sum()
            coverage = (in_hpd_count / total_points) * 100 if total_points > 0 else 0
            logger.info(
                "HPD coverage (prevalence): %d/%d (%.2f%%)",
                in_hpd_count,
                total_points,
                coverage,
            )

        if "inHPDP1" in hpd_validation_df.columns:
            total_points = len(hpd_validation_df)
            in_hpd_count = hpd_validation_df["inHPDP1"].sum()
            coverage = (in_hpd_count / total_points) * 100 if total_points > 0 else 0
            logger.info(
                "HPD coverage with +1 counts (prevalence): %d/%d (%.2f%%)",
                in_hpd_count,
                total_points,
                coverage,
            )

        return hpd_validation_df

    elif validation_type == "ne":
        # Check if logNe columns exist
        required_cols = ["logNe", "logNe_hpd_lower", "logNe_hpd_upper"]
        if not all(col in hpd_intervals.columns for col in required_cols):
            logger.warning("logNe columns not found in hpd_intervals")
            return pd.DataFrame()

        if params_df is None or params_df.empty:
            logger.warning("params_df is required for Ne validation")
            return pd.DataFrame()

        # Filter and prepare expected Ne data from trajectory
        times = hpd_intervals["timesincestart"].unique()
        expected_ne_data = interpolate_expected_ne_at_times(
            trajectory_data, params_df, times
        )
        if expected_ne_data.empty:
            return pd.DataFrame()

        # Prepare expected Ne data: keep only needed columns and sort by Deme and t
        expected_ne_prep = expected_ne_data[["Deme", "t", "Ne_expected"]].sort_values(
            ["t", "Deme"]
        )

        # Prepare HPD intervals: sort by Deme and timesincestart for merge
        hpd_prep = hpd_intervals[
            [
                "Deme",
                "timesincestart",
                "logNe",
                "logNe_hpd_lower",
                "logNe_hpd_upper",
            ]
        ].sort_values(["timesincestart", "Deme"])

        # Use merge to match each timesincestart to the nearest t for each deme
        merged = pd.merge(
            hpd_prep,
            expected_ne_prep,
            left_on=["timesincestart", "Deme"],
            right_on=["t", "Deme"],
        )

        # Check if merge was successful
        if merged.empty:
            logger.warning("No matching trajectory data found for HPD validation")
            return pd.DataFrame()

        # Calculate log Ne vectorized
        # Handle cases where value is NaN (no match) or <= 0 (log undefined)
        merged["expectedlogNe"] = np.where(
            (merged["Ne_expected"].notna()) & (merged["Ne_expected"] > 0),
            np.log(merged["Ne_expected"]),
            np.nan,
        )

        # Check if within HPD interval vectorized
        merged["inHPD"] = np.where(
            (merged["expectedlogNe"] >= merged["logNe_hpd_lower"])
            & (merged["expectedlogNe"] <= merged["logNe_hpd_upper"]),
            1,
            0,
        )

        # Set inHPD to 0 where log is undefined (value <= 0)
        merged.loc[merged["expectedlogNe"].isna(), "inHPD"] = 0
        # Rename matched_t column for clarity
        merged = merged.rename(columns={"t": "matched_t"})

        # Select and reorder columns for output
        hpd_validation_df = merged[
            [
                "Deme",
                "timesincestart",
                "matched_t",
                "logNe",
                "logNe_hpd_lower",
                "logNe_hpd_upper",
                "expectedlogNe",
                "inHPD",
            ]
        ].copy()

        # Save results to CSV
        validation_output = f"{out_prefix}_hpd_validation_ne.csv"
        hpd_validation_df["Simulation"] = basename
        hpd_validation_df.to_csv(validation_output, index=False)
        logger.info("HPD validation results (Ne) saved to %s", validation_output)

        # Print summary statistics
        if "inHPD" in hpd_validation_df.columns:
            total_points = len(hpd_validation_df)
            in_hpd_count = hpd_validation_df["inHPD"].sum()
            coverage = (in_hpd_count / total_points) * 100 if total_points > 0 else 0
            logger.info(
                "HPD coverage (Ne): %d/%d (%.2f%%)",
                in_hpd_count,
                total_points,
                coverage,
            )

        return hpd_validation_df

    elif validation_type == "cumulative_incidence":
        # Check if cumulative incidence columns exist
        required_cols = [
            "cumulativeIncidence_hpd_lower",
            "cumulativeIncidence_hpd_upper",
            "cumulativeIncidence",
        ]
        if not all(col in hpd_intervals.columns for col in required_cols):
            logger.warning("cumulative incidence columns not found in hpd_intervals")
            return pd.DataFrame()

        # Filter and prepare infected data from trajectory
        times = hpd_intervals["timesincestart"].unique()
        cumulative_incidence_data = interpolate_trajectory(
            trajectory_data, times, population="NewInfectCount"
        )
        if cumulative_incidence_data.empty:
            return pd.DataFrame()

        # Prepare trajectory data: keep only needed columns and sort by Deme and t
        trajectory_prep = cumulative_incidence_data[
            ["Deme", "t", "NewInfectCount"]
        ].sort_values(["t", "Deme"])

        # Prepare HPD intervals: sort by Deme and timesincestart for merge
        hpd_prep = hpd_intervals[
            [
                "Deme",
                "timesincestart",
                "cumulativeIncidence",
                "cumulativeIncidence_hpd_lower",
                "cumulativeIncidence_hpd_upper",
            ]
        ].sort_values(["timesincestart", "Deme"])

        # Use merge to match each timesincestart to the nearest t for each deme
        merged = pd.merge(
            hpd_prep,
            trajectory_prep,
            left_on=["timesincestart", "Deme"],
            right_on=["t", "Deme"],
        )

        # Check if merge was successful
        if merged.empty:
            logger.warning("No matching trajectory data found for HPD validation")
            return pd.DataFrame()

        # Calculate log prevalence vectorized
        # Handle cases where value is NaN (no match) or <= 0 (log undefined)
        merged["expectedcumulativeIncidence"] = np.where(
            (merged["NewInfectCount"].notna()),
            merged["NewInfectCount"],
            np.nan,
        )

        # Check if within HPD interval vectorized
        merged["inHPD"] = np.where(
            (
                merged["expectedcumulativeIncidence"]
                >= merged["cumulativeIncidence_hpd_lower"]
            )
            & (
                merged["expectedcumulativeIncidence"]
                <= merged["cumulativeIncidence_hpd_upper"]
            ),
            1,
            0,
        )

        # Set inHPD to 0 where log is undefined (value <= 0)
        merged.loc[merged["expectedcumulativeIncidence"].isna(), "inHPD"] = 0
        # Rename matched_t column for clarity
        merged = merged.rename(columns={"t": "matched_t"})

        # Select and reorder columns for output
        hpd_validation_df = merged[
            [
                "Deme",
                "timesincestart",
                "matched_t",
                "cumulativeIncidence",
                "cumulativeIncidence_hpd_lower",
                "cumulativeIncidence_hpd_upper",
                "expectedcumulativeIncidence",
                "inHPD",
            ]
        ].copy()

        # Save results to CSV
        validation_output = f"{out_prefix}_hpd_validation_cumulative_incidence.csv"
        hpd_validation_df["Simulation"] = basename
        hpd_validation_df.to_csv(validation_output, index=False)
        logger.info(
            "HPD validation results (cumulative incidence) saved to %s",
            validation_output,
        )

        if "inHPD" in hpd_validation_df.columns:
            total_points = len(hpd_validation_df)
            in_hpd_count = hpd_validation_df["inHPD"].sum()
            coverage = (in_hpd_count / total_points) * 100 if total_points > 0 else 0
            logger.info(
                "HPD coverage (cumulative incidence): %d/%d (%.2f%%)",
                in_hpd_count,
                total_points,
                coverage,
            )

        return hpd_validation_df

    else:
        logger.warning(
            "Unknown validation_type '%s'. Must be 'prevalence' or 'ne' or 'cumulative_incidence'.",
            validation_type,
        )
        return pd.DataFrame()


def get_deme_popsize(df: pd.DataFrame) -> dict:
    """
    Compute the maximum population size per deme from a trajectory DataFrame.

    For each deme/index, use the S and I populations and, at each time point,
    sum their values to obtain the total population size. The population size
    for a deme is taken as the maximum of this S+I total over time.

    Args:
        df: Trajectory DataFrame with columns including at least
            'population', 'value', and either 'Deme' or 'index'.

    Returns:
        dict: Mapping {deme: max_population_size}.
    """
    if df is None or df.empty:
        raise ValueError(
            "Trajectory data is empty or None; cannot determine deme sizes."
        )

    deme_col = "Deme" if "Deme" in df.columns else "index"
    if deme_col not in df.columns:
        raise ValueError("Trajectory data does not contain 'Deme' or 'index' column.")

    # Keep only S and I, since total population size is S + I
    mask = (df["t"] == 0.0) & df["population"].isin(["S", "I"])
    df_si = df[mask].copy()
    if df_si.empty:
        raise ValueError("No S or I populations found in trajectory data.")

    # Sum S+I per (deme, time)
    grouped = (
        df_si.groupby([deme_col], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "pop_size"})
    )

    max_per_deme = grouped.set_index(deme_col)["pop_size"].to_dict()

    # Return as a plain dict with integer keys where possible
    result: dict = {}
    for k, v in max_per_deme.items():
        try:
            key = int(k)
        except (TypeError, ValueError):
            key = k
        result[key] = float(v)

    return result


def get_deme_display_label(deme: int, starting_deme: int) -> str:
    """
    Map a numeric deme index to a human-readable label.

    The deme matching ``starting_deme`` is labeled "Start deme" and the other
    deme is labeled "Secondary deme".
    """
    return "Start deme" if int(deme) == int(starting_deme) else "Secondary deme"


def _load_input_files(args):
    """Load trajectory, datastream, parameter, and deme-switch files.

    Returns:
        dict with keys: trajectory_data, case_counts_data, seroprevalence_data,
        wastewater_data, params_df, deme_switches_df, expected_ne_data,
        mostrecent_sample_t, starting_deme, deme_popsizes.
    """
    trajectory_data = None
    if args.trajectory_file:
        trajectory_data = load_trajectory_file(args.trajectory_file)

    deme_popsizes = get_deme_popsize(trajectory_data)
    starting_deme = get_outbreak_start_deme(trajectory_data)
    mostrecent_sample_t = extract_mostrecent_sample_t(trajectory_data)
    logger.info("Most recent sample time: %s", mostrecent_sample_t)

    case_counts_data = None
    if args.case_counts_file:
        case_counts_data = load_case_counts_file(args.case_counts_file)

    seroprevalence_data = None
    if args.seroprevalence_file:
        seroprevalence_data = load_seroprevalence_file(args.seroprevalence_file)

    wastewater_data = None
    if args.wastewater_file:
        wastewater_data = load_wastewater_file(args.wastewater_file)

    params_df = (
        load_params_csv(args.params_csv)
        if getattr(args, "params_csv", None)
        else pd.DataFrame()
    )
    if params_df is not None and not params_df.empty:
        logger.debug("Loaded params CSV with shape %s", params_df.shape)

    deme_switches_df = None
    if getattr(args, "deme_switches_csv", None):
        deme_switches_df = pd.read_csv(args.deme_switches_csv)
        logger.debug("Loaded deme switches CSV with shape %s", deme_switches_df.shape)

    expected_ne_data = (
        compute_expected_ne_from_trajectory(trajectory_data, params_df)
        if (
            trajectory_data is not None
            and not trajectory_data.empty
            and params_df is not None
            and not params_df.empty
        )
        else pd.DataFrame()
    )

    return {
        "trajectory_data": trajectory_data,
        "case_counts_data": case_counts_data,
        "seroprevalence_data": seroprevalence_data,
        "wastewater_data": wastewater_data,
        "params_df": params_df,
        "deme_switches_df": deme_switches_df,
        "expected_ne_data": expected_ne_data,
        "mostrecent_sample_t": mostrecent_sample_t,
        "starting_deme": starting_deme,
        "deme_popsizes": deme_popsizes,
    }


def _parse_beast_logs(args, mostrecent_sample_t):
    """Read BEAST logs, apply burn-in, and transform skyline columns.

    Returns:
        dict with keys: log_content_original, log_content_datastream,
        skyline_long_original, skyline_long_datastream,
        tree_height_datastream, gridpointshifts_datastream.
    """
    log_content_original = read_beast_log(args.log_file_original)
    log_content_datastream, rateshifts_datastream, gridpointshifts_datastream = (
        read_beast_log(args.log_file_datastream, read_rateshifts=True)
    )
    logger.debug("Rateshifts for datastream: %s", rateshifts_datastream)
    logger.debug("Gridpoint shifts for datastream: %s", gridpointshifts_datastream)

    log_content_datastream = log_content_datastream[
        log_content_datastream["Sample"]
        > log_content_datastream["Sample"].max() * args.burnin
    ]
    log_content_original = log_content_original[
        log_content_original["Sample"]
        > log_content_original["Sample"].max() * args.burnin
    ]

    tree_height_datastream = log_content_datastream[["Sample", "Tree.height"]]

    skyline_long_datastream = transform_skyline_columns(
        log_content_datastream,
        relative_rateshifts=False,
        rateshifts=rateshifts_datastream,
        mostrecent_sample_t=mostrecent_sample_t,
    )
    skyline_long_original = transform_skyline_columns(
        log_content_original,
        relative_rateshifts=True,
        mostrecent_sample_t=mostrecent_sample_t,
    )

    skyline_long_datastream = skyline_long_datastream[
        skyline_long_datastream["Sample"]
        > skyline_long_datastream["Sample"].max() * args.burnin
    ]
    skyline_long_original = skyline_long_original[
        skyline_long_original["Sample"]
        > skyline_long_original["Sample"].max() * args.burnin
    ]

    return {
        "log_content_original": log_content_original,
        "log_content_datastream": log_content_datastream,
        "skyline_long_original": skyline_long_original,
        "skyline_long_datastream": skyline_long_datastream,
        "tree_height_datastream": tree_height_datastream,
        "rateshifts_datastream": rateshifts_datastream,
        "gridpointshifts_datastream": gridpointshifts_datastream,
    }


def _compute_hpd_intervals(args, logs, datastream_files):
    """Interpolate skylines and compute HPD intervals for Ne and cumulative incidence.

    Args:
        args: Parsed CLI arguments.
        logs: dict from ``_parse_beast_logs``.
        datastream_files: dict with case_counts_data, seroprevalence_data,
            wastewater_data, trajectory_data (used only for max-time clipping).

    Returns:
        dict with keys: hpd_original, hpd_datastream, cumulative_incidence_hpd,
        max_time, trajectory_data (possibly clipped to max_time).
    """
    skyline_long_original = logs["skyline_long_original"]
    skyline_long_datastream = logs["skyline_long_datastream"]
    tree_height_datastream = logs["tree_height_datastream"]
    gridpointshifts = logs["gridpointshifts_datastream"]

    case_counts_data = datastream_files["case_counts_data"]
    seroprevalence_data = datastream_files["seroprevalence_data"]
    wastewater_data = datastream_files["wastewater_data"]
    trajectory_data = datastream_files["trajectory_data"]

    # Determine the maximum time across all data sources
    time_maxes = [
        skyline_long_original["timesincestart"].max(),
        skyline_long_datastream["timesincestart"].max(),
    ]
    if case_counts_data is not None and not case_counts_data.empty:
        time_maxes.append(case_counts_data["t_case_counts_fromsimstart"].max())
    if seroprevalence_data is not None and not seroprevalence_data.empty:
        time_maxes.append(seroprevalence_data["t_seroprevalence_fromsimstart"].max())
    if wastewater_data is not None and not wastewater_data.empty:
        time_maxes.append(wastewater_data["t_wastewater_fromsimstart"].max())
    max_time = max(time_maxes)

    if trajectory_data is not None:
        trajectory_data = trajectory_data.loc[trajectory_data["t"] <= max_time]

    time_grid = np.linspace(
        skyline_long_original["timesincestart"].min(),
        skyline_long_original["timesincestart"].max(),
        100,
    )

    # Original variant: linear interpolation (relative rate shifts → piecewise constant)
    if time_grid is not None and not skyline_long_original.empty:
        hpd_intervals_original = interpolate_skyline(
            skyline_long_original,
            time_grid,
            interpolation_method="linear",
        )
    else:
        hpd_intervals_original = (
            skyline_long_original.groupby(["Deme", "timesincestart"])["logNe"]
            .median()
            .reset_index()
        )

    # Datastream variant: prefer NeDynamics logs; fall back to spline interpolation
    hpd_intervals_datastream = None
    has_nedynamics = args.nedynamics_deme1 or args.nedynamics_deme2
    if has_nedynamics:
        logger.info(
            "Using NeDynamics log files for datastreams - skipping interpolation"
        )
        nedynamics_list = []
        for deme_file, deme_idx in [
            (args.nedynamics_deme1, 0),
            (args.nedynamics_deme2, 1),
        ]:
            if deme_file:
                ne_deme = process_nedynamics_log(
                    deme_file,
                    deme_idx,
                    tree_height_datastream,
                    args.burnin,
                    rateshifts=gridpointshifts,
                )
                if not ne_deme.empty:
                    nedynamics_list.append(ne_deme)
        if nedynamics_list:
            hpd_intervals_datastream = pd.concat(nedynamics_list, ignore_index=True)
    else:
        hpd_intervals_datastream = interpolate_skyline(
            skyline_long_datastream,
            time_grid,
            interpolation_method="cubic_normal_spline",
        )

    # Cumulative incidence HPD from dedicated logs
    cumulative_incidence_hpd = None
    cuminc_files = [
        (args.cumulative_incidence_deme1, 0),
        (args.cumulative_incidence_deme2, 1),
    ]
    cuminc_list = []
    for cuminc_file, deme_idx in cuminc_files:
        if cuminc_file:
            cuminc = process_cumulative_incidence_log(
                cuminc_file,
                deme_idx,
                tree_height_datastream,
                rateshifts=gridpointshifts,
            )
            if not cuminc.empty:
                cuminc_list.append(cuminc)
    if cuminc_list:
        cumulative_incidence_hpd = pd.concat(cuminc_list, ignore_index=True)

    return {
        "hpd_original": hpd_intervals_original,
        "hpd_datastream": hpd_intervals_datastream,
        "cumulative_incidence_hpd": cumulative_incidence_hpd,
        "max_time": max_time,
        "trajectory_data": trajectory_data,
    }


def _validate_all_hpd(
    hpd_intervals, cumulative_incidence_hpd, trajectory_data, params_df, out_prefix
):
    """Run HPD validation for prevalence, Ne, and cumulative incidence.

    Returns:
        dict with keys: validation_data_datastreams_prevalence,
        validation_data_datastreams_ne, validation_data_datastreams_cumIncidence,
        validation_data_original_ne.
    """
    validation_prevalence = validate_hpd_intervals(
        hpd_intervals["hpd_datastream"],
        trajectory_data,
        out_prefix + "_datastreams",
        validation_type="prevalence",
    )
    validation_ne = validate_hpd_intervals(
        hpd_intervals["hpd_datastream"],
        trajectory_data,
        out_prefix + "_datastreams",
        validation_type="ne",
        params_df=params_df,
    )
    validation_cuminc = validate_hpd_intervals(
        cumulative_incidence_hpd,
        trajectory_data,
        out_prefix + "_datastreams",
        validation_type="cumulative_incidence",
    )

    validation_original_ne = None
    hpd_original = hpd_intervals["hpd_original"]
    if not hpd_original.empty and params_df is not None and not params_df.empty:
        validation_original_ne = validate_hpd_intervals(
            hpd_original,
            trajectory_data,
            out_prefix + "_original",
            validation_type="ne",
            params_df=params_df,
        )

    return {
        "validation_data_datastreams_prevalence": validation_prevalence,
        "validation_data_datastreams_ne": validation_ne,
        "validation_data_datastreams_cumIncidence": validation_cuminc,
        "validation_data_original_ne": validation_original_ne,
    }


def prepare_skyline_plot_data(args):
    """Load and compute all data needed for skyline plots.

    Orchestrates file loading, BEAST log parsing, HPD interval computation,
    and validation. Returns a dict consumed by ``plot_skyline_ne``,
    ``plot_trajectory_compartments``, and related plotting functions.
    """
    # 1. Load input files (trajectory, datastreams, params, deme switches)
    inputs = _load_input_files(args)

    # 2. Parse BEAST logs and transform skyline columns
    logs = _parse_beast_logs(args, inputs["mostrecent_sample_t"])

    # 3. Compute HPD intervals (interpolate skylines, process NeDynamics)
    hpd_intervals = _compute_hpd_intervals(args, logs, inputs)
    # _compute_hpd_intervals may clip trajectory_data to max_time
    inputs["trajectory_data"] = hpd_intervals["trajectory_data"]

    # 4. Validate HPD intervals against ground truth
    validations = _validate_all_hpd(
        hpd_intervals,
        hpd_intervals["cumulative_incidence_hpd"],
        inputs["trajectory_data"],
        inputs["params_df"],
        args.out_prefix,
    )

    diff_treeroot_start = (
        inputs["mostrecent_sample_t"]
        - logs["tree_height_datastream"]["Tree.height"].max()
    )

    return {
        "hpd_original": hpd_intervals["hpd_original"],
        "hpd_datastream": hpd_intervals["hpd_datastream"],
        "max_time": hpd_intervals["max_time"],
        "trajectory_data": inputs["trajectory_data"],
        "case_counts_data": inputs["case_counts_data"],
        "seroprevalence_data": inputs["seroprevalence_data"],
        "wastewater_data": inputs["wastewater_data"],
        "expected_ne_data": inputs["expected_ne_data"],
        "cumulative_incidence_hpd": hpd_intervals["cumulative_incidence_hpd"],
        **validations,
        "diff_treeroot_start": diff_treeroot_start,
        "starting_deme": inputs["starting_deme"],
        "deme_popsizes": inputs["deme_popsizes"],
        "params_df": inputs["params_df"],
        "deme_switches_df": inputs["deme_switches_df"],
        "log_content_original": logs["log_content_original"],
        "log_content_datastream": logs["log_content_datastream"],
        "rateshifts_datastream": logs["rateshifts_datastream"],
        "gridpointshifts_datastream": logs["gridpointshifts_datastream"],
    }


def main():
    """Main function to execute the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = parse_arguments()
    data = prepare_skyline_plot_data(args)

    data["expected_ne_data"].to_csv(f"{args.out_prefix}_expected_ne.csv", index=False)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    print(data["log_content_datastream"].columns, data["log_content_datastream"].empty)
    print(data["log_content_original"].columns, data["log_content_original"].empty)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    if not data["log_content_datastream"].empty:
        print("HEEEEEERRREEEEE")
        save_datastream_params_hpd_validation_csv(
            data["log_content_datastream"],
            data["params_df"],
            args.out_prefix,
        )

    if (
        not data["log_content_original"].empty
        and not data["log_content_datastream"].empty
    ):
        save_migration_rates_hpd_csv(
            data["log_content_original"],
            data["log_content_datastream"],
            args.out_prefix,
            params_df=data["params_df"],
            deme_switches_df=data["deme_switches_df"],
            starting_deme=data["starting_deme"],
        )

    if not data["hpd_original"].empty:
        output_file = f"{args.out_prefix}_comparison_originalvsdatastream.png"
        plot_skyline_ne(
            data["hpd_original"],
            data["hpd_datastream"],
            output_file=output_file,
            max_time=data["max_time"],
            trajectory_data=data["trajectory_data"],
            case_counts_data=data["case_counts_data"],
            seroprevalence_data=data["seroprevalence_data"],
            wastewater_data=data["wastewater_data"],
            expected_ne_data=data["expected_ne_data"],
            cumulative_incidence_hpd=data["cumulative_incidence_hpd"],
            validation_data_datastreams_prevalence=data[
                "validation_data_datastreams_prevalence"
            ],
            validation_data_datastreams_ne=data["validation_data_datastreams_ne"],
            validation_data_datastreams_cumIncidence=data[
                "validation_data_datastreams_cumIncidence"
            ],
            validation_data_original_ne=data["validation_data_original_ne"],
            time_unit="days",
            # diff_treeroot_start=data["diff_treeroot_start"],
            case_counts_P1=False,
            starting_deme=data["starting_deme"],
            deme_popsizes=data["deme_popsizes"],
        )

        output_file = f"{args.out_prefix}_comparison_originalvsdatastream_log.png"
        plot_skyline_ne(
            data["hpd_original"],
            data["hpd_datastream"],
            output_file=output_file,
            max_time=data["max_time"],
            trajectory_data=data["trajectory_data"],
            case_counts_data=data["case_counts_data"],
            seroprevalence_data=data["seroprevalence_data"],
            wastewater_data=data["wastewater_data"],
            expected_ne_data=data["expected_ne_data"],
            cumulative_incidence_hpd=data["cumulative_incidence_hpd"],
            validation_data_datastreams_prevalence=data[
                "validation_data_datastreams_prevalence"
            ],
            validation_data_datastreams_ne=data["validation_data_datastreams_ne"],
            validation_data_datastreams_cumIncidence=data[
                "validation_data_datastreams_cumIncidence"
            ],
            validation_data_original_ne=data["validation_data_original_ne"],
            show_logscale=True,
            time_unit="days",
            # diff_treeroot_start=data["diff_treeroot_start"],
            case_counts_P1=False,
            starting_deme=data["starting_deme"],
            deme_popsizes=data["deme_popsizes"],
        )

        raw_output = f"{args.out_prefix}_original_skyline_ne_raw.csv"
        data["hpd_original"].to_csv(raw_output, index=False)
        logger.info("HPD intervals of MASCOT original saved to %s", raw_output)
        raw_output = f"{args.out_prefix}_datastreams_skyline_ne_raw.csv"
        data["hpd_datastream"].to_csv(raw_output, index=False)
        logger.info("HPD intervals of MASCOT datastreams saved to %s", raw_output)
    else:
        logger.warning("No SkylineNe data found in the log file.")


if __name__ == "__main__":
    main()
