#!/usr/bin/env python3
"""Simulate wastewater, seroprevalence, and case-count datastreams from epidemic trajectories.

Data-generating model (abstracted):

Let I(t) = infectious prevalence at time t (individuals).
Let C(t) = cumulative incidence at time t (individuals ever infected).
Let N = population size.

--- Wastewater concentration ---
  Input: I(t)
  Time shift: t_ww = t + detect_delay (days)
  Sampling: indices closest to targets 0, 7, 14, ... every sample_frequency days

  μ_real = (I / N) · scaling_factor  (concentration scale, cp/g)
  μ_ln   = log(μ_real + ε) - σ²/2   (mean on log scale; ε=1e-2 avoids log(0))

  ww ~ LogNormal(μ_ln, σ)

--- Seroprevalence ---
  Input: C(t)
  Time shift: t_sp = t + detect_delay (days)
  Sampling: exactly 3 timepoints (0–20%, 40–60%, 80–100% of time range)

  p = clip(scaling_factor · C/N, ε, 1−ε)
  n_tested ~ DiscreteUniform(10, min(1000, N))
  n_positive ~ Binomial(n_tested, p)

  seroprevalence = n_positive / n_tested

--- Case counts ---
  Input: I(t)
  Time shift: t_cc = t + detect_delay (days)
  Sampling: indices closest to targets 0, 7, 14, ... every sample_frequency days

  μ = I · scaling_factor  (expected reported cases)
  λ ~ Gamma(shape=1/α, scale=μ·α)   (Gamma–Poisson = Negative Binomial)
  Y ~ Poisson(λ)

  case_counts = Y
  (Variance = μ + α·μ²; α = nb_dispersion)
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import get_outbreak_start_deme, t_first_infected_in_deme

# =============================================================
# Constants
# =============================================================
DEFAULT_SEED: int = 42


def plot_trajectories(
    trajectories,
    ax,
    xaxis="t",
    yaxis="value",
    xlabel="Time",
    ylabel="Number of individuals",
    colors=None,
    linewidth=1,
    legend=False,
):
    """Plot time series trajectories for selected populations.

    Args:
        trajectories (pd.DataFrame): Long-format dataframe with at least columns
            ['Sample', xaxis, yaxis, 'population'].
        ax (matplotlib.axes.Axes): Axis to draw on.
        xaxis (str): Column name for x values (default 't').
        yaxis (str): Column name for y values (default 'value').
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        colors (Optional[Dict[str, str]]): Mapping from population to color.
        linewidth (float): Line width.
        legend (bool): Whether to show legend.

    """
    if colors is None:
        colors = {"S": "blue", "I": "orange", "Rh": "grey"}  # , 'sample': 'green'}
    ts = np.linspace(0, trajectories[xaxis].max())
    for sample_idx in np.arange(trajectories.Sample.max() + 1):
        tmp = trajectories.loc[trajectories.Sample == sample_idx]

        for pop, color in colors.items():
            a = tmp.loc[tmp.population == pop]
            if sample_idx == 0:
                ax.plot(
                    a[xaxis].values,
                    a[yaxis].values,
                    color=color,
                    linewidth=linewidth,
                    alpha=0.5,
                    label=pop,
                )
            else:
                ax.plot(
                    a[xaxis].values,
                    a[yaxis].values,
                    color=color,
                    linewidth=linewidth,
                    alpha=0.5,
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()


def find_closest_timepoints(t, interval=7):
    """Find indices of time points closest to the left of fixed-interval targets.

    Args:
        t (np.ndarray or pd.Series): Time values.
        interval (int): Interval size for target time points.

    Returns:
        tuple: (target_times, selected_indices) - equally sized arrays where
               selected_indices[i] is the index of the closest time point <= target_times[i].
               Duplicate indices are allowed.
    """
    # Generate target timepoints
    quarter_day = 0.25 / 365
    all_target_times = np.arange(0, t.max() + quarter_day, interval)

    # Initialize arrays to store results
    target_times = []
    selected_indices = []

    # For each target time, find the closest measurement to the left (smaller or equal)
    for target in all_target_times:
        # Find indices where time is <= target
        valid_mask = t <= target
        if not np.any(valid_mask):
            continue  # Skip if no valid time points to the left

        # Among valid times, find the one closest to target (i.e., the maximum)
        valid_indices = np.where(valid_mask)[0]
        idx = valid_indices[np.argmax(t[valid_mask])]

        # Append to both arrays (duplicates allowed)
        target_times.append(target)
        selected_indices.append(idx)

    return np.array(target_times), np.array(selected_indices)


def find_three_percentile_samples(
    t,
    rng: Optional[np.random.Generator] = None,
    mostrecent_sample_t: Optional[float] = None,
):
    """Find three sample indices at different time percentiles.

    Randomly samples indices uniformly from within each specified percentile range.

    Args:
        t (np.ndarray or pd.Series): Time values.
        rng: Optional numpy random generator. If None, uses default_rng(DEFAULT_SEED).
        mostrecent_sample_t: Optional float for the time of the most recent sample. If None, uses the maximum time.
    Returns:
        np.ndarray: Indices of three chosen time points:
            - One randomly sampled within first 20% of time range (0-20%)
            - One randomly sampled within 40-60% of time range
            - One randomly sampled within 80-100% of time range
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)

    t_array = np.array(t)
    t_min = t_array.min()
    t_max = t_array.max()
    if mostrecent_sample_t is not None:
        t_max = mostrecent_sample_t
    t_range = t_max - t_min
    selected_indices = []

    # Sample 1: randomly from within first 20% (0-20%)
    mask1 = (t_array >= t_min) & (t_array <= t_min + 0.2 * t_range)
    if mask1.any():
        candidates1 = np.where(mask1)[0]
        idx1 = rng.choice(candidates1)
    else:
        # Fallback: pick first index
        idx1 = 0
    selected_indices.append(idx1)

    # Sample 2: randomly from within 40-60%
    mask2 = (t_array >= t_min + 0.4 * t_range) & (t_array <= t_min + 0.6 * t_range)
    if mask2.any():
        candidates2 = np.where(mask2)[0]
        # Ensure it's different from idx1
        candidates2_filtered = candidates2[candidates2 != idx1]
        if len(candidates2_filtered) > 0:
            idx2 = rng.choice(candidates2_filtered)
        else:
            idx2 = candidates2[0]  # Fallback if only one candidate
    else:
        # Fallback: pick closest to 50% of range
        idx2 = np.abs(t_array - (t_min + 0.5 * t_range)).argmin()
    selected_indices.append(idx2)

    # Sample 3: randomly from within 80-100%
    mask3 = (t_array >= t_min + 0.8 * t_range) & (t_array <= t_max)
    if mask3.any():
        candidates3 = np.where(mask3)[0]
        # Ensure it's different from previous indices
        candidates3_filtered = candidates3[~np.isin(candidates3, selected_indices)]
        if len(candidates3_filtered) > 0:
            idx3 = rng.choice(candidates3_filtered)
        else:
            idx3 = candidates3[0]  # Fallback if no unique candidates
    else:
        # Fallback: pick closest to 90% of range
        idx3 = np.abs(t_array - (t_min + 0.9 * t_range)).argmin()
    selected_indices.append(idx3)

    return np.array(selected_indices)


def simulate_wastewater(
    prevalence: pd.DataFrame,
    N: float,
    detect_delay: int = 0,
    sample_frequency: Optional[int] = 7,
    sigma: float = 0.3,
    scaling_factor: float = 100,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Simulate wastewater concentration measurements from prevalence.

    Prevalence is informing the real median of the lognormal distribution of wastewater concentration.

    Args:
        prevalence: Dataframe with columns ['t', 'value'] for the I population per sample/index.
        N: Population size for normalization.
        detect_delay: Days to shift detection to the right.
        sample_frequency: Interval (days) between samples. If None, all time points used.
        sigma: Lognormal sigma (std on log-scale).
        rng: Optional numpy random generator.

    Returns:
        pd.DataFrame: Input dataframe with added t_wastewater and wastewater_conc columns.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)
    prevalence["t_days_wastewater"] = (
        prevalence["t_days"] + detect_delay
    )  # + rng.normal(detect_delay,0.1, prevalence.shape[0])
    prevalence["wastewater"] = np.nan
    # sampling takes place only every X days
    if sample_frequency is None:
        sampled_time_idx = np.arange(prevalence.shape[0])
        target_times = prevalence["t_days_wastewater"].values
    else:
        target_times, sampled_time_idx = find_closest_timepoints(
            prevalence["t_days_wastewater"], interval=sample_frequency
        )
    # only observe wastewater when at least one person is infected
    infected_mask = prevalence["value"].values[sampled_time_idx] >= 1
    sampled_time_idx = sampled_time_idx[infected_mask]
    target_times = np.asarray(target_times)[infected_mask]
    tmp = prevalence["value"].values[sampled_time_idx]
    # real median of the lognormal distribution of wastewater concentration
    median_real = tmp / N * scaling_factor
    # clip to small epsilon (could be considered a detection limit) to avoid (log(0) but also not too small to avoid numerical issues when fitting with MASCOT (since MASCOT prevalence can't fit 0 prevalence and will instead try to fit super small values)
    median_real = np.clip(median_real, 1e-3, None)

    mu_ln = np.log(median_real)
    # PMMoV normalised pathogen concentration (cp/g)
    conc = rng.lognormal(mu_ln, sigma)
    new_prevalence = prevalence.loc[sampled_time_idx].copy().reset_index(drop=True)
    new_prevalence["wastewater"] = conc
    new_prevalence["t_days_measurement_wastewater"] = target_times
    new_prevalence["t_wastewater"] = (
        new_prevalence["t_days_measurement_wastewater"] / 365
    )
    return new_prevalence


def simulate_seroprevalence(
    cumulativeincidence: pd.DataFrame,
    N: float,
    detect_delay: int = 14,
    mostrecent_sample_t: Optional[float] = None,
    scaling_factor: float = 1,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Simulate seroprevalence measurements based on cumulative prevalence.

    Takes exactly 3 samples:
    - One within the first 20% of t_days_seroprev
    - One within 40-60% of t_days_seroprev
    - One within 80-100% of t_days_seroprev

    Args:
        cumulativeincidence: Dataframe with columns ['t', 'value'] for the cumulative incidence per sample/index.
        N: Population size for normalization.
        detect_delay: Days to shift detection to the right (e.g., 14 days).
        mostrecent_sample_t: Optional float for the time of the most recent sample. If None, uses the maximum time.
        scaling_factor: Scaling factor for the seroprevalence.
        rng: Optional numpy random generator.
    Returns:
        pd.DataFrame: Input dataframe with added t_seroprevalence and seroprevalence columns.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)
    cumulativeincidence["t_days_seroprev"] = (
        cumulativeincidence["t_days"] + detect_delay
    )
    cumulativeincidence["seroprevalence"] = np.nan
    # sampling: one within first 20%, one within 40-60%, one within 80-100% of t_days_seroprev
    sampled_time_idx = find_three_percentile_samples(
        cumulativeincidence["t"], rng=rng, mostrecent_sample_t=mostrecent_sample_t
    )
    tmp = cumulativeincidence["value"].values[sampled_time_idx]
    prop_exposed_population = tmp / N
    cumhazard = -1 * np.log(1.0 - prop_exposed_population)
    prop_exposed_testpop = 1 - np.exp(-1 * scaling_factor * cumhazard)
    # sample number of people to be tested uniformly
    num_people_to_test = rng.choice(
        np.arange(10, min(1000, N)), size=len(sampled_time_idx)
    )
    # sample number of people that will show antibodies based on percentage of exposed
    # Calculate hazard-scaled seroprevalence:
    p = prop_exposed_testpop
    # make sure p is [0,1]
    p = np.clip(p, 0 + 1e-16, 1 - 1e-16)
    num_people_with_antibodies = rng.binomial(num_people_to_test, p)
    new_cumulativeincidence = (
        cumulativeincidence.loc[sampled_time_idx].copy().reset_index(drop=True)
    )
    new_cumulativeincidence["seroprevalence"] = (
        num_people_with_antibodies / num_people_to_test
    )
    new_cumulativeincidence["seroprevalence_numpeopletested"] = num_people_to_test
    new_cumulativeincidence["seroprevalence_numpeoplewithantibodies"] = (
        num_people_with_antibodies
    )
    new_cumulativeincidence["t_days_measurement_seroprevalence"] = (
        new_cumulativeincidence["t_days_seroprev"].round().astype(int)
    )

    new_cumulativeincidence["t_seroprevalence"] = (
        new_cumulativeincidence["t_days_measurement_seroprevalence"] / 365
    )
    return new_cumulativeincidence


def simulate_case_counts(
    prevalence: pd.DataFrame,
    N: float,
    detect_delay: int = 0,
    sample_frequency: Optional[int] = 7,
    scaling_factor: float = 0.1,
    nb_dispersion: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Simulate reported case counts from prevalence with detection probability.

    Args:
        prevalence: Dataframe with columns ['t', 'value'] for the I population per sample/index.
        N: Population size (unused in current logic but retained for consistency).
        detect_delay: Days to shift detection to the right.
        sample_frequency: Interval (days) between samples. If None, all time points used.
        scaling_factor: Probability of detecting a case (mean scaling).
        nb_dispersion: Negative binomial dispersion parameter alpha (>0). Variance = mu + alpha * mu^2.
        rng: Optional numpy random generator.

    Returns:
        pd.DataFrame: Input dataframe with added t_case_counts and case_counts columns.
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)
    if nb_dispersion <= 0:
        raise ValueError("nb_dispersion must be > 0.")
    prevalence["t_days_case_counts"] = prevalence["t_days"] + detect_delay
    prevalence["case_counts"] = np.nan

    # Determine which time points to sample
    if sample_frequency is None:
        sampled_time_idx = np.arange(prevalence.shape[0])
    else:
        target_times, sampled_time_idx = find_closest_timepoints(
            prevalence["t_days_case_counts"], interval=sample_frequency
        )

    # Scale prevalence by detection probability and round to nearest integer
    tmp = prevalence["value"].values[sampled_time_idx]
    mu = tmp * scaling_factor
    # Sample Negative Binomial via Gamma-Poisson mixture: lambda ~ Gamma(k, scale=mu/k); y ~ Poisson(lambda)
    # Variance: mu + alpha * mu^2
    # larger alpha means variance -> Inf -> more variability in case counts
    # smaller alpha means variance -> mu -> less variability in case counts
    alpha = nb_dispersion
    # Avoid division by zero in scale; k is validated > 0 above
    lam = rng.gamma(shape=1 / alpha, scale=mu * alpha)
    case_counts = rng.poisson(lam).astype(int)
    new_prevalence = prevalence.loc[sampled_time_idx].copy().reset_index(drop=True)
    new_prevalence["t_days_measurement_case_counts"] = target_times
    new_prevalence["case_counts"] = case_counts
    new_prevalence["t_case_counts"] = (
        new_prevalence["t_days_measurement_case_counts"] / 365
    )
    return new_prevalence


def get_first_detection_t_per_deme(
    trajectories_prevalence_I: pd.DataFrame,
    threshold: float = 1.0,
) -> Dict[Tuple[int, int], float]:
    """First time (in years) infection is detected per (Sample, index).

    Only demes with at least one time point where I >= threshold are included.
    Demes that never have infection are omitted from the returned dict and
    should be excluded from exported datastreams when constraining to first detection.

    Args:
        trajectories_prevalence_I: DataFrame with population "I", columns Sample, index, t, value.
        threshold: Minimum prevalence (value) to count as "detected".

    Returns:
        Dict mapping (Sample, index) -> t_first (years). Only entries for demes
        that have at least one row with value >= threshold.
    """
    out: Dict[Tuple[int, int], float] = {}
    for (sample, index), group in trajectories_prevalence_I.groupby(
        ["Sample", "index"]
    ):
        above = group.loc[group["value"] >= threshold, "t"]
        if above.empty:
            continue
        out[(int(sample), int(index))] = float(above.min())
    return out


def plot_trajectory_compartments(
    trajectories_prevalence: pd.DataFrame,
    case_counts: pd.DataFrame,
    wastewater_prevalence: pd.DataFrame,
    seroprevalence: pd.DataFrame,
    trajectories_incidence: pd.DataFrame,
    ndemes: int,
    fig_png: Path,
    max_time: Optional[float] = None,
) -> None:

    if max_time is not None:
        trajectories_prevalence = trajectories_prevalence.loc[
            trajectories_prevalence["t"] <= max_time
        ]
        case_counts = case_counts.loc[case_counts["t_case_counts"] <= max_time]
        wastewater_prevalence = wastewater_prevalence.loc[
            wastewater_prevalence["t_wastewater"] <= max_time
        ]
        seroprevalence = seroprevalence.loc[
            seroprevalence["t_seroprevalence"] <= max_time
        ]
        trajectories_incidence = trajectories_incidence.loc[
            trajectories_incidence["t"] <= max_time
        ]
    fig, ax = plt.subplots(4, ndemes + 1, figsize=(20, 14), sharex=True)

    for i in range(ndemes):
        tmp = trajectories_prevalence.loc[trajectories_prevalence["index"] == i]
        plot_trajectories(tmp, ax[0, i], linewidth=2, ylabel=None)
        ax[0, i].set_title(f"Location {i+1}", fontsize=16)
        ax[0, i].tick_params(axis="both", labelsize=14)
        ax[0, i].set_xlabel("Time", fontsize=16)
    ax[0, 0].set_ylabel("# individuals", fontsize=16)
    ax[0, 0].legend(fontsize=14)

    tmp = trajectories_prevalence.loc[trajectories_prevalence.population == "sample"]
    for sample_idx in np.arange(tmp.Sample.max() + 1):
        a = tmp.loc[(tmp.Sample == sample_idx) & (tmp["index"] == 0)]
        if sample_idx == 0:
            ax[0, -1].plot(
                a.t.values,
                a.value.values,
                color="green",
                linewidth=2,
                alpha=0.5,
                label="sample",
            )
        else:
            ax[0, -1].plot(
                a.t.values, a.value.values, color="green", linewidth=2, alpha=0.5
            )
    ax[0, -1].legend(fontsize=14)
    ax[0, -1].tick_params(axis="both", labelsize=14)

    for i in range(ndemes):
        tmp = case_counts.loc[case_counts["index"] == i].dropna()
        plot_trajectories(
            tmp,
            ax[1, i],
            xaxis="t_case_counts",
            yaxis="case_counts",
            xlabel="Time",
            ylabel=None,
            colors={"I": "purple"},
            linewidth=2,
        )
        ax[1, i].tick_params(axis="both", labelsize=14)
        ax[1, i].set_xlabel("Time", fontsize=16)
    ax[1, 0].set_ylabel("case counts", fontsize=16)

    for i in range(ndemes):
        tmp = wastewater_prevalence.loc[wastewater_prevalence["index"] == i].dropna()
        plot_trajectories(
            tmp,
            ax[2, i],
            xaxis="t_wastewater",
            yaxis="wastewater",
            xlabel="Time",
            ylabel=None,
            colors={"I": "brown"},
            linewidth=2,
        )
        ax[2, i].tick_params(axis="both", labelsize=14)
        ax[2, i].set_xlabel("Time", fontsize=16)
    ax[2, 0].set_ylabel("viral concentration", fontsize=16)

    for i in range(ndemes):
        tmp = seroprevalence.loc[seroprevalence["index"] == i].dropna()
        plot_trajectories(
            tmp,
            ax[3, i],
            xaxis="t_seroprevalence",
            yaxis="seroprevalence",
            xlabel="Time",
            ylabel=None,
            colors={"NewInfectCount": "red"},
            linewidth=2,
        )
        ax[3, i].set_xlabel("Time", fontsize=16)
        ax[3, i].tick_params(axis="both", labelsize=14)
    ax[3, 0].set_ylabel("Seroprevalence", fontsize=16)

    for deme, df in trajectories_incidence.groupby("index"):
        for sample_idx in np.arange(df.Sample.max() + 1):
            a = df.loc[(df.Sample == sample_idx)]
            ax[3, -1].plot(
                a["t"].values,
                a["value"].values,
                linewidth=2,
                alpha=0.5,
                label=f"deme {deme}",
            )

    ax[3, -1].set_xlabel("Time", fontsize=16)
    ax[3, -1].set_ylabel("Incidence", fontsize=16)
    ax[3, -1].legend(fontsize=14)
    ax[3, -1].tick_params(axis="both", labelsize=14)

    plt.tight_layout()
    plt.savefig(fig_png)


def parse_json_to_dataframe(json_path: Path) -> pd.DataFrame:
    """Parse JSON format and convert to long-format DataFrame.

    Args:
        json_path: Path to the JSON file.

    Returns:
        pd.DataFrame with columns ['Sample', 't', 'population', 'index', 'value'].
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract time array
    t_array = np.array(data["t"])
    n_timepoints = len(t_array)

    # Convert each population to long format
    rows = []

    # Get all population names (excluding metadata)
    population_names = [key for key in data.keys() if key not in ["t", "sim"]]

    # Get number of demes
    n_demes = len(data[population_names[0]]) if population_names else 1

    for pop_name in population_names:
        pop_data = data[pop_name]

        # Each population has a list of arrays (one array per deme)
        for index in range(n_demes):
            # Each array contains values for each time point
            if isinstance(pop_data, list) and index < len(pop_data):
                values = pop_data[index]

                # values is an array of length n_timepoints
                for time_idx in range(min(len(values), len(t_array))):
                    rows.append(
                        {
                            "Sample": 0,  # Single sample per simulation
                            "t": t_array[time_idx],
                            "population": pop_name,
                            "index": index,
                            "value": values[time_idx],
                        }
                    )

    df = pd.DataFrame(rows)
    return df


def export_datastream(
    df, datacol, mostrecent_sample_t, column_keep=[], filterbymostrecentsample=True
):
    if df.empty:
        return pd.DataFrame(columns=column_keep) if column_keep else pd.DataFrame()

    df_export = df.loc[~df[datacol].isnull()].copy()
    if filterbymostrecentsample:
        df_export = df_export.loc[df_export[f"t_{datacol}"] <= mostrecent_sample_t]
    # express time relative to most recent sample
    df_export[f"t_{datacol}_frommostrecentsample"] = (
        mostrecent_sample_t - df_export[f"t_{datacol}"]
    )
    df_export[f"t_{datacol}_fromsimstart"] = df_export[f"t_{datacol}"]
    if len(column_keep) > 0:
        df_export = df_export[column_keep]

    return df_export


def _simulate_and_concat(
    grouped,
    simulate_fn,
    first_detection: Dict[Tuple[int, int], float],
    constrain: bool,
) -> pd.DataFrame:
    """Apply *simulate_fn* to each (Sample, index) group and concatenate.

    Handles the common filter-by-first-detection logic shared by all three
    datastream simulation blocks.

    Args:
        grouped: DataFrame GroupBy on ["Sample", "index"].
        simulate_fn: Callable(group_df, sample, index) -> DataFrame.
        first_detection: Map of (sample, deme) → first detection time.
        constrain: Whether to apply first-detection filtering.

    Returns:
        Concatenated DataFrame, or empty DataFrame if no groups produced output.
    """
    results = []
    for (sample, index), group in grouped:
        key = (int(sample), int(index))
        if constrain and key not in first_detection:
            continue
        tmp = group.copy().reset_index(drop=True)
        if constrain:
            tmp = tmp.loc[tmp["t"] >= first_detection[key]].copy().reset_index(drop=True)
        if tmp.empty:
            continue
        result = simulate_fn(tmp, int(sample), int(index))
        results.append(result)
    return pd.concat(results).reset_index(drop=True) if results else pd.DataFrame()


def run_pipeline(
    traj_file: Path,
    params_csv: Path,
    out_prefix: Path,
    constrain_to_first_detection: bool = False,
) -> None:
    """Run the full simulation and plotting pipeline.

    Args:
        traj_file: Path to the input trajectory file (.traj) or JSON file to read.
        params_csv: Path to the input parameters CSV file.
        out_prefix: Prefix for the output files.
        constrain_to_first_detection: If True, restrict wastewater/seroprevalence/case_counts
            to time points from first detection of infection in the deme onward; demes that
            never have infection are omitted from exports.
    """
    # Read input
    if not traj_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
    if not params_csv.exists():
        raise FileNotFoundError(f"Parameters CSV file not found: {params_csv}")

    if traj_file.suffix == ".json":
        trajectories_prevalence = parse_json_to_dataframe(traj_file)
    else:
        trajectories_prevalence = pd.read_table(traj_file, sep="\t")

    params = pd.read_csv(params_csv)
    ds_params = params[params["parameter"].str.startswith("ds_")]

    pop_sizes_demes = {}
    if "population_size" in params["parameter"].unique():
        tmp = params.loc[params["parameter"] == "population_size"]
        for deme, df in tmp.groupby("deme"):
            pop_sizes_demes[int(deme)] = df["value"].values[0]
    else:
        raise ValueError("No population_size parameter found in parameters CSV")

    # Determine the time (in years) of the most recent sample
    sample_times = trajectories_prevalence.loc[
        (trajectories_prevalence.population == "sample")
        & (trajectories_prevalence["index"] == 0)
    ].sort_values("t")

    if not sample_times.empty:
        max_sample_value = sample_times["value"].values.max()
        tmp = sample_times.loc[sample_times["value"] == max_sample_value]
        mostrecent_sample_t = tmp["t"].values[0]
    else:
        mostrecent_sample_t = trajectories_prevalence["t"].max()

    trajectories_prevalence["t_days"] = trajectories_prevalence["t"] * 365
    ndemes = int(trajectories_prevalence["index"].max()) + 1

    trajectories_incidence = trajectories_prevalence.loc[
        trajectories_prevalence.population == "NewInfectCount"
    ]
    trajectories_prevalence_I = trajectories_prevalence.loc[
        trajectories_prevalence.population == "I"
    ]

    # --- Simulation metadata CSV (start deme + first-I times per deme) ---
    sim_name = Path(out_prefix).name
    start_deme = get_outbreak_start_deme(trajectories_prevalence)
    secondary_deme = 1 - start_deme
    sim_meta_rows = [
        {
            "simulation": sim_name,
            "deme": start_deme,
            "deme_type": "start",
            "t_of_first_infect": t_first_infected_in_deme(trajectories_prevalence_I, start_deme),
        },
        {
            "simulation": sim_name,
            "deme": secondary_deme,
            "deme_type": "secondary",
            "t_of_first_infect": t_first_infected_in_deme(trajectories_prevalence_I, secondary_deme),
        },
    ]
    pd.DataFrame(sim_meta_rows).to_csv(f"{out_prefix}_sim_metadata.csv", index=False)

    first_detection: Dict[Tuple[int, int], float] = {}
    if constrain_to_first_detection:
        first_detection = get_first_detection_t_per_deme(trajectories_prevalence_I)

    constrain = constrain_to_first_detection

    # --- Wastewater ---
    sigma = ds_params.loc[ds_params["parameter"] == "ds_ww_sigma", "value"].values[0]
    if sigma is None:
        raise ValueError("No sigma parameter found in parameters CSV")
    rng_ww = np.random.default_rng(DEFAULT_SEED)

    def _sim_wastewater(tmp, sample, index):
        scaling = ds_params.loc[
            (ds_params["parameter"] == "ds_ww_scaling")
            & (ds_params["deme"] == str(index)),
            "value",
        ].values[0]
        return simulate_wastewater(
            tmp, pop_sizes_demes[index],
            detect_delay=0, sample_frequency=1,
            sigma=sigma, scaling_factor=scaling, rng=rng_ww,
        )

    wastewater_prevalence = _simulate_and_concat(
        trajectories_prevalence_I.groupby(["Sample", "index"]),
        _sim_wastewater, first_detection, constrain,
    )

    # --- Seroprevalence ---
    rng_sp = np.random.default_rng(DEFAULT_SEED)

    def _sim_seroprevalence(tmp, sample, index):
        scaling = ds_params.loc[
            (ds_params["parameter"] == "ds_sp_scaling")
            & (ds_params["deme"] == str(index)),
            "value",
        ].values[0]
        return simulate_seroprevalence(
            tmp, pop_sizes_demes[index],
            detect_delay=0, mostrecent_sample_t=mostrecent_sample_t,
            scaling_factor=scaling, rng=rng_sp,
        )

    seroprevalence = _simulate_and_concat(
        trajectories_incidence.groupby(["Sample", "index"]),
        _sim_seroprevalence, first_detection, constrain,
    )

    # --- Case counts ---
    rng_cc = np.random.default_rng(DEFAULT_SEED)
    dispersion = ds_params.loc[
        ds_params["parameter"] == "ds_cc_dispersion", "value"
    ].values[0]

    def _sim_case_counts(tmp, sample, index):
        N = trajectories_prevalence.loc[
            (trajectories_prevalence.t == 0)
            & (trajectories_prevalence.Sample == sample)
            & (trajectories_prevalence["index"] == index)
        ]["value"].sum()
        scaling = ds_params.loc[
            (ds_params["parameter"] == "ds_cc_scaling")
            & (ds_params["deme"] == str(index)),
            "value",
        ].values[0]
        return simulate_case_counts(
            tmp, N,
            detect_delay=0, sample_frequency=1,
            scaling_factor=scaling, nb_dispersion=dispersion, rng=rng_cc,
        )

    case_counts = _simulate_and_concat(
        trajectories_prevalence_I.groupby(["Sample", "index"]),
        _sim_case_counts, first_detection, constrain,
    )

    # --- Export datastreams ---
    case_counts_export = export_datastream(
        case_counts,
        "case_counts",
        mostrecent_sample_t,
        column_keep=[
            "case_counts",
            "t_case_counts_fromsimstart",
            "t_case_counts_frommostrecentsample",
            "index",
        ],
        filterbymostrecentsample=True,
    )
    seroprevalence_export = export_datastream(
        seroprevalence,
        "seroprevalence",
        mostrecent_sample_t,
        column_keep=[
            "seroprevalence",
            "seroprevalence_numpeopletested",
            "seroprevalence_numpeoplewithantibodies",
            "t_seroprevalence_fromsimstart",
            "t_seroprevalence_frommostrecentsample",
            "index",
        ],
        filterbymostrecentsample=True,
    )
    wastewater_prevalence_export = export_datastream(
        wastewater_prevalence,
        "wastewater",
        mostrecent_sample_t,
        column_keep=[
            "wastewater",
            "t_wastewater_fromsimstart",
            "t_wastewater_frommostrecentsample",
            "index",
        ],
        filterbymostrecentsample=True,
    )

    # Output paths use input basename
    case_counts_csv = f"{out_prefix}_casecounts.csv"
    seroprevalence_csv = f"{out_prefix}_seroprevalence.csv"
    wastewater_prevalence_csv = f"{out_prefix}_wastewater.csv"
    fig_png = f"{out_prefix}_trajectories.png"

    case_counts_export.to_csv(case_counts_csv, index=False)
    seroprevalence_export.to_csv(seroprevalence_csv, index=False)
    wastewater_prevalence_export.to_csv(wastewater_prevalence_csv, index=False)

    if not (case_counts.empty and wastewater_prevalence.empty and seroprevalence.empty):
        plot_trajectory_compartments(
            trajectories_prevalence,
            case_counts,
            wastewater_prevalence,
            seroprevalence,
            trajectories_incidence,
            ndemes,
            fig_png,
            max_time=mostrecent_sample_t,
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Simulate datastreams from trajectories and produce CSV + plots."
    )
    parser.add_argument(
        "--traj_file",
        type=Path,
        help="Path to the input trajectory file (.traj or .json)",
    )
    parser.add_argument(
        "--params_csv", type=Path, help="Path to the input parameters CSV file"
    )
    parser.add_argument("--out_prefix", type=Path, help="Prefix for the output files")
    parser.add_argument(
        "--constrain_to_first_detection",
        action="store_true",
        help="Restrict datastreams to times from first infection detection per deme; omit demes that never have infection.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        args.traj_file,
        args.params_csv,
        args.out_prefix,
        constrain_to_first_detection=args.constrain_to_first_detection,
    )


if __name__ == "__main__":
    main()
