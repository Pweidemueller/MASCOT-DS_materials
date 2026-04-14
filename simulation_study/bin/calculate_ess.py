#!/usr/bin/env python3
"""
Calculate Effective Sample Size (ESS) from BEAST log files.

This script reads a BEAST log file, applies burn-in, and calculates ESS
for each parameter using autocorrelation analysis.
"""

import numpy as np
import argparse
import csv


def effective_size(x, max_lag=None):
    """
    Calculate effective sample size using autocorrelation.

    This matches the R effectiveSize function behavior:
    - Demeans the data
    - Calculates autocorrelation up to max_lag
    - Sums positive autocorrelations
    - ESS = N / (1 + 2 * sum)

    Parameters
    ----------
    x : array-like
        Time series data
    max_lag : int, optional
        Maximum lag for autocorrelation calculation. If None, uses
        min(N-1, 10000) where N is the length of x.

    Returns
    -------
    ess : float
        Effective sample size
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    if N < 2:
        return 0.0

    # Demean the data
    x_demeaned = x - np.mean(x)

    # Set max_lag if not provided
    if max_lag is None:
        max_lag = min(N - 1, 10000)
    else:
        max_lag = min(max_lag, N - 1)

    # Calculate autocorrelation function (matching R's acf with type="correlation")
    # Autocorrelation at lag k: r_k = Cov(X_t, X_{t+k}) / Var(X)
    variance = np.var(x_demeaned, ddof=0)
    if variance == 0:
        return float("inf")  # Constant series has infinite ESS

    # Calculate autocorrelations
    acf_lags = []
    for lag in range(1, max_lag + 1):
        covariance = np.sum(x_demeaned[:-lag] * x_demeaned[lag:]) / (N - lag)
        acf_val = covariance / variance
        if acf_val <= 0:
            break
        acf_lags.append(acf_val)

    # Calculate ESS
    pos_sum = sum(acf_lags)
    ess = N / (1 + 2 * pos_sum)

    return ess


def find_ess_threshold(data, threshold=200, max_lag=None):
    """
    Find the minimum number of samples needed to achieve ESS >= threshold.

    Parameters
    ----------
    data : array-like
        Time series data
    threshold : float
        ESS threshold to achieve
    max_lag : int, optional
        Maximum lag for autocorrelation calculation

    Returns
    -------
    n_samples : int or None
        Minimum number of samples needed, or None if threshold not reached
    """
    data = np.asarray(data)
    N = len(data)

    # Check progressively larger sample sizes
    for n in range(100, N + 1, 100):
        ess = effective_size(data[:n], max_lag)
        if ess >= threshold:
            # Refine search in smaller increments
            for n_refined in range(max(100, n - 100), n + 1, 10):
                if effective_size(data[:n_refined], max_lag) >= threshold:
                    return n_refined
            return n

    # Check full dataset
    if effective_size(data, max_lag) >= threshold:
        return N

    return None


def read_log_file(filename):
    """
    Read BEAST log file and extract data.

    Parameters
    ----------
    filename : str
        Path to log file

    Returns
    -------
    header : list
        Column names
    data : numpy.ndarray
        Data matrix (samples x parameters)
    """
    header = None
    data_rows = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue

            if header is None:
                header = line.split("\t")
                continue

            if line:
                values = line.split("\t")
                try:
                    row = [float(v) for v in values]
                    data_rows.append(row)
                except ValueError:
                    continue

    if header is None:
        raise ValueError("No header found in log file")

    if len(data_rows) == 0:
        raise ValueError("No data rows found in log file")

    data = np.array(data_rows)

    if data.shape[1] != len(header):
        raise ValueError(
            f"Data columns ({data.shape[1]}) don't match header ({len(header)})"
        )

    return header, data


def calculate_ess_for_parameters(header, data_burnin, skip_columns=None):
    """
    Calculate ESS for all parameters in the data.

    Parameters
    ----------
    header : list
        Column names
    data_burnin : numpy.ndarray
        Data after burn-in (samples x parameters)
    skip_columns : set, optional
        Set of column names (lowercase) to skip

    Returns
    -------
    ess_results : list of tuples
        List of (parameter_name, ess_value) tuples
    """
    if skip_columns is None:
        skip_columns = {"sample"}

    ess_results = []
    for i, param_name in enumerate(header):
        if param_name.lower() in skip_columns:
            continue
        ess = effective_size(data_burnin[:, i])
        ess_results.append((param_name, ess))

    return sorted(ess_results, key=lambda x: x[0])


def print_ess_table(ess_results):
    """Print ESS results in a formatted table."""
    print("\n" + "=" * 80)
    print(f"{'Parameter':<60} {'ESS':>15}")
    print("=" * 80)
    for param_name, ess in ess_results:
        print(f"{param_name:<60} {ess:>15.2f}")
    print("=" * 80)


def save_ess_table(ess_results, filename, run_name=None):
    """
    Save ESS results to a CSV file.

    Parameters
    ----------
    ess_results : list of tuples
        List of (parameter_name, ess_value) tuples
    filename : str
        Path to output CSV file
    run_name : str, optional
        Name of the BEAST run to include in the output
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        if run_name:
            writer.writerow(["beast_run_name", "parameter_name", "ESS"])
            for param_name, ess in ess_results:
                writer.writerow([run_name, param_name, ess])
        else:
            writer.writerow(["Parameter", "ESS"])
            for param_name, ess in ess_results:
                writer.writerow([param_name, ess])


def find_column_index(header, column_name):
    """Find the index of a column by name (case-insensitive)."""
    for i, name in enumerate(header):
        if name.lower() == column_name.lower():
            return i
    return None


def report_threshold_achievement(data, header, burnin_samples, column_name, threshold):
    """
    Report when ESS threshold was achieved for a specific column.

    Parameters
    ----------
    data : numpy.ndarray
        Full data (before burn-in)
    header : list
        Column names
    burnin_samples : int
        Number of samples to use as burn-in
    column_name : str
        Name of column to check
    threshold : float
        ESS threshold

    Returns
    -------
    message : str
        Formatted message about threshold achievement
    """
    col_idx = find_column_index(header, column_name)
    if col_idx is None:
        return f"{column_name.capitalize()} column not found"

    col_data = data[:, col_idx]
    n_samples = find_ess_threshold(col_data, threshold)

    if n_samples is not None:
        return f"{column_name.capitalize()}: ESS > {threshold} achieved after {n_samples} samples"

    final_ess = effective_size(col_data[burnin_samples:])
    return f"{column_name.capitalize()}: ESS > {threshold} not achieved (final ESS: {final_ess:.2f})"


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Effective Sample Size (ESS) from BEAST log files"
    )
    parser.add_argument("log_file", type=str, help="Path to BEAST log file")
    parser.add_argument(
        "--burnin",
        type=float,
        default=0.1,
        help="Burn-in percentage (0-1, default: 0.1)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=200,
        help="ESS threshold for reporting (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (optional)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name of the BEAST run (for summary table format)",
    )

    args = parser.parse_args()

    # Validate burnin
    if not 0 <= args.burnin < 1:
        parser.error("Burn-in must be in range [0, 1)")

    # Read log file
    print(f"Reading log file: {args.log_file}")
    header, data = read_log_file(args.log_file)

    n_samples = data.shape[0]
    print(f"Found {n_samples} samples with {len(header)} parameters")

    # Apply burn-in
    burnin_samples = int(n_samples * args.burnin)
    data_burnin = data[burnin_samples:]
    n_samples_burnin = data_burnin.shape[0]
    print(f"After {args.burnin*100:.1f}% burn-in: {n_samples_burnin} samples")

    # Calculate ESS for all parameters
    print("\nCalculating ESS for each parameter...")
    ess_results = calculate_ess_for_parameters(header, data_burnin)
    print_ess_table(ess_results)

    # Save to CSV if output file specified
    if args.output:
        save_ess_table(ess_results, args.output, run_name=args.run_name)
        print(f"\nESS results saved to: {args.output}")

    # Report threshold achievement for posterior and prior
    print(f"\nFinding when ESS > {args.threshold} was achieved...")
    for column_name in ["posterior", "prior"]:
        message = report_threshold_achievement(
            data, header, burnin_samples, column_name, args.threshold
        )
        print(message)


if __name__ == "__main__":
    main()
