"""Relate case count dynamics to Epsilon/Alpha variant frequencies in California.

Reads GISAID-derived lineage counts (data/USAClusters_data.json) and a
per-deme case count CSV, computes the 7-day running average of total case
counts, finds the peak week, and reports the percentage of the Epsilon
(21C) and Alpha (20I, V1) variants among all sequenced samples at the
first, peak, and last dates covered by the case count file.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

USACLUSTERS_JSON = Path("data/USAClusters_data.json")
EPSILON_KEY = "21C (Epsilon)"
ALPHA_KEY = "20I (Alpha, V1)"


def load_variant_data(json_path):
    with open(json_path) as f:
        data = json.load(f)
    ca = data["countries"]["California"]
    weeks = pd.to_datetime(ca["week"])
    return pd.DataFrame(
        {
            "week": weeks,
            "total_sequences": ca["total_sequences"],
            "epsilon": ca[EPSILON_KEY],
            "alpha": ca[ALPHA_KEY],
        }
    ).sort_values("week").reset_index(drop=True)


def load_case_counts(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    daily_total = df.groupby("date")["case_counts"].sum().sort_index()
    daily_by_deme = df.groupby(["deme", "date"])["case_counts"].sum()
    return daily_total, daily_by_deme


def week_row_for_date(variant_df, date):
    eligible = variant_df[variant_df["week"] <= date]
    if eligible.empty:
        return variant_df.iloc[0]
    return eligible.iloc[-1]


def report_variant_percentages(variant_df, date, label):
    row = week_row_for_date(variant_df, date)
    total = row["total_sequences"]
    epsilon_pct = 100 * row["epsilon"] / total if total else float("nan")
    alpha_pct = 100 * row["alpha"] / total if total else float("nan")
    print(
        f"{label} ({date.date()}, matched week {row['week'].date()}, "
        f"n={int(total)} sequences): "
        f"Epsilon (21C) = {epsilon_pct:.1f}%, Alpha (20I, V1) = {alpha_pct:.1f}%"
    )


def report_epsilon_peak_week(variant_df, first_date, last_date):
    in_range = variant_df[
        (variant_df["week"] >= first_date) & (variant_df["week"] <= last_date)
    ].copy()
    in_range["epsilon_pct"] = 100 * in_range["epsilon"] / in_range["total_sequences"]
    in_range["alpha_pct"] = 100 * in_range["alpha"] / in_range["total_sequences"]
    peak_row = in_range.loc[in_range["epsilon_pct"].idxmax()]
    print(
        f"Epsilon (21C) peaks at {peak_row['epsilon_pct']:.1f}% "
        f"in the week of {peak_row['week'].date()} "
        f"(n={int(peak_row['total_sequences'])} sequences), "
        f"Alpha (20I, V1) = {peak_row['alpha_pct']:.1f}% that same week"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Report Epsilon/Alpha variant frequencies at the first, peak, "
            "and last dates of a case count time series."
        )
    )
    parser.add_argument(
        "case_counts_csv",
        help="Path to a case_counts.csv file with date, case_counts, deme columns",
    )
    args = parser.parse_args()

    variant_df = load_variant_data(USACLUSTERS_JSON)
    daily_total, daily_by_deme = load_case_counts(args.case_counts_csv)

    first_date = daily_total.index.min()
    last_date = daily_total.index.max()

    report_variant_percentages(variant_df, first_date, "First date")
    print()

    for deme in daily_by_deme.index.get_level_values("deme").unique():
        deme_series = daily_by_deme.loc[deme].sort_index()
        rolling_avg = deme_series.rolling(window=7, min_periods=1).mean()
        peak_date = rolling_avg.idxmax()
        print(
            f"{deme}: peak 7-day average case count = "
            f"{rolling_avg.loc[peak_date]:.1f} on {peak_date.date()}"
        )
        report_variant_percentages(variant_df, peak_date, f"  {deme} peak date")

    print()
    report_variant_percentages(variant_df, last_date, "Last date")
    print()
    report_epsilon_peak_week(variant_df, first_date, last_date)


if __name__ == "__main__":
    main()
