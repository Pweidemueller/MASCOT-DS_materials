"""
Filter datastreams (case counts, seroprevalence, wastewater) by counties and date range.

Loads three datastreams, filters by counties of interest and user-specified date range,
then writes per-deme (per-county) CSV files into a user-specified output directory.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Ensure repo root is on path for sibling import
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from bin.filter_gisaid_metadata import COUNTIES_OF_INTEREST

# Final available NCHS Commercial Laboratory Seroprevalence Survey download (retrieved 2026-03-02).
# Update this constant if a newer survey file is used.
SEROPREVALENCE_FILENAME = (
    "Nationwide_Commercial_Laboratory_Seroprevalence_Survey_20260302.csv"
)

# Seroprevalence column names (long names from CSV)
COL_N_ANTIN_ALL_AGES = "n [Anti-N, All Ages Cumulative Prevalence, Rounds 1-30 only]"
COL_RATE_ANTIN_ALL_AGES = (
    "Rate (%) [Anti-N, All Ages Cumulative Prevalence, Rounds 1-30 only]"
)

# Base column set for seroprevalence (plus "All Ages" columns without Male/Female)
SERO_BASE_COLUMNS = [
    "Site",
    "Date Range of Specimen Collection",
    "Round",
    "Catchment FIPS Code Description",
    "Catchment Area Description",
    "Catchment population",
]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Filter datastreams by counties and date range; write per-county CSVs."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for per-county CSV files (created if missing).",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start of date range (inclusive).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--metadata",
        type=Path,
        help=(
            "Path to metadata CSV. If provided, the end date is inferred as the "
            "maximum value of the 'Collection date' column."
        ),
    )
    group.add_argument(
        "--end-date",
        help="End of date range (inclusive).",
    )
    data_dir = Path(__file__).resolve().parent.parent / "data"
    parser.add_argument(
        "--case-counts",
        type=Path,
        default=data_dir / "covid19cases_test.csv",
        help="Path to case counts CSV.",
    )
    parser.add_argument(
        "--seroprevalence",
        type=Path,
        default=data_dir / SEROPREVALENCE_FILENAME,
        help="Path to seroprevalence CSV.",
    )
    parser.add_argument(
        "--wastewater",
        type=Path,
        default=data_dir / "wastewatersurveillancecalifornia.csv",
        help="Path to wastewater CSV.",
    )
    parser.add_argument(
        "--counties",
        type=str,
        default=None,
        help="Comma-separated county names; if not set, use COUNTIES_OF_INTEREST.",
    )
    return parser.parse_args()


def county_to_filename(county: str) -> str:
    """Return a filesystem-safe filename stem for a county name (e.g. spaces -> underscores)."""
    return county.replace(" ", "_")


def last_date_of_range(s: str) -> pd.Timestamp:
    """Parse 'Date Range of Specimen Collection': take part after ' - ' and parse as '%b %d, %Y'."""
    part = str(s).split(" - ")[-1].strip()
    return pd.to_datetime(part, format="%b %d, %Y")


def load_and_prepare_case_counts(path: Path) -> pd.DataFrame:
    """Load case counts CSV, parse dates, remove masked rows, cast cases to int. No county/date filter."""
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["cases"] != "Masked") & df["cases"].notna()
    df = df.loc[mask].copy()
    df["cases"] = df["cases"].astype(int)
    return df


def load_and_prepare_seroprevalence(path: Path) -> pd.DataFrame:
    """Load seroprevalence CSV, restrict to Site==CA and Round<=30, select columns, add date. No date filter."""
    df = pd.read_csv(path, low_memory=False)
    df = df[(df["Site"] == "CA") & (df["Round"] <= 30)].copy()
    all_ages_cols = [
        c
        for c in df.columns
        if "All Ages" in c and "Male" not in c and "Female" not in c
    ]
    cols = [c for c in SERO_BASE_COLUMNS if c in df.columns] + all_ages_cols
    df = df[[c for c in cols if c in df.columns]].copy()
    df["date"] = df["Date Range of Specimen Collection"].apply(last_date_of_range)
    return df


def load_and_prepare_wastewater(path: Path) -> pd.DataFrame:
    """Load wastewater CSV, parse dates, apply pcr/hum_frac filters, add pcr_target_normalised. No county/date filter."""
    df = pd.read_csv(path, low_memory=False)
    df["sample_collect_date"] = pd.to_datetime(df["sample_collect_date"])
    mask = (
        (df["pcr_target"] == "sars-cov-2")
        & (df["pcr_gene_target"].isin(["n"]))
        & (
            df["hum_frac_target_mic"].isin(
                ["pepper mild mottle virus", "Pepper mild mottle virus"]
            )
        )
    )
    df = df.loc[mask].copy()
    df["pcr_target_normalised"] = (
        df["pcr_target_avg_conc"] / df["hum_frac_mic_conc"] * 1_000_000
    )

    df = df[df["pcr_target_normalised"] > 0]
    return df


def filter_case_counts_by_counties_and_dates(
    df: pd.DataFrame,
    counties: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Restrict to County areas, given counties, exclude weekends, and date range (inclusive)."""
    df = df[df["area_type"] == "County"].copy()
    pat = "|".join(counties)
    df = df[df["area"].str.contains(pat, na=False)].copy()
    df = df[df["date"].dt.dayofweek < 5].copy()  # 0–4 = Mon–Fri
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    return df


def filter_wastewater_by_counties_and_dates(
    df: pd.DataFrame,
    counties: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Restrict to label_name containing any county and sample_collect_date in [start_date, end_date] (inclusive)."""
    pat = "|".join(counties)
    df = df[df["label_name"].str.contains(pat, na=False)].copy()
    df = df[
        (df["sample_collect_date"] >= start_date)
        & (df["sample_collect_date"] <= end_date)
    ].copy()
    return df


def filter_seroprevalence_by_dates(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Restrict date to [start_date, end_date] (inclusive)."""
    return df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()


def build_case_counts_per_county(
    df: pd.DataFrame, counties: list[str]
) -> dict[str, pd.DataFrame]:
    """For each county, build a DataFrame with columns date, counts (from cases). Empty -> header-only DataFrame."""
    result = {}
    for county in counties:
        sub = df[df["area"].str.contains(county, na=False)].copy()
        if sub.empty:
            result[county] = pd.DataFrame(columns=["date", "case_counts", "deme"])
        else:
            out = sub[["date", "cases"]].rename(columns={"cases": "case_counts"})
            out["deme"] = county
            out = out.sort_values("date").drop_duplicates()
            result[county] = out
    return result


def build_wastewater_per_county(
    df: pd.DataFrame, counties: list[str]
) -> dict[str, pd.DataFrame]:
    """For each county, build a DataFrame with date (sample_collect_date), wastewater (mean per date). Empty -> header-only. If there are multiple sewer catchment areas per county, take the mean of the pcr_target_normalised values."""
    result = {}
    for county in counties:
        sub = df[df["label_name"].str.contains(county, na=False)].copy()
        if sub.empty:
            result[county] = pd.DataFrame(columns=["date", "wastewater", "deme"])
        else:
            agg = (
                sub.groupby("sample_collect_date", as_index=False)[
                    "pcr_target_normalised"
                ]
                .mean()
                .rename(
                    columns={
                        "sample_collect_date": "date",
                        "pcr_target_normalised": "wastewater",
                    }
                )
            )
            agg["deme"] = county
            result[county] = agg.sort_values("date")
    return result


def build_seroprevalence_output(df: pd.DataFrame, counties: list[str]) -> pd.DataFrame:
    """Single DataFrame with date, seroprevalence_numpeopletested, seroprevalence_numpeoplewithantibodies (n * 0.01 * Rate)."""
    if (
        df.empty
        or COL_N_ANTIN_ALL_AGES not in df.columns
        or COL_RATE_ANTIN_ALL_AGES not in df.columns
    ):
        return pd.DataFrame(
            columns=[
                "date",
                "seroprevalence_numpeopletested",
                "seroprevalence_numpeoplewithantibodies",
            ]
        )
    n_raw = df[COL_N_ANTIN_ALL_AGES].astype(str).str.replace(",", "", regex=False)
    rate_raw = df[COL_RATE_ANTIN_ALL_AGES].astype(str).str.replace(",", "", regex=False)
    n = pd.to_numeric(n_raw, errors="coerce")
    rate = pd.to_numeric(rate_raw, errors="coerce")
    out = pd.DataFrame(
        {
            "date": df["date"].values,
            "seroprevalence_numpeopletested": n.values,
            "seroprevalence_numpeoplewithantibodies": (n * 0.01 * rate).values,
        }
    )
    out = out.sort_values("date")
    result = {}
    for county in counties:
        out["deme"] = county
        result[county] = out.sort_values("date")
    return result


def write_datastream_files(
    output_dir: Path,
    case_counts: pd.DataFrame,
    wastewater: pd.DataFrame,
    seroprevalence: pd.DataFrame,
) -> None:
    """Create output_dir and write <county>_casecounts.csv, _wastewater.csv, _seroprevalence.csv for each county."""
    output_dir.mkdir(parents=True, exist_ok=True)
    case_counts.to_csv(output_dir / "case_counts.csv", index=False)
    wastewater.to_csv(output_dir / "wastewater.csv", index=False)
    seroprevalence.to_csv(output_dir / "seroprevalence.csv", index=False)


def plot_datastreams_per_deme(
    output_dir: Path,
    counties: list[str],
    case_counts_per_county: dict[str, pd.DataFrame],
    wastewater_per_county: dict[str, pd.DataFrame],
    seroprevalence_out: dict[str, pd.DataFrame],
) -> Path:
    """
    Plot a 3xN grid (rows: case counts, wastewater, seroprevalence; cols: counties)
    with a shared x-axis, and save into output_dir.
    """
    ncols = len(counties)
    fig, axes = plt.subplots(
        nrows=3,
        ncols=ncols,
        figsize=(4.5 * ncols, 9),
        sharex=True,
        sharey=False,
    )

    if ncols == 1:
        axes = axes.reshape(3, 1)

    for j, county in enumerate(counties):
        # Row 0: case counts
        ax = axes[0, j]
        cc = case_counts_per_county.get(county, pd.DataFrame())
        if not cc.empty and {"date", "case_counts"}.issubset(cc.columns):
            ax.bar(x=pd.to_datetime(cc["date"]), height=cc["case_counts"])
        ax.set_title(county)
        if j == 0:
            ax.set_ylabel("Case counts")

        # Row 1: wastewater
        ax = axes[1, j]
        ww = wastewater_per_county.get(county, pd.DataFrame())
        if not ww.empty and {"date", "wastewater"}.issubset(ww.columns):
            ax.plot(pd.to_datetime(ww["date"]), ww["wastewater"], lw=1)
        if j == 0:
            ax.set_ylabel("Wastewater")

        # Row 2: seroprevalence fraction
        ax = axes[2, j]
        sero_ratio = seroprevalence_out.get(county, pd.DataFrame())
        sero_ratio["fraction_with_antibodies"] = (
            sero_ratio["seroprevalence_numpeoplewithantibodies"]
            / sero_ratio["seroprevalence_numpeopletested"]
        )
        if not sero_ratio.empty:
            ax.plot(sero_ratio["date"], sero_ratio["fraction_with_antibodies"], lw=1)
        if j == 0:
            ax.set_ylabel("Seroprev (with_ab / tested)")
        ax.set_xlabel("Date")

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    out_path = output_dir / "datastreams_per_deme.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    args = _parse_args()
    counties = (
        [s.strip() for s in args.counties.split(",") if s.strip()]
        if args.counties
        else list(COUNTIES_OF_INTEREST)
    )
    start_date = pd.to_datetime(args.start_date)
    if args.metadata is not None:
        metadata = pd.read_csv(args.metadata, low_memory=False)
        if "Collection date" not in metadata.columns:
            raise KeyError(
                "Expected column 'Collection date' in metadata CSV to infer end date."
            )
        metadata["Collection date"] = pd.to_datetime(metadata["Collection date"])
        end_date = metadata["Collection date"].max()
    else:
        end_date = pd.to_datetime(args.end_date)

    case_counts = load_and_prepare_case_counts(args.case_counts)
    seroprevalence = load_and_prepare_seroprevalence(args.seroprevalence)
    wastewater = load_and_prepare_wastewater(args.wastewater)

    case_counts = filter_case_counts_by_counties_and_dates(
        case_counts, counties, start_date, end_date
    )
    wastewater = filter_wastewater_by_counties_and_dates(
        wastewater, counties, start_date, end_date
    )

    seroprevalence = filter_seroprevalence_by_dates(
        seroprevalence, start_date, end_date
    )

    case_counts_per_county = build_case_counts_per_county(case_counts, counties)
    wastewater_per_county = build_wastewater_per_county(wastewater, counties)
    seroprevalence_out = build_seroprevalence_output(seroprevalence, counties)

    case_counts = pd.concat(case_counts_per_county.values())
    wastewater = pd.concat(wastewater_per_county.values())
    seroprevalence = pd.concat(seroprevalence_out.values())

    write_datastream_files(
        args.output_dir,
        case_counts,
        wastewater,
        seroprevalence,
    )
    plot_datastreams_per_deme(
        args.output_dir,
        counties,
        case_counts_per_county,
        wastewater_per_county,
        seroprevalence_out,
    )


if __name__ == "__main__":
    main()
