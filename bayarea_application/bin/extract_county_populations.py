"""Extract population per county of interest from the case-counts CSV."""

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from bin.filter_gisaid_metadata import COUNTIES_OF_INTEREST


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract population for counties of interest from the case-counts CSV."
    )
    data_dir = _REPO_ROOT / "data"
    parser.add_argument(
        "--case-counts",
        type=Path,
        default=data_dir / "covid19cases_test.csv",
        help="Path to case counts CSV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--counties",
        type=str,
        default=None,
        help="Comma-separated county names; defaults to COUNTIES_OF_INTEREST.",
    )
    return parser.parse_args()


def extract_populations(case_counts_path: Path, counties: list[str]) -> pd.DataFrame:
    """Return a DataFrame with columns [county, population] for each requested county."""
    df = pd.read_csv(case_counts_path, low_memory=False)
    df = df[df["area_type"] == "County"]
    pat = "|".join(counties)
    df = df[df["area"].str.contains(pat, na=False)]
    populations = (
        df.groupby("area")["population"]
        .first()
        .reset_index()
        .rename(columns={"area": "county"})
    )
    missing = set(counties) - set(populations["county"])
    if missing:
        print(f"Warning: no data found for counties: {missing}", file=sys.stderr)
    return populations.sort_values("county").reset_index(drop=True)


def main() -> None:
    args = _parse_args()
    counties = (
        [s.strip() for s in args.counties.split(",") if s.strip()]
        if args.counties
        else list(COUNTIES_OF_INTEREST)
    )
    populations = extract_populations(args.case_counts, counties)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    populations.to_csv(args.output, index=False)
    print(f"Wrote {len(populations)} rows to {args.output}")


if __name__ == "__main__":
    main()
