"""
Filter GISAID merged metadata by counties of interest.

Reads TSV metadata and splits into:
- seq_meta_demes: entries where Location contains any county of interest
- seq_meta_background: all other entries
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

COUNTIES_OF_INTEREST = [
    "San Francisco",
    # "San Mateo",
    # "Alameda",
    "Santa Clara",
    "Sacramento",
]

WEST_COAST_STATES = {
    "Alaska",
    "California",
    "Hawaii",
    "Oregon",
    "Washington",
}

EAST_COAST_STATES = {
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Maine",
    "Maryland",
    "Massachusetts",
    "New Hampshire",
    "New Jersey",
    "New York",
    "North Carolina",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "Virginia",
}


def load_metadata(tsv_path: Path) -> pd.DataFrame:
    """Load GISAID metadata from TSV."""
    return pd.read_csv(tsv_path, sep="\t")


def filter_by_variant(
    metadata: pd.DataFrame,
    variant: str,
    variant_col: str = "variant",
) -> pd.DataFrame:
    """
    Retain only rows whose variant exactly matches *variant*.

    Parameters
    ----------
    metadata : pd.DataFrame
        Full metadata table.
    variant : str
        Variant string to keep (e.g. "B.1.427").
    variant_col : str
        Column containing variant values.

    Returns
    -------
    pd.DataFrame
        Filtered metadata containing only rows matching *variant*.
    """
    if variant_col not in metadata.columns:
        raise ValueError(
            f"Column {variant_col!r} not found in metadata. "
            f"Available columns: {list(metadata.columns)}"
        )
    filtered = metadata[metadata[variant_col] == variant]
    print(
        f"Variant filter '{variant}': {len(filtered)} / {len(metadata)} rows retained."
    )
    return filtered


def filter_by_counties(
    metadata: pd.DataFrame,
    location_col: str = "Location",
    counties: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split metadata into demes (matching counties) and background (all others).

    Parameters
    ----------
    metadata : pd.DataFrame
        Full metadata table.
    location_col : str
        Name of the column containing location strings.
    counties : list[str] or None
        County substrings to match in Location. Defaults to COUNTIES_OF_INTEREST.

    Returns
    -------
    seq_meta_demes : pd.DataFrame
        Rows where Location contains any of the county names.
    seq_meta_background : pd.DataFrame
        All rows not in seq_meta_demes.
    """
    if counties is None:
        counties = COUNTIES_OF_INTEREST

    mask = (
        metadata[location_col]
        .astype(str)
        .str.contains(
            "|".join(counties),
            case=False,
            na=False,
            regex=True,
        )
    )
    seq_meta_demes = metadata.loc[mask].copy()
    seq_meta_background = metadata.loc[~mask].copy()

    seq_meta_demes["County"] = None
    for county in counties:
        county_mask = (
            seq_meta_demes[location_col]
            .astype(str)
            .str.contains(county, case=False, na=False)
        )
        seq_meta_demes.loc[county_mask, "County"] = county

    return seq_meta_demes, seq_meta_background


def _split_equal_among_locations(
    n_total: int,
    location_counts: list[tuple[str, int]],
) -> dict[str, int]:
    """
    Split n_total across locations as equally as possible, capping each by its count.

    Used when every location has more sequences than its fair share; allocates
    integers that sum to n_total with each location getting at most its count.
    """
    if not location_counts or n_total <= 0:
        return {loc: 0 for loc, _ in location_counts}
    n_loc = len(location_counts)
    base = n_total // n_loc
    remainder = n_total % n_loc
    # Desired: base+1 for first `remainder` locations, base for the rest; then cap.
    desired = [
        (loc, base + (1 if i < remainder else 0), cap)
        for i, (loc, cap) in enumerate(location_counts)
    ]
    allocation = {loc: min(want, cap) for loc, want, cap in desired}
    shortfall = n_total - sum(allocation.values())
    # Distribute shortfall to locations that have room (were capped)
    while shortfall > 0:
        gave = False
        for loc, want, cap in desired:
            if shortfall <= 0:
                break
            if allocation[loc] < cap:
                allocation[loc] += 1
                shortfall -= 1
                gave = True
        if not gave:
            break
    return allocation


def _allocate_stratified_recursive(
    n_total: int,
    location_counts: list[tuple[str, int]],
) -> dict[str, int]:
    """
    Allocate n_total sequences across locations so each gets ~equal share.

    Locations with count <= n_total/num_locations get all their sequences;
    the remaining slots are divided equally across the rest (recursively).
    """
    if not location_counts or n_total <= 0:
        return {loc: 0 for loc, _ in location_counts}
    n_loc = len(location_counts)
    target_per = n_total / n_loc
    small = [(loc, c) for loc, c in location_counts if c <= target_per]
    large = [(loc, c) for loc, c in location_counts if c > target_per]
    result = {loc: c for loc, c in small}
    taken = sum(result.values())
    remaining = n_total - taken
    if not large:
        return result
    if not small:
        return _split_equal_among_locations(n_total, location_counts)
    sub = _allocate_stratified_recursive(remaining, large)
    result.update(sub)
    return result


def _sample_metadata_n(
    metadata: pd.DataFrame,
    n: int,
    pool_name: str,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Randomly sample n rows from a metadata DataFrame.

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata table to sample from.
    n : int
        Number of rows to sample.
    pool_name : str
        Name of the pool (e.g. "demes", "background") for error messages.
    random_state : int or None
        Seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        Subset of metadata with n randomly sampled rows.
    """
    if n < 0:
        raise ValueError(f"Sample size must be non-negative, got {n}")
    if n > len(metadata):
        raise ValueError(
            f"Requested sample size {n} exceeds {pool_name} size {len(metadata)}"
        )
    return metadata.sample(n=n, replace=False, random_state=random_state)


def sample_uniform_over_time(
    df_loc: pd.DataFrame, n: int, date_col: str = "Collection date"
) -> pd.DataFrame:
    df_loc = df_loc.copy()
    df_loc[date_col] = pd.to_datetime(df_loc[date_col])
    df_loc = df_loc.sort_values(date_col)
    m = len(df_loc)
    if m <= n:
        return df_loc
    times = df_loc[date_col].astype("int64").to_numpy()  # nanoseconds since epoch
    t_min, t_max = times.min(), times.max()
    if t_min == t_max:
        # all same date → fall back to random or first n
        return df_loc.sample(n=n, random_state=None, replace=False)
    target_times = np.linspace(t_min, t_max, num=n)
    chosen_idx = []
    used = np.zeros(m, dtype=bool)
    for tt in target_times:
        # find closest unused point
        diffs = np.abs(times - tt)
        diffs[used] = np.inf
        j = int(diffs.argmin())
        used[j] = True
        chosen_idx.append(df_loc.index[j])
    return df_loc.loc[chosen_idx]


def sample_uniform_bins(
    df: pd.DataFrame,
    n: int,
    bin_type: str = "isoweek",
    date_col: str = "Collection date",
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Sample n sequences as uniformly as possible across time bins.

    Algorithm
    ---------
    1. Bin sequences by isoweek or month.
    2. Compute baseline quota q = n // B (B = non-empty bins).
    3. Take min(q, bin_size) from each bin.
    4. Distribute remaining slots via round-robin, prioritising bins with
       the most remaining sequences.

    Within each bin, sequences are selected at random.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if len(df) <= n:
        return df

    rng = np.random.default_rng(random_state)

    if bin_type == "isoweek":
        iso = df[date_col].dt.isocalendar()
        df["_bin"] = (
            iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
        )
    elif bin_type == "month":
        df["_bin"] = df[date_col].dt.to_period("M").astype(str)
    else:
        raise ValueError(f"Unknown bin_type: {bin_type!r}")

    bin_indices: dict[str, np.ndarray] = {}
    for bin_name, group in df.groupby("_bin"):
        bin_indices[bin_name] = group.index.to_numpy()

    n_bins = len(bin_indices)
    if n_bins == 0:
        return df.head(0).drop(columns=["_bin"], errors="ignore")

    q = n // n_bins

    sampled: list = []
    remaining: dict[str, list] = {}

    for bin_name, indices in bin_indices.items():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        take = min(q, len(shuffled))
        sampled.extend(shuffled[:take].tolist())
        leftover = shuffled[take:].tolist()
        if leftover:
            remaining[bin_name] = leftover

    slots_left = n - len(sampled)

    while slots_left > 0 and remaining:
        sorted_bins = sorted(remaining, key=lambda b: -len(remaining[b]))
        gave_any = False
        bins_to_delete: list[str] = []
        for bname in sorted_bins:
            if slots_left <= 0:
                break
            if remaining[bname]:
                sampled.append(remaining[bname].pop(0))
                slots_left -= 1
                gave_any = True
                if not remaining[bname]:
                    bins_to_delete.append(bname)
        for bname in bins_to_delete:
            del remaining[bname]
        if not gave_any:
            break

    return df.loc[sampled].drop(columns=["_bin"])


def _sample_from_pool(
    df: pd.DataFrame,
    n: int,
    temporal_strategy: str = "random",
    date_col: str = "Collection date",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Sample n sequences from *df* using the given temporal strategy."""
    if n <= 0:
        return df.head(0)
    if len(df) <= n:
        return df
    if temporal_strategy == "random":
        return df.sample(n=n, replace=False, random_state=random_state)
    if temporal_strategy in ("uniform_isoweek", "uniform_month"):
        bin_type = "isoweek" if temporal_strategy == "uniform_isoweek" else "month"
        return sample_uniform_bins(
            df, n, bin_type=bin_type, date_col=date_col, random_state=random_state
        )
    raise ValueError(f"Unknown temporal_strategy: {temporal_strategy!r}")


# ---------------------------------------------------------------------------
# Spatial helpers for background sampling
# ---------------------------------------------------------------------------


def _extract_us_state(location: str) -> str | None:
    """Extract the US state from a GISAID Location string.

    Expects format like ``North America / USA / California / ...``.
    Returns None when the state cannot be determined.
    """
    parts = [p.strip() for p in location.split("/")]
    if len(parts) >= 3 and "USA" in parts[1]:
        return parts[2]
    return None


def _classify_us_region(state: str | None) -> str:
    """Map a US state name to west_coast / east_coast / other."""
    if state is None:
        return "other"
    state_clean = state.strip()
    if state_clean in WEST_COAST_STATES:
        return "west_coast"
    if state_clean in EAST_COAST_STATES:
        return "east_coast"
    return "other"


def _sample_background_balanced_regions(
    df: pd.DataFrame,
    n: int,
    temporal_strategy: str = "random",
    date_col: str = "Collection date",
    location_col: str = "Location",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Sample background sequences balancing west-coast, east-coast, and other US regions."""
    df = df.copy()
    df["_us_state"] = df[location_col].astype(str).apply(_extract_us_state)
    df["_region"] = df["_us_state"].apply(_classify_us_region)

    region_counts = [(region, len(group)) for region, group in df.groupby("_region")]
    allocation = _allocate_stratified_recursive(n, region_counts)

    sampled_parts: list[pd.DataFrame] = []
    for region, alloc_n in allocation.items():
        if alloc_n <= 0:
            continue
        region_df = df[df["_region"] == region]
        part = _sample_from_pool(
            region_df, alloc_n, temporal_strategy, date_col, random_state
        )
        sampled_parts.append(part)

    if not sampled_parts:
        return df.head(0)

    result = pd.concat(sampled_parts, ignore_index=True)
    cols_to_drop = [c for c in ("_us_state", "_region") if c in result.columns]
    return result.drop(columns=cols_to_drop)


def sample_background(
    df: pd.DataFrame,
    n: int,
    spatial_strategy: str = "random",
    temporal_strategy: str = "random",
    date_col: str = "Collection date",
    location_col: str = "Location",
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Sample n background sequences with configurable spatial and temporal strategy.

    Parameters
    ----------
    spatial_strategy : {"random", "balanced_regions"}
        "random" ignores geography; "balanced_regions" balances across
        west-coast, east-coast, and other US states.
    temporal_strategy : {"random", "uniform_isoweek", "uniform_month"}
        "random" samples uniformly at random; the uniform variants aim for
        equal representation across isoweek or month bins.
    """
    if spatial_strategy == "random":
        return _sample_from_pool(df, n, temporal_strategy, date_col, random_state)
    if spatial_strategy == "balanced_regions":
        return _sample_background_balanced_regions(
            df, n, temporal_strategy, date_col, location_col, random_state
        )
    raise ValueError(f"Unknown spatial_strategy: {spatial_strategy!r}")


# ---------------------------------------------------------------------------
# Deme sampling
# ---------------------------------------------------------------------------


def sample_demes_accessions(
    seq_meta_demes: pd.DataFrame,
    n: int,
    location_col: str = "Location",
    random_state: int | None = None,
    temporal_strategy: str = "random",
) -> pd.DataFrame:
    """
    Sample n sequences for each deme.

    Locations with fewer than n sequences are included in full.
    """
    if n < 0:
        raise ValueError(f"Sample size must be non-negative, got {n}")
    if n == 0:
        return seq_meta_demes.head(0)

    groups = seq_meta_demes.groupby(location_col, sort=False)
    sampled = []
    for _, grp in groups:
        tmp = _sample_from_pool(grp, n, temporal_strategy, random_state=random_state)
        sampled.append(tmp)

    return pd.concat(sampled, axis=0, ignore_index=True)


def extract_accession_from_header(header: str) -> str | None:
    """
    Extract the accession ID from a FASTA header line.

    Assumes GISAID-style headers with the accession as one of the
    pipe-separated fields, e.g. "...|EPI_ISL_1098314|...".
    """
    line = header.lstrip(">").strip()
    if not line:
        return None
    parts = line.split("|")
    for part in parts:
        token = part.strip()
        if token.startswith("EPI_"):
            return token
    return None


def filter_fasta_by_accessions(
    fasta_in: Path,
    fasta_out: Path,
    accession_to_label: dict[str, str],
) -> tuple[int, dict[str, str]]:
    """
    Write a FASTA with only sequences whose accession is in accession_to_label.

    Each retained header is suffixed with "|<label>" where label is the
    county name (whitespace replaced by "_") or "background".

    Returns
    -------
    written : int
        Number of sequences written to fasta_out.
    header_by_accession : dict[str, str]
        Mapping from accession ID to the full written header line.
    """
    if not accession_to_label:
        return 0, {}

    written = 0
    header_by_accession: dict[str, str] = {}
    accession_count: dict[str, int] = {}
    counter = 0
    with fasta_in.open("r") as in_f, fasta_out.open("w") as out_f:
        current_header: str | None = None
        current_seq_lines: list[str] = []
        current_label: str | None = None
        current_accession: str | None = None

        def flush() -> None:
            nonlocal written
            if current_header is None:
                return
            if current_label is None:
                return
            if current_accession is None:
                return
            label_clean = current_label.replace(" ", "_")
            header_stripped = current_header.rstrip("\n")
            full_header = f"{header_stripped}|{label_clean}"
            out_f.write(f"{full_header}\n")
            out_f.writelines(current_seq_lines)
            header_by_accession[current_accession] = full_header
            written += 1

        for line in in_f:
            if line.startswith(">"):
                flush()
                current_header = line
                current_seq_lines = []
                acc = extract_accession_from_header(line)
                current_accession = acc
                current_label = accession_to_label.get(acc) if acc is not None else None
                if current_label is not None:
                    if acc in accession_count:
                        accession_count[acc] += 1
                    else:
                        accession_count[acc] = 1
            else:
                current_seq_lines.append(line)

        flush()

    # more htan one
    for acc, count in accession_count.items():
        if count > 1:
            print(f"Accession {acc} appears {count} times")

    return written, header_by_accession


from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


def plot_cumulative_sequences_over_time(
    seq_meta_demes: Optional[pd.DataFrame],
    seq_meta_background: Optional[pd.DataFrame],
    date_col: str,
    deme_col: str = "County",
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot cumulative number of sequences over time, split by deme (top subplot)
    and background (bottom subplot).

    Parameters
    ----------
    seq_meta_demes
        DataFrame with at least `date_col` and `deme_col` for deme sequences.
    seq_meta_background
        DataFrame with at least `date_col` for background sequences.
    date_col
        Name of the column containing the collection date.
    deme_col
        Name of the column defining demes (e.g. "County").
    output_path
        If provided, path where the figure will be saved. If None, the plot
        is shown interactively.
    """
    if seq_meta_demes is None and seq_meta_background is None:
        return

    # Prepare data
    if seq_meta_demes is not None:
        demes_df = seq_meta_demes.copy()
        demes_df[date_col] = pd.to_datetime(demes_df[date_col])

        deme_counts = (
            demes_df.groupby([deme_col, date_col])
            .size()
            .reset_index(name="count")
            .sort_values(date_col)
        )
        deme_counts["cum_count"] = deme_counts.groupby(deme_col)["count"].cumsum()
    else:
        deme_counts = None

    if seq_meta_background is not None:
        bg_df = seq_meta_background.copy()
        bg_df[date_col] = pd.to_datetime(bg_df[date_col])

        bg_counts = (
            bg_df.groupby(date_col)
            .size()
            .reset_index(name="count")
            .sort_values(date_col)
        )
        bg_counts["cum_count"] = bg_counts["count"].cumsum()
    else:
        bg_counts = None

    # Set up figure
    n_subplots = int(deme_counts is not None) + int(bg_counts is not None)
    fig, axes = plt.subplots(
        n_subplots,
        1,
        figsize=(10, 4 * n_subplots),
        sharex=True,
        constrained_layout=True,
    )

    if n_subplots == 1:
        axes = [axes]

    ax_idx = 0

    # Demes subplot: one line per deme
    if deme_counts is not None:
        ax_demes = axes[ax_idx]
        for deme, sub_df in deme_counts.groupby(deme_col):
            ax_demes.plot(
                sub_df[date_col],
                sub_df["cum_count"],
                label=str(deme),
                linewidth=1.5,
            )
        ax_demes.set_ylabel("Cumulative sequences")
        ax_demes.set_title("Demes (by {0})".format(deme_col))
        ax_demes.legend(loc="upper left", fontsize="small", ncol=2)
        ax_idx += 1

    # Background subplot: all background combined
    if bg_counts is not None:
        ax_bg = axes[ax_idx]
        ax_bg.plot(
            bg_counts[date_col],
            bg_counts["cum_count"],
            color="black",
            linewidth=1.5,
        )
        ax_bg.set_ylabel("Cumulative sequences")
        ax_bg.set_title("Background")
        ax_bg.set_xlabel("Collection date")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter GISAID merged metadata by counties of interest."
    )
    parser.add_argument(
        "tsv_path",
        type=Path,
        help="Path to the merged metadata TSV file.",
    )
    parser.add_argument(
        "fasta_path",
        type=Path,
        help="Path to the merged FASTA file.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        metavar="VARIANT",
        help=(
            "If provided, retain only sequences where the 'variant' column "
            "exactly matches this value (e.g. 'B1427'). Default: no variant filter."
        ),
    )
    parser.add_argument(
        "--most_recent_date",
        type=str,
        help="Latest date to include sequences. Only sequences with Collection date before or at this date will be included.",
    )
    parser.add_argument(
        "--n_background",
        type=int,
        default=None,
        metavar="N",
        help="Number of sequences to randomly sample from the background (for downstream use).",
    )
    parser.add_argument(
        "--n_demes",
        type=int,
        default=None,
        metavar="N",
        help="Number of sequences to randomly sample from the demes (counties of interest).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for sampling (reproducible runs). Default is 42.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help=(
            "Output directory for sampled sequences. "
            "Defaults to the current directory."
        ),
    )

    temporal_choices = ["random", "uniform_isoweek", "uniform_month"]
    parser.add_argument(
        "--deme_temporal_strategy",
        type=str,
        choices=temporal_choices,
        default="random",
        help=(
            "Temporal sampling strategy for deme sequences. "
            "'random' = purely random; 'uniform_isoweek' / 'uniform_month' = "
            "aim for equal representation across isoweek / month bins. "
            "Default: random."
        ),
    )
    parser.add_argument(
        "--bg_temporal_strategy",
        type=str,
        choices=temporal_choices,
        default="random",
        help=(
            "Temporal sampling strategy for background sequences. "
            "Same choices as --deme_temporal_strategy. Default: random."
        ),
    )
    parser.add_argument(
        "--bg_spatial_strategy",
        type=str,
        choices=["random", "balanced_regions"],
        default="random",
        help=(
            "Spatial sampling strategy for background sequences. "
            "'random' = ignore geography; 'balanced_regions' = balance "
            "across US west-coast, east-coast, and other states. "
            "Default: random."
        ),
    )

    return parser.parse_args()


def convert_date_to_numerical_date(date_series: pd.Series) -> pd.Series:

    date_series = pd.to_datetime(date_series)
    years = date_series.dt.year
    year_start = pd.to_datetime(years.astype(str) + "-01-01")
    next_year_start = pd.to_datetime((years + 1).astype(str) + "-01-01")
    days_in_year = (next_year_start - year_start).dt.days
    day_of_year = date_series.dt.dayofyear
    return years + (day_of_year / days_in_year)


def main() -> None:
    """Load metadata, filter by counties, and report split."""
    args = parse_args()

    if not args.tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {args.tsv_path}")
    if not args.fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {args.fasta_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.tsv_path)
    if args.variant:
        metadata = filter_by_variant(metadata, args.variant)
    if args.most_recent_date:
        metadata = metadata[metadata["Collection date"] <= args.most_recent_date]
    seq_meta_demes, seq_meta_background = filter_by_counties(metadata)

    most_recent_deme_date = seq_meta_demes["Collection date"].max()
    seq_meta_background = seq_meta_background[
        seq_meta_background["Collection date"] <= most_recent_deme_date
    ]

    print(
        f"Total rows: {len(metadata)}\n"
        f"most recent deme date: {most_recent_deme_date}\n"
        f"seq_meta_demes (counties of interest): {len(seq_meta_demes)}\n"
        f"seq_meta_background (background) before most recent deme date: {len(seq_meta_background)}"
    )

    seq_meta_demes_sampled = None
    if args.n_demes is not None:
        seq_meta_demes_sampled = sample_demes_accessions(
            seq_meta_demes,
            location_col="County",
            n=args.n_demes,
            random_state=args.random_state,
            temporal_strategy=args.deme_temporal_strategy,
        )
        print(
            f"Sampled {len(seq_meta_demes_sampled)} demes accessions "
            f"(temporal: {args.deme_temporal_strategy})."
        )
        for location in seq_meta_demes_sampled["County"].unique():
            n_loc = (seq_meta_demes_sampled["County"] == location).sum()
            print(f"  {location}: {n_loc} sequences")

    seq_meta_background_sampled = None
    if args.n_background is not None:
        seq_meta_background_sampled = sample_background(
            seq_meta_background,
            n=args.n_background,
            spatial_strategy=args.bg_spatial_strategy,
            temporal_strategy=args.bg_temporal_strategy,
            random_state=args.random_state,
        )
        print(
            f"Sampled {len(seq_meta_background_sampled)} background accessions "
            f"(spatial: {args.bg_spatial_strategy}, temporal: {args.bg_temporal_strategy})."
        )
        if args.bg_spatial_strategy == "balanced_regions":
            bg_regions = (
                seq_meta_background_sampled["Location"]
                .astype(str)
                .apply(_extract_us_state)
                .apply(_classify_us_region)
            )
            for region, count in bg_regions.value_counts().items():
                print(f"  {region}: {count} sequences")

    plot_cumulative_sequences_over_time(
        seq_meta_demes_sampled,
        seq_meta_background_sampled,
        date_col="Collection date",
        deme_col="County",
        output_path=output_dir / "sampled_cumulative_sequences_over_time.png",
    )

    # COMPARISON
    plot_cumulative_sequences_over_time(
        seq_meta_demes,
        seq_meta_background,
        date_col="Collection date",
        deme_col="County",
        output_path=output_dir / "full_cumulative_sequences_over_time.png",
    )

    # Write a single sampled metadata table to CSV, if present
    sampled_meta: pd.DataFrame | None = None
    combined_sampled_frames: list[pd.DataFrame] = []
    if seq_meta_demes_sampled is not None:
        combined_sampled_frames.append(seq_meta_demes_sampled.copy())

    if seq_meta_background_sampled is not None:
        background_with_county = seq_meta_background_sampled.copy()
        background_with_county["County"] = "background"
        combined_sampled_frames.append(background_with_county)

    if combined_sampled_frames:
        sampled_meta = pd.concat(combined_sampled_frames, ignore_index=True)
        sampled_meta_out = output_dir / "sampled_sequences_metadata.csv"
        sampled_meta.to_csv(sampled_meta_out, index=False)
        print(
            f"Wrote combined sampled metadata to CSV ({len(sampled_meta)} rows): {sampled_meta_out}"
        )

    # Build accession -> label mapping for sampled demes and background
    accession_to_label: dict[str, str] = {}
    if seq_meta_demes_sampled is not None:
        for _, row in seq_meta_demes_sampled.iterrows():
            acc = str(row["Accession ID"])
            county = str(row["County"])
            accession_to_label[acc] = county
    if seq_meta_background_sampled is not None:
        for _, row in seq_meta_background_sampled.iterrows():
            acc = str(row["Accession ID"])
            if acc not in accession_to_label:
                accession_to_label[acc] = "background"

    if accession_to_label:
        output_fasta = output_dir / "sampled_sequences.fasta"

        n_written, header_by_accession = filter_fasta_by_accessions(
            fasta_in=args.fasta_path,
            fasta_out=output_fasta,
            accession_to_label=accession_to_label,
        )
        print(f"Wrote {n_written} sequences to sampled FASTA: {output_fasta}")
        # Also write sampled_dates.csv with full header and collection date
        if sampled_meta is not None and not sampled_meta.empty:
            names: list[str] = []
            dates: list[str] = []
            for _, row in sampled_meta.iterrows():
                acc = str(row["Accession ID"])
                header = (
                    header_by_accession.get(acc).strip(">").replace("=", "_").strip()
                )
                if header is None:
                    continue
                names.append(header)

                dates.append(str(row["Collection date"]))
            dates_df = pd.DataFrame({"name": names, "date": dates})
            dates_df["date"] = convert_date_to_numerical_date(dates_df["date"])
            if names:
                sampled_dates_out = output_dir / "sampled_dates.csv"
                dates_df.to_csv(sampled_dates_out, index=False)
                print(f"Wrote sampled dates CSV: {sampled_dates_out}")
    else:
        print("No sampled demes or background sequences; no FASTA written.")


if __name__ == "__main__":
    main()
