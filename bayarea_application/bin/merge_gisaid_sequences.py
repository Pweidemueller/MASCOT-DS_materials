"""
Merge all TSV and FASTA files in data/GISAID_sequences by variant (subfolder name).
Adds a 'variant' column from the subfolder name and prints sequence counts.
Excludes Accession IDs listed in any file with "nonhuman" in the filename from
the written metadata and FASTA (counts are printed before exclusion).
"""

import argparse
import re
from pathlib import Path

import pandas as pd

GISAID_DIR = Path(__file__).resolve().parent / "../data" / "GISAID_sequences"
# Default output directory — can be overridden via --output_dir.
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "../results" / "GISAID_merged"


def iter_fasta_records(path: Path):
    """Yield (header, sequence) for each record in a FASTA file."""
    header = None
    seq_parts = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts)


# GISAID FASTA headers: >...|ACCESSION|... ; accession is typically EPI_ISL_<digits>
ACCESSION_PATTERN = re.compile(r"EPI_ISL_\d+")


def accession_from_fasta_header(header: str) -> str | None:
    """Extract Accession ID from a GISAID FASTA header (first EPI_ISL_* segment)."""
    m = ACCESSION_PATTERN.search(header)
    return m.group(0) if m else None


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merge GISAID FASTA and TSV files across variant subfolders."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory to write merged_metadata.tsv and merged_sequences.fasta "
             f"(default: {_DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    gisaid = Path(GISAID_DIR)
    if not gisaid.is_dir():
        raise FileNotFoundError(f"GISAID_sequences folder not found: {gisaid}")

    # Collect variant subfolders (directories only, not files)
    variant_dirs = [d for d in gisaid.iterdir() if d.is_dir()]
    if not variant_dirs:
        raise FileNotFoundError(f"No variant subfolders found under {gisaid}")

    all_tsv_dfs = []
    all_fasta_records = []
    nonhuman_accession_ids = set()

    for var_dir in sorted(variant_dirs):
        variant_name = var_dir.name
        tsv_files = list(var_dir.glob("*.tsv"))
        fasta_files = list(var_dir.glob("*.fasta"))

        # Collect Accession IDs from any TSV with "nonhuman" in the filename
        for tsv_path in tsv_files:
            if "nonhuman" in tsv_path.name.lower():
                df_nh = pd.read_csv(tsv_path, sep="\t")
                if "Accession ID" in df_nh.columns:
                    nonhuman_accession_ids.update(
                        df_nh["Accession ID"].astype(str).dropna().tolist()
                    )

        # Merge TSVs for this variant
        var_tsv_dfs = []
        for tsv_path in sorted(tsv_files):
            df = pd.read_csv(tsv_path, sep="\t")
            df["variant"] = variant_name
            var_tsv_dfs.append(df)
        if var_tsv_dfs:
            var_merged = pd.concat(var_tsv_dfs, ignore_index=True)
            all_tsv_dfs.append(var_merged)

        # Merge FASTAs for this variant (attach variant in header)
        var_fasta_count = 0
        for fasta_path in sorted(fasta_files):
            for header, seq in iter_fasta_records(fasta_path):
                # Attach variant to header as extra field
                new_header = f"{header}|variant_{variant_name}"
                all_fasta_records.append((new_header, seq))
                var_fasta_count += 1

    # Combined merged TSV and counts
    merged_tsv = (
        pd.concat(all_tsv_dfs, ignore_index=True) if all_tsv_dfs else pd.DataFrame()
    )

    # Print per-variant and combined counts (before removing nonhuman)

    # Remove all nonhuman Accession IDs from metadata and FASTA (only written files are filtered)
    if nonhuman_accession_ids:
        merged_tsv = merged_tsv[
            ~merged_tsv["Accession ID"].astype(str).isin(nonhuman_accession_ids)
        ]
        all_fasta_records = [
            (header, seq)
            for header, seq in all_fasta_records
            if accession_from_fasta_header(header) not in nonhuman_accession_ids
        ]
        print(
            f"Excluded {len(nonhuman_accession_ids)} nonhuman Accession ID(s) from written outputs."
        )

    # make sure there are no duplicates
    print("Checking for duplicates...")
    print(f"Number of metadata rows: {len(merged_tsv)}")
    print(f"Number of sequences: {len(all_fasta_records)}")
    merged_tsv = merged_tsv.drop_duplicates()
    all_fasta_records = list(set(all_fasta_records))
    print(f"Number of metadata rows after deduplication: {len(merged_tsv)}")
    print(f"Number of sequences after deduplication: {len(all_fasta_records)}")
    variant_tsv_counts = {
        variant_name: len(merged_tsv[merged_tsv["variant"] == variant_name])
        for variant_name in merged_tsv["variant"].unique()
    }
    variant_fasta_counts = {}
    for header, seq in all_fasta_records:
        variant_name = header.split("|")[-1].split("_")[1]
        if variant_name not in variant_fasta_counts:
            variant_fasta_counts[variant_name] = 0
        variant_fasta_counts[variant_name] += 1

    total_tsv = len(merged_tsv)
    total_fasta = len(all_fasta_records)

    print("Sequences after merging:")
    for variant_name in sorted(variant_tsv_counts.keys()):
        n_tsv = variant_tsv_counts[variant_name]
        n_fasta = variant_fasta_counts[variant_name]
        print(
            f"  {variant_name}: {n_tsv} metadata rows (TSV), {n_fasta} sequences (FASTA)"
        )
    print(
        f"  Combined: {total_tsv} metadata rows (TSV), {total_fasta} sequences (FASTA)"
    )

    # Write merged files to the output directory.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not merged_tsv.empty:
        tsv_path = out_dir / "merged_metadata.tsv"
        merged_tsv.to_csv(tsv_path, sep="\t", index=False)
        print(f"Wrote merged metadata: {tsv_path} ({len(merged_tsv)} rows)")

    if all_fasta_records:
        fasta_path = out_dir / "merged_sequences.fasta"
        with open(fasta_path, "w") as f:
            for header, seq in all_fasta_records:
                f.write(header + "\n")
                # Write sequence in 80-char lines
                for i in range(0, len(seq), 80):
                    f.write(seq[i : i + 80] + "\n")
        print(
            f"Wrote merged sequences: {fasta_path} ({len(all_fasta_records)} sequences)"
        )


if __name__ == "__main__":
    main()
