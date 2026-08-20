"""
Remove outlier sequences from a FASTA file.

Reads an outliers TSV (first column = sequence IDs) and a FASTA file,
writes a new FASTA with outlier sequences excluded to the specified output directory.
"""

import argparse
from pathlib import Path

import pandas as pd
from Bio import SeqIO


def load_outlier_ids(outliers_path: Path) -> set[str]:
    """
    Load sequence IDs from the outliers TSV file.

    The first column of the TSV is assumed to contain sequence IDs
    (matching FASTA headers without the leading '>').

    Parameters
    ----------
    outliers_path : Path
        Path to the outliers TSV file.

    Returns
    -------
    set[str]
        Set of sequence IDs to exclude.
    """
    df = pd.read_csv(outliers_path, sep="\t")
    first_col = df.columns[0]
    return set(df[first_col].astype(str).str.strip())


def remove_outliers_from_fasta(
    fasta_path: Path,
    outlier_ids: set[str],
    output_path: Path,
) -> tuple[int, int]:
    """
    Filter FASTA records, excluding those whose IDs are in outlier_ids.

    Parameters
    ----------
    fasta_path : Path
        Path to input FASTA file.
    outlier_ids : set[str]
        Set of sequence IDs to exclude (without leading '>').
    output_path : Path
        Path for the output FASTA file.

    Returns
    -------
    tuple[int, int]
        (total_records, records_written)
    """
    total = 0
    kept = 0
    records_to_write = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        total += 1
        seq_id = record.id
        if seq_id in outlier_ids:
            continue
        records_to_write.append(record)
        kept += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records_to_write, output_path, "fasta")

    return total, kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove outlier sequences from a FASTA file."
    )
    parser.add_argument(
        "outliers_tsv",
        type=Path,
        help="Path to outliers TSV (first column = sequence IDs).",
    )
    parser.add_argument(
        "fasta",
        type=Path,
        help="Path to input FASTA file.",
    )
    parser.add_argument(
        "metadata",
        type=Path,
        help="Path to input metadata CSV file.",
    )
    parser.add_argument(
        "dates",
        type=Path,
        help="Path to input dates CSV file.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the filtered FASTA file.",
    )
    parser.add_argument(
        "-n",
        "--output-name",
        type=str,
        default=None,
        help="Output filename (default: input name with '_no_outliers' suffix).",
    )
    args = parser.parse_args()

    outlier_ids = load_outlier_ids(args.outliers_tsv)
    print(f"Loaded {len(outlier_ids)} outlier sequence ID(s) from {args.outliers_tsv}")

    metadata = pd.read_csv(args.metadata)
    dates = pd.read_csv(args.dates)

    output_name = args.output_name
    if output_name is None:
        stem = args.fasta.stem
        output_name = f"{stem}_no_outliers.fasta"
        output_metadata_name = f"{stem}_no_outliers_metadata.csv"
        output_dates_name = f"{stem}_no_outliers_dates.csv"
    else:
        output_stem = Path(output_name).stem
        output_metadata_name = f"{output_stem}_metadata.csv"
        output_dates_name = f"{output_stem}_dates.csv"
    output_path = args.output_dir / output_name
    output_metadata_path = args.output_dir / output_metadata_name
    output_dates_path = args.output_dir / output_dates_name

    total, kept = remove_outliers_from_fasta(
        args.fasta,
        outlier_ids,
        output_path,
    )
    acc_ids = [i.split("|")[1] for i in outlier_ids]
    metadata = metadata[~metadata["Accession ID"].isin(acc_ids)]
    dates = dates[~dates["name"].isin(outlier_ids)]
    metadata.to_csv(output_metadata_path, index=False)
    dates.to_csv(output_dates_path, index=False)
    removed = total - kept

    print(f"Input: {total} sequences")
    print(f"Removed: {removed} outlier(s)")
    print(f"Output: {kept} sequences -> {output_path}")


if __name__ == "__main__":
    main()
