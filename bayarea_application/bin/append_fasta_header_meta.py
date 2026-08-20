"""
Append FASTA header identifiers to a metadata table.

Given:
- a metadata CSV file with an `Accession ID` column
- a FASTA file where each header line contains the accession as the second
  `|`-separated field (e.g. >name|EPI_ISL_123456|date|...)

This script matches each metadata row to the corresponding FASTA header using
the accession, and writes an updated metadata CSV with an additional column
`sequence_ID` containing the full FASTA header (without the leading `>`).
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_fasta_headers(fasta_path: Path) -> dict[str, str]:
    """
    Build a mapping from accession ID to FASTA header.

    Parameters
    ----------
    fasta_path : Path
        Path to the FASTA file.

    Returns
    -------
    dict[str, str]
        Mapping from accession ID (e.g. EPI_ISL_123456) to the full header
        string without the leading `>`.
    """
    accession_to_header: dict[str, str] = {}

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            if not header:
                continue
            parts = header.split("|")
            if len(parts) < 2:
                continue
            accession = parts[1].strip()
            if accession:
                accession_to_header[accession] = header

    return accession_to_header


def append_sequence_ids_to_metadata(
    metadata_path: Path,
    fasta_path: Path,
    output_path: Path,
) -> None:
    """
    Append a `sequence_ID` column to metadata using FASTA headers.

    Parameters
    ----------
    metadata_path : Path
        Input metadata CSV with an `Accession ID` column.
    fasta_path : Path
        FASTA file whose headers contain the accession as the second
        `|`-separated field.
    output_path : Path
        Destination CSV file path for the updated metadata.
    """
    metadata = pd.read_csv(metadata_path)

    if "Accession ID" not in metadata.columns:
        raise KeyError(
            "Metadata file must contain an 'Accession ID' column, "
            f"found columns: {list(metadata.columns)}"
        )

    accession_to_header = parse_fasta_headers(fasta_path)

    # Map each accession to its corresponding FASTA header (or NaN if missing)
    metadata["sequence_ID"] = metadata["Accession ID"].map(accession_to_header)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=False)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append FASTA header identifiers to a metadata CSV.\n\n"
            "The metadata file must contain an 'Accession ID' column. The FASTA "
            "headers are expected to be '|' separated, with the accession in the "
            "second field (e.g. >name|EPI_ISL_123456|date|...)."
        )
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Path to input metadata CSV (with 'Accession ID' column).",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help="Path to input FASTA file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for output metadata CSV with added 'sequence_ID' column.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    append_sequence_ids_to_metadata(
        metadata_path=args.metadata,
        fasta_path=args.fasta,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

