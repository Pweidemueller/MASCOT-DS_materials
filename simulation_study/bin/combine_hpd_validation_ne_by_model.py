#!/usr/bin/env python3
"""
Combine Ne HPD validation CSVs from MASCOT (original) and MASCOT-DS (datastreams)
into a single CSV with a Model column.

Two modes:
  1. Explicit files (for Nextflow):
       --original-ne-csv all_ne_original.csv --datastreams-ne-csv all_ne_datastreams.csv
  2. Directory scan (standalone / legacy):
       --analysis-dir results/3_analysis
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from constants import MODEL_MASCOT, MODEL_MASCOT_DS

MODEL_ORIGINAL = MODEL_MASCOT
MODEL_DATASTREAMS = MODEL_MASCOT_DS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine original (MASCOT) and datastreams (MASCOT-DS) Ne HPD "
            "validation CSVs, tagging each row with a Model column."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path for the combined CSV (parent directory is created if needed)",
    )

    # Mode 1: explicit file paths (preferred for Nextflow)
    parser.add_argument(
        "--original-ne-csv",
        type=Path,
        default=None,
        help="Pre-concatenated Ne HPD validation CSV for the original (MASCOT) variant.",
    )
    parser.add_argument(
        "--datastreams-ne-csv",
        type=Path,
        default=None,
        help="Pre-concatenated Ne HPD validation CSV for the datastreams (MASCOT-DS) variant.",
    )

    # Mode 2: directory scan (legacy / standalone)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing *_*_simulation folders. "
            "Used only when --original-ne-csv / --datastreams-ne-csv are not provided."
        ),
    )
    return parser.parse_args()


def simulation_folders(analysis_dir: Path) -> list[Path]:
    if not analysis_dir.is_dir():
        raise SystemExit(f"Analysis directory does not exist: {analysis_dir}")
    folders = sorted(
        p for p in analysis_dir.glob("*_*_simulation") if p.is_dir()
    )
    return folders


def paths_for_simulation(sim_dir: Path) -> tuple[Path, Path]:
    name = sim_dir.name
    noclip = sim_dir / "datastreams_noclip"
    original = noclip / f"{name}_datastreams_noclip_original_hpd_validation_ne.csv"
    datastreams = noclip / f"{name}_datastreams_noclip_datastreams_hpd_validation_ne.csv"
    return original, datastreams


def read_rows_with_model(path: Path, model: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], []
        base_fields = list(reader.fieldnames)
        if "Model" in base_fields:
            raise SystemExit(f"Unexpected existing Model column in {path}")
        rows: list[dict[str, str]] = []
        for row in reader:
            row["Model"] = model
            rows.append(row)
        out_fields = base_fields + ["Model"]
        return out_fields, rows


def combine_from_files(
    original_csv: Path,
    datastreams_csv: Path,
    output_path: Path,
) -> None:
    """Combine two pre-concatenated CSVs, tagging rows with Model."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    header: list[str] | None = None

    for path, model in (
        (original_csv, MODEL_ORIGINAL),
        (datastreams_csv, MODEL_DATASTREAMS),
    ):
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")
        fields, rows = read_rows_with_model(path, model)
        if not fields:
            print(f"Skip empty file: {path}")
            continue
        if header is None:
            header = fields
        elif fields != header:
            raise SystemExit(
                f"Header mismatch in {path}.\nExpected: {header}\nGot: {fields}"
            )
        all_rows.extend(rows)

    if header is None or not all_rows:
        raise SystemExit("No data rows were collected; nothing to write.")

    _write_csv(header, all_rows, output_path)
    print(f"Wrote {len(all_rows)} rows (2 files) to {output_path}")


def combine_from_directory(analysis_dir: Path, output_path: Path) -> None:
    """Legacy mode: scan directory for per-simulation CSV pairs."""
    folders = simulation_folders(analysis_dir)
    if not folders:
        raise SystemExit(f"No *_*_simulation folders found under {analysis_dir}")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    header: list[str] | None = None

    for sim_dir in folders:
        original_path, ds_path = paths_for_simulation(sim_dir)
        if not original_path.is_file():
            print(f"Skip {sim_dir.name}: missing {original_path.name}")
            continue
        if not ds_path.is_file():
            print(f"Skip {sim_dir.name}: missing {ds_path.name}")
            continue

        for path, model in (
            (original_path, MODEL_ORIGINAL),
            (ds_path, MODEL_DATASTREAMS),
        ):
            fields, rows = read_rows_with_model(path, model)
            if not fields:
                print(f"Skip empty file: {path}")
                continue
            if header is None:
                header = fields
            elif fields != header:
                raise SystemExit(
                    f"Header mismatch in {path}.\nExpected: {header}\nGot: {fields}"
                )
            all_rows.extend(rows)

    if header is None or not all_rows:
        raise SystemExit("No data rows were collected; nothing to write.")

    _write_csv(header, all_rows, output_path)
    print(
        f"Wrote {len(all_rows)} rows from {len(folders)} simulation folder(s) to {output_path}"
    )


def _write_csv(
    header: list[str], rows: list[dict[str, str]], output_path: Path
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.original_ne_csv is not None and args.datastreams_ne_csv is not None:
        combine_from_files(args.original_ne_csv, args.datastreams_ne_csv, args.output)
    elif args.analysis_dir is not None:
        combine_from_directory(args.analysis_dir.resolve(), args.output)
    else:
        raise SystemExit(
            "Provide either --original-ne-csv and --datastreams-ne-csv, "
            "or --analysis-dir."
        )


if __name__ == "__main__":
    main()
