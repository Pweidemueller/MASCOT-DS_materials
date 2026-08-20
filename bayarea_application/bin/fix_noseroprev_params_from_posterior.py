#!/usr/bin/env python3
"""
Helper: produce two noseroprevalence BEAST2 XMLs from an existing
``*_datastreams_noseroprevalence.xml`` by fixing parameters at their posterior
medians from the full-datastreams combined log.

Two outputs:
  * ``<basename>_fixedNeScalers.xml``        — per-deme ``NeScaler.DemeN``
    parameters fixed (priors / operators / AVMN refs removed).
  * ``<basename>_fixedCaseCountsScaling.xml`` — per-deme
    ``caseCounts.scaling.DemeN`` parameters fixed (priors / operator refs
    removed). ``caseCounts.dispersion`` stays estimated.

Borrows from :mod:`create_mascot_xml`:
  * XML pretty-printing convention (strip blank lines, preserve deme_map
    comment line).
"""
import argparse
import os
import re
from xml.dom import minidom
from xml.etree import ElementTree as ET

import pandas as pd


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------


def read_log_medians(combined_log_path, column_regex):
    """
    Read a BEAST combined .log file and return ``{column: median}`` for every
    column whose name matches *column_regex*.
    """
    df = pd.read_csv(combined_log_path, sep="\t", comment="#")
    pattern = re.compile(column_regex)
    return {c: float(df[c].median()) for c in df.columns if pattern.fullmatch(c)}


# ---------------------------------------------------------------------------
# XML modifications (mirror the patterns in
# create_mascot_xml.configure_ne_scaler_mean / configure_seroprevalence_scaling)
# ---------------------------------------------------------------------------


def _format_value(v):
    return f"{v:.10g}"


def fix_ne_scaler_demes(root, ne_scaler_medians):
    """
    Fix per-deme ``NeScaler.DemeN`` parameters at posterior medians.

    For each ``NeScaler.DemeN`` column (e.g. ``NeScaler.Deme1``) in
    *ne_scaler_medians*:
      * Set ``estimate="false"`` on the state parameter and overwrite its value.
      * Remove ``NeScaler.DemeN.Prior.t:SimDataset`` from ``<prior>``.
      * Remove ``NeScaler.DemeN.Scaler.t:SimDataset`` operator from ``<mcmc>``.
      * Strip ``NeScaler.DemeN.t:SimDataset`` reference from
        ``AVMNLogTransform.Mascot.SimDataset``.
      * Strip ``<up idref="NeScaler.DemeN.t:SimDataset"/>`` from
        ``NeScalerUpDown.t:SimDataset`` if present.
    """
    state = root.find(".//*[@id='state']")
    prior_dist = root.find(".//*[@id='prior']")
    run = root.find(".//*[@id='mcmc']")
    avmn_log = root.find(".//*[@id='AVMNLogTransform.Mascot.SimDataset']")
    ne_up_down = (
        run.find("./operator[@id='NeScalerUpDown.t:SimDataset']")
        if run is not None
        else None
    )

    for col, median in ne_scaler_medians.items():
        deme = col.split("NeScaler.", 1)[1]
        param_id = f"NeScaler.{deme}.t:SimDataset"

        for p in state.findall("parameter"):
            if p.get("id") == param_id:
                p.set("estimate", "false")
                p.text = _format_value(median)

        for child in list(prior_dist):
            if child.get("id") == f"NeScaler.{deme}.Prior.t:SimDataset":
                prior_dist.remove(child)

        for op in list(run.findall("operator")):
            if op.get("id") == f"NeScaler.{deme}.Scaler.t:SimDataset":
                run.remove(op)

        if avmn_log is not None:
            for child in list(avmn_log):
                if child.get("idref") == param_id:
                    avmn_log.remove(child)

        if ne_up_down is not None:
            for child in list(ne_up_down):
                if child.tag == "up" and child.get("idref") == param_id:
                    ne_up_down.remove(child)


def fix_case_counts_scaling(root, cc_scaling_medians):
    """
    Fix per-deme ``caseCounts.scaling.DemeN:SimDataset`` parameters at posterior
    medians.

    For each ``caseCounts.scaling.DemeN:SimDataset`` column in
    *cc_scaling_medians*:
      * Set ``estimate="false"`` on the state parameter and overwrite its value.
      * Remove ``CaseCountsScaling.DemeN.Prior:SimDataset`` from ``<prior>``.
      * Strip the parameter reference from
        ``AVMNLogTransform.Mascot.SimDataset``.
      * Strip ``<downRealParameter idref=".../>`` from
        ``UpDownPrevScaling.DemeN:SimDataset``.

    ``caseCounts.dispersion`` is left estimated.
    """
    state = root.find(".//*[@id='state']")
    prior_dist = root.find(".//*[@id='prior']")
    run = root.find(".//*[@id='mcmc']")
    avmn_log = root.find(".//*[@id='AVMNLogTransform.Mascot.SimDataset']")

    for col, median in cc_scaling_medians.items():
        param_id = col  # already includes :SimDataset
        deme = col.split("caseCounts.scaling.", 1)[1].split(":", 1)[0]

        for p in state.findall("parameter"):
            if p.get("id") == param_id:
                p.set("estimate", "false")
                p.text = _format_value(median)

        for child in list(prior_dist):
            if child.get("id") == f"CaseCountsScaling.{deme}.Prior:SimDataset":
                prior_dist.remove(child)

        if avmn_log is not None:
            for child in list(avmn_log):
                if child.get("idref") == param_id:
                    avmn_log.remove(child)

        if run is not None:
            for op in run.findall("operator"):
                if op.get("id") == f"UpDownPrevScaling.{deme}:SimDataset":
                    for child in list(op):
                        if (
                            child.tag == "downRealParameter"
                            and child.get("idref") == param_id
                        ):
                            op.remove(child)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _extract_deme_map_comment(xml_path):
    """
    Return the ``<!-- deme_map: ... -->`` line from the source XML, or ``None``
    if absent. ElementTree drops top-level comments, so we read it from raw
    text and re-insert when writing.
    """
    with open(xml_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("<!-- deme_map:"):
                return stripped
            if stripped.startswith("<beast"):
                break
    return None


def write_xml(tree, output_path, deme_map_comment=None):
    """Pretty-print *tree* to *output_path*, preserving an optional deme_map comment."""
    xml_str = ET.tostring(tree.getroot(), encoding="utf-8")
    parsed = minidom.parseString(xml_str)
    pretty = parsed.toprettyxml(indent="  ")
    lines = [line for line in pretty.splitlines() if line.strip() != ""]
    if deme_map_comment:
        lines.insert(1, deme_map_comment)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Produce two noseroprevalence MASCOT-DS XMLs with parameters fixed "
            "at posterior medians from the full-datastreams combined log."
        )
    )
    parser.add_argument(
        "--input_xml",
        required=True,
        help="Path to an existing *_datastreams_noseroprevalence.xml",
    )
    parser.add_argument(
        "--combined_log",
        required=True,
        help="Path to the combined .log from the full _datastreams run",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: same directory as --input_xml).",
    )
    args = parser.parse_args()

    input_xml = os.path.abspath(args.input_xml)
    if not os.path.isfile(input_xml):
        parser.error(f"--input_xml not found: {input_xml}")
    if not os.path.isfile(args.combined_log):
        parser.error(f"--combined_log not found: {args.combined_log}")

    output_dir = args.output_dir or os.path.dirname(input_xml)
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_xml))[0]

    ne_scaler_medians = read_log_medians(args.combined_log, r"NeScaler\.Deme\d+")
    cc_scaling_medians = read_log_medians(
        args.combined_log, r"caseCounts\.scaling\.Deme\d+:SimDataset"
    )
    if not ne_scaler_medians:
        raise ValueError("No NeScaler.DemeN columns found in combined log.")
    if not cc_scaling_medians:
        raise ValueError(
            "No caseCounts.scaling.DemeN:SimDataset columns found in combined log."
        )

    print("Posterior medians (NeScaler):")
    for k, v in ne_scaler_medians.items():
        print(f"  {k} = {v:.6g}")
    print("Posterior medians (caseCounts.scaling):")
    for k, v in cc_scaling_medians.items():
        print(f"  {k} = {v:.6g}")

    deme_map_comment = _extract_deme_map_comment(input_xml)

    # Variant 1: NeScalers fixed
    tree = ET.parse(input_xml)
    fix_ne_scaler_demes(tree.getroot(), ne_scaler_medians)
    out_ne = os.path.join(output_dir, base + "_fixedNeScalers.xml")
    write_xml(tree, out_ne, deme_map_comment=deme_map_comment)
    print(f"Wrote {out_ne}")

    # Variant 2: caseCounts.scaling fixed
    tree = ET.parse(input_xml)
    fix_case_counts_scaling(tree.getroot(), cc_scaling_medians)
    out_cc = os.path.join(output_dir, base + "_fixedCaseCountsScaling.xml")
    write_xml(tree, out_cc, deme_map_comment=deme_map_comment)
    print(f"Wrote {out_cc}")


if __name__ == "__main__":
    main()
