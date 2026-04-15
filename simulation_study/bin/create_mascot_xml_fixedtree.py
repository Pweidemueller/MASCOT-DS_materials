#!/usr/bin/env python3
import argparse
import re
import string
import os
from xml.etree import ElementTree as ET
from Bio import Phylo
from io import StringIO
import dendropy
from xml.dom import minidom
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import sys

from constants import DATASTREAM_VARIANTS, VARIANT_ORIGINAL

sys.setrecursionlimit(10000)


def get_state0_newick(nexus_text):
    """
    Given the full text of a NEXUS file, extract the tree (STATE_0 or TREE)
    as a Newick string with integer IDs replaced by leaf names.
    """

    # ------------------------------------------------------
    # 1. Extract the TRANSLATE block to build a dictionary:
    #    numeric ID => leaf name (if present)
    # ------------------------------------------------------
    translate_block_pattern = re.compile(
        r"(?is)Begin\s+trees\s*;.*?Translate\s*(.*?)\s*;", re.DOTALL
    )
    translate_match = translate_block_pattern.search(nexus_text)
    translation_dict = {}
    if translate_match:
        translate_block = translate_match.group(1)
        # Build a dictionary from the translate block
        # Each line is typically: 1 leaf_0, or 1 leaf_0,
        for line in translate_block.split("\n"):
            line = line.strip().rstrip(",;")
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                num_id, label = parts
                translation_dict[num_id] = label
    else:
        # No Translate block - will use numeric IDs directly
        print("Warning: No 'Translate' block found. Using numeric IDs.")

    # ------------------------------------------------------
    # 2. Extract the tree STATE_0 or TREE = ( ... );
    #    We'll match up to the semicolon
    # ------------------------------------------------------
    # Try STATE_0 first, then TREE
    state0_pattern = re.compile(r"tree\s+STATE_0\s*=\s*(.*?)\s*;", re.DOTALL)
    state0_match = state0_pattern.search(nexus_text)

    if not state0_match:
        # Try TREE instead
        state0_pattern = re.compile(r"tree\s+TREE\s*=\s*(.*?)\s*;", re.DOTALL)
        state0_match = state0_pattern.search(nexus_text)

    if not state0_match:
        raise ValueError(
            "Could not find 'tree STATE_0 = ...;' or 'tree TREE = ...;' in the NEXUS text."
        )

    tree_str = state0_match.group(1).strip()

    # ------------------------------------------------------
    # 3. Remove bracketed metadata like [&type="D0",time=0.0]
    # ------------------------------------------------------
    def strip_bracketed_metadata(s):
        return re.sub(r"\[.*?\]", "", s)

    tree_str = strip_bracketed_metadata(tree_str)

    # ------------------------------------------------------
    # 4. Safely replace only integer IDs with leaf names
    #    (leave floating-point numbers alone)
    # ------------------------------------------------------
    def replace_labels_safely(s, dictionary):
        """
        Go character-by-character, capturing numeric tokens:
         - If token is purely digits => replace if in dictionary
         - If token contains '.' => treat as float => do not replace
        """
        result = []
        i = 0
        while i < len(s):
            c = s[i]
            if c.isdigit():
                j = i + 1
                while j < len(s) and s[j].isdigit():
                    j += 1
                numeric_token = s[i:j]
                if j < len(s) and s[j] == ":":
                    if numeric_token in dictionary:
                        result.append(dictionary[numeric_token])
                    else:
                        result.append(numeric_token)
                else:
                    result.append(numeric_token)
                i = j
            else:
                result.append(c)
                i += 1
        return "".join(result)

    tree_str = replace_labels_safely(tree_str, translation_dict)

    # ------------------------------------------------------
    # 5. Remove extraneous whitespace, then return
    # ------------------------------------------------------
    tree_str = re.sub(r"\s+", "", tree_str)

    # Return a valid Newick with trailing semicolon
    return tree_str + ";"


def collapse_single_child_nodes(clade):
    """
    Recursively collapse nodes with a single child, adding branch lengths.
    """
    # If the clade is terminal, return it unchanged
    if clade.is_terminal():
        return clade

    # Process children first (postorder traversal)
    new_clades = []
    for child in clade.clades:
        simplified_child = collapse_single_child_nodes(child)
        new_clades.append(simplified_child)

    clade.clades = new_clades

    # If this node has only one child, collapse it
    if len(clade.clades) == 1:
        child = clade.clades[0]
        # Add this clade's branch length to the child's branch length
        child.branch_length = (child.branch_length or 0) + (clade.branch_length or 0)
        return child  # Return the collapsed child

    # If this node has more than one child, keep it
    return clade


def remove_excluded_datastream_elements(
    root,
    exclude_case_counts=False,
    exclude_seroprevalence=False,
    exclude_wastewater=False,
):
    """
    Remove XML elements related to excluded datastream types.

    Uses a data-driven approach: each datastream type defines its parameter
    patterns, distribution/prior/operator/logger prefixes. A single set of
    loops handles all types uniformly.
    """
    # ── Data-driven element configuration ────────────────────────────────
    # Each entry: (should_exclude, param_patterns, dist_prefixes,
    #              prior_prefixes, operator_prefixes, log_prefixes)
    DATASTREAM_ELEMENTS = {
        "case_counts": {
            "exclude": exclude_case_counts,
            "param_patterns": [
                "caseCounts.",
                "caseTimes.",
                "caseCounts.scaling.",
                "caseCounts.dispersion",
            ],
            "distribution_prefixes": ["caseCountLikelihood."],
            "prior_prefixes": [
                "CaseCountsScaling.",
                "CaseCountDispersionPrior.",
            ],
            "standalone_operator_prefixes": ["CaseCountsDispersionScalerX."],
            "log_prefixes": ["caseCountLikelihood."],
            "logger_prefixes": [],
        },
        "seroprevalence": {
            "exclude": exclude_seroprevalence,
            "param_patterns": [
                "seroTestedCounts.",
                "seroWithAntibodiesCounts.",
                "seroTestedTimes.",
                "seroprevalence.scaling.",
            ],
            "distribution_prefixes": ["seroprevalenceLikelihood."],
            "prior_prefixes": ["SeroprevalenceScaling."],
            "standalone_operator_prefixes": [],
            "log_prefixes": ["seroprevalenceLikelihood."],
            "logger_prefixes": ["cumulativeIncidenceLogger."],
        },
        "wastewater": {
            "exclude": exclude_wastewater,
            "param_patterns": [
                "wastewaterConcentration.",
                "wastewaterConcentrationTimes.",
                "wastewater.scaling.",
                "wastewater.sigma",
            ],
            "distribution_prefixes": ["wastewaterLikelihood."],
            "prior_prefixes": [
                "WastewaterScaling.",
                "WastewaterSigmaPrior.",
            ],
            # Template uses typo "WasterwaterSigmaScalerX" (see Mascot_datastreams_template_fixedtree.xml)
            "standalone_operator_prefixes": [
                "WastewaterSigmaScalerX.",
            ],
            "log_prefixes": ["wastewaterLikelihood."],
            "logger_prefixes": [],
        },
    }

    # Collect active patterns (only for excluded types)
    active_param_patterns = []
    active_dist_prefixes = []
    active_prior_prefixes = []
    active_standalone_op_prefixes = []
    active_log_prefixes = []
    active_logger_prefixes = []
    active_param_patterns_by_type = {}

    for ds_type, cfg in DATASTREAM_ELEMENTS.items():
        if not cfg["exclude"]:
            continue
        active_param_patterns.extend(cfg["param_patterns"])
        active_dist_prefixes.extend(cfg["distribution_prefixes"])
        active_prior_prefixes.extend(cfg["prior_prefixes"])
        active_standalone_op_prefixes.extend(cfg["standalone_operator_prefixes"])
        active_log_prefixes.extend(cfg["log_prefixes"])
        active_logger_prefixes.extend(cfg["logger_prefixes"])
        active_param_patterns_by_type[ds_type] = cfg["param_patterns"]

    if not active_param_patterns:
        return  # Nothing to exclude

    def _matches(id_str, prefixes):
        return any(id_str.startswith(p) for p in prefixes)

    def _remove_from_parent(parent, child):
        if parent is not None and child in list(parent):
            parent.remove(child)

    # Remove state parameters
    for parent in root.iter():
        for param in list(parent.findall("parameter")):
            if _matches(param.attrib.get("id", ""), active_param_patterns):
                _remove_from_parent(parent, param)

    # Remove likelihood distributions
    for parent in root.iter():
        for dist in list(parent.findall("distribution")):
            if _matches(dist.attrib.get("id", ""), active_dist_prefixes):
                _remove_from_parent(parent, dist)

    # Remove priors
    for parent in root.iter():
        for prior in list(parent.findall("prior")):
            if _matches(prior.attrib.get("id", ""), active_prior_prefixes):
                _remove_from_parent(parent, prior)

    # Remove parameter references from operators
    for operator in root.findall(".//operator"):
        for tag in ["parameter", "upLogParameter", "downRealParameter",
                     "upRealParameter", "downLogParameter"]:
            for elem in list(operator.findall(f".//{tag}[@idref]")):
                if _matches(elem.attrib.get("idref", ""), active_param_patterns):
                    for parent in operator.iter():
                        if elem in list(parent):
                            _remove_from_parent(parent, elem)
                            break

    # Remove standalone operators that ONLY reference excluded parameters
    if active_standalone_op_prefixes:
        for parent in root.iter():
            for operator in list(parent.findall("operator")):
                oid = operator.attrib.get("id", "")
                if not _matches(oid, active_standalone_op_prefixes):
                    continue
                all_refs = operator.findall(".//*[@idref]")
                if all(
                    _matches(ref.attrib.get("idref", ""), active_param_patterns)
                    for ref in all_refs
                ):
                    _remove_from_parent(parent, operator)

    # Remove log references in logger elements
    for logger in root.findall(".//logger"):
        for log_elem in list(logger.findall(".//log[@idref]")):
            if _matches(log_elem.attrib.get("idref", ""), active_log_prefixes):
                logger.remove(log_elem)
        for param_log in list(logger.findall(".//parameter[@idref]")):
            if param_log.attrib.get("name") == "log":
                if _matches(param_log.attrib.get("idref", ""), active_param_patterns):
                    logger.remove(param_log)

    # Remove entire logger elements (e.g. cumulativeIncidenceLogger when sero excluded)
    if active_logger_prefixes:
        for parent in root.iter():
            for logger in list(parent.findall("logger")):
                if _matches(logger.attrib.get("id", ""), active_logger_prefixes):
                    _remove_from_parent(parent, logger)


# Priors and operators for datastreams_onlytree variant (tree-only model; no datastream likelihoods).
ONLYTREE_PRIORS_XML = """
<prior id="SkylinePrev.Deme1.Prior.t:SimDataset" name="distribution">
            <x id="diff.SkylinePrev.Deme1.t:SimDataset" spec="mascot.util.Difference" arg="@SkylinePrev.Deme1.t:SimDataset"/>
            <Normal id="Normal.1" name="distr">
                <parameter id="RealParameter.73" spec="parameter.RealParameter" estimate="false" name="mean">0.0</parameter>
                <parameter id="RealParameter.83" spec="parameter.RealParameter" estimate="false" name="sigma">1.0</parameter>
            </Normal>
        </prior>
        <prior id="SkylinePrev.Deme2.Prior.t:SimDataset" name="distribution">
            <x id="diff.SkylinePrev.Deme2.t:SimDataset" spec="mascot.util.Difference" arg="@SkylinePrev.Deme2.t:SimDataset"/>
            <Normal id="Normal.2" name="distr">
                <parameter id="RealParameter.70" spec="parameter.RealParameter" estimate="false" name="mean">0.0</parameter>
                <parameter id="RealParameter.80" spec="parameter.RealParameter" estimate="false" name="sigma">1.0</parameter>
            </Normal>
        </prior>
        <prior id="SkylinePrev.Deme1.FirstPrior.t:SimDataset" name="distribution">
            <x id="first.SkylinePrev.Deme1.t:SimDataset" spec="mascot.util.Final" arg="@SkylinePrev.Deme1.t:SimDataset"/>
            <Normal id="Normal.3" name="distr">
                <parameter id="RealParameter.72" spec="parameter.RealParameter" estimate="false" name="mean">0.0</parameter>
                <parameter id="RealParameter.82" spec="parameter.RealParameter" estimate="false" name="sigma">1.0</parameter>
            </Normal>
        </prior>
        <prior id="SkylinePrev.Deme2.FirstPrior.t:SimDataset" name="distribution">
            <x id="first.SkylinePrev.Deme2.t:SimDataset" spec="mascot.util.Final" arg="@SkylinePrev.Deme2.t:SimDataset"/>
            <Normal id="Normal.4" name="distr">
                <parameter id="RealParameter.71" spec="parameter.RealParameter" estimate="false" name="mean">0.0</parameter>
                <parameter id="RealParameter.81" spec="parameter.RealParameter" estimate="false" name="sigma">1.0</parameter>
            </Normal>
        </prior>
"""


def apply_onlytree_operators_and_priors(root):
    """
    For datastreams_onlytree variant: remove PopSize parameters and add SkylinePrev Difference/First priors
    to the prior distribution.
    """
    # Remove PopSize state parameters (only used by datastream likelihoods)
    for parent in root.iter():
        for param in list(parent.findall("parameter")):
            pid = param.attrib.get("id", "")
            if pid.startswith("PopSize."):
                parent.remove(param)

    run = root.find(".//run[@id='mcmc']")
    if run is None:
        run = root.find(".//run")
    if run is None:
        return

    # Find the prior compound distribution (id="prior" under posterior)
    posterior = root.find(".//distribution[@id='posterior']")
    if posterior is None:
        return
    prior_dist = None
    for child in posterior:
        if child.attrib.get("id") == "prior":
            prior_dist = child
            break
    if prior_dist is None:
        return

    # Parse and append the 4 SkylinePrev priors
    priors_wrapper = ET.fromstring("<wrapper>" + ONLYTREE_PRIORS_XML + "</wrapper>")
    for prior_elem in priors_wrapper:
        prior_dist.append(prior_elem)


def _replace_xml_element(root, tag, element_id, new_xml_str):
    """Find an XML element by tag and id, replace it in-place with new XML string."""
    for parent in root.iter():
        for idx, child in enumerate(list(parent)):
            if child.tag == tag and child.attrib.get("id") == element_id:
                parent.remove(child)
                parent.insert(idx, ET.fromstring(new_xml_str))
                return
    print(f"Warning: element <{tag} id='{element_id}'> not found in template")


def _inject_traits(root, trait_block, type_trait_block, infer_tree=False):
    """Replace the dateTrait and typeTraitSet blocks in the template."""
    if infer_tree:
        _replace_xml_element(root, "trait", "dateTrait.t:SimDataset", trait_block)
    _replace_xml_element(root, "typeTrait", "typeTraitSet.t:SimDataset", type_trait_block)


# Mapping from XML parameter id prefix -> (datastream dict key for values, datastream dict key for times)
_DATASTREAM_PARAM_MAP = {
    "caseCounts.": ("case_counts_by_deme", [("caseCounts.", "counts"), ("caseTimes.", "times")]),
    "seroTestedCounts.": (
        "seroprevalence_by_deme",
        [
            ("seroTestedCounts.", "tested_counts"),
            ("seroWithAntibodiesCounts.", "with_antibodies_counts"),
            ("seroTestedTimes.", "times"),
        ],
    ),
    "wastewaterConcentration.": (
        "wastewater_by_deme",
        [("wastewaterConcentration.", "wastewater"), ("wastewaterConcentrationTimes.", "times")],
    ),
}


def _inject_datastream_params(root, gamma, case_counts_by_deme, seroprevalence_by_deme, wastewater_by_deme):
    """Update <state> parameter values for case counts, seroprevalence, and wastewater."""
    sources = {
        "case_counts_by_deme": case_counts_by_deme,
        "seroprevalence_by_deme": seroprevalence_by_deme,
        "wastewater_by_deme": wastewater_by_deme,
    }

    for param in root.findall(".//parameter"):
        pid = param.attrib.get("id", "")
        if not pid.endswith(":SimDataset"):
            continue

        # Handle gamma (uninfectious rate)
        if pid == "uninfectiousRate.t:SimDataset" and gamma is not None:
            param.text = str(gamma)
            continue

        # Match against datastream parameter prefixes
        for _trigger_prefix, (source_key, mappings) in _DATASTREAM_PARAM_MAP.items():
            by_deme = sources.get(source_key)
            if by_deme is None:
                continue
            for prefix, value_key in mappings:
                if pid.startswith(prefix):
                    deme = pid.split(prefix, 1)[1].split(":", 1)[0]
                    if deme in by_deme and value_key in by_deme[deme]:
                        param.text = by_deme[deme][value_key]
                    break


def _configure_rate_shifts(root, max_age, min_age):
    """Update SkygrowthRateShifts and SplineGridRateShifts with linspace values."""
    if max_age is None or min_age is None:
        return
    max_age += 0.001  # small epsilon to avoid floating point boundary issues

    skygrowth = root.find(".//rateShifts[@id='SkygrowthRateShifts']")
    if skygrowth is not None:
        skygrowth.text = " ".join(f"{v:.4f}" for v in np.linspace(min_age, max_age, 11))

    spline_grid = root.find(".//gridRateShifts[@id='SplineGridRateShifts']")
    if spline_grid is not None:
        spline_grid.text = " ".join(f"{v:.4f}" for v in np.linspace(min_age, max_age, 1001))


def _configure_mcmc(root, chain_length, use_coupled_mcmc, chains, target,
                     log_heated_chains, delta_temperature, optimise, resample_every):
    """Set MCMC or CoupledMCMC spec and attributes on the <run> element."""
    for run in root.findall(".//run"):
        if run.attrib.get("id") != "mcmc":
            continue
        if use_coupled_mcmc:
            run.set("spec", "coupledMCMC.CoupledMCMC")
            run.set("chainLength", str(chain_length))
            run.set("chains", str(chains))
            run.set("target", str(target))
            run.set("logHeatedChains", "true" if log_heated_chains else "false")
            run.set("deltaTemperature", str(delta_temperature))
            run.set("optimise", "true" if optimise else "false")
            run.set("resampleEvery", str(resample_every))
        else:
            run.set("spec", "MCMC")
            run.set("chainLength", str(chain_length))


def _write_xml(root, output_path):
    """Pretty-print XML tree and write to file."""
    xml_str = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(xml_str)
    pretty_xml = parsed.toprettyxml(indent="  ")
    # Remove empty lines introduced by minidom
    lines = [line for line in pretty_xml.splitlines() if line.strip() != ""]
    print(f"Writing XML to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _build_output_suffix(output_suffix_override, clip_trans_rate,
                          exclude_case_counts, exclude_seroprevalence,
                          exclude_wastewater, is_datastream_template):
    """Determine the output filename suffix."""
    if is_datastream_template:
        if output_suffix_override is not None:
            return "_" + output_suffix_override
        suffix_parts = ["_datastreams"]
        if exclude_case_counts:
            suffix_parts.append("_nocasecounts")
        if exclude_seroprevalence:
            suffix_parts.append("_noseroprevalence")
        if exclude_wastewater:
            suffix_parts.append("_nowastewater")
        return "".join(suffix_parts)
    else:
        if output_suffix_override != VARIANT_ORIGINAL:
            return "_" + output_suffix_override
        return "_" + VARIANT_ORIGINAL


def replace_blocks_template(
    template_path,
    xml_name,
    mascot_alignment_block,
    trait_block,
    type_trait_block,
    gamma,
    case_counts_by_deme=None,
    seroprevalence_by_deme=None,
    wastewater_by_deme=None,
    chain_length=1000000,
    use_coupled_mcmc=False,
    chains=4,
    target=0.234,
    log_heated_chains=True,
    delta_temperature=0.1,
    optimise=True,
    resample_every=1000,
    max_age=None,
    min_age=None,
    newick_tree=None,
    exclude_case_counts=False,
    exclude_seroprevalence=False,
    exclude_wastewater=False,
    use_fixed_tree=True,
    output_suffix_override=None,
    clip_trans_rate=True,
):
    """
    Replace <data>, <trait>, and datastream parameter blocks in a Mascot template
    XML, configure MCMC, and write the resulting XML file.
    """
    tree = ET.parse(template_path)
    root = tree.getroot()

    is_datastream_template = (
        case_counts_by_deme is not None
        or seroprevalence_by_deme is not None
        or wastewater_by_deme is not None
        or (output_suffix_override is not None and output_suffix_override != VARIANT_ORIGINAL)
    )

    # Set clipTransRate on Spline elements for datastream templates
    if is_datastream_template:
        clip_val = "true" if clip_trans_rate else "false"
        for elem in root.iter():
            if elem.get("spec") == "mascotdatastreams.dynamics.Spline":
                elem.set("clipTransRate", clip_val)

    # Remove existing SimDataset data block
    for data_elem in list(root.findall(".//data")):
        if data_elem.attrib.get("id") == "SimDataset":
            root.remove(data_elem)

    # Inject alignment, traits
    _inject_traits(root, trait_block, type_trait_block, infer_tree=use_fixed_tree == False)

    # Inject datastream parameter values
    if is_datastream_template:
        _inject_datastream_params(
            root, gamma, case_counts_by_deme, seroprevalence_by_deme, wastewater_by_deme
        )

    # Remove excluded elements and handle onlytree variant
    if exclude_case_counts or exclude_seroprevalence or exclude_wastewater:
        remove_excluded_datastream_elements(
            root,
            exclude_case_counts=exclude_case_counts,
            exclude_seroprevalence=exclude_seroprevalence,
            exclude_wastewater=exclude_wastewater,
        )
        if exclude_case_counts and exclude_seroprevalence and exclude_wastewater:
            apply_onlytree_operators_and_priors(root)

    _configure_rate_shifts(root, max_age, min_age)

    # Insert alignment at root
    root.insert(0, ET.fromstring(mascot_alignment_block))

    _configure_mcmc(root, chain_length, use_coupled_mcmc, chains, target,
                     log_heated_chains, delta_temperature, optimise, resample_every)

    # Inject fixed Newick tree
    if newick_tree is not None:
        init_elem = root.find(".//init[@spec='beast.base.evolution.tree.TreeParser']")
        if init_elem is not None:
            init_elem.set("newick", newick_tree.strip())

    # Disable tree likelihood for nomascotll variant
    if not use_fixed_tree:
        mascot_logp = root.find(
            ".//*[@spec='mascotdatastreams.distribution.MascotLogPflag']"
        )
        if mascot_logp is not None:
            mascot_logp.set("compute_likelihood", "false")

    output_suffix = _build_output_suffix(
        output_suffix_override, clip_trans_rate,
        exclude_case_counts, exclude_seroprevalence, exclude_wastewater,
        is_datastream_template,
    )
    _write_xml(root, xml_name + output_suffix + ".xml")


def _build_datastream_by_deme(csv_file, time_col, value_columns, label):
    """
    Generic reader for per-deme datastream CSVs.

    Args:
        csv_file: Path to CSV.
        time_col: Name of the time column (e.g. 't_case_counts_frommostrecentsample').
        value_columns: Dict mapping output key -> CSV column name.
            Example: {'counts': 'case_counts', 'times': time_col}
        label: Human-readable label for error messages (e.g. 'Case counts').

    Returns:
        (by_deme_dict, deme_map, max_age, min_age)
    """
    df = pd.read_csv(csv_file)

    possible_deme_cols = ["deme", "index", "location"]
    deme_col = next((c for c in possible_deme_cols if c in df.columns), None)
    if deme_col is None:
        raise ValueError(
            f"{label} CSV must contain a 'deme' (or 'index'/'location') column."
        )

    required = [time_col] + list(value_columns.values())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} CSV missing required columns: {missing}")

    df_sorted = df.sort_values(by=[deme_col, time_col])

    deme_map = {
        str(deme): f"Deme{idx+1}"
        for idx, deme in enumerate(df_sorted[deme_col].unique())
    }

    by_deme = {}
    for deme, sub in df_sorted.groupby(deme_col):
        entry = {
            out_key: " ".join(sub[csv_col].astype(str))
            for out_key, csv_col in value_columns.items()
        }
        entry["times"] = " ".join(sub[time_col].astype(str))
        by_deme[deme_map[str(deme)]] = entry

    max_age = df_sorted[time_col].max()
    min_age = df_sorted[time_col].min()
    return by_deme, deme_map, max_age, min_age


def build_case_counts_by_deme(
    case_counts_file, remove_small_counts=False, add1tocounts=False
):
    """
    Read case counts CSV and build mapping per-deme for insertion into <state> parameters.

    Returns: (dict, deme_map, max_age, min_age)
        dict like {'Deme1': {'counts': '5 7 6', 'times': '0.0 0.1 0.2'}, ...}
    """
    if remove_small_counts or add1tocounts:
        # Pre-process before handing to generic reader
        df = pd.read_csv(case_counts_file)
        if remove_small_counts:
            df = df.loc[df["case_counts"] > 10]
        if add1tocounts:
            df["case_counts"] = df["case_counts"] + 1
        # Write to temp file so generic reader can process it
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            df.to_csv(tmp, index=False)
            case_counts_file = tmp.name

    return _build_datastream_by_deme(
        case_counts_file,
        time_col="t_case_counts_frommostrecentsample",
        value_columns={"counts": "case_counts"},
        label="Case counts",
    )


def build_seroprevalence_by_deme(seroprevalence_file):
    """Read seroprevalence CSV and build mapping per-deme."""
    return _build_datastream_by_deme(
        seroprevalence_file,
        time_col="t_seroprevalence_frommostrecentsample",
        value_columns={
            "tested_counts": "seroprevalence_numpeopletested",
            "with_antibodies_counts": "seroprevalence_numpeoplewithantibodies",
        },
        label="Seroprevalence",
    )


def build_wastewater_by_deme(wastewater_file):
    """Read wastewater CSV and build mapping per-deme."""
    return _build_datastream_by_deme(
        wastewater_file,
        time_col="t_wastewater_frommostrecentsample",
        value_columns={"wastewater": "wastewater"},
        label="Wastewater",
    )


def extract_leaf_states(tree_path):
    """
    Ensures the TRANSLATE block has no trailing comma before the semicolon, then uses DendroPy to parse the first tree and extract a dictionary mapping real leaf names to their state/deme/type.
    """

    # Read and fix the .trees file if necessary
    with open(tree_path, "r") as f:
        lines = f.readlines()

    in_translate = False
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("translate"):
            in_translate = True
            continue
        if in_translate:
            # Find the end of the TRANSLATE block
            if ";" in line:
                # Remove trailing comma from the line before if present
                prev_idx = i - 1
                # Only check if previous line is not 'Translate' and not empty
                while prev_idx > 0 and lines[prev_idx].strip() == "":
                    prev_idx -= 1
                if prev_idx > 0:
                    prev_line = lines[prev_idx].rstrip("\n")
                    if prev_line.rstrip().endswith(","):
                        lines[prev_idx] = prev_line.rstrip(",\n") + "\n"
                in_translate = False
                break

    # Write to a temporary file
    with open("tmp.trees", "w") as tmp:
        tmp.writelines(lines)

    # Now load with DendroPy
    tree = dendropy.Tree.get(
        path="tmp.trees", schema="nexus", preserve_underscores=True
    )
    taxon_namespace = tree.taxon_namespace

    # # Clean up the temporary file
    # os.remove("tmp.trees")

    # Build translation dictionary: numeric name -> real name
    translate_dict = {}
    for idx, taxon in enumerate(taxon_namespace):
        # Try to extract mapping from taxon.label (which should be the real name)
        # Numeric name is idx+1 as string (BEAST2 convention)
        translate_dict[str(idx + 1)] = taxon.label

    leaf_state = {}
    leaf_time = {}
    for leaf in tree.leaf_node_iter():
        real_label = leaf.taxon.label
        if leaf.annotations:
            state = leaf.annotations.get_value("type")
            # state = leaf.annotations.get_value("location")
            time = leaf.annotations.get_value("time")
        else:
            state = None
            time = None
        leaf_state[real_label] = state.replace("{", "").replace("}", "")
        leaf_time[real_label] = time
    return leaf_state, leaf_time


def get_uninfectious_rate(parameters_path):
    df = pd.read_csv(parameters_path)
    df = df[df["parameter"] == "gamma"]
    if df.shape[0] == 0:
        raise ValueError("No gamma parameter found in parameters file")
    return df["value"].values[0]


# ---------------------------------------------------------------------------
# main() helpers — extracted to keep main() under ~60 lines
# ---------------------------------------------------------------------------

def _detect_exclusions(args):
    """Determine which datastreams to exclude based on --variant_type or file checks."""
    if args.variant_type is not None:
        variant = args.variant_type
        exclude_cc = variant in ("datastreams_nocasecounts", "datastreams_onlytree", VARIANT_ORIGINAL)
        exclude_sp = variant in ("datastreams_noseroprevalence", "datastreams_onlytree", VARIANT_ORIGINAL)
        exclude_ww = variant in ("datastreams_nowastewater", "datastreams_onlytree", VARIANT_ORIGINAL)
        use_fixed_tree = variant != "datastreams_nomascotll"
    else:
        exclude_cc = (
            args.case_counts is None
            or args.case_counts == ""
            or (os.path.exists(args.case_counts) and os.path.getsize(args.case_counts) == 0)
        )
        exclude_sp = (
            args.seroprevalence is None
            or args.seroprevalence == ""
            or (os.path.exists(args.seroprevalence) and os.path.getsize(args.seroprevalence) == 0)
        )
        exclude_ww = (
            args.wastewater is None
            or args.wastewater == ""
            or (os.path.exists(args.wastewater) and os.path.getsize(args.wastewater) == 0)
        )
        use_fixed_tree = True
    return exclude_cc, exclude_sp, exclude_ww, use_fixed_tree


def _process_tree_data(args):
    """
    Read tree file, extract leaf states/times, convert to dates, and save state_time CSV.

    Returns (leaf_state_dict, leaf_time_dict, trees_content, max_age, min_age).
    """
    with open(args.tree, "r") as f:
        trees_content = f.read()

    leaf_state_dict, leaf_time_dict = extract_leaf_states(args.tree)
    leaf_time_dict_relativetoroot = leaf_time_dict.copy()

    max_age = 0.0
    min_age = 0.0

    if args.artificial_date is not None:
        artificial_date_dt = datetime.strptime(args.artificial_date, "%Y/%m/%d")
        for leaf in sorted(leaf_time_dict.keys()):
            rel_time = float(leaf_time_dict[leaf])
            max_age = max(max_age, rel_time)
            min_age = min(min_age, rel_time)
            rel_time_days = int(rel_time * 365)
            leaf_time_dict[leaf] = (
                artificial_date_dt + timedelta(days=rel_time_days)
            ).strftime("%Y/%m/%d")

    # Save state and time CSV
    state_time_csv = pd.DataFrame.from_dict(
        leaf_state_dict, orient="index", columns=["state"]
    )
    tmp = pd.DataFrame.from_dict(leaf_time_dict, orient="index", columns=["time"])
    tmp_relative = pd.DataFrame.from_dict(
        leaf_time_dict_relativetoroot, orient="index", columns=["time_relativetoroot"]
    )
    state_time_csv = (
        state_time_csv
        .merge(tmp, left_index=True, right_index=True)
        .merge(tmp_relative, left_index=True, right_index=True)
        .reset_index(drop=False)
        .rename(columns={"index": "sample_id"})
    )
    state_time_csv.sort_values(by="sample_id").to_csv(
        args.xml_name + "_state_time.csv", index=False
    )

    return leaf_state_dict, leaf_time_dict, trees_content, max_age, min_age


def _build_all_datastreams(args, exclude_cc, exclude_sp, exclude_ww, max_age, min_age):
    """
    Build per-deme dicts for each datastream type and compute encompassing age range.

    Returns (case_counts_by_deme, seroprevalence_by_deme, wastewater_by_deme, max_age, min_age).
    """
    builders = []
    if not exclude_cc and args.case_counts:
        if os.path.exists(args.case_counts) and os.path.getsize(args.case_counts) > 0:
            builders.append(("cc", lambda: build_case_counts_by_deme(
                args.case_counts, remove_small_counts=False, add1tocounts=args.add1tocounts
            )))
    if not exclude_sp and args.seroprevalence:
        if os.path.exists(args.seroprevalence) and os.path.getsize(args.seroprevalence) > 0:
            builders.append(("sp", lambda: build_seroprevalence_by_deme(args.seroprevalence)))
    if not exclude_ww and args.wastewater:
        if os.path.exists(args.wastewater) and os.path.getsize(args.wastewater) > 0:
            builders.append(("ww", lambda: build_wastewater_by_deme(args.wastewater)))

    results = {}
    age_values = [max_age, min_age]
    for key, builder_fn in builders:
        by_deme, _deme_map, ds_max, ds_min = builder_fn()
        results[key] = by_deme
        age_values.extend([ds_max, ds_min])

    max_age = max(v for v in age_values if v is not None)
    min_age = min(v for v in age_values if v is not None)
    if min_age < 0:
        raise ValueError("Min age is less than 0")

    return results.get("cc"), results.get("sp"), results.get("ww"), max_age, min_age


def _build_alignment_block(leaf_state_dict):
    """Generate placeholder <data> alignment XML block from leaf names."""
    seq_lines = [
        f'    <sequence id="seq_{leaf}" spec="Sequence" taxon="{leaf}" totalcount="4" value="????"/>'
        for leaf in sorted(leaf_state_dict.keys())
    ]
    return (
        '<data\nid="SimDataset"\nspec="Alignment"\nname="alignment">\n'
        + "\n".join(seq_lines)
        + "\n</data>"
    )


def _extract_newick(trees_content):
    """Parse NEXUS content, collapse single-child nodes, return Newick string."""
    newick_state0 = get_state0_newick(trees_content)
    tree = Phylo.read(StringIO(newick_state0), "newick")
    tree.root = collapse_single_child_nodes(tree.root)
    return tree.format("newick")


def _build_trait_block(leaf_time_dict):
    """Build the dateTrait XML block from leaf time mappings."""
    trait_value = ",".join(
        f"{leaf}={leaf_time_dict[leaf]}" for leaf in sorted(leaf_time_dict.keys())
    )
    return (
        '<trait id="dateTrait.t:SimDataset" spec="beast.base.evolution.tree.TraitSet"'
        f' traitname="date" value="{trait_value}">\n'
        '  <taxa id="TaxonSet.SimDataset" spec="TaxonSet">\n'
        '    <data idref="SimDataset" name="alignment"/>\n'
        "  </taxa>\n"
        "</trait>"
    )


def _build_type_trait_block(leaf_state_dict):
    """Build the typeTraitSet XML block from leaf state mappings."""
    type_trait_value = ",".join(
        f"{leaf}={str(leaf_state_dict[leaf]).replace('{','').replace('}','')}"
        for leaf in sorted(leaf_state_dict.keys())
    )
    return (
        '<typeTrait id="typeTraitSet.t:SimDataset" spec="mascot.util.InitializedTraitSet"'
        f' traitname="type" value="{type_trait_value}">\n'
        '  <taxa id="TaxonSet.1" spec="TaxonSet" alignment="@SimDataset"/>\n'
        "</typeTrait>"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create Mascot XML by injecting alignment, traits, and case counts."
    )
    parser.add_argument(
        "--datastream_template",
        type=str,
        required=True,
        help="Path to Mascot+datastreams template XML",
    )
    parser.add_argument(
        "--standard_template",
        type=str,
        required=True,
        help="Path to Mascot template XML",
    )
    parser.add_argument(
        "--xml_name",
        type=str,
        required=True,
        help="Name of the XML file to write",
    )
    parser.add_argument(
        "--tree",
        type=str,
        required=True,
        help="Path to the Nexus tree file (output from Remaster simulation)",
    )
    parser.add_argument(
        "--case_counts",
        type=str,
        required=False,
        default=None,
        help="Path to the case counts file (optional)",
    )
    parser.add_argument(
        "--seroprevalence",
        type=str,
        required=False,
        default=None,
        help="Path to the sero prevalence file (optional)",
    )
    parser.add_argument(
        "--wastewater",
        type=str,
        required=False,
        default=None,
        help="Path to the wastewater concentrations file (optional)",
    )
    parser.add_argument(
        "--parameters",
        type=str,
        required=True,
        help="Path to the parameters file",
    )
    parser.add_argument(
        "--artificial_date",
        type=str,
        default="2000/01/01",
        help="Artificial date of the root",
    )
    parser.add_argument(
        "--chain_length",
        type=int,
        default=1000000,
        help='Override MCMC chainLength in the <run id="mcmc"> element (optional)',
    )
    parser.add_argument(
        "--coupled_mcmc",
        action="store_true",
        help='If set, use beast.coupledMCMC.CoupledMCMC for <run id="mcmc"> instead of standard MCMC',
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of chains for coupled MCMC (only used if --coupled_mcmc)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.234,
        help="Target acceptance probability for coupled MCMC (only used if --coupled_mcmc)",
    )
    parser.add_argument(
        "--log_heated_chains",
        action="store_true",
        help="Log heated chains for coupled MCMC (only used if --coupled_mcmc)",
    )
    parser.add_argument(
        "--delta_temperature",
        type=float,
        default=0.1,
        help="Delta temperature for coupled MCMC (only used if --coupled_mcmc)",
    )
    parser.add_argument(
        "--optimise",
        dest="optimise",
        action="store_true",
        help="Enable optimisation for coupled MCMC (only used if --coupled_mcmc)",
    )
    parser.add_argument(
        "--resample_every",
        type=int,
        default=1000,
        help="Resample every N steps for coupled MCMC (only used if --coupled_mcmc)",
    )
    parser.add_argument(
        "--add1tocounts",
        action="store_true",
        help="Add 1 to case counts",
    )
    parser.add_argument(
        "--variant_type",
        type=str,
        default=None,
        help="Variant for output naming and exclusions: datastreams, datastreams_nocasecounts, datastreams_noseroprevalence, datastreams_nowastewater, datastreams_nomascotll, datastreams_onlytree, or original. When set, only one XML is written (xml_name + '_' + variant_type + '.xml').",
    )
    parser.add_argument(
        "--clip-trans-rate",
        type=str,
        choices=("true", "false"),
        default="false",
        help="For datastream template only: set clipTransRate on Spline elements to true or false. Default: false. Ignored when writing original (standard) template.",
    )
    args = parser.parse_args()
    clip_trans_rate = args.clip_trans_rate == "true"

    # Detect which data types are excluded (overridden by variant_type when provided)
    exclude_case_counts, exclude_seroprevalence, exclude_wastewater, use_fixed_tree = (
        _detect_exclusions(args)
    )

    # Process tree: extract leaf states, build alignment/trait XML blocks
    (
        leaf_state_dict,
        leaf_time_dict,
        trees_content,
        max_age,
        min_age,
    ) = _process_tree_data(args)

    gamma = get_uninfectious_rate(args.parameters)

    # Build datastream dicts and compute encompassing max/min age
    case_counts_by_deme, seroprevalence_by_deme, wastewater_by_deme, max_age, min_age = (
        _build_all_datastreams(args, exclude_case_counts, exclude_seroprevalence,
                                exclude_wastewater, max_age, min_age)
    )

    # Generate XML blocks from tree data
    mascot_alignment_block = _build_alignment_block(leaf_state_dict)
    newick_string = _extract_newick(trees_content)
    trait_block = _build_trait_block(leaf_time_dict)
    type_trait_block = _build_type_trait_block(leaf_state_dict)

    # Shared keyword arguments for replace_blocks_template
    common_kw = dict(
        chain_length=args.chain_length,
        use_coupled_mcmc=args.coupled_mcmc,
        chains=args.chains,
        target=args.target,
        log_heated_chains=args.log_heated_chains,
        delta_temperature=args.delta_temperature,
        optimise=args.optimise,
        resample_every=args.resample_every,
        newick_tree=newick_string,
    )

    # Determine which XMLs to write
    write_datastream = args.variant_type is None or args.variant_type in DATASTREAM_VARIANTS
    write_standard = args.variant_type is None or args.variant_type == VARIANT_ORIGINAL

    if write_datastream:
        replace_blocks_template(
            args.datastream_template,
            args.xml_name,
            mascot_alignment_block,
            trait_block,
            type_trait_block,
            gamma=gamma,
            case_counts_by_deme=case_counts_by_deme,
            seroprevalence_by_deme=seroprevalence_by_deme,
            wastewater_by_deme=wastewater_by_deme,
            max_age=max_age,
            min_age=min_age,
            exclude_case_counts=exclude_case_counts,
            exclude_seroprevalence=exclude_seroprevalence,
            exclude_wastewater=exclude_wastewater,
            use_fixed_tree=use_fixed_tree,
            output_suffix_override=(
                args.variant_type if args.variant_type in DATASTREAM_VARIANTS else None
            ),
            clip_trans_rate=clip_trans_rate,
            **common_kw,
        )
    if write_standard:
        replace_blocks_template(
            args.standard_template,
            args.xml_name,
            mascot_alignment_block,
            trait_block,
            type_trait_block,
            gamma=gamma,
            output_suffix_override=(
                VARIANT_ORIGINAL if args.variant_type == VARIANT_ORIGINAL else None
            ),
            **common_kw,
        )


if __name__ == "__main__":
    main()
