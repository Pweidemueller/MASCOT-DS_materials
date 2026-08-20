#!/usr/bin/env python3
import argparse
import copy
import os
from xml.etree import ElementTree as ET
from Bio import Phylo
from Bio import SeqIO
from io import StringIO
import dendropy
from xml.dom import minidom
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import sys

sys.setrecursionlimit(10000)

SARS_COV2_AGE = 2019.8333
MAX_RATE_SHIFT_OUTSIDE_DEME = SARS_COV2_AGE  # 2020.4

# Variant naming. Datastream variants all use the datastream template; the
# `original` variant uses the standard (Mascot-without-datastreams) template.
VARIANT_ORIGINAL = "original"
VARIANT_DATASTREAMS = "datastreams"
VARIANT_NO_CASE_COUNTS = "datastreams_nocasecounts"
VARIANT_NO_SEROPREVALENCE = "datastreams_noseroprevalence"
VARIANT_NO_WASTEWATER = "datastreams_nowastewater"
VARIANT_NO_GENETIC = "datastreams_nomascotll"  # no genetic data (sequences/tree)
VARIANT_ONLY_TREE = "datastreams_onlytree"  # just genetic data, no datastreams

DATASTREAM_VARIANTS = (
    VARIANT_DATASTREAMS,
    VARIANT_NO_CASE_COUNTS,
    VARIANT_NO_SEROPREVALENCE,
    VARIANT_NO_WASTEWATER,
    VARIANT_NO_GENETIC,
    VARIANT_ONLY_TREE,
)

ALL_VARIANTS = DATASTREAM_VARIANTS + (VARIANT_ORIGINAL,)

# Operators in the infer-tree datastream template that propose moves on the
# tree topology / node heights. Stripped for VARIANT_NO_GENETIC when the tree
# is being inferred, so the tree stays at its random initialization.
TREE_OPERATOR_IDS = (
    "MascotScaleAll.t:SimDataset",
    "MascotIntervalScaleOperator.t:SimDataset",
    "MascotRangeSlide.t:SimDataset",
    "MascotWeightBasedNarrow.t:SimDataset",
    "MascotHeightBasedNarrow.t:SimDataset",
    "MascotUntargetedWide.t:SimDataset",
    "MascotTargetedWide.t:SimDataset",
    "MascotTargetedWilsonBalding.t:SimDataset",
    "MascotTargetedWilsonBalding2.t:SimDataset",
)


def get_newick_tree(nexus_text):
    """
    Parse a NEXUS (or plain Newick) tree and return a clean Newick string
    with all node annotations stripped.  Uses dendropy for robust parsing.
    """
    try:
        tree = dendropy.Tree.get(data=nexus_text, schema="nexus")
    except Exception:
        tree = dendropy.Tree.get(data=nexus_text, schema="newick")
    return tree.as_string(schema="newick").strip()


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


# ---------------------------------------------------------------------------
# Dynamic deme expansion
# ---------------------------------------------------------------------------


def _replace_deme_refs(elem, src, dst):
    """Recursively replace *src* with *dst* in every attribute value and text node."""
    for attr_name, attr_val in list(elem.attrib.items()):
        if src in attr_val:
            elem.set(attr_name, attr_val.replace(src, dst))
    if elem.text and src in elem.text:
        elem.text = elem.text.replace(src, dst)
    if elem.tail and src in elem.tail:
        elem.tail = elem.tail.replace(src, dst)
    for child in elem:
        _replace_deme_refs(child, src, dst)


def _clone_element_for_deme(elem, dst_deme, src_deme="Deme1"):
    """Deep-copy *elem* and replace every occurrence of *src_deme* with *dst_deme*."""
    new_elem = copy.deepcopy(elem)
    _replace_deme_refs(new_elem, src_deme, dst_deme)
    return new_elem


def _clone_children_for_demes(parent, additional_demes, filter_tags=None):
    """
    For every direct child of *parent* whose ``id`` or ``idref`` contains
    ``"Deme1"``, insert deep-copies renamed for each deme in *additional_demes*.

    Processing in reverse index order so earlier insertions don't shift indices
    of later originals.
    """
    deme1_items = []
    for idx, child in enumerate(list(parent)):
        if filter_tags and child.tag not in filter_tags:
            continue
        combined = child.get("id", "") + child.get("idref", "")
        if "Deme1" in combined:
            deme1_items.append((idx, child))

    for idx, elem in reversed(deme1_items):
        for d_offset, deme in enumerate(additional_demes):
            new_elem = _clone_element_for_deme(elem, deme)
            parent.insert(idx + 1 + d_offset, new_elem)


def _clone_ne_dynamics_for_deme(deme1_ne, dst_deme):
    """
    Clone the Deme1 NeDynamics element for *dst_deme*.

    The first deme defines ``rateShifts`` and ``gridRateShifts`` inline (with an
    ``id`` attribute).  Additional demes must reference them via ``idref`` only.
    """
    new_ne = _clone_element_for_deme(deme1_ne, dst_deme)
    for spline in new_ne.iter():
        if spline.tag != "spline":
            continue
        for tag_name, shared_id in [
            ("rateShifts", "SkygrowthRateShifts"),
            ("gridRateShifts", "SplineGridRateShifts"),
        ]:
            for child in list(spline):
                if child.tag == tag_name and child.get("id"):
                    tail = child.tail
                    spline.remove(child)
                    ref = ET.SubElement(spline, tag_name)
                    ref.set("idref", shared_id)
                    ref.tail = tail
    return new_ne


def _add_inter_deme_spline_coupling(root, n_demes, skip_demes=()):
    """
    Insert ``<otherSpline>``, ``<incomingForwardMigration>`` and
    ``<forwardMigrationIndices>`` into each ``<neDynamics>`` block so the new
    MASCOT-DS structured-coalescent term can couple each deme's prevalence
    spline to the others.

    For deme X (1-indexed) we append, in ascending deme-index order over
    Y != X:

      * ``<otherSpline idref="splinePrev.DemeY.t:SimDataset"/>``

    followed by a single

      * ``<incomingForwardMigration idref="migrationRatesSkyline.t:SimDataset"/>``

    and a single

      * ``<forwardMigrationIndices id="fwdMigIdx.DemeX.t:SimDataset" .../>``

    whose values are the flat indices of the forward-migration rates
    ``(src=Y, dst=X)`` for each Y != X.  The matrix is laid out row-major
    over ``(src, dst)`` with the diagonal skipped, matching
    ``_build_migration_indicators``.

    ``skip_demes`` names demes whose own ``<neDynamics>`` block stays
    uncoupled (old wiring).  The coupled demes still reference the skipped
    demes' splines and migration indices, so the skip only suppresses
    elements *inside* the skipped deme's block.
    """
    ne_list = root.find(".//*[@id='NeDynamicsList.t:SimDataset']")
    if ne_list is None:
        return

    skip = set(skip_demes)

    deme_ne = {}
    for ne in ne_list.findall("neDynamics"):
        nid = ne.get("id", "")
        if nid.startswith("NeDynamics."):
            deme_ne[nid.split(".", 2)[1]] = ne

    for x_idx in range(n_demes):
        x_name = f"Deme{x_idx + 1}"
        if x_name in skip:
            continue
        ne = deme_ne.get(x_name)
        if ne is None:
            continue

        for tag in (
            "otherSpline",
            "incomingForwardMigration",
            "forwardMigrationIndices",
        ):
            for old in list(ne.findall(tag)):
                ne.remove(old)

        indices = []
        for y_idx in range(n_demes):
            if y_idx == x_idx:
                continue
            y_name = f"Deme{y_idx + 1}"
            other = ET.SubElement(ne, "otherSpline")
            other.set("idref", f"splinePrev.{y_name}.t:SimDataset")
            flat = y_idx * (n_demes - 1) + (x_idx if x_idx < y_idx else x_idx - 1)
            indices.append(flat)

        inc = ET.SubElement(ne, "incomingForwardMigration")
        inc.set("idref", "migrationRatesSkyline.t:SimDataset")

        fmi = ET.SubElement(ne, "forwardMigrationIndices")
        fmi.set("id", f"fwdMigIdx.{x_name}.t:SimDataset")
        fmi.set("spec", "parameter.IntegerParameter")
        fmi.set("estimate", "false")
        fmi.text = " ".join(str(i) for i in indices)


def _add_local_transmission_priors(root, n_demes, skip_demes=()):
    """
    Add a ``LocalTransmissionSmallerThan`` prior per deme so MCMC rejects
    states where the local transmission rate β^local goes non-positive at any
    spline grid point.  Required by the new MASCOT-DS coupling wiring.

    The priors are inserted at the top of the ``id="prior"`` compound
    distribution, immediately before the existing ``regularizeTransmissionRate``
    block (BEAST evaluates priors in order and this group must run first).

    ``skip_demes`` names demes whose prior should NOT be added (e.g. the
    background deme when it isn't coupled).  Idempotent: skips demes whose
    prior already exists.
    """
    prior_dist = root.find(".//*[@id='prior']")
    if prior_dist is None:
        return

    insert_idx = next(
        (
            i
            for i, child in enumerate(list(prior_dist))
            if child.get("id", "").startswith("regularizeTransmissionRate.")
        ),
        0,
    )

    skip = set(skip_demes)
    for x_idx in range(n_demes):
        x_name = f"Deme{x_idx + 1}"
        if x_name in skip:
            continue

        ne_id = f"NeDynamics.{x_name}.t:SimDataset"
        if root.find(f".//*[@id='{ne_id}']") is None:
            continue

        prior_id = f"regularizeLocalTransmissionRate.{x_name}.t:SimDataset"
        if root.find(f".//*[@id='{prior_id}']") is not None:
            continue

        dist_elem = ET.Element("distribution")
        dist_elem.set("id", prior_id)
        dist_elem.set("spec", "mascotdatastreams.util.LocalTransmissionSmallerThan")
        dynamics_ref = ET.SubElement(dist_elem, "dynamics")
        dynamics_ref.set("idref", ne_id)
        prior_dist.insert(insert_idx, dist_elem)
        insert_idx += 1


def _build_migration_indicators(n_demes):
    """
    Build the indicator string for the migration-rate matrix.

    Migration from any non-background deme into the background deme (last
    deme, highest index) is masked (``false``); all other pairwise migrations
    are ``true``.  The flat ordering is row-major over
    ``(src, dst)`` with ``src != dst``.
    """
    bg = n_demes - 1
    indicators = []
    for src in range(n_demes):
        for dst in range(n_demes):
            if dst == src:
                continue
            indicators.append("false" if dst == bg and src != bg else "true")
    return " ".join(indicators)


def expand_template_for_demes(
    root,
    n_demes,
    couple_deme_splines=False,
    outside_deme=False,
    couple_background_deme=False,
):
    """
    The datastream template contains per-deme elements only for ``Deme1``.
    This function replicates them for ``Deme2`` … ``Deme<n_demes>`` and
    updates dimension attributes that depend on the number of demes.

    When ``couple_deme_splines`` is true, each cloned ``<neDynamics>`` block
    is also given ``<otherSpline>`` / ``<incomingForwardMigration>`` /
    ``<forwardMigrationIndices>`` children for the new MASCOT-DS structured
    coalescent wiring.  Defaults to false to preserve the pre-coupling XML.

    When ``outside_deme`` is true the last-indexed deme is the background /
    ghost deme.  By default the background deme keeps the old (uncoupled)
    wiring even when the rest of the demes are coupled; set
    ``couple_background_deme`` to true to couple it the same way.
    """
    if n_demes < 2:
        raise ValueError("MASCOT requires at least 2 demes")

    additional = [f"Deme{i}" for i in range(2, n_demes + 1)]

    # 1. State parameters
    state = root.find(".//*[@id='state']")
    _clone_children_for_demes(state, additional, filter_tags={"parameter"})

    # 2. Prior / distribution block
    prior_dist = root.find(".//*[@id='prior']")
    _clone_children_for_demes(
        prior_dist, additional, filter_tags={"distribution", "prior"}
    )

    # 3. NeDynamics (special: inline → idref for rateShifts / gridRateShifts)
    ne_list = root.find(".//*[@id='NeDynamicsList.t:SimDataset']")
    deme1_ne = None
    for child in list(ne_list):
        if child.tag == "neDynamics" and "Deme1" in child.get("id", ""):
            deme1_ne = child
            break
    if deme1_ne is not None:
        insert_idx = list(ne_list).index(deme1_ne)
        for d_offset, deme in enumerate(additional):
            ne_list.insert(
                insert_idx + 1 + d_offset, _clone_ne_dynamics_for_deme(deme1_ne, deme)
            )

    # 3b. Inter-deme spline coupling (new MASCOT-DS package wiring; opt-in)
    if couple_deme_splines:
        skip_demes = ()
        if outside_deme and not couple_background_deme:
            skip_demes = (f"Deme{n_demes}",)
        _add_inter_deme_spline_coupling(root, n_demes, skip_demes=skip_demes)
        _add_local_transmission_priors(root, n_demes, skip_demes=skip_demes)

    # 4. AVMN operator – add per-deme parameter refs inside transformations
    for transform_id in (
        "AVMNLogTransform.Mascot.SimDataset",
        "AVMNNoTransform.Mascot.SimDataset",
    ):
        transform = root.find(f".//*[@id='{transform_id}']")
        if transform is None:
            continue
        _clone_children_for_demes(transform, additional, filter_tags={"parameter"})

    # 5. Per-deme operators (under <run>)
    run = root.find(".//*[@id='mcmc']")
    _clone_children_for_demes(run, additional, filter_tags={"operator"})

    # 6. Tracelog – clone per-deme log / parameter entries
    tracelog = root.find(".//*[@id='tracelog']")
    _clone_children_for_demes(tracelog, additional, filter_tags={"log", "parameter"})

    # 7. Per-deme loggers (under <run>)
    _clone_children_for_demes(run, additional, filter_tags={"logger"})

    # 8. Update dimension attributes
    n_mig = n_demes * (n_demes - 1)
    for elem_id, dim in [
        ("migrationRatesSkyline.t:SimDataset", n_mig),
        ("StructuredSkyline.t:SimDataset", n_demes),
        ("indicatorsSkyline.t:SimDataset", n_mig),
    ]:
        elem = root.find(f".//*[@id='{elem_id}']")
        if elem is not None:
            elem.set("dimension", str(dim))

    # 9. Set migration indicators: mask migration INTO the last background deme
    indicators_elem = root.find(".//*[@id='indicatorsSkyline.t:SimDataset']")
    if indicators_elem is not None:
        indicators_elem.text = _build_migration_indicators(n_demes)

    # 10. Set explicit types so MASCOT uses all demes (including ghost demes
    #     without sequences) rather than counting unique types from the tree.
    dynamics_elem = root.find(".//*[@id='StructuredSkyline.t:SimDataset']")
    if dynamics_elem is not None:
        all_deme_names = " ".join(f"Deme{i}" for i in range(1, n_demes + 1))
        dynamics_elem.set("types", all_deme_names)


def strip_background_deme_datastreams(root, bg_deme):
    """
    Remove all datastream observation elements (case counts, seroprevalence,
    wastewater, PopSize) for the background deme while keeping prevalence /
    NeDynamics / SkylinePrev elements intact.
    """
    datastream_state_prefixes = [
        f"caseCounts.{bg_deme}:",
        f"caseTimes.{bg_deme}:",
        f"caseCounts.scaling.{bg_deme}:",
        f"seroTestedCounts.{bg_deme}:",
        f"seroWithAntibodiesCounts.{bg_deme}:",
        f"seroTestedTimes.{bg_deme}:",
        f"seroprevalence.scaling.{bg_deme}:",
        f"wastewaterConcentration.{bg_deme}:",
        f"wastewaterConcentrationTimes.{bg_deme}:",
        f"wastewater.scaling.{bg_deme}:",
        f"PopSize.{bg_deme}:",
    ]

    # 1. State parameters
    state = root.find(".//*[@id='state']")
    for param in list(state.findall("parameter")):
        pid = param.get("id", "")
        if any(pid.startswith(p) for p in datastream_state_prefixes):
            state.remove(param)

    # 2. Likelihood distributions and scaling priors
    dist_id_prefixes = [
        f"caseCountLikelihood.{bg_deme}.",
        f"seroprevalenceLikelihood.{bg_deme}.",
        f"wastewaterLikelihood.{bg_deme}.",
        f"CaseCountsScaling.{bg_deme}.",
        f"SeroprevalenceScaling.{bg_deme}.",
        f"WastewaterScaling.{bg_deme}.",
    ]
    prior_dist = root.find(".//*[@id='prior']")
    for child in list(prior_dist):
        cid = child.get("id", "")
        if any(cid.startswith(p) for p in dist_id_prefixes):
            prior_dist.remove(child)

    # 3. UpDownPrevScaling operator (references scaling params that no longer exist)
    run = root.find(".//*[@id='mcmc']")
    for op in list(run.findall("operator")):
        if op.get("id", "") == f"UpDownPrevScaling.{bg_deme}:SimDataset":
            run.remove(op)

    # 4. AVMN LogTransform – remove scaling parameter refs for background
    avmn_log = root.find(".//*[@id='AVMNLogTransform.Mascot.SimDataset']")
    if avmn_log is not None:
        scaling_idrefs = {
            f"caseCounts.scaling.{bg_deme}:SimDataset",
            f"seroprevalence.scaling.{bg_deme}:SimDataset",
            f"wastewater.scaling.{bg_deme}:SimDataset",
        }
        for param in list(avmn_log.findall("parameter")):
            if param.get("idref", "") in scaling_idrefs:
                avmn_log.remove(param)

    # 5. Tracelog entries
    tracelog = root.find(".//*[@id='tracelog']")
    tracelog_remove_idrefs = {
        f"caseCountLikelihood.{bg_deme}.t:SimDataset",
        f"seroprevalenceLikelihood.{bg_deme}.t:SimDataset",
        f"wastewaterLikelihood.{bg_deme}.t:SimDataset",
        f"caseCounts.scaling.{bg_deme}:SimDataset",
        f"seroprevalence.scaling.{bg_deme}:SimDataset",
        f"wastewater.scaling.{bg_deme}:SimDataset",
    }
    for child in list(tracelog):
        idref = child.get("idref", "")
        if idref in tracelog_remove_idrefs:
            tracelog.remove(child)

    # 6. cumulativeIncidenceLogger
    for logger in list(run.findall("logger")):
        if logger.get("id", "") == f"cumulativeIncidenceLogger.{bg_deme}.t:SimDataset":
            run.remove(logger)

    # 7. NeScaler – remove ghost deme state param, prior, operator, AVMN ref,
    #    NeDynamics child element, and tracelog entry.
    # EDIT: decided to keep the NeScaler for the ghost deme
    # ne_scaler_id = f"NeScaler.{bg_deme}.t:SimDataset"

    # for param in list(state.findall("parameter")):
    #     if param.get("id", "") == ne_scaler_id:
    #         state.remove(param)

    # for child in list(prior_dist):
    #     if child.get("id", "") == f"NeScaler.{bg_deme}.Prior.t:SimDataset":
    #         prior_dist.remove(child)

    # for op in list(run.findall("operator")):
    #     if op.get("id", "") == f"NeScaler.{bg_deme}.Scaler.t:SimDataset":
    #         run.remove(op)

    # if avmn_log is not None:
    #     for param in list(avmn_log.findall("parameter")):
    #         if param.get("idref", "") == ne_scaler_id:
    #             avmn_log.remove(param)

    # ne_dynamics = root.find(f".//*[@id='NeDynamics.{bg_deme}.t:SimDataset']")
    # if ne_dynamics is not None:
    #     for child in list(ne_dynamics):
    #         if child.tag == "NeScaler" and child.get("idref", "") == ne_scaler_id:
    #             ne_dynamics.remove(child)

    # for child in list(tracelog):
    #     if child.get("idref", "") == ne_scaler_id:
    #         tracelog.remove(child)


def inject_ne_scaler_up_down(
    root, n_demes, outside_deme, estimate_clock_rate=True, estimate_ne_scaler_mean=True
):
    """
    Inject the NeScalerUpDown BactrianUpDownOperator after the per-deme
    NeScaler scaler operators.

    Couples NeScaler.MEAN (when *estimate_ne_scaler_mean* is True) and all
    local per-deme NeScaler parameters (up) against all local SkylinePrev
    parameters and (when *estimate_clock_rate* is True) the strict clock
    rate (down).  This single operator addresses the joint ridge: Ne ↑ →
    deeper tree → clockRate must ↓, while SkylinePrev adjusts down to
    compensate for the larger NeScaler in the Ne calculation.

    The ghost/outside deme is excluded because it has no NeScaler.

    When both *estimate_clock_rate* and *estimate_ne_scaler_mean* are False
    the operator is nonsensical (no down elements, no MEAN up element) and
    is not injected at all.
    """
    if not estimate_clock_rate and not estimate_ne_scaler_mean:
        return

    n_local = n_demes - 1 if outside_deme else n_demes
    local_demes = [f"Deme{i}" for i in range(1, n_local + 1)]

    ne_scaler_up = (
        '<up idref="NeScaler.MEAN.t:SimDataset"/>' if estimate_ne_scaler_mean else ""
    ) + "".join(f'<up idref="NeScaler.{d}.t:SimDataset"/>' for d in local_demes)
    down_elements = ""
    # down_elements = "".join(
    #     f'<down idref="SkylinePrev.{d}.t:SimDataset"/>' for d in local_demes
    # )
    if estimate_clock_rate:
        down_elements += '<down idref="clockRate.c:SimDataset"/>'

    op_xml = (
        '<operator id="NeScalerUpDown.t:SimDataset"'
        ' spec="operator.kernel.BactrianUpDownOperator"'
        ' scaleFactor="0.75" weight="3.0">'
        + ne_scaler_up
        + down_elements
        + "</operator>"
    )

    run = root.find(".//*[@id='mcmc']")
    if run is None:
        return

    # Insert after the last NeScaler.DemeX.Scaler operator
    insert_after_idx = None
    for idx, child in enumerate(list(run)):
        op_id = child.get("id", "")
        if op_id.startswith("NeScaler.Deme") and op_id.endswith(".Scaler.t:SimDataset"):
            insert_after_idx = idx

    if insert_after_idx is None:
        return

    run.insert(insert_after_idx + 1, ET.fromstring(op_xml))


def configure_seroprevalence_scaling(root, estimate):
    """
    Toggle estimation of seroprevalence scaling parameters.

    When *estimate* is False (the default), the scaling parameters are fixed
    at their initial values and removed from all operators so BEAST does not
    sample them.  The seroprevalence likelihood still contributes — only the
    per-deme scaling multiplier is frozen.

    When *estimate* is True, estimation is enabled: ``estimate="false"`` is
    removed, a LogNormal prior is added for each deme's scaling parameter,
    and the parameter is included in the AVMN Mascot operator.
    """
    state = root.find(".//*[@id='state']")
    run = root.find(".//*[@id='mcmc']")
    prior_dist = root.find(".//*[@id='prior']")
    avmn_log = root.find(".//*[@id='AVMNLogTransform.Mascot.SimDataset']")

    # Collect seroprevalence.scaling parameters that exist in the state
    sero_params = [
        p
        for p in state.findall("parameter")
        if p.get("id", "").startswith("seroprevalence.scaling.")
    ]

    if not estimate:
        # Ensure estimate="false" on each parameter
        for p in sero_params:
            p.set("estimate", "false")

        # Remove from UpDownPrevScaling operators
        for op in run.findall("operator"):
            if op.get("id", "").startswith("UpDownPrevScaling."):
                for child in list(op):
                    if (
                        child.tag == "downRealParameter"
                        and "seroprevalence.scaling" in child.get("idref", "")
                    ):
                        op.remove(child)
    else:
        # Enable estimation — remove the fixed flag
        for p in sero_params:
            if "estimate" in p.attrib:
                del p.attrib["estimate"]

        # Add per-deme SeroprevalenceScaling priors (lost when ET strips
        # the XML comments in the template)
        for p in sero_params:
            deme = p.get("id").split("seroprevalence.scaling.", 1)[1].split(":", 1)[0]
            prior_xml = (
                f'<prior id="SeroprevalenceScaling.{deme}.Prior:SimDataset"'
                f' name="distribution"'
                f' x="@seroprevalence.scaling.{deme}:SimDataset">'
                f'<LogNormal id="SeroprevalenceScaling.{deme}.LogNormal"'
                f' name="distr">'
                f'<parameter id="SeroprevalenceScaling.{deme}.MeanLog"'
                f' spec="parameter.RealParameter" estimate="false"'
                f' name="M">0.0</parameter>'
                f'<parameter id="SeroprevalenceScaling.{deme}.SDLog"'
                f' spec="parameter.RealParameter" estimate="false"'
                f' name="S">0.5</parameter>'
                f"</LogNormal>"
                f"</prior>"
            )
            prior_dist.append(ET.fromstring(prior_xml))

        # Re-add to AVMN LogTransform (lost when ET strips the comment)
        if avmn_log is not None:
            for p in sero_params:
                ref = ET.SubElement(avmn_log, "parameter")
                ref.set("idref", p.get("id"))
                ref.set("name", "f")


def add_skyline_prev_priors(root, deme_names):
    """
    Add SkylinePrev Difference and First priors for the specified demes.

    These smoothing priors constrain the prevalence spline for demes that
    lack datastream observations (e.g. background/ghost demes).  They are
    intentionally NOT applied to demes of interest whose prevalence is
    already informed by case-count / seroprevalence / wastewater data.
    """
    prior_dist = root.find(".//*[@id='prior']")
    if prior_dist is None:
        return

    for deme in deme_names:
        diff_prior_xml = (
            f'<prior id="SkylinePrev.{deme}.Prior.t:SimDataset" name="distribution">'
            f'<x id="diff.SkylinePrev.{deme}.t:SimDataset" '
            f'spec="mascot.util.Difference" arg="@SkylinePrev.{deme}.t:SimDataset"/>'
            f'<Normal id="Normal.SkylinePrev.{deme}.Prior" name="distr">'
            f'<parameter id="Mean.SkylinePrev.{deme}.Prior" '
            f'spec="parameter.RealParameter" estimate="false" name="mean">0.0</parameter>'
            f'<parameter id="Sigma.SkylinePrev.{deme}.Prior" '
            f'spec="parameter.RealParameter" estimate="false" name="sigma">1.0</parameter>'
            f"</Normal>"
            f"</prior>"
        )
        # first_prior_xml = (
        #     f'<prior id="SkylinePrev.{deme}.FirstPrior.t:SimDataset" name="distribution">'
        #     f'<x id="first.SkylinePrev.{deme}.t:SimDataset" '
        #     f'spec="mascot.util.Final" arg="@SkylinePrev.{deme}.t:SimDataset"/>'
        #     f'<Normal id="Normal.SkylinePrev.{deme}.FirstPrior" name="distr">'
        #     f'<parameter id="Mean.SkylinePrev.{deme}.FirstPrior" '
        #     f'spec="parameter.RealParameter" estimate="false" name="mean">8.0</parameter>'
        #     f'<parameter id="Sigma.SkylinePrev.{deme}.FirstPrior" '
        #     f'spec="parameter.RealParameter" estimate="false" name="sigma">2.0</parameter>'
        #     f"</Normal>"
        #     f"</prior>"
        # )
        prior_dist.append(ET.fromstring(diff_prior_xml))
        # prior_dist.append(ET.fromstring(first_prior_xml))


def configure_ne_scaler_mean(root, estimate, value=None):
    """
    Toggle estimation of the NeScaler.MEAN parameter.

    When *estimate* is True (the default), everything stays as-is — the
    NeScaler.MEAN is sampled by existing operators.

    When *estimate* is False, the NeScaler.MEAN is fixed:
    - ``estimate="false"`` is set on the parameter (with optional *value*)
    - ``NeScalerMean.Scaler`` operator is removed
    - ``NeScalerMeanPrior`` is removed from the prior distribution
    - NeScaler.MEAN reference is stripped from ``AVMNLogTransform.Mascot``
    - NeScaler.MEAN ``up`` reference is stripped from ``NeScalerUpDown``
    """
    if estimate:
        return

    ne_mean_param = root.find(".//*[@id='NeScaler.MEAN.t:SimDataset']")
    if ne_mean_param is None:
        return
    ne_mean_param.set("estimate", "false")
    if value is not None:
        ne_mean_param.text = str(value)

    run = root.find(".//*[@id='mcmc']")
    if run is None:
        return

    # Remove the NeScalerMean scaler operator
    scaler = run.find("./operator[@id='NeScalerMean.Scaler.t:SimDataset']")
    if scaler is not None:
        run.remove(scaler)

    # Remove the NeScalerMeanPrior from the prior distribution
    prior_dist = root.find(".//*[@id='prior']")
    if prior_dist is not None:
        mean_prior = prior_dist.find(".//*[@id='NeScalerMeanPrior.t:SimDataset']")
        if mean_prior is not None:
            prior_dist.remove(mean_prior)

    # Remove NeScaler.MEAN from AVMNLogTransform.Mascot
    avmn_log = root.find(".//*[@id='AVMNLogTransform.Mascot.SimDataset']")
    if avmn_log is not None:
        for param in list(avmn_log):
            if param.get("idref") == "NeScaler.MEAN.t:SimDataset":
                avmn_log.remove(param)

    # Remove NeScaler.MEAN up-reference from NeScalerUpDown (may be injected
    # later, but guard here for robustness)
    ne_up_down = run.find("./operator[@id='NeScalerUpDown.t:SimDataset']")
    if ne_up_down is not None:
        for child in list(ne_up_down):
            if child.tag == "up" and child.get("idref") == "NeScaler.MEAN.t:SimDataset":
                ne_up_down.remove(child)


def configure_clock_rate(root, estimate, rate=None):
    """
    Toggle estimation of the strict clock rate.

    When *estimate* is True (the default), everything stays as-is — the clock
    rate is sampled by the existing operators.

    When *estimate* is False, the clock rate is fixed:
    - ``estimate="false"`` is set on the parameter (with optional *rate* value)
    - ``StrictClockRateScaler`` is removed (the nested ``AVMNOperator`` is
      re-parented so other operators keep working)
    - ``strictClockUpDownOperator`` is removed
    - ``MascotScaleAll`` ``down`` attribute referencing the clock is cleared
    - Clock-rate reference is stripped from ``AVMNLogTransform.SimDataset``
    """
    if estimate:
        return

    clock_param = root.find(".//*[@id='clockRate.c:SimDataset']")
    if clock_param is None:
        return
    clock_param.set("estimate", "false")
    if rate is not None:
        clock_param.text = str(rate)

    run = root.find(".//*[@id='mcmc']")

    # The AVMNOperator.SimDataset is *defined* inside StrictClockRateScaler.
    # Extract it before removing the scaler so other operators that reference
    # it (gammaShapeScaler, KappaScaler, FrequenciesExchanger) keep working.
    scaler = run.find("./operator[@id='StrictClockRateScaler.c:SimDataset']")
    if scaler is not None:
        avmn_def = scaler.find("./operator[@id='AVMNOperator.SimDataset']")
        if avmn_def is not None:
            # Remove clock-rate parameter from AVMNLogTransform.SimDataset
            log_transform = avmn_def.find(".//*[@id='AVMNLogTransform.SimDataset']")
            if log_transform is not None:
                for param in list(log_transform):
                    if param.get("idref") == "clockRate.c:SimDataset":
                        log_transform.remove(param)

            scaler_idx = list(run).index(scaler)
            run.insert(scaler_idx, avmn_def)

        run.remove(scaler)

    up_down = run.find("./operator[@id='strictClockUpDownOperator.c:SimDataset']")
    if up_down is not None:
        run.remove(up_down)

    # Remove clock-rate coupling from MascotScaleAll
    scale_all = run.find("./operator[@id='MascotScaleAll.t:SimDataset']")
    if scale_all is not None and "down" in scale_all.attrib:
        del scale_all.attrib["down"]

    # Remove clock-rate from NeScalerUpDown (injected later, but guard here)
    ne_up_down = run.find("./operator[@id='NeScalerUpDown.t:SimDataset']")
    if ne_up_down is not None:
        for child in list(ne_up_down):
            if child.tag == "down" and child.get("idref") == "clockRate.c:SimDataset":
                ne_up_down.remove(child)


def remove_excluded_datastream_elements(
    root,
    exclude_case_counts=False,
    exclude_seroprevalence=False,
    exclude_wastewater=False,
):
    """
    Remove XML elements related to excluded datastream types.

    Args:
        root: XML root element
        exclude_case_counts: If True, remove case count related elements
        exclude_seroprevalence: If True, remove seroprevalence related elements
        exclude_wastewater: If True, remove wastewater related elements
    """
    # Patterns for parameter IDs to remove
    case_counts_param_patterns = [
        "caseCounts.",
        "caseTimes.",
        "caseCounts.scaling.",
        "caseCounts.dispersion",
    ]
    seroprevalence_param_patterns = [
        "seroTestedCounts.",
        "seroWithAntibodiesCounts.",
        "seroTestedTimes.",
        "seroprevalence.scaling.",
    ]
    wastewater_param_patterns = [
        "wastewaterConcentration.",
        "wastewaterConcentrationTimes.",
        "wastewater.scaling.",
        "wastewater.sigma",
    ]

    # Helper function to find and remove elements from their parents
    def remove_from_parent(parent, child):
        """Remove child element from parent."""
        if parent is not None and child in list(parent):
            parent.remove(child)

    # Remove state parameters - iterate through all elements to find parents
    for parent in root.iter():
        for param in list(parent.findall("parameter")):
            pid = param.attrib.get("id", "")
            should_remove = False
            if exclude_case_counts and any(
                pid.startswith(p) for p in case_counts_param_patterns
            ):
                should_remove = True
            elif exclude_seroprevalence and any(
                pid.startswith(p) for p in seroprevalence_param_patterns
            ):
                should_remove = True
            elif exclude_wastewater and any(
                pid.startswith(p) for p in wastewater_param_patterns
            ):
                should_remove = True

            if should_remove:
                remove_from_parent(parent, param)

    # Remove likelihood distributions
    for parent in root.iter():
        for dist in list(parent.findall("distribution")):
            did = dist.attrib.get("id", "")
            should_remove = False
            if exclude_case_counts and did.startswith("caseCountLikelihood."):
                should_remove = True
            elif exclude_seroprevalence and did.startswith("seroprevalenceLikelihood."):
                should_remove = True
            elif exclude_wastewater and did.startswith("wastewaterLikelihood."):
                should_remove = True

            if should_remove:
                remove_from_parent(parent, dist)

    # Remove priors
    for parent in root.iter():
        for prior in list(parent.findall("prior")):
            pid = prior.attrib.get("id", "")
            should_remove = False
            if exclude_case_counts and (
                pid.startswith("CaseCountsScaling.")
                or pid.startswith("CaseCountDispersionPrior.")
            ):
                should_remove = True
            elif exclude_seroprevalence and pid.startswith("SeroprevalenceScaling."):
                should_remove = True
            elif exclude_wastewater and (
                pid.startswith("WastewaterScaling.")
                or pid.startswith("WastewaterSigmaPrior.")
            ):
                should_remove = True

            if should_remove:
                remove_from_parent(parent, prior)

    # Remove parameter references from operators (not entire operators)
    for operator in root.findall(".//operator"):
        # Find all parameter/idref elements within this operator
        for param_ref in list(operator.findall(".//parameter[@idref]")):
            idref = param_ref.attrib.get("idref", "")
            should_remove = False
            if exclude_case_counts and any(
                idref.startswith(p) for p in case_counts_param_patterns
            ):
                should_remove = True
            elif exclude_seroprevalence and any(
                idref.startswith(p) for p in seroprevalence_param_patterns
            ):
                should_remove = True
            elif exclude_wastewater and any(
                idref.startswith(p) for p in wastewater_param_patterns
            ):
                should_remove = True

            if should_remove:
                # Find parent of param_ref within operator
                for parent in operator.iter():
                    if param_ref in list(parent):
                        remove_from_parent(parent, param_ref)
                        break

        # Also check for upLogParameter, downRealParameter, etc.
        for elem_type in [
            "upLogParameter",
            "downRealParameter",
            "upRealParameter",
            "downLogParameter",
        ]:
            for elem in list(operator.findall(f".//{elem_type}[@idref]")):
                idref = elem.attrib.get("idref", "")
                should_remove = False
                if exclude_case_counts and any(
                    idref.startswith(p) for p in case_counts_param_patterns
                ):
                    should_remove = True
                elif exclude_seroprevalence and any(
                    idref.startswith(p) for p in seroprevalence_param_patterns
                ):
                    should_remove = True
                elif exclude_wastewater and any(
                    idref.startswith(p) for p in wastewater_param_patterns
                ):
                    should_remove = True

                if should_remove:
                    # Find parent of elem within operator
                    for parent in operator.iter():
                        if elem in list(parent):
                            remove_from_parent(parent, elem)
                            break

    # Remove standalone operators that ONLY operate on excluded parameters
    if exclude_case_counts:
        for parent in root.iter():
            for operator in list(parent.findall("operator")):
                oid = operator.attrib.get("id", "")
                if oid.startswith("CaseCountsDispersionScalerX."):
                    # Check if this operator only references case counts parameters
                    all_refs = operator.findall(".//*[@idref]")
                    only_case_counts = True
                    for ref in all_refs:
                        idref = ref.attrib.get("idref", "")
                        if not any(
                            idref.startswith(p) for p in case_counts_param_patterns
                        ):
                            only_case_counts = False
                            break
                    if only_case_counts:
                        remove_from_parent(parent, operator)

    if exclude_wastewater:
        for parent in root.iter():
            for operator in list(parent.findall("operator")):
                oid = operator.attrib.get("id", "")
                if oid.startswith("WasterwaterSigmaScalerX.") or oid.startswith(
                    "WastewaterSigmaScalerX."
                ):
                    # Check if this operator only references wastewater parameters
                    all_refs = operator.findall(".//*[@idref]")
                    only_wastewater = True
                    for ref in all_refs:
                        idref = ref.attrib.get("idref", "")
                        if not any(
                            idref.startswith(p) for p in wastewater_param_patterns
                        ):
                            only_wastewater = False
                            break
                    if only_wastewater:
                        remove_from_parent(parent, operator)

    # Remove log references in logger elements
    for logger in root.findall(".//logger"):
        for log_elem in list(logger.findall(".//log[@idref]")):
            idref = log_elem.attrib.get("idref", "")
            should_remove = False
            if exclude_case_counts and idref.startswith("caseCountLikelihood."):
                should_remove = True
            elif exclude_seroprevalence and idref.startswith(
                "seroprevalenceLikelihood."
            ):
                should_remove = True
            elif exclude_wastewater and idref.startswith("wastewaterLikelihood."):
                should_remove = True

            if should_remove:
                logger.remove(log_elem)

        # Also remove parameter log references
        for param_log in list(logger.findall(".//parameter[@idref]")):
            if param_log.attrib.get("name") == "log":
                idref = param_log.attrib.get("idref", "")
                should_remove = False
                if exclude_case_counts and any(
                    idref.startswith(p) for p in case_counts_param_patterns
                ):
                    should_remove = True
                elif exclude_seroprevalence and any(
                    idref.startswith(p) for p in seroprevalence_param_patterns
                ):
                    should_remove = True
                elif exclude_wastewater and any(
                    idref.startswith(p) for p in wastewater_param_patterns
                ):
                    should_remove = True

                if should_remove:
                    logger.remove(param_log)

    # Remove cumulativeIncidenceLogger elements when seroprevalence is excluded
    if exclude_seroprevalence:
        for parent in root.iter():
            for logger in list(parent.findall("logger")):
                logger_id = logger.attrib.get("id", "")
                if logger_id.startswith("cumulativeIncidenceLogger."):
                    remove_from_parent(parent, logger)


def disable_genetic_data(root, remove_tree_operators, outside_deme=False, bg_deme=None):
    """
    Strip all genetic-data contributions for VARIANT_NO_GENETIC.

    - Sets ``compute_likelihood="false"`` on the MascotLogPflag distribution so
      the coalescent likelihood doesn't condition on the tree.
    - Removes the substitution-model ``treeLikelihood`` distribution if present
      (only emitted by the infer-tree template), plus any ``<log>`` references
      to it inside loggers (otherwise BEAST errors with "Could not find object
      associated with idref treeLikelihood.SimDataset").
    - When *remove_tree_operators* is True (infer-tree mode), removes the
      tree-move operators listed in TREE_OPERATOR_IDS so the tree stays at its
      RandomTree initialization.
    - When *outside_deme* is True, removes every operator that proposes moves
      on the background/ghost deme's ``SkylinePrev.<bg_deme>`` parameter (its
      own AdaptableOperatorSampler + BactrianRandomWalkBlockOperator, and its
      idref inside the shared ``AVMNNoTransform.Mascot.SimDataset``), and
      likewise for ``NeScaler.<bg_deme>`` (its own BactrianScaleOperator, and
      its idref inside ``AVMNLogTransform.Mascot.SimDataset``). With the
      coalescent likelihood disabled and the background deme carrying no
      datastream likelihoods of its own (see
      ``strip_background_deme_datastreams``), nothing informs either
      parameter in VARIANT_NO_GENETIC, so no operator should be spending
      weight sampling them.
    """
    mascot_logp = root.find(
        ".//*[@spec='mascotdatastreams.distribution.MascotLogPflag']"
    )
    if mascot_logp is not None:
        mascot_logp.set("compute_likelihood", "false")

    # Drop substitution-model likelihood (infer-tree template only) and any
    # dangling log references to it.
    treelik_id = "treeLikelihood.SimDataset"
    for parent in root.iter():
        for dist in list(parent.findall("distribution")):
            if dist.attrib.get("id") == treelik_id:
                parent.remove(dist)
    for logger in root.findall(".//logger"):
        for log_elem in list(logger.findall("log")):
            if log_elem.attrib.get("idref") == treelik_id:
                logger.remove(log_elem)

    if outside_deme and bg_deme is not None:
        run = root.find(".//*[@id='mcmc']")
        if run is not None:
            for op_id in (
                f"SkylinePrev.{bg_deme}.Scaler.t:SimDataset",
                f"SkylinePrev.{bg_deme}.ScalerXX.t:SimDataset",
            ):
                op = run.find(f"./operator[@id='{op_id}']")
                if op is not None:
                    run.remove(op)
        avmn_no_transform = root.find(".//*[@id='AVMNNoTransform.Mascot.SimDataset']")
        if avmn_no_transform is not None:
            bg_skyline_idref = f"SkylinePrev.{bg_deme}.t:SimDataset"
            for param in list(avmn_no_transform.findall("parameter")):
                if param.attrib.get("idref") == bg_skyline_idref:
                    avmn_no_transform.remove(param)

        if run is not None:
            ne_scaler_op = run.find(
                f"./operator[@id='NeScaler.{bg_deme}.Scaler.t:SimDataset']"
            )
            if ne_scaler_op is not None:
                run.remove(ne_scaler_op)
        avmn_log_transform = root.find(".//*[@id='AVMNLogTransform.Mascot.SimDataset']")
        if avmn_log_transform is not None:
            bg_ne_scaler_idref = f"NeScaler.{bg_deme}.t:SimDataset"
            for param in list(avmn_log_transform.findall("parameter")):
                if param.attrib.get("idref") == bg_ne_scaler_idref:
                    avmn_log_transform.remove(param)

    if not remove_tree_operators:
        return

    run = root.find(".//*[@id='mcmc']")
    if run is None:
        return
    for op in list(run.findall("operator")):
        if op.attrib.get("id") in TREE_OPERATOR_IDS:
            run.remove(op)


def apply_onlytree_operators_and_priors(root, n_demes):
    """
    For datastreams_onlytree variant: remove PopSize parameters and add
    SkylinePrev Difference / First priors for demes that don't already
    have them (e.g. background deme priors may have been added earlier).
    """
    for parent in root.iter():
        for param in list(parent.findall("parameter")):
            pid = param.attrib.get("id", "")
            if pid.startswith("PopSize."):
                parent.remove(param)

    demes_needing_priors = [
        f"Deme{i}"
        for i in range(1, n_demes + 1)
        if root.find(f".//*[@id='SkylinePrev.Deme{i}.Prior.t:SimDataset']") is None
    ]
    if demes_needing_priors:
        add_skyline_prev_priors(root, demes_needing_priors)


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
    population_by_deme=None,
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
    n_demes=2,
    estimate_clock_rate=True,
    fixed_clock_rate_value=None,
    use_relative_rateshifts=False,
    outside_deme=False,
    max_root_age=None,
    estimate_seroprevalence_scaling=False,
    estimate_ne_scaler_mean=True,
    fixed_ne_scaler_mean_value=None,
    deme_map=None,
    couple_deme_splines=False,
    couple_background_deme=False,
):
    """
    Replace the <data> and <trait> blocks in the Mascot template XML with new data.

    The template only contains per-deme elements for ``Deme1``.
    ``expand_template_for_demes`` replicates them for the required number of demes.
    """
    # Parse the template XML
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Expand the single-deme template to n_demes
    expand_template_for_demes(
        root,
        n_demes,
        couple_deme_splines=couple_deme_splines,
        outside_deme=outside_deme,
        couple_background_deme=couple_background_deme,
    )

    # The background/ghost = outside deme (highest index) has no datastream observations
    if outside_deme:
        bg_deme = f"Deme{n_demes}"
        strip_background_deme_datastreams(root, bg_deme)
        add_skyline_prev_priors(root, [bg_deme])

        # Set the ghost deme SkylinePrev initial value to 6 (log-prevalence ~400)
        # to reflect the larger background epidemic outside the local demes.
        bg_skyline = root.find(f".//*[@id='SkylinePrev.{bg_deme}.t:SimDataset']")
        if bg_skyline is not None:
            bg_skyline.text = "6.0"

    inject_ne_scaler_up_down(
        root,
        n_demes,
        outside_deme,
        estimate_clock_rate=estimate_clock_rate,
        estimate_ne_scaler_mean=estimate_ne_scaler_mean,
    )
    configure_seroprevalence_scaling(root, estimate=estimate_seroprevalence_scaling)
    configure_clock_rate(
        root, estimate=estimate_clock_rate, rate=fixed_clock_rate_value
    )
    configure_ne_scaler_mean(
        root, estimate=estimate_ne_scaler_mean, value=fixed_ne_scaler_mean_value
    )

    # Set per-deme PopSize from population CSV
    if population_by_deme is not None:
        for param in root.findall(".//parameter"):
            pid = param.attrib.get("id", "")
            if pid.startswith("PopSize.") and pid.endswith(":SimDataset"):
                deme = pid.split("PopSize.", 1)[1].split(":", 1)[0]
                if deme in population_by_deme:
                    param.text = str(population_by_deme[deme])

    # Always disable clipTransRate on Spline elements: clipping was found to
    # artificially suppress migration rate estimates.
    is_datastream_template = (
        case_counts_by_deme is not None
        or seroprevalence_by_deme is not None
        or wastewater_by_deme is not None
        or (
            output_suffix_override is not None
            and output_suffix_override != VARIANT_ORIGINAL
        )
    )
    if is_datastream_template:
        for elem in root.iter():
            if elem.get("spec") == "mascotdatastreams.dynamics.Spline":
                elem.set("clipTransRate", "false")

    # Remove existing <data> blocks
    for data_elem in list(root.findall(".//data")):
        if data_elem.attrib.get("id") == "SimDataset":
            root.remove(data_elem)

    # Find and replace the trait block at the correct nesting
    trait_parent = None
    trait_index = None
    for parent in root.iter():
        for idx, child in enumerate(list(parent)):
            if (
                child.tag == "trait"
                and child.attrib.get("id") == "dateTrait.t:SimDataset"
            ):
                trait_parent = parent
                trait_index = idx
                break
        if trait_parent is not None:
            break
    if trait_parent is not None and trait_index is not None:
        # Remove old trait
        trait_parent.remove(trait_parent[trait_index])
        # Insert new trait at the same position
        new_trait_elem = ET.fromstring(trait_block)
        trait_parent.insert(trait_index, new_trait_elem)

    # Find and replace the typeTrait block at the correct nesting
    type_trait_parent = None
    type_trait_index = None
    for parent in root.iter():

        for idx, child in enumerate(list(parent)):
            if (
                child.tag == "typeTrait"
                and child.attrib.get("id") == "typeTraitSet.t:SimDataset"
            ):
                type_trait_parent = parent
                type_trait_index = idx
                break
        if type_trait_parent is not None:
            break
    if type_trait_parent is not None and type_trait_index is not None:
        # Remove old typeTrait
        type_trait_parent.remove(type_trait_parent[type_trait_index])
        # Insert new typeTrait at the same position
        new_type_trait_elem = ET.fromstring(type_trait_block)
        type_trait_parent.insert(type_trait_index, new_type_trait_elem)

    # uninfectiousRate is shared across all datastream likelihoods, so always
    # overwrite the template default (else the wrong gamma sticks for variants
    # that exclude case counts).
    if is_datastream_template and gamma is not None:
        for param in root.findall(".//parameter"):
            if param.attrib.get("id") == "uninfectiousRate.t:SimDataset":
                param.text = str(gamma)
                break

    # Optionally update case count parameters inside <state>
    # New convention: case counts are given as <parameter> entries in the state
    # with ids like caseCounts.DemeX:SimDataset and caseTimes.DemeX:SimDataset
    if (
        case_counts_by_deme is not None
        or seroprevalence_by_deme is not None
        or wastewater_by_deme is not None
    ):
        if case_counts_by_deme is not None:
            # Iterate through parameters and update any caseCounts./caseTimes. that we have values for
            for param in root.findall(".//parameter"):
                pid = param.attrib.get("id", "")
                if pid.startswith("caseCounts.") and pid.endswith(":SimDataset"):
                    # Extract deme name between 'caseCounts.' and ':'
                    try:
                        deme = pid.split("caseCounts.", 1)[1].split(":", 1)[0]
                    except Exception:
                        continue
                    if (
                        deme in case_counts_by_deme
                        and "counts" in case_counts_by_deme[deme]
                    ):
                        param.text = case_counts_by_deme[deme]["counts"]
                elif pid.startswith("caseTimes.") and pid.endswith(":SimDataset"):
                    try:
                        deme = pid.split("caseTimes.", 1)[1].split(":", 1)[0]
                    except Exception:
                        continue
                    if (
                        deme in case_counts_by_deme
                        and "times" in case_counts_by_deme[deme]
                    ):
                        param.text = case_counts_by_deme[deme]["times"]
        if seroprevalence_by_deme is not None:
            for param in root.findall(".//parameter"):
                pid = param.attrib.get("id", "")
                if pid.startswith("seroTestedCounts.") and pid.endswith(":SimDataset"):
                    deme = pid.split("seroTestedCounts.", 1)[1].split(":", 1)[0]
                    if (
                        deme in seroprevalence_by_deme
                        and "tested_counts" in seroprevalence_by_deme[deme]
                    ):
                        param.text = seroprevalence_by_deme[deme]["tested_counts"]
                elif pid.startswith("seroWithAntibodiesCounts.") and pid.endswith(
                    ":SimDataset"
                ):
                    deme = pid.split("seroWithAntibodiesCounts.", 1)[1].split(":", 1)[0]
                    if (
                        deme in seroprevalence_by_deme
                        and "with_antibodies_counts" in seroprevalence_by_deme[deme]
                    ):
                        param.text = seroprevalence_by_deme[deme][
                            "with_antibodies_counts"
                        ]
                elif pid.startswith("seroTestedTimes.") and pid.endswith(":SimDataset"):
                    deme = pid.split("seroTestedTimes.", 1)[1].split(":", 1)[0]
                    if (
                        deme in seroprevalence_by_deme
                        and "times" in seroprevalence_by_deme[deme]
                    ):
                        param.text = seroprevalence_by_deme[deme]["times"]
        if wastewater_by_deme is not None:
            for param in root.findall(".//parameter"):
                pid = param.attrib.get("id", "")
                if pid.startswith("wastewaterConcentration.") and pid.endswith(
                    ":SimDataset"
                ):
                    deme = pid.split("wastewaterConcentration.", 1)[1].split(":", 1)[0]
                    if (
                        deme in wastewater_by_deme
                        and "wastewater" in wastewater_by_deme[deme]
                    ):
                        param.text = wastewater_by_deme[deme]["wastewater"]
                elif pid.startswith("wastewaterConcentrationTimes.") and pid.endswith(
                    ":SimDataset"
                ):
                    deme = pid.split("wastewaterConcentrationTimes.", 1)[1].split(
                        ":", 1
                    )[0]
                    if (
                        deme in wastewater_by_deme
                        and "times" in wastewater_by_deme[deme]
                    ):
                        param.text = wastewater_by_deme[deme]["times"]

        # Build output suffix based on exclusions (unless overridden)
        if output_suffix_override is not None:
            output_suffix = "_" + output_suffix_override
        else:
            suffix_parts = ["_datastreams"]
            if exclude_case_counts:
                suffix_parts.append("_nocasecounts")
            if exclude_seroprevalence:
                suffix_parts.append("_noseroprevalence")
            if exclude_wastewater:
                suffix_parts.append("_nowastewater")
            output_suffix = "".join(suffix_parts)

    else:
        if output_suffix_override != VARIANT_ORIGINAL:
            output_suffix = "_" + output_suffix_override
        else:
            output_suffix = "_" + VARIANT_ORIGINAL
    print(f"Output suffix: {output_suffix}")
    # Remove excluded datastream elements when any are excluded (e.g. datastreams_onlytree with no data)
    if exclude_case_counts or exclude_seroprevalence or exclude_wastewater:
        remove_excluded_datastream_elements(
            root,
            exclude_case_counts=exclude_case_counts,
            exclude_seroprevalence=exclude_seroprevalence,
            exclude_wastewater=exclude_wastewater,
        )
        # For onlytree variant: replace all operators with tree-only set and add SkylinePrev priors
        if exclude_case_counts and exclude_seroprevalence and exclude_wastewater:
            apply_onlytree_operators_and_priors(root, n_demes)

    # Update rateShifts values
    if use_relative_rateshifts:
        # Relative rate shifts: fractions of tree height with tree reference
        n_skygrowth = 10
        n_grid = 1000
        skygrowth_rate_shifts = root.find(".//rateShifts[@id='SkygrowthRateShifts']")
        if skygrowth_rate_shifts is not None:
            new_values = np.linspace(0, 1, n_skygrowth + 1)
            skygrowth_rate_shifts.text = " ".join(f"{v:.2f}" for v in new_values)
            skygrowth_rate_shifts.set("tree", "@Tree.t:SimDataset")

        splinegridpoints_rate_shifts = root.find(
            ".//gridRateShifts[@id='SplineGridRateShifts']"
        )
        if splinegridpoints_rate_shifts is not None:
            new_values = np.linspace(0, 1, n_grid + 1)
            splinegridpoints_rate_shifts.text = " ".join(f"{v:.4f}" for v in new_values)
            splinegridpoints_rate_shifts.set("tree", "@Tree.t:SimDataset")

        # Adjust SkylinePrev dimension to match the number of rate shift points
        for param in root.findall(".//parameter"):
            pid = param.attrib.get("id", "")
            if pid.startswith("SkylinePrev.") and pid.endswith(".t:SimDataset"):
                param.set("dimension", str(n_skygrowth))

    elif max_age is not None and min_age is not None:
        # Absolute rate shifts
        # add one more interval half a month to the max age
        # add and interval 0.5 month older than max age and as a last rateshift fill up to the next year
        overhang = 0.5 / 12
        print(f"Max age: {max_age}")
        print(f"Min age: {min_age}")
        skygrowth_rate_shifts = root.find(".//rateShifts[@id='SkygrowthRateShifts']")
        n_shifts = 11
        if skygrowth_rate_shifts is not None:
            new_values = np.linspace(min_age, max_age, n_shifts)
            new_values = np.append(new_values, max_age + overhang)
            next_year = np.ceil(max_age + overhang)
            new_values = np.append(new_values, next_year)
            skygrowth_rate_shifts.text = " ".join(f"{v:.4f}" for v in new_values)

        splinegridpoints_rate_shifts = root.find(
            ".//gridRateShifts[@id='SplineGridRateShifts']"
        )
        if splinegridpoints_rate_shifts is not None:
            new_values = np.linspace(min_age, max_age + overhang, 1001)
            # add 5 additional points to fill up to the next year
            next_year = np.ceil(max_age + overhang)
            new_values = np.append(
                new_values,
                np.linspace(max_age + overhang, next_year, 6)[1:],
            )
            splinegridpoints_rate_shifts.text = " ".join(f"{v:.4f}" for v in new_values)

        if outside_deme and max_root_age is not None:
            bg_deme = f"Deme{n_demes}"
            bg_spline = root.find(f".//spline[@id='splinePrev.{bg_deme}.t:SimDataset']")
            if bg_spline is not None:
                for rs in bg_spline.findall("rateShifts"):
                    bg_spline.remove(rs)
                for grs in bg_spline.findall("gridRateShifts"):
                    bg_spline.remove(grs)

                bg_skygrowth = ET.SubElement(bg_spline, "rateShifts")
                bg_skygrowth.set("id", f"SkygrowthRateShifts.{bg_deme}")
                bg_skygrowth.set("spec", "mascot.dynamics.RateShifts")
                next_year = np.ceil(max_age + overhang)
                bg_skygrowth_values = np.linspace(min_age, next_year, n_shifts + 1)
                # TODO: this is inconsistent with the SplineGridRateShifts that have their max at max_root_age and not next_year+1
                bg_skygrowth_values = np.append(bg_skygrowth_values, next_year + 1)
                bg_skygrowth.text = " ".join(f"{v:.4f}" for v in bg_skygrowth_values)

                bg_grid = ET.SubElement(bg_spline, "gridRateShifts")
                bg_grid.set("id", f"SplineGridRateShifts.{bg_deme}")
                bg_grid.set("spec", "mascot.dynamics.RateShifts")
                bg_grid_values = np.linspace(min_age, next_year, 1001)
                if next_year < max_root_age:
                    bg_grid_values = np.append(
                        bg_grid_values,
                        np.arange(next_year, max_root_age + 0.1, 0.1)[1:],
                    )
                max_value = np.round(np.max(bg_grid_values), 4)
                bg_grid.text = " ".join(f"{v:.4f}" for v in bg_grid_values)
                print(
                    f"Outside deme {bg_deme}: rate shifts span "
                    f"{min_age:.4f} to {max_value:.4f}"
                )

    # Populate mascotshifts from max_root_age
    if max_root_age is not None:
        mascot_shifts = root.find(".//rateShifts[@id='mascotshifts']")
        if mascot_shifts is not None:
            mascot_shift_values = np.arange(0, max_value + 0.005, 0.005)
            mascot_shifts.text = " ".join(f"{v:.4f}" for v in mascot_shift_values)

    # Insert alignment block at root
    alignment_data_elem = ET.fromstring(mascot_alignment_block)
    root.insert(0, alignment_data_elem)
    # Write the modified XML to output

    # Pretty-print the XML
    # Configure MCMC or CoupledMCMC
    for run in root.findall(".//run"):
        if run.attrib.get("id") == "mcmc":
            if use_coupled_mcmc:
                # Switch to coupled MCMC and set attributes
                run.set("spec", "coupledMCMC.CoupledMCMC")
                run.set("chainLength", str(chain_length))
                run.set("chains", str(chains))
                run.set("target", str(target))
                run.set("logHeatedChains", "true" if log_heated_chains else "false")
                run.set("deltaTemperature", str(delta_temperature))
                run.set("optimise", "true" if optimise else "false")
                run.set("resampleEvery", str(resample_every))
            else:
                # Ensure standard MCMC spec and optionally override chain length
                print(f"Setting chain length to {chain_length}")
                run.set("spec", "MCMC")
                run.set("chainLength", str(chain_length))
    # Examples:
    # <run id="mcmc" spec="MCMC" chainLength="....." numInitializationAttempts="....">
    # <run id="mcmc" spec="beast.coupledMCMC.CoupledMCMC" chainLength="10000000" chains="4" target="0.234" logHeatedChains="true" deltaTemperature="0.1" optimise="true" resampleEvery="1000" >

    # Inject fixed tree if provided (still set newick for datastreams_nomascotll variant so TreeParser has a tree; likelihood is disabled via MascotLogPflag)
    if newick_tree is not None:
        # Find the init element with TreeParser
        init_elem = root.find(".//init[@spec='beast.base.evolution.tree.TreeParser']")
        if init_elem is not None:
            init_elem.set("newick", newick_tree.strip())

    # VARIANT_NO_GENETIC (datastreams_nomascotll): disable all genetic-data
    # contributions. With a fixed tree, disabling MascotLogPflag is sufficient
    # (no substitution-model likelihood is present and the tree never moves).
    # With an inferred tree, also drop the substitution-model treeLikelihood
    # and remove tree-move operators so the tree is not estimated.
    if not use_fixed_tree:
        disable_genetic_data(
            root,
            remove_tree_operators=newick_tree is None,
            outside_deme=outside_deme,
            bg_deme=f"Deme{n_demes}" if outside_deme else None,
        )

    xml_str = ET.tostring(tree.getroot(), encoding="utf-8")
    parsed = minidom.parseString(xml_str)

    print(f"Writing XML to {xml_name + output_suffix + '.xml'}")

    with open(xml_name + output_suffix + ".xml", "w", encoding="utf-8") as f:
        # Remove superfluous empty lines introduced by minidom.toprettyxml
        pretty_xml = parsed.toprettyxml(indent="  ")
        # Remove lines that are empty or contain only whitespace
        lines = [line for line in pretty_xml.splitlines() if line.strip() != ""]
        # Insert deme_map comment on the second line (after the XML declaration)
        # so that analyse_posteriors.py can read it back for validation.
        if deme_map:
            mapping_str = ", ".join(
                f"{deme}={state}" for deme, state in sorted(deme_map.items())
            )
            comment_line = f"<!-- deme_map: {mapping_str} -->"
            lines.insert(1, comment_line)
        f.write("\n".join(lines))


def convert_date_to_numerical_date(date_series: pd.Series) -> pd.Series:

    date_series = pd.to_datetime(date_series)
    years = date_series.dt.year
    year_start = pd.to_datetime(years.astype(str) + "-01-01")
    next_year_start = pd.to_datetime((years + 1).astype(str) + "-01-01")
    days_in_year = (next_year_start - year_start).dt.days
    day_of_year = date_series.dt.dayofyear
    return years + (day_of_year / days_in_year)


def _normalize_deme_name(name):
    """Normalize a deme name so CSV names (spaces) match tree names (underscores)."""
    return str(name).replace(" ", "_")


def _resolve_csv_deme_label(csv_deme, state_to_deme):
    """Map a CSV deme name to its DemeN label using the authoritative mapping."""
    normalized = _normalize_deme_name(csv_deme)
    if normalized not in state_to_deme:
        raise ValueError(
            f"CSV deme '{csv_deme}' (normalized: '{normalized}') "
            f"not found in tree states: {list(state_to_deme.keys())}"
        )
    return state_to_deme[normalized]


def build_case_counts_by_deme(
    case_counts_file,
    most_recent_sample_time,
    state_to_deme,
    remove_small_counts=False,
    add1tocounts=False,
):
    """
    Read case counts CSV and build mapping per-deme for insertion into <state> parameters.

    Uses *state_to_deme* (tree-state → DemeN) as the authoritative deme assignment.
    """
    df = pd.read_csv(case_counts_file)
    for col in ("deme", "date", "case_counts"):
        if col not in df.columns:
            raise ValueError(f"Case counts CSV missing required column: {col}")

    if remove_small_counts:
        df = df.loc[df["case_counts"] > 10]

    if add1tocounts:
        df["case_counts"] = df["case_counts"] + 1

    df["numerical_date"] = convert_date_to_numerical_date(df["date"])
    # make sure there are no measurements that are more recent than the most recent sequence sample time
    df = df.loc[df["numerical_date"] <= most_recent_sample_time]
    df["t_case_counts_frommostrecentsample"] = (
        most_recent_sample_time - df["numerical_date"]
    )
    df_sorted = df.sort_values(by=["deme", "t_case_counts_frommostrecentsample"])

    case_counts_by_deme = {}
    for deme, sub in df_sorted.groupby("deme"):
        label = _resolve_csv_deme_label(deme, state_to_deme)
        counts_str = " ".join(sub["case_counts"].astype(str))
        times_str = " ".join(sub["t_case_counts_frommostrecentsample"].astype(str))
        case_counts_by_deme[label] = {"counts": counts_str, "times": times_str}

    max_age = df_sorted["t_case_counts_frommostrecentsample"].max()
    min_age = df_sorted["t_case_counts_frommostrecentsample"].min()
    return case_counts_by_deme, max_age, min_age


def build_seroprevalence_by_deme(
    seroprevalence_file, most_recent_sample_time, state_to_deme
):
    df = pd.read_csv(seroprevalence_file)
    df["numerical_date"] = convert_date_to_numerical_date(df["date"])
    # make sure there are no measurements that are more recent than the most recent sequence sample time
    df = df.loc[df["numerical_date"] <= most_recent_sample_time]
    df["t_seroprevalence_frommostrecentsample"] = (
        most_recent_sample_time - df["numerical_date"]
    )
    df_sorted = df.sort_values(by=["deme", "t_seroprevalence_frommostrecentsample"])

    seroprevalence_by_deme = {}
    for deme, sub in df_sorted.groupby("deme"):
        label = _resolve_csv_deme_label(deme, state_to_deme)
        tested_counts_str = " ".join(sub["seroprevalence_numpeopletested"].astype(str))
        with_antibodies_counts_str = " ".join(
            sub["seroprevalence_numpeoplewithantibodies"].astype(str)
        )
        times_str = " ".join(sub["t_seroprevalence_frommostrecentsample"].astype(str))
        seroprevalence_by_deme[label] = {
            "tested_counts": tested_counts_str,
            "with_antibodies_counts": with_antibodies_counts_str,
            "times": times_str,
        }

    max_age = df_sorted["t_seroprevalence_frommostrecentsample"].max()
    min_age = df_sorted["t_seroprevalence_frommostrecentsample"].min()
    return seroprevalence_by_deme, max_age, min_age


def build_wastewater_by_deme(wastewater_file, most_recent_sample_time, state_to_deme):
    df = pd.read_csv(wastewater_file)
    df["numerical_date"] = convert_date_to_numerical_date(df["date"])
    # make sure there are no measurements that are more recent than the most recent sequence sample time
    df = df.loc[df["numerical_date"] <= most_recent_sample_time]
    df["t_wastewater_frommostrecentsample"] = (
        most_recent_sample_time - df["numerical_date"]
    )
    df_sorted = df.sort_values(by=["deme", "t_wastewater_frommostrecentsample"])

    wastewater_by_deme = {}
    for deme, sub in df_sorted.groupby("deme"):
        label = _resolve_csv_deme_label(deme, state_to_deme)
        wastewater_str = " ".join(sub["wastewater"].astype(str))
        times_str = " ".join(sub["t_wastewater_frommostrecentsample"].astype(str))
        wastewater_by_deme[label] = {"wastewater": wastewater_str, "times": times_str}

    max_age = df_sorted["t_wastewater_frommostrecentsample"].max()
    min_age = df_sorted["t_wastewater_frommostrecentsample"].min()
    return wastewater_by_deme, max_age, min_age


def build_population_by_deme(population_csv, state_to_deme):
    """
    Read a county-population CSV and return {DemeN: population} using the
    authoritative state_to_deme mapping.
    """
    df = pd.read_csv(population_csv)
    for col in ("county", "population"):
        if col not in df.columns:
            raise ValueError(f"Population CSV missing required column: {col}")

    population_by_deme = {}
    for _, row in df.iterrows():
        label = _resolve_csv_deme_label(row["county"], state_to_deme)
        population_by_deme[label] = float(row["population"])
    return population_by_deme


def extract_leaf_states(tree_path):
    """
    Parse a (time-)tree in NEXUS format and extract:
      - a mapping from tip label -> state/deme
      - a mapping from tip label -> time (decimal year or relative time)

    This is tailored to work with timetrees such as
    `results/final_sequences/timetree.nexus`, where tip labels have the form
    `<sample_id>|<EPI_ID>|<YYYY-MM-DD>|<variant>|<deme>` and tips may also
    carry BEAST-style comment metadata like `[&date=2021.17]`.
    """

    # Load the tree directly from the NEXUS file and extract BEAST-style
    # comment metadata (e.g. [&date=...]) into annotations.
    tree = dendropy.Tree.get(
        path=tree_path,
        schema="nexus",
        preserve_underscores=True,
        extract_comment_metadata=True,
    )

    leaf_state = {}
    leaf_time = {}
    for leaf in tree.leaf_node_iter():
        real_label = leaf.taxon.label
        state = None
        time = None

        # 1) Prefer explicit annotations from the tree (BEAST/timetree output)
        if leaf.annotations:
            # Try multiple possible keys for the state/deme

            for key in ("type", "location", "deme"):
                if key in leaf.annotations:
                    state = leaf.annotations.get_value(key)
                    break
            # TODO: Uncomment once upstream date calculation is fixed
            # # Try multiple possible keys for the time
            # for key in ("time", "date"):
            #     if key in leaf.annotations:
            #         time = leaf.annotations.get_value(key)
            #         break

        # 2) If still missing, fall back to parsing the taxon label itself.
        #    For `timetree.nexus` the label looks like:
        #    <sample_id>|<EPI_ID>|<YYYY-MM-DD>|<variant>|<deme>
        parts = real_label.split("|")
        if state is None and len(parts) >= 5:
            state = parts[-1]

        if time is None and len(parts) >= 3:
            date_str = parts[2]
            # Convert YYYY-MM-DD to a decimal year, which downstream code
            # interprets as a continuous time scale.
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            time = convert_date_to_numerical_date(pd.Series([dt])).iloc[0]

        # Final clean-up and assignment
        clean_state = (
            state.replace("{", "").replace("}", "") if isinstance(state, str) else None
        )
        leaf_state[real_label] = clean_state
        leaf_time[real_label] = time
    return leaf_state, leaf_time


def compute_tree_root_age(tree_path, leaf_time_dict):
    """
    Compute the root age (years before most recent sample) from a timetree.

    Uses tip dates and root-to-tip distances to recover the root's calendar
    time, returning how far back the root is relative to the most recent sample.
    """
    tree = dendropy.Tree.get(
        path=tree_path,
        schema="nexus",
        preserve_underscores=True,
    )
    most_recent_sample_time = max(leaf_time_dict.values())

    root_times = []
    for leaf in tree.leaf_node_iter():
        label = leaf.taxon.label
        if label in leaf_time_dict:
            root_to_tip = leaf.distance_from_root()
            root_times.append(leaf_time_dict[label] - root_to_tip)

    if not root_times:
        raise ValueError("No leaf labels match leaf_time_dict for root age computation")

    root_time = np.median(root_times)
    return most_recent_sample_time - root_time


def get_uninfectious_rate(parameters_path):
    df = pd.read_csv(parameters_path)
    df = df[df["parameter"] == "gamma"]
    if df.shape[0] == 0:
        raise ValueError("No gamma parameter found in parameters file")
    return df["value"].values[0]


def build_alignment_block_from_fasta(fasta_path):
    """
    Read a FASTA alignment and return the <data id="SimDataset"> XML block
    with actual nucleotide sequences (lower-case).
    """
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        raise ValueError(f"No sequences found in {fasta_path}")

    seq_xml_blocks = []
    for record in records:
        seq_xml_blocks.append(
            f'    <sequence id="seq_{record.id}" spec="Sequence" '
            f'taxon="{record.id}" totalcount="4" '
            f'value="{str(record.seq).lower()}"/>'
        )
    return (
        "<data\n"
        'id="SimDataset"\n'
        'spec="Alignment"\n'
        'name="alignment">\n' + "\n".join(seq_xml_blocks) + "\n</data>"
    )


def read_metadata_csv(metadata_path):
    """
    Read sample metadata CSV and return (leaf_state, leaf_time) dicts.

    Expected columns: sequence_ID,Collection date,County
    Returns:
        leaf_state: {sample_id: deme_name}
        leaf_time:  {sample_id: decimal_year}
    """
    df = pd.read_csv(metadata_path)
    for col in ("sequence_ID", "Collection date", "County"):
        if col not in df.columns:
            raise ValueError(f"Metadata CSV missing required column: {col}")

    df["numerical_date"] = convert_date_to_numerical_date(df["Collection date"])

    leaf_state = {}
    leaf_time = {}
    for _, row in df.iterrows():
        sid = str(row["sequence_ID"])
        leaf_state[sid] = _normalize_deme_name(row["County"])
        leaf_time[sid] = row["numerical_date"]
    return leaf_state, leaf_time


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
        required=False,
        default=None,
        help="Path to Mascot template XML (required for fixed-tree mode)",
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
        required=False,
        default=None,
        help="Path to the Nexus tree file (required for fixed-tree mode)",
    )
    parser.add_argument(
        "--burate",
        type=float,
        required=True,
        help="Becoming uninfectious rate in 1/years",
    )
    parser.add_argument(
        "--infer_tree",
        action="store_true",
        help="Infer the tree from a sequence alignment instead of using a fixed tree",
    )
    parser.add_argument(
        "--use_relative_rateshifts",
        action="store_true",
        help="Use relative rateshifts instead of absolute rateshifts",
    )
    parser.add_argument(
        "--couple_deme_splines",
        action="store_true",
        help="Insert otherSpline / incomingForwardMigration / "
        "forwardMigrationIndices children into each <neDynamics> block "
        "in the MASCOT-DS package. Off by default (preserves "
        "the pre-coupling XML wiring).",
    )
    parser.add_argument(
        "--couple_background_deme",
        action="store_true",
        help="Only meaningful with --couple_deme_splines and "
        "--ghost_outsidedeme: also apply the new coupling wiring to the "
        "background/ghost deme. Off by default (background deme keeps the "
        "old uncoupled wiring).",
    )
    parser.add_argument(
        "--alignment",
        type=str,
        required=False,
        default=None,
        help="Path to a FASTA alignment file (required when --infer_tree is set)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=False,
        default=None,
        help="Path to metadata CSV with columns: sample_id, date, deme "
        "(required when --infer_tree is set)",
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
        "--ghost_outsidedeme",
        action="store_true",
        help="Add a ghost outside deme to the tree",
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
        choices=ALL_VARIANTS,
        help=(
            "Which XML variant to produce. When unset, both a full-datastream "
            "XML and (if --standard_template is provided in fixed-tree mode) "
            "an 'original' XML are written. Options: "
            f"'{VARIANT_DATASTREAMS}' (all datastreams), "
            f"'{VARIANT_NO_CASE_COUNTS}' (no case counts), "
            f"'{VARIANT_NO_SEROPREVALENCE}' (no seroprevalence), "
            f"'{VARIANT_NO_WASTEWATER}' (no wastewater), "
            f"'{VARIANT_NO_GENETIC}' (no genetic data — disables tree/sequence "
            "likelihoods; with --infer_tree, also removes tree operators so "
            "the tree is not estimated), "
            f"'{VARIANT_ONLY_TREE}' (just genetic data, no datastreams), "
            f"'{VARIANT_ORIGINAL}' (standard Mascot template, fixed-tree only)."
        ),
    )
    parser.add_argument(
        "--population_csv",
        type=str,
        required=True,
        help="Path to county population CSV (columns: county, population). "
        "Used to set PopSize per deme in the datastream template.",
    )
    parser.add_argument(
        "--estimate_seroprevalence_scaling",
        action="store_true",
        default=False,
        help="Estimate the per-deme seroprevalence scaling parameter. "
        "Default: off (scaling fixed at initial value). When enabled, "
        "adds a LogNormal prior and includes the parameter in AVMN/UpDown operators.",
    )
    parser.add_argument(
        "--fixed_clock_rate",
        type=float,
        default=None,
        help="Fix the strict clock rate to this value instead of estimating it. "
        "When provided, clock rate operators and AVMN/UpDown references are removed.",
    )
    parser.add_argument(
        "--fixed_ne_scaler_mean",
        type=float,
        default=None,
        help="Fix the NeScaler.MEAN parameter to this value instead of estimating it. "
        "When provided, NeScalerMean operator, prior, and AVMN/UpDown references "
        "are removed.",
    )
    parser.add_argument(
        "--add_sentinel_zero_case_counts",
        action="store_true",
        default=False,
        help="Append two sentinel zero case counts per local deme just before "
        "the analysis window (at max_age + overhang and the closest older year "
        "boundary), so MASCOT-DS sees a definitive zero prevalence before the "
        "observation window begins. Default: off (only true case counts from the "
        "case-counts file are used).",
    )
    parser.add_argument(
        "--max_age",
        type=float,
        default=None,
        help="Override the internally computed max_age used to set "
        "SkygrowthRateShifts / SplineGridRateShifts. Useful for matching "
        "rate-shift grids across variants that have different datastream "
        "coverage (e.g. force the datastreams_onlytree variant to use the "
        "max_age computed by the full datastreams variant).",
    )
    args = parser.parse_args()

    # --- Input validation ---
    if args.infer_tree:
        if not args.alignment:
            parser.error("--alignment is required when --infer_tree is set")
        if not args.metadata:
            parser.error("--metadata is required when --infer_tree is set")
    else:
        if not args.tree:
            parser.error("--tree is required when --infer_tree is not set")
        if not args.standard_template:
            print(
                "Warning: --standard_template not provided; "
                "'original' variant will not be written."
            )

    print(f"Mode: {'infer_tree' if args.infer_tree else 'fixed_tree'}")
    print(f"Case counts file provided: {args.case_counts}")
    print(f"Sero prevalence file provided: {args.seroprevalence}")
    print(f"Wastewater concentrations file provided: {args.wastewater}")

    if args.variant_type is not None and args.variant_type not in ALL_VARIANTS:
        parser.error(
            f"--variant_type must be one of {ALL_VARIANTS} (got {args.variant_type!r})"
        )

    # Detect which data types are excluded (overridden by variant_type when provided)
    if args.variant_type is not None:
        variant = args.variant_type
        exclude_case_counts = variant in (VARIANT_NO_CASE_COUNTS, VARIANT_ONLY_TREE)
        exclude_seroprevalence = variant in (
            VARIANT_NO_SEROPREVALENCE,
            VARIANT_ONLY_TREE,
        )
        exclude_wastewater = variant in (VARIANT_NO_WASTEWATER, VARIANT_ONLY_TREE)
        # use_fixed_tree=False is the flag that triggers disable_genetic_data().
        # It is only meaningful for VARIANT_NO_GENETIC. The name is historical:
        # it predates the infer-tree support and refers to whether the genetic
        # likelihood contributes (kept for backward compat in the helper API).
        use_fixed_tree = variant != VARIANT_NO_GENETIC
    else:
        exclude_case_counts = (
            args.case_counts is None
            or args.case_counts == ""
            or (
                os.path.exists(args.case_counts)
                and os.path.getsize(args.case_counts) == 0
            )
        )
        exclude_seroprevalence = (
            args.seroprevalence is None
            or args.seroprevalence == ""
            or (
                os.path.exists(args.seroprevalence)
                and os.path.getsize(args.seroprevalence) == 0
            )
        )
        exclude_wastewater = (
            args.wastewater is None
            or args.wastewater == ""
            or (
                os.path.exists(args.wastewater)
                and os.path.getsize(args.wastewater) == 0
            )
        )
        use_fixed_tree = True

    # ---- Mode-specific data extraction ----
    if args.infer_tree:
        print(f"Alignment file: {args.alignment}")
        print(f"Metadata file: {args.metadata}")

        leaf_state_dict, leaf_time_dict = read_metadata_csv(args.metadata)

        mascot_alignment_block = build_alignment_block_from_fasta(args.alignment)

        # Validate that alignment sample IDs match metadata sample IDs
        fasta_ids = {r.id for r in SeqIO.parse(args.alignment, "fasta")}
        meta_ids = set(leaf_state_dict.keys())
        missing_in_meta = fasta_ids - meta_ids
        missing_in_fasta = meta_ids - fasta_ids
        if missing_in_meta:
            raise ValueError(
                f"Sequences in FASTA but not in metadata: {missing_in_meta}"
            )
        if missing_in_fasta:
            raise ValueError(
                f"Samples in metadata but not in FASTA: {missing_in_fasta}"
            )

        newick_string = None
    else:
        print(f"Tree file: {args.tree}")

        with open(args.tree, "r") as f:
            trees_content = f.read()

        leaf_state_dict, leaf_time_dict = extract_leaf_states(args.tree)

        # Generate placeholder sequences from tree
        seq_xml_blocks = []
        for leaf in sorted(leaf_state_dict.keys()):
            seq_xml_blocks.append(
                f'    <sequence id="seq_{leaf}" spec="Sequence" taxon="{leaf}" '
                f'totalcount="4" value="????"/>'
            )
        mascot_alignment_block = (
            "<data\n"
            'id="SimDataset"\n'
            'spec="Alignment"\n'
            'name="alignment">\n' + "\n".join(seq_xml_blocks) + "\n</data>"
        )

        # Extract and collapse tree
        newick_state0 = get_newick_tree(trees_content)
        tree = Phylo.read(StringIO(newick_state0), "newick")
        tree.root = collapse_single_child_nodes(tree.root)
        newick_string = tree.format("newick")

    # ---- Shared logic: deme detection, datastream loading, XML generation ----
    max_age = 0
    min_age = 0

    all_states = set(v for v in leaf_state_dict.values() if v is not None)
    non_bg = sorted(all_states - {"background"})
    unique_demes = non_bg + (["background"] if "background" in all_states else [])
    n_demes = len(unique_demes)
    print(f"Detected {n_demes} demes: {unique_demes}")
    if args.ghost_outsidedeme:
        unique_demes.append("ghost_outside_deme")
        n_demes += 1
    state_to_deme = {state: f"Deme{i + 1}" for i, state in enumerate(unique_demes)}
    if args.ghost_outsidedeme or "background" in all_states:
        outside_deme = True
    else:
        outside_deme = False

    print(f"State-to-deme mapping: {state_to_deme}")

    population_by_deme = build_population_by_deme(args.population_csv, state_to_deme)
    print(f"Population per deme: {population_by_deme}")

    # Save state and time as csv
    state_time_csv = pd.DataFrame.from_dict(
        leaf_state_dict, orient="index", columns=["state"]
    )
    tmp = pd.DataFrame.from_dict(leaf_time_dict, orient="index", columns=["time"])
    state_time_csv = pd.merge(state_time_csv, tmp, left_index=True, right_index=True)
    state_time_csv = state_time_csv.reset_index(drop=False)
    state_time_csv = state_time_csv.rename(columns={"index": "sample_id"})
    state_time_csv.sort_values(by="sample_id").to_csv(
        args.xml_name + "_state_time.csv", index=False
    )

    most_recent_sample_time = max(leaf_time_dict.values())
    max_age = most_recent_sample_time - min(leaf_time_dict.values())
    max_root_age = most_recent_sample_time - MAX_RATE_SHIFT_OUTSIDE_DEME
    print(
        f"Oldest sample: min(leaf_time_dict.values()): {min(leaf_time_dict.values())}"
    )
    print(f"Most recent sample: most_recent_sample_time: {most_recent_sample_time}")
    min_age = most_recent_sample_time - most_recent_sample_time

    gamma = args.burate

    # Build per-deme datastream mappings
    case_counts_by_deme = None
    max_age_counts = None
    min_age_counts = None
    if not exclude_case_counts and args.case_counts:
        if os.path.exists(args.case_counts) and os.path.getsize(args.case_counts) > 0:
            case_counts_by_deme, max_age_counts, min_age_counts = (
                build_case_counts_by_deme(
                    args.case_counts,
                    most_recent_sample_time=most_recent_sample_time,
                    state_to_deme=state_to_deme,
                    remove_small_counts=False,
                    add1tocounts=args.add1tocounts,
                )
            )

    seroprevalence_by_deme = None
    max_age_sero = None
    min_age_sero = None
    if not exclude_seroprevalence and args.seroprevalence:
        if (
            os.path.exists(args.seroprevalence)
            and os.path.getsize(args.seroprevalence) > 0
        ):
            seroprevalence_by_deme, max_age_sero, min_age_sero = (
                build_seroprevalence_by_deme(
                    args.seroprevalence, most_recent_sample_time, state_to_deme
                )
            )

    wastewater_by_deme = None
    max_age_wastewater = None
    min_age_wastewater = None
    if not exclude_wastewater and args.wastewater:
        if os.path.exists(args.wastewater) and os.path.getsize(args.wastewater) > 0:
            wastewater_by_deme, max_age_wastewater, min_age_wastewater = (
                build_wastewater_by_deme(
                    args.wastewater, most_recent_sample_time, state_to_deme
                )
            )

    # Set max_age and min_age to encompass all included datastream types
    age_values = [max_age, min_age]
    age_values_datastreams = []
    if max_age_counts is not None:
        age_values.append(max_age_counts)
        age_values_datastreams.append(max_age_counts)
    if min_age_counts is not None:
        age_values.append(min_age_counts)
        age_values_datastreams.append(min_age_counts)
    if max_age_sero is not None:
        age_values.append(max_age_sero)
        age_values_datastreams.append(max_age_sero)
    if min_age_sero is not None:
        age_values.append(min_age_sero)
        age_values_datastreams.append(min_age_sero)
    if max_age_wastewater is not None:
        age_values.append(max_age_wastewater)
        age_values_datastreams.append(max_age_wastewater)
    if min_age_wastewater is not None:
        age_values.append(min_age_wastewater)
        age_values_datastreams.append(min_age_wastewater)

    max_age = max([a for a in age_values if a is not None])
    min_age = min([a for a in age_values if a is not None])

    if age_values_datastreams:
        max_age_datastreams = max(age_values_datastreams)
        min_age_datastreams = min(age_values_datastreams)
    else:
        max_age_datastreams = None
        min_age_datastreams = None

    print(f"Max age: {max_age}")
    print(f"Min age: {min_age}")
    print(f"Max age datastreams: {max_age_datastreams}")
    print(f"Min age datastreams: {min_age_datastreams}")

    if args.infer_tree:
        # Relative rate shifts are used; max_age from datastreams kept for reference
        if max_age_datastreams is not None:
            max_age = max_age_datastreams
    else:
        # Fixed tree: max_age should cover both the datastreams and the tree root
        tree_root_age = compute_tree_root_age(args.tree, leaf_time_dict)
        print(f"Tree root age: {tree_root_age}")
        candidates = [a for a in [max_age_datastreams, tree_root_age] if a is not None]
        max_age = max(candidates)
        print("Tree root age: ", tree_root_age)
        print("Max age: ", max_age)
    if args.max_age is not None:
        print(f"Overriding max_age from {max_age} to {args.max_age} (--max_age)")
        max_age = args.max_age
    if min_age < 0:
        raise ValueError(f"Min age is less than 0: {min_age}")

    # Write resolved max_age to a sidecar so downstream callers (e.g. the
    # 03_xml_generation.sh wrapper) can reuse the same value when generating
    # other variants that should share the same rate-shift grid.
    with open(args.xml_name + "_max_age.txt", "w") as f:
        f.write(f"{max_age}\n")

    # Optionally append a sentinel zero case count just before the analysis window
    # and at the closest older year boundary for each local deme, so that
    # MASCOT-DS sees a definitive zero prevalence before the actual observation
    # window begins. When --add_sentinel_zero_case_counts is not set, only the
    # true case counts from the case-counts file are used.
    overhang = 0.5 / 12
    if case_counts_by_deme is not None and args.add_sentinel_zero_case_counts:
        sentinel_time = max_age + overhang - 0.001
        sentinel_time_str = f"{sentinel_time:.10f}"
        closest_older_year = np.ceil(sentinel_time)
        for deme, data in case_counts_by_deme.items():
            data["counts"] += " 0 0"
            data["times"] += f" {sentinel_time_str} {closest_older_year}"

    # Construct trait block
    trait_value = ",".join(
        [f"{leaf}={leaf_time_dict[leaf]}" for leaf in sorted(leaf_time_dict.keys())]
    )
    trait_block = (
        '<trait id="dateTrait.t:SimDataset" spec="beast.base.evolution.tree.TraitSet" traitname="date" value="'
        + trait_value
        + '">\n'
        '  <taxa id="TaxonSet.SimDataset" spec="TaxonSet">\n'
        '    <data idref="SimDataset" name="alignment"/>\n'
        "  </taxa>\n"
        "</trait>"
    )

    # Construct typeTrait block using DemeN labels so that the alphabetical
    # ordering of trait values matches the NeDynamics list.
    type_trait_value = ",".join(
        f"{leaf}={state_to_deme[leaf_state_dict[leaf]]}"
        for leaf in sorted(leaf_state_dict.keys())
    )
    type_trait_block = (
        '<typeTrait id="typeTraitSet.t:SimDataset" spec="mascot.util.InitializedTraitSet" traitname="type" value="'
        + type_trait_value
        + '">\n'
        '  <taxa id="TaxonSet.1" spec="TaxonSet" alignment="@SimDataset"/>\n'
        "</typeTrait>"
    )

    # ---- Write XML files ----
    write_datastream = (
        args.variant_type is None or args.variant_type in DATASTREAM_VARIANTS
    )
    write_standard = (
        not args.infer_tree
        and args.standard_template is not None
        and (args.variant_type is None or args.variant_type == VARIANT_ORIGINAL)
    )

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
            population_by_deme=population_by_deme,
            chain_length=args.chain_length,
            use_coupled_mcmc=args.coupled_mcmc,
            chains=args.chains,
            target=args.target,
            log_heated_chains=args.log_heated_chains,
            delta_temperature=args.delta_temperature,
            optimise=args.optimise,
            resample_every=args.resample_every,
            max_age=max_age,
            min_age=min_age,
            newick_tree=newick_string,
            exclude_case_counts=exclude_case_counts,
            exclude_seroprevalence=exclude_seroprevalence,
            exclude_wastewater=exclude_wastewater,
            use_fixed_tree=use_fixed_tree,
            output_suffix_override=(
                args.variant_type if args.variant_type in DATASTREAM_VARIANTS else None
            ),
            n_demes=n_demes,
            estimate_clock_rate=args.fixed_clock_rate is None,
            fixed_clock_rate_value=args.fixed_clock_rate,
            use_relative_rateshifts=args.use_relative_rateshifts,
            outside_deme=outside_deme,
            max_root_age=max_root_age,
            estimate_seroprevalence_scaling=args.estimate_seroprevalence_scaling,
            estimate_ne_scaler_mean=args.fixed_ne_scaler_mean is None,
            fixed_ne_scaler_mean_value=args.fixed_ne_scaler_mean,
            deme_map={v: k for k, v in state_to_deme.items()},
            couple_deme_splines=args.couple_deme_splines,
            couple_background_deme=args.couple_background_deme,
        )
    if write_standard:
        replace_blocks_template(
            args.standard_template,
            args.xml_name,
            mascot_alignment_block,
            trait_block,
            type_trait_block,
            gamma=gamma,
            case_counts_by_deme=None,
            seroprevalence_by_deme=None,
            wastewater_by_deme=None,
            population_by_deme=population_by_deme,
            chain_length=args.chain_length,
            use_coupled_mcmc=args.coupled_mcmc,
            chains=args.chains,
            target=args.target,
            log_heated_chains=args.log_heated_chains,
            delta_temperature=args.delta_temperature,
            optimise=args.optimise,
            resample_every=args.resample_every,
            max_age=None,
            min_age=None,
            newick_tree=newick_string,
            n_demes=n_demes,
            estimate_clock_rate=args.fixed_clock_rate is None,
            fixed_clock_rate_value=args.fixed_clock_rate,
            output_suffix_override=(
                VARIANT_ORIGINAL if args.variant_type == VARIANT_ORIGINAL else None
            ),
            outside_deme=outside_deme,
            max_root_age=max_root_age,
            estimate_seroprevalence_scaling=args.estimate_seroprevalence_scaling,
            deme_map={v: k for k, v in state_to_deme.items()},
            couple_deme_splines=args.couple_deme_splines,
            couple_background_deme=args.couple_background_deme,
        )


if __name__ == "__main__":
    main()
