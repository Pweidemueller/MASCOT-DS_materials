"""
Shared HPD validation time-series prep and plotting (coverage, bias, relative HPD width).

Used by make_figure_simstudy.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import MODEL_MASCOT, MODEL_MASCOT_DS, MODEL_COLORS
from plot_utils import COLORS, FONTSIZES_LIST, configure_pdf_fonts, save_figure_png_and_pdf
from plot_utils import set_axis_fontsizes

MODEL_COLORS_DEFAULT: tuple[tuple[str, str], ...] = tuple(MODEL_COLORS.items())

SIMULATION_QUANTILE_LOW = 0.025
SIMULATION_QUANTILE_HIGH = 0.975

# Alpha for 2.5–97.5% simulation-level band (behind median line)
DEFAULT_METRIC_BAND_ALPHA = 0.22



def _frac_index_on_reference_grid(t: float, t_ref: np.ndarray) -> float:
    """
    Fractional time index on sorted ``t_ref`` (MASCOT-DS grid): index i plus
    fraction between ``t_ref[i]`` and ``t_ref[i+1]`` (linear interpolation).
    Outside the grid, extrapolate using the first or last segment.
    """
    t_ref = np.asarray(t_ref, dtype=float)
    n = len(t_ref)
    if n == 1:
        return 0.0
    if t <= t_ref[0]:
        return (t - t_ref[0]) / (t_ref[1] - t_ref[0]) + 0.0
    if t >= t_ref[-1]:
        return (n - 1) + (t - t_ref[-1]) / (t_ref[-1] - t_ref[-2])
    return float(np.interp(t, t_ref, np.arange(n, dtype=float)))


def _ds_integer_time_index_for_timesincestart(
    t: float,
    t_ref: np.ndarray,
    simulation: str,
) -> int:
    """Integer index k with ``t`` matching ``t_ref[k]`` (within tolerance)."""
    t_ref = np.asarray(t_ref, dtype=float)
    diffs = np.abs(t_ref - t)
    k = int(np.argmin(diffs))
    if diffs[k] > 1e-9 * max(1.0, abs(t)):
        raise ValueError(
            f"MASCOT-DS timesincestart {t!r} not on reference grid for Simulation "
            f"{simulation!r} (nearest gap {diffs[k]:.3e})."
        )
    return k


def add_ne_time_index_mascot_ds_reference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ``Simulation``, build the time grid from **MASCOT-DS** rows only.

    MASCOT-DS rows get integer indices 0, 1, … on that grid. MASCOT rows get a
    fractional index from ``_frac_index_on_reference_grid``, then **rounded** to
    the nearest integer so filtering and aggregation share the same discrete axis
    as prevalence (MASCOT-DS grid).
    """
    pieces: list[pd.DataFrame] = []
    for sim, g in df.groupby("Simulation", sort=False):
        g_ds = g[g["Model"] == MODEL_MASCOT_DS]
        if g_ds.empty:
            raise ValueError(f"No MASCOT-DS rows for Simulation {sim!r}")
        t_ref = np.unique(np.sort(g_ds["timesincestart"].to_numpy(dtype=float)))
        g = g.copy()
        ti = np.empty(len(g), dtype=int)
        for i in range(len(g)):
            t = float(g.iloc[i]["timesincestart"])
            if g.iloc[i]["Model"] == MODEL_MASCOT_DS:
                ti[i] = _ds_integer_time_index_for_timesincestart(t, t_ref, str(sim))
            else:
                frac = _frac_index_on_reference_grid(t, t_ref)
                ti[i] = int(np.round(frac))
        g["time_index"] = ti
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def _is_ne_two_model_dataframe(df: pd.DataFrame) -> bool:
    if "Model" not in df.columns:
        return False
    models = set(df["Model"].astype(str).unique())
    return MODEL_MASCOT in models and MODEL_MASCOT_DS in models


def add_prevalence_time_index_matched_to_ne_ds(
    df_prev: pd.DataFrame,
    df_ne: pd.DataFrame,
) -> pd.DataFrame:
    """
    Same integer ``time_index`` as MASCOT-DS Ne: for each ``Simulation``, the grid
    is ``np.unique`` sorted ``timesincestart`` from Ne rows with ``Model`` MASCOT-DS.
    """
    pieces: list[pd.DataFrame] = []
    for sim, g in df_prev.groupby("Simulation", sort=False):
        sim_str = str(sim)
        g_ne_ds = df_ne[
            (df_ne["Simulation"].astype(str) == sim_str)
            & (df_ne["Model"] == MODEL_MASCOT_DS)
        ]
        if g_ne_ds.empty:
            raise ValueError(
                f"No MASCOT-DS Ne rows for Simulation {sim_str!r} (cannot align prevalence)."
            )
        t_ref = np.unique(np.sort(g_ne_ds["timesincestart"].to_numpy(dtype=float)))
        g = g.copy()
        ti = np.empty(len(g), dtype=int)
        for i in range(len(g)):
            t = float(g.iloc[i]["timesincestart"])
            ti[i] = _ds_integer_time_index_for_timesincestart(t, t_ref, sim_str)
        g["time_index"] = ti
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def add_prevalence_time_index_per_simulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ``Simulation``, assign ``time_index`` 0,1,... for each distinct
    ``timesincestart`` in ascending order (``np.unique`` order).
    """
    pieces: list[pd.DataFrame] = []
    for _, g in df.groupby("Simulation", sort=False):
        g = g.copy()
        _, inv = np.unique(
            g["timesincestart"].to_numpy(dtype=float),
            return_inverse=True,
        )
        g["time_index"] = inv.astype(int)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def add_prevalence_deme_role_column(
    df: pd.DataFrame,
    starting_deme_by_sim: dict[str, int],
) -> pd.DataFrame:
    """Tag each row as ``start`` or ``secondary`` using outbreak start deme."""
    out = df.copy()
    starts = out["Simulation"].map(starting_deme_by_sim)
    if starts.isna().any():
        missing = out.loc[starts.isna(), "Simulation"].unique()
        raise ValueError(
            "Missing outbreak start deme for Simulation id(s) not in lookup: "
            f"{list(missing)}"
        )
    deme_int = out["Deme"].astype(int)
    start_int = starts.astype(int)
    out["deme_role"] = np.where(deme_int == start_int, "start", "secondary")
    return out


def add_prevalence_infection_arrived_column(
    df: pd.DataFrame,
    trajectory_meta: dict[str, tuple[int, float, float]],
) -> pd.DataFrame:
    """
    ``infection_arrived``: ``timesincestart`` is at/after first ``I`` in this deme
    (from trajectory: first-I time in start deme vs secondary deme).
    """
    out = df.copy()
    meta_series = out["Simulation"].map(trajectory_meta)
    if meta_series.isna().any():
        missing = out.loc[meta_series.isna(), "Simulation"].unique()
        raise ValueError(
            "Missing trajectory metadata for Simulation id(s): "
            f"{list(missing)}"
        )
    t = out["timesincestart"].to_numpy(dtype=float)
    deme = out["Deme"].to_numpy(dtype=int)
    start_deme = np.array([m[0] for m in meta_series], dtype=int)
    t_first_start = np.array([m[1] for m in meta_series], dtype=float)
    t_first_secondary = np.array([m[2] for m in meta_series], dtype=float)
    threshold = np.where(deme == start_deme, t_first_start, t_first_secondary)
    out["infection_arrived"] = t >= threshold - 1e-12
    return out


def filter_prevalence_rows_where_all_sims_infection_arrived(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep rows where, for each ``(time_index, deme_role)``, every simulation has
    ``infection_arrived`` True (so coverage is only plotted after all sims have
    I in that deme).
    """
    out = df.copy()
    out["all_arrived"] = out.groupby(
        ["time_index", "deme_role"], sort=False
    )["infection_arrived"].transform(lambda s: s.all())
    return out[out["all_arrived"]].drop(columns=["all_arrived"])


def prevalence_coverage_by_time_index_and_role(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ``inHPD`` grouped by ``time_index`` and ``deme_role``."""
    return (
        df.groupby(["time_index", "deme_role"], sort=True)
        .agg(coverage=("inHPD", "mean"))
        .reset_index()
    )


def ne_coverage_by_time_index_role_model(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ``inHPD`` grouped by ``time_index``, ``deme_role``, and ``Model``."""
    return (
        df.groupby(["time_index", "deme_role", "Model"], sort=True)
        .agg(coverage=("inHPD", "mean"))
        .reset_index()
    )


def _prepare_validation_timeseries_df_ne(
    df: pd.DataFrame,
    trajectory_meta: dict[str, tuple[int, float, float]],
    starting_deme_by_sim: dict[str, int],
) -> pd.DataFrame:
    """
    Ne combined CSV: time index from MASCOT-DS grid; MASCOT mapped to nearest
    integer index. Infection filter uses MASCOT-DS rows only (same semantics as
    prevalence on that grid).
    """
    df = add_ne_time_index_mascot_ds_reference(df)
    df["inHPD"] = pd.to_numeric(df["inHPD"], errors="coerce")
    df = add_prevalence_infection_arrived_column(df, trajectory_meta)
    df = add_prevalence_deme_role_column(df, starting_deme_by_sim)
    df_key = df[df["Model"] == MODEL_MASCOT_DS].drop_duplicates(
        subset=["Simulation", "time_index", "deme_role"], keep="first"
    )
    df_filt = filter_prevalence_rows_where_all_sims_infection_arrived(df_key)
    valid_keys = df_filt[["Simulation", "time_index", "deme_role"]].drop_duplicates()
    return df.merge(valid_keys, on=["Simulation", "time_index", "deme_role"], how="inner")


def prepare_validation_timeseries_df(
    df: pd.DataFrame,
    trajectory_meta: dict[str, tuple[int, float, float]],
    starting_deme_by_sim: dict[str, int],
    ne_reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add ``time_index``, ``deme_role``, ``infection_arrived``, and drop timepoints
    before every simulation has infection in that deme role.

    For **prevalence** (no ``Model`` or single model): ``time_index`` is the usual
    per-simulation index from sorted unique ``timesincestart``, **unless**
    ``ne_reference_df`` is given (combined Ne CSV): then indices match the
    MASCOT-DS grid from Ne so prevalence aligns with Ne figures.

    For **Ne** with both MASCOT and MASCOT-DS: ``time_index`` is defined from the
    MASCOT-DS ``timesincestart`` grid; MASCOT rows use fractional position on that
    grid, rounded to the nearest integer. The infection filter is applied to
    MASCOT-DS rows only, then both models are merged back on
    ``(Simulation, time_index, deme_role)``.
    """
    if _is_ne_two_model_dataframe(df):
        return _prepare_validation_timeseries_df_ne(
            df, trajectory_meta, starting_deme_by_sim
        )
    if ne_reference_df is not None:
        df = add_prevalence_time_index_matched_to_ne_ds(df, ne_reference_df)
    else:
        df = add_prevalence_time_index_per_simulation(df)
    df["inHPD"] = pd.to_numeric(df["inHPD"], errors="coerce")
    df = add_prevalence_infection_arrived_column(df, trajectory_meta)
    df = add_prevalence_deme_role_column(df, starting_deme_by_sim)
    df_key = df.drop_duplicates(
        subset=["Simulation", "time_index", "deme_role"], keep="first"
    )
    df_filt = filter_prevalence_rows_where_all_sims_infection_arrived(df_key)
    valid_keys = df_filt[["Simulation", "time_index", "deme_role"]].drop_duplicates()
    return df.merge(valid_keys, on=["Simulation", "time_index", "deme_role"], how="inner")


def min_time_index_secondary_deme(
    df: pd.DataFrame,
    *,
    model: str | None = None,
) -> float:
    """Smallest ``time_index`` among rows with ``deme_role == 'secondary'``."""
    sub = df[df["deme_role"] == "secondary"]
    if model is not None:
        sub = sub[sub["Model"] == model]
    if sub.empty:
        raise ValueError(
            "No rows in secondary deme after filtering (cannot compute min time_index)."
        )
    return float(sub["time_index"].min())


def assert_matching_secondary_min_time_index(
    df_prev: pd.DataFrame,
    df_ne: pd.DataFrame,
) -> None:
    """
    Require prevalence and Ne (MASCOT-DS) to agree on the earliest ``time_index``
    in the secondary deme after filtering.
    """
    mp = min_time_index_secondary_deme(df_prev)
    mn = min_time_index_secondary_deme(df_ne, model=MODEL_MASCOT_DS)
    if abs(mp - mn) > 1e-9:
        raise ValueError(
            "Secondary deme minimum time_index must match between prevalence and Ne "
            f"(MASCOT-DS): prevalence {mp} vs Ne {mn}. "
            "Prevalence must use the same MASCOT-DS timesincestart grid per Simulation as Ne."
        )


# Output column suffix for ``bias_<suf>`` / ``rel_hpd_width_<suf>`` (not always median_col.lower()).
_BIAS_REL_OUT_SUFFIX: dict[str, str] = {
    "logNe": "logne",
    "logPrevalence": "logprev",
}


def add_bias_and_hpd_width_columns(
    df: pd.DataFrame,
    *,
    expected_col: str,
    median_col: str,
    rel_hpd_width: bool = False,
) -> pd.DataFrame:
    """
    Bias (posterior median minus expected) and relative HPD width ((upper−lower)/|median|),
    or absolute HPD width, on the log scale implied by ``median_col``.

    Uses ``{median_col}_hpd_lower`` and ``{median_col}_hpd_upper`` for the HPD bounds.
    Writes ``bias_<suffix>`` and ``rel_hpd_width_<suffix>`` (or ``hpd_width_<suffix>``).
    """
    if median_col not in _BIAS_REL_OUT_SUFFIX:
        raise ValueError(
            f"median_col must be one of {sorted(_BIAS_REL_OUT_SUFFIX)}, got {median_col!r}"
        )
    suf = _BIAS_REL_OUT_SUFFIX[median_col]
    lo_col = f"{median_col}_hpd_lower"
    hi_col = f"{median_col}_hpd_upper"
    out = df.copy()
    ev = pd.to_numeric(out[expected_col], errors="coerce")
    med = pd.to_numeric(out[median_col], errors="coerce")
    lo = pd.to_numeric(out[lo_col], errors="coerce")
    hi = pd.to_numeric(out[hi_col], errors="coerce")
    out[f"bias_{suf}"] = med - ev
    denom = np.maximum(np.abs(med), 1e-12)
    if rel_hpd_width:
        out[f"rel_hpd_width_{suf}"] = (hi - lo) / denom
    else:
        out[f"hpd_width_{suf}"] = hi - lo
    return out


def add_prevalence_bias_and_hpd_width_real_space(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bias and relative HPD width on the **natural prevalence scale**, from ln-space
    columns in the validation CSV.

    ``bias_prev_real = exp(median_ln) - exp(expected_ln)`` (additive error in
    prevalence units). **Not** ``exp(median_ln - expected_ln)``, which would be the
    ratio ``median/truth``, not a difference.

    ``rel_hpd_width_prev_real = (exp(hi_ln) - exp(lo_ln)) / |exp(median_ln)|``.
    **Not** ``exp(hi_ln - lo_ln)``, which does not equal the HPD span on the natural
    scale.
    """
    out = df.copy()
    med_ln = pd.to_numeric(out["logPrevalence"], errors="coerce")
    ev_ln = pd.to_numeric(out["expectedlogPrev"], errors="coerce")
    lo_ln = pd.to_numeric(out["logPrevalence_hpd_lower"], errors="coerce")
    hi_ln = pd.to_numeric(out["logPrevalence_hpd_upper"], errors="coerce")
    med = np.exp(med_ln)
    ev = np.exp(ev_ln)
    lo = np.exp(lo_ln)
    hi = np.exp(hi_ln)
    out["bias_prev_real"] = med - ev
    out["bias_prev_real_rel"] = (med - ev) / np.maximum(ev, 1e-12)
    width = hi - lo
    out["rel_hpd_width_prev_real"] = width / np.maximum(np.abs(med), 1e-12)
    return out


def _quantile_low(s: pd.Series) -> float:
    return float(s.quantile(SIMULATION_QUANTILE_LOW))


def _quantile_high(s: pd.Series) -> float:
    return float(s.quantile(SIMULATION_QUANTILE_HIGH))


def median_quantile_metric_by_time_index_role_model(
    df: pd.DataFrame,
    metric_col: str,
) -> pd.DataFrame:
    """
    Per (time_index, deme_role, Model): median and 2.5% / 97.5% quantiles of
    ``metric_col`` across simulations.
    """
    sub = df[["time_index", "deme_role", "Model", metric_col]].copy()
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    return (
        sub.groupby(["time_index", "deme_role", "Model"], sort=True)
        .agg(
            median_value=(metric_col, "median"),
            q025=(metric_col, _quantile_low),
            q975=(metric_col, _quantile_high),
        )
        .reset_index()
    )


def median_quantile_metric_by_time_index_role(
    df: pd.DataFrame,
    metric_col: str,
) -> pd.DataFrame:
    """Same as ``median_quantile_metric_by_time_index_role_model`` without ``Model``."""
    sub = df[["time_index", "deme_role", metric_col]].copy()
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    return (
        sub.groupby(["time_index", "deme_role"], sort=True)
        .agg(
            median_value=(metric_col, "median"),
            q025=(metric_col, _quantile_low),
            q975=(metric_col, _quantile_high),
        )
        .reset_index()
    )


def _roles() -> tuple[tuple[str, str], ...]:
    return (
        ("start", "Start deme"),
        ("secondary", "Secondary deme"),
    )


def plot_prevalence_coverage_over_time(
    df_prev: pd.DataFrame,
    output_png: Path,
    trajectory_meta: dict[str, tuple[int, float, float]],
    *,
    ne_reference_df: pd.DataFrame | None = None,
) -> None:
    """
    Two stacked panels: coverage (%) vs time index for start deme and secondary deme.

    Uses trajectory first-I times per deme; only ``time_index`` values where every
    simulation already has infection in that deme are plotted.

    Pass ``ne_reference_df`` (combined Ne CSV) so ``time_index`` matches Ne MASCOT-DS.
    """
    configure_pdf_fonts()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    required = ["Deme", "timesincestart", "inHPD", "Simulation"]
    missing = [c for c in required if c not in df_prev.columns]
    if missing:
        raise ValueError(f"Prevalence CSV missing columns: {missing}")

    starting_deme_by_sim = {s: m[0] for s, m in trajectory_meta.items()}
    df = prepare_validation_timeseries_df(
        df_prev,
        trajectory_meta,
        starting_deme_by_sim,
        ne_reference_df=ne_reference_df,
    )
    agg = prevalence_coverage_by_time_index_and_role(df)

    fig, axes = plt.subplots(2, 1, figsize=(4.5, 5.2), sharex=True)
    color = COLORS[3]
    for ax, (role_key, panel_title) in zip(axes, _roles()):
        sub = agg[agg["deme_role"] == role_key].sort_values("time_index")
        ax.plot(
            sub["time_index"],
            sub["coverage"] * 100.0,
            color=color,
            marker="o",
            ms=2.5,
            lw=0.9,
        )
        ax.set_ylim(0.0, 103.0)
        ax.set_yticks(np.linspace(0.0, 100.0, 6))
        ax.set_xlim(0.0, 103.0)
        ax.set_title(panel_title, fontsize=FONTSIZES_LIST[0])
        set_axis_fontsizes(
            ax,
            FONTSIZES_LIST,
            xlabel=None,
            ylabel="Coverage (%)",
        )
        ax.tick_params(labelsize=FONTSIZES_LIST[2])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Time index", fontsize=FONTSIZES_LIST[1])
    fig.tight_layout()
    save_figure_png_and_pdf(output_png)
    plt.close(fig)
    return agg


def plot_ne_coverage_over_time(
    df_ne: pd.DataFrame,
    output_png: Path,
    trajectory_meta: dict[str, tuple[int, float, float]],
    *,
    models_colors: tuple[tuple[str, str], ...] = MODEL_COLORS_DEFAULT,
) -> None:
    """
    Two stacked panels: mean HPD coverage (%) vs time index for start and secondary
    deme, with separate lines for MASCOT and MASCOT-DS. Uses the same trajectory
    alignment and infection filter as ``plot_prevalence_coverage_over_time``.
    """
    configure_pdf_fonts()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    required = [
        "Deme",
        "timesincestart",
        "inHPD",
        "Simulation",
        "Model",
    ]
    missing = [c for c in required if c not in df_ne.columns]
    if missing:
        raise ValueError(f"Ne validation CSV missing columns: {missing}")

    starting_deme_by_sim = {s: m[0] for s, m in trajectory_meta.items()}
    df = prepare_validation_timeseries_df(df_ne, trajectory_meta, starting_deme_by_sim)
    agg = ne_coverage_by_time_index_role_model(df)

    fig, axes = plt.subplots(2, 1, figsize=(4.5, 5.2), sharex=True)
    model_order = [m for m, _ in models_colors]
    color_by_model = {m: c for m, c in models_colors}

    for ax, (role_key, panel_title) in zip(axes, _roles()):
        sub_all = agg[agg["deme_role"] == role_key]
        for model_name in model_order:
            sub = sub_all[sub_all["Model"] == model_name].sort_values("time_index")
            if sub.empty:
                continue
            ax.plot(
                sub["time_index"],
                sub["coverage"] * 100.0,
                color=color_by_model[model_name],
                marker="o",
                ms=2.5,
                lw=0.9,
                label=model_name,
            )
        ax.set_ylim(0.0, 103.0)
        ax.set_yticks(np.linspace(0.0, 100.0, 6))
        ax.set_xlim(0.0, 103.0)
        ax.set_title(panel_title, fontsize=FONTSIZES_LIST[0])
        set_axis_fontsizes(
            ax,
            FONTSIZES_LIST,
            xlabel=None,
            ylabel="Coverage (%)",
        )
        ax.tick_params(labelsize=FONTSIZES_LIST[2])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=FONTSIZES_LIST[2])
    axes[-1].set_xlabel("Time index", fontsize=FONTSIZES_LIST[1])
    fig.tight_layout()
    save_figure_png_and_pdf(output_png)
    plt.close(fig)


def plot_two_panel_metric_multi_model(
    agg: pd.DataFrame,
    output_png: Path,
    *,
    ylabel: str,
    models_colors: tuple[tuple[str, str], ...] = MODEL_COLORS_DEFAULT,
    draw_zero_line: bool = False,
    band_alpha: float = DEFAULT_METRIC_BAND_ALPHA,
) -> None:
    """
    Two panels (start / secondary deme): per model, median line (no markers) and
    shaded band between simulation-level 2.5% and 97.5% quantiles.

    ``agg`` must include ``time_index``, ``deme_role``, ``Model``, ``median_value``,
    ``q025``, ``q975``.
    """
    configure_pdf_fonts()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    model_order = [m for m, _ in models_colors]
    color_by_model = {m: c for m, c in models_colors}

    fig, axes = plt.subplots(2, 1, figsize=(4.5, 5.2), sharex=True)
    for ax, (role_key, panel_title) in zip(axes, _roles()):
        sub_all = agg[agg["deme_role"] == role_key]
        for model_name in model_order:
            sub = sub_all[sub_all["Model"] == model_name].sort_values("time_index")
            if sub.empty:
                continue
            c = color_by_model[model_name]
            ax.fill_between(
                sub["time_index"],
                sub["q025"],
                sub["q975"],
                color=c,
                alpha=band_alpha,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                sub["time_index"],
                sub["median_value"],
                color=c,
                lw=0.9,
                zorder=2,
                label=model_name,
            )
        if draw_zero_line:
            ax.axhline(0.0, color="0.75", lw=0.7, ls="--", zorder=0)
        ax.set_xlim(0.0, 103.0)
        ax.set_title(panel_title, fontsize=FONTSIZES_LIST[0])
        set_axis_fontsizes(
            ax,
            FONTSIZES_LIST,
            xlabel=None,
            ylabel=ylabel,
        )
        ax.tick_params(labelsize=FONTSIZES_LIST[2])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=FONTSIZES_LIST[2])
    axes[-1].set_xlabel("Time index", fontsize=FONTSIZES_LIST[1])
    fig.tight_layout()
    save_figure_png_and_pdf(output_png)
    plt.close(fig)


def plot_two_panel_metric_single_series(
    agg: pd.DataFrame,
    output_png: Path,
    *,
    ylabel: str,
    color: str,
    draw_zero_line: bool = False,
    band_alpha: float = DEFAULT_METRIC_BAND_ALPHA,
) -> None:
    """
    Two panels with one series per panel: median line and 2.5–97.5% quantile band
    (no ``Model`` column). ``agg`` must have ``median_value``, ``q025``, ``q975``.
    """
    configure_pdf_fonts()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(4.5, 5.2), sharex=True)
    for ax, (role_key, panel_title) in zip(axes, _roles()):
        sub = agg[agg["deme_role"] == role_key].sort_values("time_index")
        if sub.empty:
            continue
        ax.fill_between(
            sub["time_index"],
            sub["q025"],
            sub["q975"],
            color=color,
            alpha=band_alpha,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            sub["time_index"],
            sub["median_value"],
            color=color,
            lw=0.9,
            zorder=2,
        )
        if draw_zero_line:
            ax.axhline(0.0, color="0.75", lw=0.7, ls="--", zorder=0)
        ax.set_xlim(0.0, 103.0)
        ax.set_title(panel_title, fontsize=FONTSIZES_LIST[0])
        set_axis_fontsizes(
            ax,
            FONTSIZES_LIST,
            xlabel=None,
            ylabel=ylabel,
        )
        ax.tick_params(labelsize=FONTSIZES_LIST[2])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Time index", fontsize=FONTSIZES_LIST[1])
    fig.tight_layout()
    save_figure_png_and_pdf(output_png)
    plt.close(fig)
