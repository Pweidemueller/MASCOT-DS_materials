#!/usr/bin/env python3
"""Generate a symmetric contact matrix representing migration/cross infection between demes.

Outputs two CSVs:
1) contact_matrix.csv: the contact matrix C (n x n)
2) sampled_parameters.csv: tidy table with columns [parameter, deme, value]
   - For per-deme entries, `deme` is the integer index [0..n-1]
   - For global parameters (q, offdiag_contact, target/expected migration, start_deme), `deme` is NaN

Notes
- The migration fraction p_mig is estimated in the early-epidemic regime as:
  p_mig = sum_{i!=j} C[i,j]*f_j / sum_{i,j} C[i,j]*f_j, where f_j = N_j / sum_k N_k.
- C is built as: C[j,j] = a_j (sampled per-deme), C[i!=j] = b (shared off-diagonal),
  with b solved so that p_mig ≈ target_migration_fraction given the sampled a_j and f_j.
- q is sampled uniformly in [0.05, 0.95] and reported separately. Since the simulator
  forms the mixing matrix M = q*C, the target migration fraction is independent of q.

"""

import argparse
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

TARGET_MIGRATION = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample contact matrix and parameters representing migration/cross infection between demes"
    )
    parser.add_argument(
        "--n_demes",
        type=int,
        required=True,
        help="Number of demes (n)",
    )
    parser.add_argument(
        "--population_sizes",
        type=str,
        required=True,
        help='Space-separated list of population sizes, e.g. "10000 5000 20000"',
    )
    parser.add_argument(
        "--n_rateshifts",
        type=int,
        default=10,
        help="Number of rate shifts",
    )
    parser.add_argument(
        "--matrix_type",
        type=str,
        default="contact",
        choices=["contact", "mixing"],
        help="Type of matrix to sample: 'contact' for contact matrix, 'mixing' for mixing matrix",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--matrix_csv",
        type=str,
        default="matrix.csv",
        help="Output path for the contact matrix CSV",
    )
    parser.add_argument(
        "--params_csv",
        type=str,
        default="sampled_parameters.csv",
        help="Output path for the sampled parameters CSV",
    )
    return parser.parse_args()


def parse_population_sizes(pop_str: str, n: int) -> np.ndarray:
    vals = [int(x) for x in pop_str.strip().split()]
    if len(vals) != n:
        raise ValueError(f"population_sizes length {len(vals)} != n_demes {n}")
    if any(v < 0 for v in vals):
        raise ValueError("population_sizes must be non-negative")
    if sum(vals) <= 0:
        raise ValueError("Sum of population_sizes must be positive")
    return np.asarray(vals, dtype=float)


def bounded_lognormal(
    rng: np.random.Generator,
    n: int,
    mu: float = 0.0,
    sigma: float = 0.4,
    lo: float = 0.5,
    hi: float = 2.0,
) -> np.ndarray:
    """Sample n lognormal values and clamp to [lo, hi] to avoid extremes."""
    x_single = rng.lognormal(mean=mu, sigma=sigma, size=1)
    x = np.full(n, x_single)
    return np.clip(x, lo, hi)


def solve_offdiag_for_target(
    a: np.ndarray, f: np.ndarray, target: float, n: int
) -> float:
    """Solve for a shared off-diagonal b so that the expected migration fraction ≈ target.

    For C with diag a_j and offdiag b (constant),
    p = b*(n-1) / (E_f[a] + b*(n-1)), where E_f[a] = sum_j f_j * a_j.
    Solve b = target * E_f[a] / ((1 - target) * (n - 1)).
    """
    Ef_a = float(np.sum(f * a))
    if not (0 < target < 1):
        raise ValueError("target must be in (0,1)")
    denom = (1.0 - target) * (n - 1)
    if denom <= 0:
        raise ValueError("Invalid n or target leading to non-positive denominator")
    b = target * Ef_a / denom
    return b


def expected_migration_fraction(C: np.ndarray, f: np.ndarray) -> float:
    total = float(np.sum(C * f[np.newaxis, :]))
    off = float(np.sum((C - np.diag(np.diag(C))) * f[np.newaxis, :]))
    if total <= 0:
        return float("nan")
    return off / total


def sample_contact_matrix(
    n: int, rng: np.random.Generator, f: np.ndarray, target_migration_fraction: float
) -> np.ndarray:
    # Sample diagonal contact intensities (bounded lognormal)
    # for now contacts within the demes are the same for all demes
    a = bounded_lognormal(rng, n, mu=0.0, sigma=0.4, lo=0.5, hi=10.0)

    # Solve for shared off-diagonal b representing migration/cross infection between demes
    b = solve_offdiag_for_target(a, f, target_migration_fraction, n)

    # Build symmetric contact matrix C
    C = np.full((n, n), b, dtype=float)
    np.fill_diagonal(C, a)

    # Verify expected migration fraction achieved by C
    p_exp = expected_migration_fraction(C, f)

    return C, p_exp


def main() -> None:
    args = parse_args()

    # RNG
    rng = np.random.default_rng(seed=args.seed)

    # Populations and fractions
    N = parse_population_sizes(args.population_sizes, args.n_demes)
    n_demes = args.n_demes
    f = N / N.sum()

    # Choose starting deme proportional to population (only among demes with N>0)
    positive = np.where(N > 0)[0]
    if len(positive) == 0:
        raise ValueError("All population sizes are zero")
    probs = (N[positive] / N[positive].sum()).astype(float)
    start_deme = int(rng.choice(positive, p=probs))

    # Set S and I vectors
    S = N.astype(int).copy()
    I = np.zeros(n_demes, dtype=int)
    if S[start_deme] <= 0:
        raise ValueError(
            "Selected start_deme has zero population; cannot assign initial infection"
        )
    S[start_deme] -= 1
    I[start_deme] = 1

    # Sample becoming uninfectious rate
    gamma = float(rng.uniform(50, 100))

    # Sample sampling rate
    samplingrate = float(rng.uniform(0.001, 0.002))

    # Sample mutation rate
    mutation_rate = float(rng.choice([0.5, 1]))

    # Sample reproductive number for each time period
    high_transmission_period = int(args.n_rateshifts / 3)
    decrease_transmission_period = int(args.n_rateshifts / 3)
    low_transmission_period = (
        args.n_rateshifts - high_transmission_period - decrease_transmission_period
    )

    high_transmission_R0 = float(rng.uniform(2.0, 3.0))
    low_transmission_R0 = float(rng.uniform(0.2, 0.8))

    R0 = [high_transmission_R0] * high_transmission_period
    if decrease_transmission_period > 0:
        # Linear decrease from high_transmission_R0 to low_transmission_R0 (inclusive)
        # Make endpoint inclusive (so the last value is low_transmission_R0)
        R0 += list(
            np.linspace(
                high_transmission_R0,
                low_transmission_R0,
                decrease_transmission_period + 1,
                endpoint=True,
            )[1:]
        )
    R0 += [low_transmission_R0] * low_transmission_period

    # Sample offdiag_R (which will inform the forward migration rate in MASCOT) independently for each direction (i,j)
    # create all possible pairwise combination of demes and sample offdiag_R for all possible (i, j) pairs
    for_mig_rate = rng.lognormal(mean=0.5, sigma=0.5, size=(n_demes, n_demes))
    offdiag_R = for_mig_rate / gamma

    # Sample target migration fraction
    target_migration_fraction = float(
        rng.uniform(TARGET_MIGRATION, 5 * TARGET_MIGRATION)
    )

    if args.matrix_type == "contact":

        C, p_exp = sample_contact_matrix(n_demes, rng, f, target_migration_fraction)
        # Sample per-contact transmission probability q
        q = float(rng.uniform(0.05, 0.5))
        # Save contact matrix
        cm_df = pd.DataFrame(C)
        cm_df.to_csv(args.matrix_csv, header=False, index=False)
    elif args.matrix_type == "mixing":
        print("Mixing matrix not implemented yet")

    # Sample datastream parameters
    cc_dispersion = float(
        rng.lognormal(mean=-1.0, sigma=0.5)
    )  # samples k of the Negative Binomial distribution for case counts
    ww_sigma = rng.lognormal(mean=-0.7, sigma=0.3)

    cc_scaling = rng.lognormal(mean=-3.0, sigma=0.5, size=n_demes)
    # sp_scaling = rng.lognormal(mean=0.0, sigma=0.2, size=n_demes)  # formerly sampled; now fixed to 1.0
    sp_scaling = np.ones(n_demes)
    ww_scaling = rng.lognormal(mean=4.5, sigma=0.5, size=n_demes)

    # Build sampled parameters DF
    rows: List[dict] = []

    # Per-deme entries
    for d in range(n_demes):
        rows.append({"parameter": "population_size", "deme": d, "value": float(N[d])})
        rows.append({"parameter": "S", "deme": d, "value": float(S[d])})
        rows.append({"parameter": "I", "deme": d, "value": float(I[d])})

        for d2 in range(n_demes):
            if d != d2:
                rows.append(
                    {
                        "parameter": f"offdiag_R",
                        "deme": f"{d}_{d2}",
                        "value": float(offdiag_R[d][d2]),
                    }
                )

    # Time-varying entries
    for i in range(args.n_rateshifts):
        rows.append(
            {"parameter": f"R0_{i}", "deme": float("nan"), "value": float(R0[i])}
        )

    # Global entries
    rows.append({"parameter": "q", "deme": float("nan"), "value": float(q)})
    rows.append(
        {
            "parameter": "target_migration_fraction",
            "deme": float("nan"),
            "value": float(target_migration_fraction),
        }
    )
    rows.append(
        {
            "parameter": "expected_migration_fraction",
            "deme": float("nan"),
            "value": float(p_exp),
        }
    )
    rows.append(
        {"parameter": "start_deme", "deme": float("nan"), "value": float(start_deme)}
    )
    rows.append(
        {
            "parameter": "samplingrate",
            "deme": float("nan"),
            "value": float(samplingrate),
        }
    )
    rows.append({"parameter": "gamma", "deme": float("nan"), "value": float(gamma)})
    rows.append(
        {
            "parameter": "mutation_rate",
            "deme": float("nan"),
            "value": float(mutation_rate),
        }
    )

    # Datastream parameters
    rows.append(
        {
            "parameter": "ds_cc_dispersion",
            "deme": float("nan"),
            "value": float(cc_dispersion),
        }
    )
    rows.append(
        {"parameter": "ds_ww_sigma", "deme": float("nan"), "value": float(ww_sigma)}
    )
    for d in range(n_demes):
        rows.append(
            {"parameter": f"ds_cc_scaling", "deme": d, "value": float(cc_scaling[d])}
        )
        rows.append(
            {"parameter": f"ds_sp_scaling", "deme": d, "value": float(sp_scaling[d])}
        )
        rows.append(
            {"parameter": f"ds_ww_scaling", "deme": d, "value": float(ww_scaling[d])}
        )

    params_df = pd.DataFrame(rows, columns=["parameter", "deme", "value"])
    params_df.to_csv(args.params_csv, index=False)

    print(f"Wrote contact matrix to: {os.path.abspath(args.matrix_csv)}")
    print(f"Wrote sampled parameters to: {os.path.abspath(args.params_csv)}")


if __name__ == "__main__":
    main()
