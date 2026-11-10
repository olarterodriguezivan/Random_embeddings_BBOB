#!/usr/bin/env python3
"""
Standalone parallel full-space sampling + ELA extraction script.

Parallelization strategy:
- Sampling is done once per group in the full D-dimensional space (cheap)
- Evaluation + ELA are parallelized over (fid, iid)

This version removes slice/embedding sampling:
- No low-dimensional d
- No N_SLICES
- No embedding / pseudo-inverse
- One full D-dimensional design per group
- One ELA output per (fid, iid, group): full.csv
"""

from pathlib import Path
from itertools import product
from multiprocessing import get_context
from typing import List, Tuple

import numpy as np
import pandas as pd

from scipy.stats.qmc import scale

from sklearn.preprocessing import MinMaxScaler

from ioh import get_problem

# pflacco
from pflacco.classical_ela_features import (
    calculate_ela_meta,
    calculate_ela_distribution,
    calculate_ela_level,
    calculate_nbc,
    calculate_dispersion,
    calculate_information_content,
    calculate_pca,
)
from pflacco.misc_features import calculate_fitness_distance_correlation

# your sampler
from qmc_samplers import get_sampler


# =========================================================
# Configuration
# =========================================================

MAIN_DIR = Path(__file__).parent.resolve()

# FUNCTION_IDS = [1, 8, 11, 16, 20]
# FUNCTION_IDS = [2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 23, 24]
FUNCTION_IDS = list(range(1, 25))
INSTANCE_IDS = list(range(15))

D = 2
N_GROUPS = 40
INITIAL_SEED = 43
MULTIPLIER = 100

CONSTRAINTS = (-5.0, 5.0)

COMPUTE_ELA_LEVEL = True
COMPUTE_NBC = True

NORMALIZE_X = True
NORMALIZE_Y = True


SAVE_DIR = MAIN_DIR / f"sampling_outputs_{D}D_fullspace_mult_{MULTIPLIER}_groups{N_GROUPS}_seed{INITIAL_SEED}_normalized{NORMALIZE_X}_{NORMALIZE_Y}"




# =========================================================
# Utilities
# =========================================================

def determine_number_of_samples(n_multiplier: int, D: int) -> int:
    return n_multiplier * D


def compute_global_seed_array(n: int, initial_seed: int) -> List[int]:
    return [initial_seed + i for i in range(n)]


# =========================================================
# Sampling
# =========================================================

def sample_full_space(
    D: int,
    n_samples: int,
    seed: int,
    constraints: Tuple[float, float] = (-5.0, 5.0),
    sampler_name: str = "lhs",
) -> np.ndarray:
    """
    Sample directly in the full D-dimensional box.

    This replaces the old slice logic:
        low-D LHS -> random embedding -> clipped high-D samples

    with:
        full-D LHS -> scaled to [lower, upper]^D
    """

    sampler = get_sampler(sampler_name)
    samples = sampler(dim=D, n_samples=n_samples, random_seed=seed)

    samples = scale(
        samples,
        l_bounds=constraints[0] * np.ones(D),
        u_bounds=constraints[1] * np.ones(D),
    )

    return samples


# =========================================================
# ELA
# =========================================================

def extract_ela_features(
    seed: int,
    X: np.ndarray,
    fX: np.ndarray,
    dim: int,
    fid: int,
    inst_id: int,
    compute_ela_level: bool = True,
    compute_nbc: bool = True,
    normalize_x: bool = False,
    normalize_y: bool = False,
) -> pd.DataFrame:

    r"""
    Extract ELA features from the given samples and their corresponding function values.

    Args:
        seed (int): Random seed for reproducibility.
        X (np.ndarray): Sampled points in the input space. (This should be a 2D array of shape (n_samples, dim). with all values in the range [-5,5] as for BBOB)
        fX (np.ndarray): Function values corresponding to the sampled points.
        dim (int): Dimensionality of the input space.
        fid (int): Function ID for the problem.
        inst_id (int): Instance ID for the problem.
        compute_ela_level (bool): Whether to compute ELA level features.
        compute_nbc (bool): Whether to compute nearest better clustering features.
        normalize_x (bool): Whether to normalize the input samples X to the range [0,1]^dim.
        normalize_y (bool): Whether to normalize the function values fX to the range [0,1].
    
    Returns:
        pd.DataFrame: A DataFrame containing the extracted ELA features.
    """
    problem = get_problem(fid, inst_id, dim)
    opt_y = problem.optimum.y
    
    if normalize_x:
        X = (X + 5.0) / (10.0)

    if normalize_y:
        scaler = MinMaxScaler()
        scaler.fit(fX.reshape(-1, 1))
        fX = scaler.transform(fX.reshape(-1, 1)).flatten()
        opt_y = scaler.transform(np.array([[opt_y]])).flatten()[0]


    features = {
        **calculate_ela_meta(X, fX),
        **calculate_ela_distribution(X, fX),
        **(
            calculate_ela_level(
                X, fX, ela_level_quantiles=[0.1, 0.25, 0.5]
            )
            if compute_ela_level
            else {}
        ),
        **(calculate_nbc(X, fX) if compute_nbc else {}),
        **calculate_dispersion(X, fX),
        **calculate_information_content(X, fX, seed=seed),
        **calculate_pca(X, fX),
        **calculate_fitness_distance_correlation(
            X, fX, opt_y, minkowski_p=2.0
        ),
    }

    return pd.DataFrame(features, index=[0])


# =========================================================
# Worker must be top-level for multiprocessing spawn
# =========================================================

def process_problem_instance(
    fid: int,
    iid: int,
    D: int,
    seed_idx: int,
    group_idx: int,
    X_full: np.ndarray,
    save_dir: Path,
):

    problem = get_problem(fid, iid, D)

    fX = np.array([problem(x) for x in X_full])

    ela_full = extract_ela_features(
        seed=seed_idx,
        X=X_full,
        fX=fX,
        dim=D,
        fid=fid,
        inst_id=iid,
        compute_ela_level=COMPUTE_ELA_LEVEL,
        compute_nbc=COMPUTE_NBC,
        normalize_x=NORMALIZE_X,
        normalize_y=NORMALIZE_Y,
    )

    full_path = save_dir / f"f{fid}/iid_{iid}/group{group_idx}/full.csv"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    ela_full.to_csv(full_path, index=False)

    return fid, iid


# =========================================================
# Main
# =========================================================

def main():

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    n_samples = determine_number_of_samples(
        n_multiplier=MULTIPLIER,
        D=D,
    )

    # One seed per full-space group.
    group_seeds = compute_global_seed_array(
        n=N_GROUPS,
        initial_seed=INITIAL_SEED,
    )

    ctx = get_context("spawn")
    max_workers = 4
    n_workers = max(1, max_workers)

    for group_idx, seed_idx in enumerate(group_seeds):

        print(f"\n=== Group {group_idx} (seed={seed_idx}) ===")

        X_full = sample_full_space(
            D=D,
            n_samples=n_samples,
            seed=seed_idx,
            constraints=CONSTRAINTS,
            sampler_name="sobol",
        )

        tasks = [
            (fid, iid, D, seed_idx, group_idx, X_full, SAVE_DIR)
            for fid, iid in product(FUNCTION_IDS, INSTANCE_IDS)
        ]

        if n_workers == 1:
            for task in tasks:
                process_problem_instance(*task)
        else:
            with ctx.Pool(processes=n_workers) as pool:
                pool.starmap(process_problem_instance, tasks)


if __name__ == "__main__":
    main()
