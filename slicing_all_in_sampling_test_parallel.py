#!/usr/bin/env python3
"""
Standalone parallel sampling + ELA extraction script.

Parallelization strategy:
- Sampling is done once per group (cheap)
- Evaluation + ELA are parallelized over (fid, iid)
"""

from pathlib import Path
from itertools import product
from multiprocessing import cpu_count, get_context
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from scipy.linalg import pinv
from scipy.stats.qmc import scale

from ioh import get_problem
from ioh.iohcpp.problem import RealSingleObjective

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

# Set this to the directory where you want to save the outputs
MAIN_DIR = Path(__file__).parent.resolve()


#FUNCTION_IDS = [1, 8, 11, 16, 20]
#FUNCTION_IDS = [2, 3, 4, 5, 6,7, 9, 10, 12, 13, 14, 15, 17, 18, 19,21, 22, 23, 24]

# For quick testing, you can use a smaller subset of functions and instances
FUNCTION_IDS = list(range(1, 25))
INSTANCE_IDS = list(range(15))

# Ambient Dimension
D = 20

# Intrinsic Dimension
d = 10

# Number of groups (independent samplings)
N_GROUPS = 40 * (D // d)
N_SLICES = 1
INITIAL_SEED = 44
MULTIPLIER = 10

# Upper and lower bounds for sampling in the low-D space (before projection) 
# This reference is used for clipping the high-D samples after projection, and for scaling the low-D samples before projection.
CONSTRAINTS = (-5.0, 5.0)

# Save directory for ELA features (will be created if it doesn't exist)
SAVE_DIR = MAIN_DIR / f"sampling_outputs_all_in_{D}D_{d}D"

# Whether to compute ELA level and NBC features for the low-D samples (slices). These cannot be computed
# if the number of samples per slice is too small, so we make it optional.
COMPUTE_SMALL_ELA_LEVEL = True
COMPUTE_SMALL_NBC = True

# =========================================================
# Utilities
# =========================================================

def _check_input_size(d: int, D: int) -> None:
    if d >= D:
        raise ValueError(f"d={d} must be < D={D}")


def determine_number_of_samples_per_slice(n: int, S: int) -> int:
    return max(2, n // S)


def compute_global_seed_array(n: int, initial_seed: int) -> List[int]:
    return [initial_seed + i for i in range(n)]


# =========================================================
# Sampling
# =========================================================

def sample_embedding_matrix(
    d: int,
    D: int,
    n_samples: int,
    seed: int,
    constraints: Tuple[float, float] = (-5.0, 5.0),
    normalize_embedding: bool = True,
) -> Dict[str, np.ndarray]:

    _check_input_size(d, D)
    rng = np.random.default_rng(seed)

    embedding = rng.standard_normal((D, d))
    embedding /= np.linalg.norm(embedding, axis=0, keepdims=True)

    if normalize_embedding:
        U, _, Vt = np.linalg.svd(embedding, full_matrices=False)
        embedding = U @ Vt

    embedding_pinv = pinv(embedding)

    sampler = get_sampler("lhs")
    samples = sampler(dim=d, n_samples=n_samples, random_seed=seed)

    samples = scale(
        samples,
        l_bounds=constraints[0] * np.ones(d),
        u_bounds=constraints[1] * np.ones(d),
    ) * np.sqrt(D / d)

    samples_up = np.clip(samples @ embedding.T, *constraints)
    samples_down = samples_up @ embedding_pinv.T

    return {
        "low_D_samples": samples_down,
        "high_D_samples": samples_up,
    }


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
) -> pd.DataFrame:

    problem = get_problem(fid, inst_id, dim)

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
        X, fX, problem.optimum.y, minkowski_p=2.0
    ),
}


    return pd.DataFrame(features, index=[0])


# =========================================================
# Worker (must be top-level!)
# =========================================================

def process_problem_instance(
    fid: int,
    iid: int,
    D: int,
    seed_idx: int,
    group_idx: int,
    group_list: list,
    save_dir: Path,
):

    r""" 
    Process a single problem instance (fid, iid) for a given group of samplings. This function evaluates the high-D samples, 
    extracts ELA features for both the full set and the slices, and saves the results to CSV files.

    The ELA features for the full set are saved in "full.csv", while the features for each slice are saved in "slice{ii}.csv". 
    The directory structure is organized by function, instance, and group.

    The `seed_idx` is used to ensure that the ELA features for different groups are computed with different random seeds, which can affect features like information content. 
    The `group_list` contains the sampled low-D and high-D points for all slices in the current group.

    Args:
        fid (int): Function ID of the problem instance.
        iid (int): Instance ID of the problem instance.
        D (int): Ambient dimension.
        seed_idx (int): Base random seed index for ELA feature computation.
        group_idx (int): Index of the current group (for organizational purposes).
        group_list (list): List of dictionaries containing "low_D_samples" and "high_D_samples" for each slice in the group.
        save_dir (Path): Base directory where the ELA features will be saved.

    Returns:
        None. The function saves the ELA features to CSV files and does not return any value.
    """

    problem = get_problem(fid, iid, D)

    fitness_per_slice = [
        np.array([problem(x) for x in resp["high_D_samples"]])
        for resp in group_list
    ]

    combined_fitness = np.concatenate(fitness_per_slice)
    combined_samples = np.vstack(
        [resp["high_D_samples"] for resp in group_list]
    )

    ela_full = extract_ela_features(
        seed=seed_idx,
        X=combined_samples,
        fX=combined_fitness,
        dim=D,
        fid=fid,
        inst_id=iid,
        compute_ela_level=True,
        compute_nbc=True,
    )

    full_path = save_dir / f"f{fid}/iid_{iid}/group{group_idx}/full.csv"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    ela_full.to_csv(full_path, index=False)

    for ii, resp in enumerate(group_list):
        ela_slice = extract_ela_features(
            seed=seed_idx + ii,
            X=resp["low_D_samples"],
            fX=fitness_per_slice[ii],
            dim=D,
            fid=fid,
            inst_id=iid,
            compute_ela_level=COMPUTE_SMALL_ELA_LEVEL,
            compute_nbc=COMPUTE_SMALL_NBC,
        )

        slice_path = (
            save_dir
            / f"f{fid}/iid_{iid}/group{group_idx}/slice{ii + 1}.csv"
        )
        ela_slice.to_csv(slice_path, index=False)

    return fid, iid


def process_problem_instance_all_in(
    fid: int,
    iid: int,
    D: int,
    seed_idx: int,
    group_idx: int,
    group_list: list,
    save_dir: Path)->None:

    r"""
    This function is a wrapper around `process_problem_instance` that calls it and ignores the returned
    (fid, iid) tuple. It is used for parallel processing where the return value is not needed.

    Args
    ------
        fid (int): Function ID of the problem instance.
        iid (int): Instance ID of the problem instance.
        D (int): Ambient dimension.
        seed_idx (int): Base random seed index for ELA feature computation.
        group_idx (int): Index of the current group (for organizational purposes).
        group_list (list): List of dictionaries containing "low_D_samples" and "high_D_samples" for each slice in the group.
        save_dir (Path): Base directory where the ELA features will be saved.
    
    Returns
    -------
        None. The function calls `process_problem_instance` and does not return any value.

    """

    a = 1



# =========================================================
# Main
# =========================================================

def main():

    # 
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    n_samples = MULTIPLIER * D
    
    #n_samples_per_slice = determine_number_of_samples_per_slice(
    #    n=n_samples, S=N_SLICES
    #)

    n_samples_per_slice = n_samples

    global_seeds = compute_global_seed_array(
        n=N_GROUPS * N_SLICES, initial_seed=INITIAL_SEED
    )

    group_seeds = global_seeds[::N_SLICES]

    ctx = get_context("spawn")
    max_workers = 6
    n_workers = max(1, max_workers)

    for group_idx, seed_idx in enumerate(group_seeds):

        print(f"\n=== Group {group_idx} (seed={seed_idx}) ===")

        group_list = [
            sample_embedding_matrix(
                d=d,
                D=D,
                n_samples=n_samples_per_slice,
                seed=seed_idx + s,
            )
            for s in range(N_SLICES)
        ]

        tasks = [
            (fid, iid, D, seed_idx, group_idx, group_list, SAVE_DIR)
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
