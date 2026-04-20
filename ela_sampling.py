from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import os
from multiprocessing import Pool, cpu_count
from ioh import get_problem

# pflacco imports
from pflacco.classical_ela_features import (
    # Classical ELA features
    calculate_ela_meta,
    calculate_ela_distribution,
    calculate_ela_level,
    calculate_ela_local,
    calculate_ela_curvate,
    calculate_ela_conv,

    # Cell mapping features
    calculate_cm_angle,
    calculate_cm_conv,
    calculate_cm_grad,

    # Linear model features
    calculate_limo,

    #Nearest better clustering
    calculate_nbc,

    # Dispersion features Lunacek and Whitley
    calculate_dispersion,

    # Information content features Muñoz et al.
    calculate_information_content,

    # PCA features
    calculate_pca
)

from pflacco.misc_features import (
    
    calculate_fitness_distance_correlation,
    calculate_gradient_features,
    calculate_hill_climbing_features,
    calculate_length_scales_features,
    calculate_sobol_indices_features)

# ---------------------------------------------------------
# Simple helpers
# ---------------------------------------------------------
def read_csv(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)

def read_x_samples(file_path: Path) -> np.ndarray:
    return pd.read_csv(file_path).values

def save_csv(df: pd.DataFrame, out: Path):
    df.to_csv(out, index=False)


# ---------------------------------------------------------
# Directory discovery
# ---------------------------------------------------------
def get_files(directory: Path, suffix: str) -> List[Path]:
    return list(directory.rglob(suffix))


def parse_common_parts(parts):
    """Extract dim, seed, samples, objective_type."""
    dim = int([p for p in parts if p.startswith("Dimension_")][0].split("_")[1])
    seed = int([p for p in parts if p.startswith("seed_")][0].split("_")[1])
    n_samples = int([p for p in parts if p.startswith("Samples_")][0].split("_")[1])
    objective_type = [p for p in parts if p in ["ELA_extraction", "reduction"]][0]
    return dim, seed, n_samples, objective_type


def distill_x_sample_list(file_list: List[Path]) -> Dict[Tuple, Path]:
    distilled = {}
    for f in file_list:
        try:
            dim, seed, n_samples, obj = parse_common_parts(f.parts)
            key = (dim, seed, n_samples, obj)
            distilled[key] = f
        except:
            print(f"Warning: skipping unparseable X file {f}")
    return distilled


def distill_y_sample_list(file_list: List[Path]) -> Dict[Tuple, Tuple[Path, int, int]]:
    distilled = {}
    for f in file_list:
        try:
            dim, seed, n_samples, obj = parse_common_parts(f.parts)
            func_id = int([p for p in f.parts if p.startswith("f_")][0].split("_")[1])
            inst_id = int([p for p in f.parts if p.startswith("id_")][0].split("_")[1])
            key = (dim, seed, n_samples, obj)
            distilled.setdefault(key, []).append((f, func_id, inst_id))
        except:
            print(f"Warning: skipping unparseable Y file {f}")
    return distilled


# ---------------------------------------------------------
# ELA Feature Extraction
# ---------------------------------------------------------
def extract_ela_features(seed: int, 
                         X: np.ndarray,
                           fX: np.ndarray,
                           dim:int,
                           fid: int,
                           inst_id: int) -> pd.DataFrame:
    
    # Instantiate a function object
    problem = get_problem(fid, inst_id, dim)


    ### CLASSICAL ELA FEATURES ###
    # Raw data is X and fX
    ela_meta = calculate_ela_meta(X, fX)
    ela_distr = calculate_ela_distribution(X, fX)
    ela_level = calculate_ela_level(X,fX, ela_level_quantiles=[0.1,0.25,0.5])

    # Require extra problem info and more samples
    #ela_local = calculate_ela_local(X, fX, problem,dim,lower_bound=-5.0, upper_bound=5.0, seed=seed)
    #ela_curvate = calculate_ela_curvate(X,fX,problem,dim,lower_bound=-5.0, upper_bound=5.0, seed=seed)
    #ela_conv = calculate_ela_conv(X, fX,problem)

    ### CELL MAPPING FEATURES ###
    #cm_angle = calculate_cm_angle(X, fX,lower_bound=-5.0, upper_bound=5.0)
    #cm_conv = calculate_cm_conv(X, fX, lower_bound=-5.0, upper_bound=5.0)
    #cm_grad = calculate_cm_grad(X, fX, lower_bound=-5.0, upper_bound=5.0)


    ### LINEAR MODEL FEATURES ###
    #limo = calculate_limo(X, fX, upper_bound=5.0, lower_bound=-5.0)

    ### NEAREST BETTER CLUSTERING ###
    nbc = calculate_nbc(X, fX)


    ### DISPERSION FEATURES ###
    disp = calculate_dispersion(X, fX)

    ### INFORMATION CONTENT FEATURES ###
    ic = calculate_information_content(X, fX, seed=seed)

    ### PCA FEATURES ###
    pca_features = calculate_pca(X, fX)


    # Use the miscellaneous features
    fdc = calculate_fitness_distance_correlation(X, fX, problem.optimum.y, minkowski_p=2.0)
    #grad_features = calculate_gradient_features(problem, dim, lower_bound=-5.0, upper_bound=5.0, seed=seed)
    #hc_features = calculate_hill_climbing_features(problem, dim, lower_bound=-5.0, upper_bound=5.0, seed=seed)
    #ls_features = calculate_length_scales_features(problem, dim, lower_bound=-5.0, upper_bound=5.0, seed=seed)
    #sobol_features = calculate_sobol_indices_features(problem, dim, lower_bound=-5.0, upper_bound=5.0, seed=seed)

    #return pd.DataFrame({**ela_meta, **ela_distr, **ela_level, **nbc, **disp, **ic, **pca_features, **cm_angle, **cm_conv, **cm_grad, **limo}, index=[0])
    #return pd.DataFrame({**ela_meta, **ela_distr, **ela_level, **nbc, **disp, **ic, **pca_features, **limo}, index=[0])
    return pd.DataFrame({**ela_meta, **ela_distr, **ela_level, **nbc, **disp, **ic, **pca_features, **fdc}, index=[0])

# ---------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------
def worker_extract_and_save(args):
    key, x_file, y_file, func_id, inst_id, base_dir = args

    dim, seed, n_samples, obj_type = key

    X = read_x_samples(x_file)
    fX = read_csv(y_file)["fX"].values

    

    out_dir = base_dir / "ela_features_2" / obj_type / f"Dimension_{dim}" / f"seed_{seed}" \
              / f"Samples_{n_samples}" / f"f_{func_id}" / f"id_{inst_id}"

    
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_dir.joinpath("ela_features_2.csv").exists():
        #print(f"Skipping existing: {out_dir}")
        # Open the existing file to check if it's valid
        #df = pd.read_csv(out_dir / "ela_features_2.csv")

        df = extract_ela_features(seed, X, fX, dim, func_id, inst_id)

        save_csv(df, out_dir / "ela_features_2.csv")
        print(f"Saved: {out_dir}")

        
        return True
    else:
        df = extract_ela_features(seed, X, fX, dim, func_id, inst_id)

        save_csv(df, out_dir / "ela_features.csv")
        print(f"Saved: {out_dir}")

        return True


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    base_dir = Path(os.getcwd())

    x_files = get_files(base_dir, "*samples.csv")
    y_files = get_files(base_dir, "*evaluations.csv")

    x_dict = distill_x_sample_list(x_files)
    y_dict = distill_y_sample_list(y_files)

    # build list of tasks for multiprocessing
    tasks = []
    for key, x_file in x_dict.items():
        if key in y_dict:
            if key[0] <40 and key[1]>2000:  # only dimensions up to 40
                for (y_file, func_id, inst_id) in y_dict[key]:
                    tasks.append((key, x_file, y_file, func_id, inst_id, base_dir))

    print(f"Found {len(tasks)} feature extraction tasks.")

    # run multiprocessing
    #n_proc = max(1, cpu_count()//2 - 1)
    n_proc = 6
    print(f"Using {n_proc} processes...")

    if n_proc == 1:
        for task in tasks:
            worker_extract_and_save(task)
    else:
        with Pool(n_proc) as pool:
            pool.map(worker_extract_and_save, tasks)


if __name__ == "__main__":
    main()
