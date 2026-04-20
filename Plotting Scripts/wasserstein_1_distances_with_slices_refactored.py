from __future__ import annotations

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import wasserstein_distance

from dataclasses import dataclass


# =============================
# Plot configuration
# =============================
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)



# =============================
# Constants
# =============================
FUNCTION_IDS = list(range(1, 25))
INSTANCE_IDS = list(range(15))

DATASET_2000_CONSIDERED_SEEDS = list(range(2001, 2041))
DATASET_200_CONSIDERED_SEEDS = list(range(1001, 1041))

DATA_SIZES = [200, 2000]
REDUCTION_RATIOS = [0.5, 0.25, 0.1]

EPSILON = 1e-9

ROOT_DIRECTORY = Path(__file__).resolve().parent
SAVE_FIGURE_DIRECTORY = ROOT_DIRECTORY / "figures_wasserstein_refactored"

EXCLUDED_COLUMNS = {
    "embedding_seed",
    "round",
    "reduction_ratio",
    "instance_idx",
    "function_idx",
    "n_samples",
    "seed_lhs",
    "dimension",
    "group_id",
    "slice_id",
}

PLOT_METHOD_LABELS = {
    "full": r"Full/ELA$_{\mathrm{A}}$",
    "sliced_0_0.5": r"Sliced/ELA$_{\mathrm{A}},r=0.5$",
    "sliced_0_0.25": r"Sliced/ELA$_{\mathrm{A}},r=0.25$",
    "sliced_0_0.1": r"Sliced/ELA$_{\mathrm{A}},r=0.1$",
    "all_in_0_0.5": r"All\_in/ELA$_{\mathrm{A}},r=0.5$",
    "all_in_gen_0.5": r"All\_in/ELA$_{\mathrm{R}},r=0.5$",
    "all_in_0_0.25": r"All\_in/ELA$_{\mathrm{A}},r=0.25$",
    "all_in_gen_0.25": r"All\_in/ELA$_{\mathrm{R}},r=0.25$",
    "all_in_0_0.1": r"All\_in/ELA$_{\mathrm{A}},r=0.1$",
    "all_in_gen_0.1": r"All\_in/ELA$_{\mathrm{R}},r=0.1$",
}

# =============================
# File selection
# =============================
def choose_full_dataset_file(data_size: int) -> str:
    return {
        200: "complete_data_2.csv",
        2000: "complete_data_generated.csv",
    }[data_size]


def choose_reduced_feature_file(data_size: int, reduction_ratio: float) -> str:
    mapping = {
        (200, 0.25): "reduced_1_200_0.25.parquet",
        (200, 0.5): "reduced_1_200_0.5.parquet",
        (2000, 0.25): "reduced_2_2000_0.25.parquet",
        (2000, 0.5): "reduced_2_2000_0.5.parquet",
    }
    return mapping[(data_size, reduction_ratio)]


def choose_reduced_feature_file_one_shot(data_size: int, reduction_ratio: float) -> str:
    mapping = {
        (200, 0.25): "reduced_oneshot_3_200_0.25.parquet",
        (200, 0.5): "reduced_oneshot_3_200_0.5.parquet",
        (200, 0.1): "reduced_oneshot_3_200_0.1.parquet",
        (2000, 0.25): "reduced_oneshot_3_2000_0.25.parquet",
        (2000, 0.5): "reduced_oneshot_3_2000_0.5.parquet",
        (2000, 0.1): "reduced_oneshot_3_2000_0.1.parquet",
    }
    return mapping[(data_size, reduction_ratio)]


def choose_slice_file(data_size: int, reduction_ratio: float, all_in: bool = False) -> str:
    prefix = (
        f"slices_{data_size}_all_in_{reduction_ratio}"
        if all_in
        else f"slices_{data_size}_{reduction_ratio}"
    )
    return f"{prefix}.parquet"


# =============================
# Loading
# =============================
def load_dataset(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError("Unsupported file")


def preprocess(df: pd.DataFrame, dataset_size: int, filter_seeds=True) -> pd.DataFrame:
    df = df.copy()

    if filter_seeds:
        seeds = (
            DATASET_200_CONSIDERED_SEEDS
            if dataset_size == 200
            else DATASET_2000_CONSIDERED_SEEDS
        )
        df = df[df["seed_lhs"].isin(seeds)]

    df = df[df["function_idx"].isin(FUNCTION_IDS)]
    df = df[df["instance_idx"].isin(INSTANCE_IDS)]

    runtime_cols = [c for c in df.columns if "runtime" in c.lower()]
    df = df.drop(columns=runtime_cols, errors="ignore")

    return df


def load_all_datasets():
    datasets = {}

    for n in DATA_SIZES:
        datasets[("full", n, None)] = preprocess(
            load_dataset(choose_full_dataset_file(n)), n
        )

        for r in REDUCTION_RATIOS:
            if r != 0.1:
                datasets[("reduced", n, r)] = preprocess(
                    load_dataset(choose_reduced_feature_file(n, r)), n
                )

            datasets[("oneshot", n, r)] = preprocess(
                load_dataset(choose_reduced_feature_file_one_shot(n, r)), n
            )

            if n == 200:
                datasets[("slices", n, r)] = preprocess(
                    load_dataset(choose_slice_file(n, r)), n, filter_seeds=False
                )

                datasets[("slices_all_in", n, r)] = preprocess(
                    load_dataset(choose_slice_file(n, r, all_in=True)),
                    n,
                    filter_seeds=False,
                )

    return datasets


# =============================
# Wasserstein computations
# =============================
def compute_wasserstein(df1, df2, features):
    rows = []

    for f in features:
        for fid in FUNCTION_IDS:
            for iid in INSTANCE_IDS:
                x = df1[(df1.function_idx == fid) & (df1.instance_idx == iid)][f]
                y = df2[(df2.function_idx == fid) & (df2.instance_idx == iid)][f]

                d = wasserstein_distance(x, y)

                rows.append((fid, iid, f, d))

    return pd.DataFrame(
        rows, columns=["function_idx", "instance_idx", "feature", "wasserstein"]
    )


def compute_wasserstein_slices(ref, slices, features):
    df0 = slices[slices.slice_id == 0]
    dfgen = slices[slices.slice_id != 0]

    return (
        compute_wasserstein(ref, df0, features),
        compute_wasserstein(ref, dfgen, features),
    )


# =============================
# Build results (like shifts)
# =============================
def build_wasserstein_tables(datasets, features):
    results = {}
    ref = datasets[("full", 2000, None)]

    # full
    results["full"] = compute_wasserstein(
        ref, datasets[("full", 200, None)], features
    )

    # reduced
    for r in REDUCTION_RATIOS:
        results[f"reduced_{r}"] = compute_wasserstein(
            ref, datasets[("oneshot", 200, r)], features
        )

    # slices
    for r in REDUCTION_RATIOS:
        d0, dgen = compute_wasserstein_slices(
            ref, datasets[("slices", 200, r)], features
        )
        results[f"sliced_0_{r}"] = d0
        results[f"sliced_gen_{r}"] = dgen

    # all-in
    for r in REDUCTION_RATIOS:
        d0, dgen = compute_wasserstein_slices(
            ref, datasets[("slices_all_in", 200, r)], features
        )
        results[f"all_in_0_{r}"] = d0
        results[f"all_in_gen_{r}"] = dgen

    return results

@dataclass(frozen=True)
class WassersteinSpec:
    name: str
    dataset_key: tuple
    slice_filter: str | None = None  # None, "slice_0", "slice_gen"


def build_wasserstein_tables_subset(datasets, features):
    results = {}
    ref = datasets[("full", 2000, None)]

    specs = [
        # === baseline
        WassersteinSpec("full", ("full", 200, None)),

        # === reduced (example: only r=0.5)
        #WassersteinSpec("reduced_0.5", ("oneshot", 200, 0.5)),

        # === slices (ONLY r=0.5)
        WassersteinSpec("sliced_0_0.5", ("slices", 200, 0.5), "slice_0"),
        WassersteinSpec("sliced_0_0.25", ("slices", 200, 0.25), "slice_0"),
        WassersteinSpec("sliced_0_0.1", ("slices", 200, 0.1), "slice_0"),
        #WassersteinSpec("sliced_gen_0.5", ("slices", 200, 0.5), "slice_gen"),

        # === all-in slices
        WassersteinSpec("all_in_0_0.5", ("slices_all_in", 200, 0.5), "slice_0"),
        WassersteinSpec("all_in_0_0.25", ("slices_all_in", 200, 0.25), "slice_0"),
        WassersteinSpec("all_in_0_0.1", ("slices_all_in", 200, 0.1), "slice_0"),

        # All in gen slices 
        WassersteinSpec("all_in_gen_0.5", ("slices_all_in", 200, 0.5), "slice_gen"),
        WassersteinSpec("all_in_gen_0.25", ("slices_all_in", 200, 0.25), "slice_gen"),
        WassersteinSpec("all_in_gen_0.1", ("slices_all_in", 200, 0.1), "slice_gen")

        #WassersteinSpec("all_in_gen_0.5", ("slices_all_in", 200, 0.5), "slice_gen"),
    ]

    for spec in specs:
        df = datasets[spec.dataset_key]

        if spec.slice_filter == "slice_0":
            df = df[df["slice_id"] == 0]

        elif spec.slice_filter == "slice_gen":
            df = df[df["slice_id"] != 0]

        results[spec.name] = compute_wasserstein(ref, df, features)

    return results


def combine_results(results, method_labels):
    frames = []

    for name, df in results.items():
        tmp = df.copy()

        tmp["method"] = method_labels.get(name, name)

        frames.append(tmp)

    return pd.concat(frames, ignore_index=True)


# =============================
# Plotting
# =============================
def plot_violin(df, feature, function_id):
    sub = df[
        (df.feature == feature) & (df.function_idx == function_id)
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.violinplot(
        data=sub,
        x="feature",
        y="wasserstein",
        hue="method",
        ax=ax,
        cut=0,
    )

    ax.set_title(f"Wasserstein – f{function_id}")
    ax.set_ylabel("Distance")
    ax.set_xlabel("")

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    return fig


# =============================
# Main
# =============================
def main():
    datasets = load_all_datasets()

    ref_df = datasets[("reduced", 200, 0.5)]
    features = [c for c in ref_df.columns if c not in EXCLUDED_COLUMNS]

    results = build_wasserstein_tables_subset(datasets, features)
    df_long = combine_results(results, PLOT_METHOD_LABELS)

    for fid in FUNCTION_IDS:
        for feat in features:
            fig = plot_violin(df_long, feat, fid)

            path = (
                SAVE_FIGURE_DIRECTORY
                / f"function_id_{fid}"
                / f"feature_{feat}"
            )
            path.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                path / "violin_wasserstein.pdf",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


if __name__ == "__main__":
    main()