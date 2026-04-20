from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
#FUNCTION_IDS: List[int] = list(range(1, 25))
FUNCTION_IDS: List[int] = list(range(17, 25))
INSTANCE_IDS: List[int] = list(range(15))

DATASET_2000_CONSIDERED_SEEDS = list(range(2001, 2041))
DATASET_200_CONSIDERED_SEEDS = list(range(1001, 1041))

DATA_SIZES = [200, 2000]
REDUCTION_RATIOS = [0.5, 0.25, 0.1]
EPSILON = 1e-9
MODE = 2  # 1 to include PCA features, 2 to exclude them

ROOT_DIRECTORY = Path(__file__).resolve().parent
SAVE_FIGURE_DIRECTORY = ROOT_DIRECTORY / "figures_barplots_slices_comparison_full_with_line"

BASE_EXCLUDED_COLUMNS = {
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

PCA_EXCLUDED_COLUMNS = {
    "pca.expl_var.cor_init",
    "pca.expl_var.cor_x",
    "pca.expl_var.cov_init",
    "pca.expl_var.cov_x",
    "pca.expl_var_PC1.cor_init",
    "pca.expl_var_PC1.cor_x",
    "pca.expl_var_PC1.cov_init",
    "pca.expl_var_PC1.cov_x",
}

EXCLUDED_COLUMNS = BASE_EXCLUDED_COLUMNS | (PCA_EXCLUDED_COLUMNS if MODE == 2 else set())

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
# Data classes
# =============================
@dataclass(frozen=True)
class DatasetKey:
    family: str
    n_samples: int
    ratio: Optional[float] = None


@dataclass(frozen=True)
class DifferenceSpec:
    name: str
    target_key: DatasetKey
    target_group_cols: Tuple[str, ...]
    target_filters: Tuple[Tuple[str, str, int], ...] = ()


# =============================
# File selection
# =============================
def choose_full_dataset_file(data_size: int) -> str:
    mapping = {
        200: "complete_data_2.csv",
        2000: "complete_data_generated.csv"
    }
    try:
        return mapping[data_size]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset size: {data_size}") from exc


REDUCED_FILE_MAP = {
    (200, 0.25): "reduced_1_200_0.25.parquet",
    (200, 0.5): "reduced_1_200_0.5.parquet",
    (2000, 0.25): "reduced_2_2000_0.25.parquet",
    (2000, 0.5): "reduced_2_2000_0.5.parquet",
}

ONE_SHOT_FILE_MAP = {
    (200, 0.25): "reduced_oneshot_3_200_0.25.parquet",
    (200, 0.5): "reduced_oneshot_3_200_0.5.parquet",
    (200, 0.1): "reduced_oneshot_3_200_0.1.parquet",
    (2000, 0.25): "reduced_oneshot_3_2000_0.25.parquet",
    (2000, 0.5): "reduced_oneshot_3_2000_0.5.parquet",
    (2000, 0.1): "reduced_oneshot_3_2000_0.1.parquet",
}


def choose_reduced_feature_file(data_size: int, reduction_ratio: float) -> str:
    try:
        return REDUCED_FILE_MAP[(data_size, reduction_ratio)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported combination: data_size={data_size}, reduction_ratio={reduction_ratio}"
        ) from exc



def choose_reduced_feature_file_one_shot(data_size: int, reduction_ratio: float) -> str:
    try:
        return ONE_SHOT_FILE_MAP[(data_size, reduction_ratio)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported combination: data_size={data_size}, reduction_ratio={reduction_ratio}"
        ) from exc



def choose_slice_file(data_size: int, reduction_ratio: float, all_in: bool = False) -> str:
    if data_size != 200:
        raise FileNotFoundError(
            f"No slice datasets available for data_size={data_size}, reduction_ratio={reduction_ratio}"
        )

    prefix = f"slices_{data_size}_all_in_{reduction_ratio}" if all_in else f"slices_{data_size}_{reduction_ratio}"
    return f"{prefix}.parquet"


# =============================
# Loading and preprocessing
# =============================
@lru_cache(maxsize=None)
def load_dataset_as_pd_df(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")



def considered_seeds(dataset_size: int) -> List[int]:
    if dataset_size == 200:
        return DATASET_200_CONSIDERED_SEEDS
    if dataset_size == 2000:
        return DATASET_2000_CONSIDERED_SEEDS
    raise ValueError(f"Unsupported dataset size: {dataset_size}")



def erase_runtime_columns(df: pd.DataFrame) -> pd.DataFrame:
    runtime_columns = [c for c in df.columns if "runtime" in c.lower()]
    return df.drop(columns=runtime_columns, errors="ignore")



def preprocess_dataframe(
    df: pd.DataFrame,
    dataset_size: int,
    *,
    filter_seeds: bool,
    function_ids: Sequence[int],
    instance_ids: Sequence[int],
) -> pd.DataFrame:
    out = erase_runtime_columns(df)

    mask = out["function_idx"].isin(function_ids) & out["instance_idx"].isin(instance_ids)
    out = out.loc[mask]

    if filter_seeds:
        out = out.loc[out["seed_lhs"].isin(considered_seeds(dataset_size))]

    return out.copy()



def load_all_datasets() -> Dict[DatasetKey, pd.DataFrame]:
    datasets: Dict[DatasetKey, pd.DataFrame] = {}

    for n_samples in DATA_SIZES:
        full_file = choose_full_dataset_file(n_samples)
        datasets[DatasetKey("full", n_samples)] = preprocess_dataframe(
            load_dataset_as_pd_df(full_file),
            n_samples,
            filter_seeds=True,
            function_ids=FUNCTION_IDS,
            instance_ids=INSTANCE_IDS,
        )

        for ratio in REDUCTION_RATIOS:
            if ratio != 0.1:
                reduced_file = choose_reduced_feature_file(n_samples, ratio)
                datasets[DatasetKey("reduced", n_samples, ratio)] = preprocess_dataframe(
                    load_dataset_as_pd_df(reduced_file),
                    n_samples,
                    filter_seeds=True,
                    function_ids=FUNCTION_IDS,
                    instance_ids=INSTANCE_IDS,
                )

            oneshot_file = choose_reduced_feature_file_one_shot(n_samples, ratio)
            datasets[DatasetKey("oneshot", n_samples, ratio)] = preprocess_dataframe(
                load_dataset_as_pd_df(oneshot_file),
                n_samples,
                filter_seeds=True,
                function_ids=FUNCTION_IDS,
                instance_ids=INSTANCE_IDS,
            )

            if n_samples == 200:
                slice_file = choose_slice_file(n_samples, ratio, all_in=False)
                datasets[DatasetKey("slices", n_samples, ratio)] = preprocess_dataframe(
                    load_dataset_as_pd_df(slice_file),
                    n_samples,
                    filter_seeds=False,
                    function_ids=FUNCTION_IDS,
                    instance_ids=INSTANCE_IDS,
                )

                slice_all_in_file = choose_slice_file(n_samples, ratio, all_in=True)
                datasets[DatasetKey("slices_all_in", n_samples, ratio)] = preprocess_dataframe(
                    load_dataset_as_pd_df(slice_all_in_file),
                    n_samples,
                    filter_seeds=False,
                    function_ids=FUNCTION_IDS,
                    instance_ids=INSTANCE_IDS,
                )

    return datasets



def get_feature_columns(df: pd.DataFrame, excluded_columns: Iterable[str]) -> List[str]:
    excluded = set(excluded_columns)
    return [col for col in df.columns if col not in excluded]


# =============================
# Difference engine
# =============================
def apply_filters(df: pd.DataFrame, filters: Sequence[Tuple[str, str, int]]) -> pd.DataFrame:
    if not filters:
        return df

    out = df
    for column, op, value in filters:
        if op == "==":
            out = out.loc[out[column] == value]
        elif op == "!=":
            out = out.loc[out[column] != value]
        else:
            raise ValueError(f"Unsupported filter operator: {op}")
    return out



def aggregate_features(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    feature_cols: Sequence[str],
    agg: str,
) -> pd.DataFrame:
    return df.groupby(list(group_cols), observed=True)[list(feature_cols)].agg(agg).reset_index()



def compute_relative_difference(
    *,
    df_reference: pd.DataFrame,
    df_target: pd.DataFrame,
    reference_group_cols: Sequence[str],
    target_group_cols: Sequence[str],
    merge_on: Sequence[str],
    feature_cols: Sequence[str],
    agg: str = "median",
) -> pd.DataFrame:
    ref_agg = aggregate_features(df_reference, reference_group_cols, feature_cols, agg)
    tgt_agg = aggregate_features(df_target, target_group_cols, feature_cols, agg)

    merged = tgt_agg.merge(
        ref_agg,
        on=list(merge_on),
        suffixes=("_target", "_ref"),
        how="inner",
        copy=False,
    )

    target_cols = [f"{f}_target" for f in feature_cols]
    ref_cols = [f"{f}_ref" for f in feature_cols]
    ratio_cols = [f"ratio_{f}" for f in feature_cols]

    target_values = merged[target_cols].to_numpy(copy=False)
    ref_values = merged[ref_cols].to_numpy(copy=False)
    merged[ratio_cols] = (target_values - ref_values) / (np.abs(ref_values) + EPSILON)

    keep_cols = [c for c in target_group_cols if c in merged.columns] + ratio_cols
    return merged[keep_cols]



def build_difference_tables(
    datasets: Mapping[DatasetKey, pd.DataFrame],
    feature_cols: Sequence[str],
    agg: str = "median",
) -> Dict[str, pd.DataFrame]:
    differences: Dict[str, pd.DataFrame] = {}

    reference_full_2000 = datasets[DatasetKey("full", 2000)]
    full_200 = datasets[DatasetKey("full", 200)]

    differences["full"] = compute_relative_difference(
        df_reference=reference_full_2000,
        df_target=full_200,
        reference_group_cols=("function_idx", "instance_idx"),
        target_group_cols=("function_idx", "instance_idx"),
        merge_on=("function_idx", "instance_idx"),
        feature_cols=feature_cols,
        agg=agg,
    )

    specs = [
        DifferenceSpec(
            name=f"sliced_0_{ratio}",
            target_key=DatasetKey("slices", 200, ratio),
            target_group_cols=("function_idx", "instance_idx", "group_id"),
            target_filters=(("slice_id", "==", 0),),
        )
        for ratio in REDUCTION_RATIOS
    ] + [
        DifferenceSpec(
            name=f"all_in_0_{ratio}",
            target_key=DatasetKey("slices_all_in", 200, ratio),
            target_group_cols=("function_idx", "instance_idx", "group_id"),
            target_filters=(("slice_id", "==", 0),),
        )
        for ratio in REDUCTION_RATIOS
    ] + [
        DifferenceSpec(
            name=f"all_in_gen_{ratio}",
            target_key=DatasetKey("slices_all_in", 200, ratio),
            target_group_cols=("function_idx", "instance_idx", "group_id", "slice_id"),
            target_filters=(("slice_id", "!=", 0),),
        )
        for ratio in REDUCTION_RATIOS
    ]

    for spec in specs:
        target_df = apply_filters(datasets[spec.target_key], spec.target_filters)
        differences[spec.name] = compute_relative_difference(
            df_reference=reference_full_2000,
            df_target=target_df,
            reference_group_cols=("function_idx", "instance_idx"),
            target_group_cols=spec.target_group_cols,
            merge_on=("function_idx", "instance_idx"),
            feature_cols=feature_cols,
            agg=agg,
        )

    return differences



def combine_differences_results(
    named_differences: Mapping[str, pd.DataFrame],
    method_labels: Mapping[str, str],
) -> pd.DataFrame:
    long_frames = []
    for name, df in named_differences.items():
        ratio_cols = [c for c in df.columns if c.startswith("ratio_")]
        id_vars = [c for c in df.columns if not c.startswith("ratio_")]
        label = method_labels.get(name, name)

        melted = df.melt(
            id_vars=id_vars,
            value_vars=ratio_cols,
            var_name="feature",
            value_name="ratio",
        )
        melted["method"] = label
        melted["feature"] = melted["feature"].str.removeprefix("ratio_")
        long_frames.append(melted)

    return pd.concat(long_frames, ignore_index=True)


# =============================
# Plotting
# =============================
def plot_feature_violin(
    df_long: pd.DataFrame,
    feature_name: str,
    function_id: int,
    instance_ids: Sequence[int] = INSTANCE_IDS,
) -> Tuple[plt.Figure, plt.Axes]:
    plot_df = df_long.loc[
        (df_long["function_idx"] == function_id)
        & (df_long["instance_idx"].isin(instance_ids))
        & (df_long["feature"] == feature_name)
    ].copy()

    #fig, ax = plt.subplots(figsize=(4, 6))
    fig, ax = plt.subplots(figsize=(7, 5))

    sns.violinplot(
        data=plot_df,
        x="feature",
        y="ratio",
        hue="method",
        ax=ax,
        cut=0,
    )
    ax.set_title(f"Relative differences – f{function_id}")
    ax.set_ylabel("Relative difference")
    ax.set_xlabel("Feature")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    # Plot the horizontal line at y=0
    ax.axhline(0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)

    # Change the legend position (outside) and title
    legend = ax.legend(title="Method",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0)

    # 🔧 Force font sizes
    for text in legend.get_texts():
        text.set_fontsize(8)

    legend.get_title().set_fontsize(10)

    return fig, ax



def save_all_feature_plots(
    df_long: pd.DataFrame,
    feature_names: Sequence[str],
    function_ids: Sequence[int],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for function_id in function_ids:
        for feature_name in feature_names:
            fig, _ = plot_feature_violin(df_long, feature_name, function_id)

            figure_path = output_dir / f"function_id_{function_id}" / f"feature_{feature_name}"
            figure_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(figure_path / "violin_plot_comparison_all_variants.pdf", 
                        dpi=300, 
                        bbox_inches="tight")
            plt.close(fig)


# =============================
# Main pipeline
# =============================
def main() -> None:
    datasets = load_all_datasets()

    reference_df = datasets[DatasetKey("reduced", 
                                       200, 
                                       0.5)]
    
    feature_cols = get_feature_columns(reference_df, EXCLUDED_COLUMNS)

    difference_tables = build_difference_tables(datasets, feature_cols, agg="median")
    differences_long = combine_differences_results(difference_tables, PLOT_METHOD_LABELS)

    save_all_feature_plots(
        differences_long,
        feature_names=feature_cols,
        function_ids=FUNCTION_IDS,
        output_dir=SAVE_FIGURE_DIRECTORY,
    )


if __name__ == "__main__":
    main()
