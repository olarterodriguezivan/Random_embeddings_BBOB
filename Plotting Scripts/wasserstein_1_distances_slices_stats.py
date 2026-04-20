from matplotlib import colors
import pandas as pd
import seaborn as sns
import numpy as np
import os, sys
from pathlib import Path
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from typing import Dict, Tuple, List

# Import the Wasserstein distance and wilcoxon computation functions
from scipy.stats import wasserstein_distance, wilcoxon
from scipy.stats import friedmanchisquare

# Add the Nemenyi post-hoc test function
import scikit_posthocs as sp




## =============================
## GECCO Conference Settings for plots
## =============================

# Set font to be consistent before importing pyplot
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



# Import pyplot
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.rcfonts": False,
})

## =============================
## CONSTANT CONFIGURATION
## =============================

#FUNCTION_IDS:list = [1, 8, 11, 16, 20]  # Function IDs to consider
FUNCTION_IDS:list = [*range(1, 25)]  # Function IDs to consider
#FUNCTION_IDS:list = [20]  # Function IDs to consider
INSTANCE_IDS:list = [*range(15)]  # Instance IDs to consider

DATASET_2000_CONSIDERED_SEEDS = [*range(2001,2041)] # Seeds to consider for DATASET_SIZE = 2000
DATASET_200_CONSIDERED_SEEDS = [*range(1001,1041)] # Seeds to consider for DATASET_SIZE = 200

EPSILON = 1e-9 # To avoid log(0) or ./0 issues

ROOT_DIRECTORY = Path(__file__).resolve().parent
SAVE_FIGURE_DIRECTORY = ROOT_DIRECTORY.joinpath("tables_wasserstein_2_distances_slices_stats")

if not SAVE_FIGURE_DIRECTORY.exists():
    SAVE_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    

DATA_SIZES = [200, 2000]
REDUCTION_RATIOS = [0.5, 0.25, 0.1]

MODE=2 # 1 to include PCA features, 2 to exclude them


if MODE == 1:
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
elif MODE == 2:
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
    # PCA features
    "pca.expl_var.cor_init",
    "pca.expl_var.cor_x", 
    "pca.expl_var.cov_init", 
    "pca.expl_var.cov_x", 
    "pca.expl_var_PC1.cor_init", 
    "pca.expl_var_PC1.cor_x",
    "pca.expl_var_PC1.cov_init",
    "pca.expl_var_PC1.cov_x",
    }

    P_ADJUST = "none"


#/% First import the datasets (just keep in mind to make the labelling consistent)
# ELA Features on Full-Dataset without Projections


def choose_full_dataset_file(data_size:int) -> str:
    r"""
    This function chooses the appropriate full dataset file based on the data size.

    Args
    --------------
    data_size (int): The size of the dataset (e.g., 200 or 2000).

    Returns
    --------------
    str: The filename of the full dataset corresponding to the given data size.

    """
    if data_size == 200:
        return "complete_data_2.csv"
    elif data_size == 2000:
        return "complete_data_generated.csv"
    else:
        raise ValueError("Unsupported DATASET_SIZE")

def choose_reduced_feature_file(data_size:int, reduction_ratio:float) -> str:
    r"""
    This function chooses the appropriate reduced feature dataset file based on the data size and reduction ratio.

    Args
    --------------
    data_size (int): The size of the dataset (e.g., 200 or 2000).
    reduction_ratio (float): The reduction ratio (e.g., 0.25 or 0.5).

    Returns
    --------------
    str: The filename of the reduced feature dataset corresponding to the given data size and reduction ratio.

    """
    if data_size == 200 and reduction_ratio == 0.25:
        return "reduced_1_200_0.25.parquet"
    elif data_size == 200 and reduction_ratio == 0.5:
        return "reduced_1_200_0.5.parquet"
    elif data_size == 2000 and reduction_ratio == 0.25:
        return "reduced_2_2000_0.25.parquet"
    elif data_size == 2000 and reduction_ratio == 0.5:
        return "reduced_2_2000_0.5.parquet"
    else:
        raise ValueError("Unsupported combination of DATASET_SIZE and REDUCTION_RATIO")
    
def choose_reduced_feature_file_one_shot(data_size:int, reduction_ratio:float) -> str:
    r"""
    This function chooses the appropriate reduced feature dataset file for one-shot reduction based on the data size and reduction ratio.   

    Args
    --------------
    data_size (int): The size of the dataset (e.g., 200 or 2000).
    reduction_ratio (float): The reduction ratio (e.g., 0.25 or 0.5).
    
    Returns
    --------------
    str: The filename of the reduced feature dataset corresponding to the given data size and reduction ratio.
    
    """

    if data_size == 200 and reduction_ratio == 0.25:
        return "reduced_oneshot_3_200_0.25.parquet"
    elif data_size == 200 and reduction_ratio == 0.5:
        return "reduced_oneshot_3_200_0.5.parquet"
    elif data_size == 200 and reduction_ratio == 0.1:
        return "reduced_oneshot_3_200_0.1.parquet"
    elif data_size == 2000 and reduction_ratio == 0.25:
        return "reduced_oneshot_3_2000_0.25.parquet"
    elif data_size == 2000 and reduction_ratio == 0.5:
        return "reduced_oneshot_3_2000_0.5.parquet"
    elif data_size == 2000 and reduction_ratio == 0.1:
        return "reduced_oneshot_3_2000_0.1.parquet"
    else:
        raise ValueError("Unsupported combination of DATASET_SIZE and REDUCTION_RATIO")

def choose_reduced_feature_file_slice(data_size:int, reduction_ratio:float, slice_id:int) -> str:
    r"""
    This function chooses the appropriate reduced feature dataset file based on the data size, reduction ratio, and slice ID.

    Args
    --------------
    data_size (int): The size of the dataset (e.g., 200 or 2000).
    reduction_ratio (float): The reduction ratio (e.g., 0.25 or 0.5).
    slice_id (int): The slice ID.

    Returns
    --------------
    str: The filename of the reduced feature dataset corresponding to the given data size, reduction ratio, and slice ID.

    """
    if data_size == 200 and reduction_ratio == 0.25:
        return f"slices_{data_size}_{reduction_ratio}.parquet"
    elif data_size == 200 and reduction_ratio == 0.5:
        return f"slices_{data_size}_{reduction_ratio}.parquet"
    elif data_size == 200 and reduction_ratio == 0.1:
        return f"slices_{data_size}_{reduction_ratio}.parquet"
    elif data_size == 2000 and reduction_ratio == 0.25:
        raise FileNotFoundError("No slice datasets for DATASET_SIZE = 2000 and REDUCTION_RATIO = 0.25")
    elif data_size == 2000 and reduction_ratio == 0.5:
        raise FileNotFoundError("No slice datasets for DATASET_SIZE = 2000 and REDUCTION_RATIO = 0.5")
    elif data_size == 2000 and reduction_ratio == 0.1:
        raise FileNotFoundError("No slice datasets for DATASET_SIZE = 2000 and REDUCTION_RATIO = 0.1")
    else:
        raise ValueError("Unsupported combination of DATASET_SIZE and REDUCTION_RATIO")

def choose_reduced_feature_file_slice_all_in(data_size:int, reduction_ratio:float, slice_id:int) -> str:
    r"""
    This function chooses the appropriate reduced feature dataset file based on the data size, reduction ratio, and slice ID,
    which are the all in versions of the slice datasets (i.e. they have one slice/group pairing so all the samples lie 
    in the same slice).

    Args
    --------------
    data_size (int): The size of the dataset (e.g., 200 or 2000).
    reduction_ratio (float): The reduction ratio (e.g., 0.25 or 0.5).
    slice_id (int): The slice ID.

    Returns
    --------------
    str: The filename of the reduced feature dataset corresponding to the given data size, reduction ratio, and slice ID.

    """
    if data_size == 200 and reduction_ratio == 0.25:
        return f"slices_{data_size}_all_in_{reduction_ratio}.parquet"
    elif data_size == 200 and reduction_ratio == 0.5:
        return f"slices_{data_size}_all_in_{reduction_ratio}.parquet"
    elif data_size == 200 and reduction_ratio == 0.1:
        return f"slices_{data_size}_all_in_{reduction_ratio}.parquet"
    elif data_size == 2000 and reduction_ratio == 0.25:
        raise FileNotFoundError("No slice datasets for DATASET_SIZE = 2000 and REDUCTION_RATIO = 0.25")
    elif data_size == 2000 and reduction_ratio == 0.5:
        raise FileNotFoundError("No slice datasets for DATASET_SIZE = 2000 and REDUCTION_RATIO = 0.5")
    elif data_size == 2000 and reduction_ratio == 0.1:
        raise FileNotFoundError("No slice datasets for DATASET_SIZE = 2000 and REDUCTION_RATIO = 0.1")
    else:
        raise ValueError("Unsupported combination of DATASET_SIZE and REDUCTION_RATIO")

def load_dataset_as_pd_df(file_path:str) -> pd.DataFrame:
    r"""
    Load a dataset from a given file path into a pandas DataFrame.

    Args
    --------------
    file_path (str): The path to the dataset file.
    
    Returns
    --------------
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """

    # Check first the file exists
    if Path(file_path).exists is False:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .parquet or .csv file.")
    
    return df

def filter_considered_seeds(df:pd.DataFrame, dataset_size:int) -> pd.DataFrame:
    r"""
    Filter the DataFrame to include only the considered seeds based on the DATASET_SIZE.

    Args
    --------------
    df (pd.DataFrame): The input DataFrame containing a 'seed_lhs' column.
    dataset_size (int): The size of the dataset (e.g., 200 or 2000).
    
    Returns
    --------------
    pd.DataFrame: The filtered DataFrame containing only the considered seeds.
    """

    if dataset_size == 200:
        considered_seeds = DATASET_200_CONSIDERED_SEEDS
    elif dataset_size == 2000:
        considered_seeds = DATASET_2000_CONSIDERED_SEEDS
    else:
        raise ValueError("Unsupported DATASET_SIZE")

    filtered_df = df[df['seed_lhs'].isin(considered_seeds)].copy()
    return filtered_df

def erase_runtime_columns(df:pd.DataFrame) -> pd.DataFrame:
    r"""
    Erase runtime-related columns from the DataFrame.

    Args
    --------------
    df (pd.DataFrame): The input DataFrame.
    
    Returns
    --------------
    pd.DataFrame: The DataFrame with runtime-related columns removed.
    """

    runtime_columns = [col for col in df.columns if 'runtime' in col.lower()]
    df_cleaned = df.drop(columns=runtime_columns)
    return df_cleaned

def select_only_required_function_ids(df:pd.DataFrame) -> pd.DataFrame:
    r"""
    Select only the rows corresponding to the required function IDs.

    Args
    --------------
    df (pd.DataFrame): The input DataFrame containing a 'function_idx' column.
    
    Returns
    --------------
    pd.DataFrame: The DataFrame filtered to include only the required function IDs.
    """

    filtered_df = df[df['function_idx'].isin(FUNCTION_IDS)].copy()
    return filtered_df


def process_dataframe(df:pd.DataFrame, dataset_size:int) -> pd.DataFrame:
    r"""
    Process the DataFrame by filtering considered seeds and erasing runtime columns.

    Args
    --------------
    df (pd.DataFrame): The input DataFrame.
    dataset_size (int): The size of the dataset (e.g., 200 or 2000).
    
    Returns
    --------------
    pd.DataFrame: The processed DataFrame.
    """

    df_filtered = filter_considered_seeds(df, dataset_size)
    df_processed = erase_runtime_columns(df_filtered)
    df_processed = select_only_required_function_ids(df_processed)
    return df_processed

def process_slice_dataframe(df:pd.DataFrame, dataset_size:int) -> pd.DataFrame:
    r"""
    Process the slice DataFrame by filtering considered seeds and erasing runtime columns.

    Args
    --------------
    df (pd.DataFrame): The input DataFrame.
    dataset_size (int): The size of the dataset (e.g., 200 or 2000).
    
    Returns
    --------------
    pd.DataFrame: The processed DataFrame.
    """

    df_filtered = df.copy() # Slices do not need seed filtering
    df_processed = erase_runtime_columns(df_filtered)
    df_processed = select_only_required_function_ids(df_processed)
    return df_processed


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------


def load_all_datasets():
    datasets = {}


    for n_samples in DATA_SIZES:
    # Full dataset
        full_file = choose_full_dataset_file(n_samples)
        datasets[("full", n_samples, None)] = process_dataframe(
        load_dataset_as_pd_df(full_file), n_samples
        )


        # Reduced datasets
        for ratio in REDUCTION_RATIOS:
            
            if not ratio==0.1:
                reduced_file = choose_reduced_feature_file(n_samples, ratio)
                datasets[("reduced", n_samples, ratio)] = process_dataframe(
                load_dataset_as_pd_df(reduced_file), n_samples
                )


            oneshot_file = choose_reduced_feature_file_one_shot(n_samples, ratio)
            datasets[("oneshot", n_samples, ratio)] = process_dataframe(
            load_dataset_as_pd_df(oneshot_file), n_samples
            )
    
            # Slice datasets (only for n_samples=200)
            if n_samples == 200:
                slice_file = choose_reduced_feature_file_slice(n_samples, ratio, slice_id=0)
                datasets[("slices", n_samples, ratio)] = process_slice_dataframe(
                load_dataset_as_pd_df(slice_file), n_samples
                )
            
            if n_samples == 200:
                slice_all_in_file = choose_reduced_feature_file_slice_all_in(n_samples, ratio, slice_id=0)
                datasets[("slices_all_in", n_samples, ratio)] = process_slice_dataframe(
                load_dataset_as_pd_df(slice_all_in_file), n_samples
                )


    return datasets


# -----------------------------------------------------------------------------
# Wasserstein distance computations (per instance, per function, per feature)
# -----------------------------------------------------------------------------

def compute_wasserstein_distance(
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame,
    feature_name_list,
    function_id_list,
    instance_id_list
):

    results = []

    # Pre-group datasets
    g1 = dataset1.groupby(['function_idx', 'instance_idx'])
    g2 = dataset2.groupby(['function_idx', 'instance_idx'])

    for function_id in function_id_list:
        for instance_id in instance_id_list:

            key = (function_id, instance_id)

            if key not in g1.groups or key not in g2.groups:
                continue

            df1 = g1.get_group(key)
            df2 = g2.get_group(key)

            for feature_name in feature_name_list:

                data1 = df1[feature_name].values
                data2 = df2[feature_name].values

                distance = wasserstein_distance(data1, data2)

                results.append({
                    'function_id': function_id,
                    'instance_id': instance_id,
                    'feature_name': feature_name,
                    'wasserstein_distance': distance
                })

    return pd.DataFrame(results)

def compute_wasserstein_distance_slices(ref_dataset:pd.DataFrame,
                                        slice_dataset:pd.DataFrame,
                                        feature_name_list:List[str],
                                        function_id_list:List[int],
                                        instance_id_list:List[int]) -> Tuple[pd.DataFrame]:
    r"""
    Compute the Wasserstein distance for each slice of the slice dataset compared to the reference dataset.
    
    Args
    --------------
        ref_dataset (pd.DataFrame): The reference dataset to compare against.
        slice_dataset (pd.DataFrame): The dataset containing the slices to compare.
        feature_name_list (List[str]): The list of feature names to compute the Wasserstein distance for.
        function_id_list (List[int]): The list of function IDs to compute the Wasserstein distance for.
        instance_id_list (List[int]): The list of instance IDs to compute the Wasserstein distance for.

    Returns
    --------------
        Tuple[pd.DataFrame]: A tuple of DataFrames, each containing the computed Wasserstein distances for a general and combined slice ID.
    """

    assert all(feature_name in ref_dataset.columns for feature_name in feature_name_list), f"Features {feature_name_list} not found in ref_dataset"
    assert all(feature_name in slice_dataset.columns for feature_name in feature_name_list), f"Features {feature_name_list} not found in slice_dataset"
    assert 'function_idx' in ref_dataset.columns, "Column 'function_idx' not found in ref_dataset"
    assert 'function_idx' in slice_dataset.columns, "Column 'function_idx' not found in slice_dataset"
    assert 'instance_idx' in ref_dataset.columns, "Column 'instance_idx' not found in ref_dataset"
    assert 'instance_idx' in slice_dataset.columns, "Column 'instance_idx' not found in slice_dataset"
    assert 'slice_id' in slice_dataset.columns, "Column 'slice_id' not found in slice_dataset"

    # Filter the reference dataset for the specific function IDs and instance IDs
    ref_data = ref_dataset[(ref_dataset['function_idx'].isin(function_id_list)) & (ref_dataset['instance_idx'].isin(instance_id_list))]

    # Filter the slice dataset for the specific function IDs and instance IDs
    slice_data = slice_dataset[(slice_dataset['function_idx'].isin(function_id_list)) & (slice_dataset['instance_idx'].isin(instance_id_list))]

    # Extract the slice 0 and the others
    slice_0_data = slice_data[slice_data['slice_id'] == 0]
    other_slices_data = slice_data[slice_data['slice_id'] != 0]

    # Compute the Wasserstein distance for slice 0
    df_wasserstein_slice_0 = compute_wasserstein_distance(ref_data, slice_0_data, feature_name_list, function_id_list, instance_id_list)

    # Compute the Wasserstein distance for the other slices combined
    df_wasserstein_other_slices = compute_wasserstein_distance(ref_data, other_slices_data, feature_name_list, function_id_list, instance_id_list)  

    return df_wasserstein_slice_0, df_wasserstein_other_slices



def heatmap_wasserstein_rankings_2(combined_df: pd.DataFrame, 
                                   function_id_list:List[int],
                                   feature_name_list:List[str],
                                   agg:str="median",
                                   significance_df: pd.DataFrame = None) -> Tuple[plt.Figure, plt.Axes]:
    r"""
    A simplified version of the heatmap_wasserstein_rankings function that takes a single combined DataFrame as input.
    
    Args
    --------------
        combined_df (pd.DataFrame): A DataFrame containing columns 'function_id', 'feature_name', 'method', and 'wasserstein_distance'.
        feature_name_list (List[str]): The list of feature names to include in the heatmap.
        function_id_list (List[int]): The list of function IDs to include in the heatmap.
        agg (str): The aggregation method to use when averaging Wasserstein distances across instances. 
                   Must be either 'mean' or 'median'. Default is 'median'.
        significance_df (pd.DataFrame, optional): A DataFrame containing significance information for the heatmap.
    Returns
    --------------
        Tuple[plt.Figure, plt.Axes]: The matplotlib Figure and Axes objects containing the plot.
    """

    dataset_order = combined_df["method"].unique().tolist()

    # 16 unique markers
    markers = [
        "o", "s", "D", "^", "v", "<", ">", "P",
        "X", "*", "h", "H", "8", "p", "d", "|"
    ]

    # distinct colors
    # tab20 or tab10
    dataset_colors = plt.get_cmap("tab10").colors[:len(dataset_order)]

    dataset_to_marker = {
        ds: markers[i] for i, ds in enumerate(dataset_order)
    }

    dataset_to_code = {
        ds: i for i, ds in enumerate(dataset_order)
    }

    df_all = combined_df[
        combined_df["feature_name"].isin(feature_name_list)
        & combined_df["function_id"].isin(function_id_list)
    ]

    # ---------------------------------------------------------
    # 2) Aggregate
    # ---------------------------------------------------------

    df_agg = (
        df_all
        .groupby(["function_id", "feature_name", "method"])["wasserstein_distance"]
        .agg(agg)
        .reset_index()
    )

    # ---------------------------------------------------------
    # 3) Rank (LOWER is better)
    # ---------------------------------------------------------

    df_ranked = (
        df_agg
        .sort_values(["function_id", "feature_name", "wasserstein_distance"])
        .groupby(["function_id", "feature_name"], group_keys=False)
        .apply(lambda x: x.assign(rank=np.arange(1, len(x) + 1)))
    )

    df_winner = df_ranked[df_ranked["rank"] == 1][
        ["function_id", "feature_name", "method"]
    ].rename(columns={"method": "winner"})

    df_second = df_ranked[df_ranked["rank"] == 2][
        ["function_id", "feature_name", "method"]
    ].rename(columns={"method": "second"})

    df_plot = df_winner.merge(
        df_second,
        on=["function_id", "feature_name"],
        how="left",
    )

    df_plot["winner_code"] = df_plot["winner"].map(dataset_to_code)
    df_plot["second_marker"] = df_plot["second"].map(dataset_to_marker)

    if significance_df is not None:
        sig_lookup = {
            (r.function_id, r.feature_name): r.significant
            for _, r in significance_df.iterrows()
        }
    else:
        sig_lookup = {}

    # ---------------------------------------------------------
    # 4) Pivot
    # ---------------------------------------------------------

    heatmap = df_plot.pivot(
        index="function_id",
        columns="feature_name",
        values="winner_code",
    )

    heatmap = heatmap.reindex(index=function_id_list)
    heatmap = heatmap[feature_name_list]

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    cmap = ListedColormap(dataset_colors)
    norm = BoundaryNorm(
        np.arange(-0.5, len(dataset_colors) + 0.5, 1),
        cmap.N,
    )

    fig, ax = plt.subplots(
        figsize=(0.3 * heatmap.shape[1], 0.5 * heatmap.shape[0])
    )

    sns.heatmap(
        heatmap,
        cmap=cmap,
        norm=norm,
        linewidths=0.3,
        cbar=False,
        ax=ax,
    )

    # ---------------------------------------------------------
    # Overlay runner-up markers
    # ---------------------------------------------------------

    feature_to_x = {f: i for i, f in enumerate(heatmap.columns)}
    function_to_y = {f: i for i, f in enumerate(heatmap.index)}

    for _, r in df_plot.iterrows():

        if pd.isna(r["second_marker"]):
            continue

        is_significant = False
        if significance_df is not None:
            key = (r["function_id"], r["feature_name"])
            is_significant = key in sig_lookup and sig_lookup[key]

        if r["feature_name"] not in feature_to_x:
            continue

        if r["function_id"] not in function_to_y:
            continue

        ax.scatter(
                    feature_to_x[r["feature_name"]] + 0.5,
                    function_to_y[r["function_id"]] + 0.5,
                    marker=r["second_marker"],
                    s=80 if not is_significant else 60,              # slightly bigger
                    facecolors="black" if not is_significant else "none",  # filled if significant
                    edgecolors="black",
                    linewidths=2 if not is_significant else 1,      # thicker edge = bold
                    zorder=10,
                )

    ax.set_xlabel("Feature")
    ax.set_ylabel("Function")

    # ---------------------------------------------------------
    # Legends
    # ---------------------------------------------------------

    color_legend = ax.legend(
        handles=[
            Patch(color=dataset_colors[i], label=dataset_order[i])
            for i in range(len(dataset_order))
        ],
        title="Best (lowest Wasserstein)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.add_artist(color_legend)

    ax.legend(
        handles=[
            Line2D(
                [0], [0],
                marker=m,
                color="black",
                linestyle="None",
                markerfacecolor="none",
                markersize=8,
                label=ds,
            )
            for ds, m in dataset_to_marker.items()
        ],
        title="Second-best",
        bbox_to_anchor=(1.02, 0.1),
        loc="upper left",
    )

    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    return fig, ax


def heatmap_wasserstein_rankings_3(
    combined_df: pd.DataFrame, 
    function_id_list: List[int],
    feature_name_list: List[str],
    agg: str = "median",
) -> Tuple[plt.Figure, plt.Axes]:

    # ---------------------------------------------------------
    # Extract reduction ratio r from method
    # ---------------------------------------------------------
    df_all = combined_df.copy()
    df_all["r"] = (
        df_all["method"]
        .str.extract(r"r=([0-9.]+)")
        .astype(float)
    )

    # Optional: assign value to "Full" (no r)
    df_all["r"] = df_all["r"].fillna(1.0)

    dataset_order = df_all["method"].unique().tolist()

    # markers for second-best
    markers = [
        "o", "s", "D", "^", "v", "<", ">", "P",
        "X", "*", "h", "H", "8", "p", "d", "|"
    ]

    dataset_to_marker = {
        ds: markers[i] for i, ds in enumerate(dataset_order)
    }

    # ---------------------------------------------------------
    # Filter
    # ---------------------------------------------------------
    df_all = df_all[
        df_all["feature_name"].isin(feature_name_list)
        & df_all["function_id"].isin(function_id_list)
    ]

    # ---------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------
    df_agg = (
        df_all
        .groupby(["function_id", "feature_name", "method"])["wasserstein_distance"]
        .agg(agg)
        .reset_index()
    )

    # ---------------------------------------------------------
    # Rank (LOWER is better)
    # ---------------------------------------------------------
    df_ranked = (
        df_agg
        .sort_values(["function_id", "feature_name", "wasserstein_distance"])
        .groupby(["function_id", "feature_name"], group_keys=False)
        .apply(lambda x: x.assign(rank=np.arange(1, len(x) + 1)))
    )

    df_winner = df_ranked[df_ranked["rank"] == 1][
        ["function_id", "feature_name", "method"]
    ].rename(columns={"method": "winner"})

    df_second = df_ranked[df_ranked["rank"] == 2][
        ["function_id", "feature_name", "method"]
    ].rename(columns={"method": "second"})

    df_plot = df_winner.merge(
        df_second,
        on=["function_id", "feature_name"],
        how="left",
    )

    # ---------------------------------------------------------
    # Map method -> r
    # ---------------------------------------------------------
    method_to_r = (
        df_all[["method", "r"]]
        .drop_duplicates()
        .set_index("method")["r"]
        .to_dict()
    )

    df_plot["winner_r"] = df_plot["winner"].map(method_to_r)

    # ---------------------------------------------------------
    # Encode r as colors
    # ---------------------------------------------------------
    r_values = sorted(df_plot["winner_r"].dropna().unique())
    r_to_code = {r: i for i, r in enumerate(r_values)}
    df_plot["winner_code"] = df_plot["winner_r"].map(r_to_code)

    # markers for second-best
    #df_plot["second_marker"] = df_plot["second"].map(dataset_to_marker)
    df_plot["winner_marker"] = df_plot["winner"].map(dataset_to_marker)

    # ---------------------------------------------------------
    # Pivot
    # ---------------------------------------------------------
    heatmap = df_plot.pivot(
        index="function_id",
        columns="feature_name",
        values="winner_code",
    )

    heatmap = heatmap.reindex(index=function_id_list)
    heatmap = heatmap[feature_name_list]

    # ---------------------------------------------------------
    # Colormap based on r
    # ---------------------------------------------------------
    r_colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(r_values)))

    cmap = ListedColormap(r_colors)
    norm = BoundaryNorm(
        np.arange(-0.5, len(r_values) + 0.5, 1),
        cmap.N,
    )

    fig, ax = plt.subplots(
        figsize=(0.3 * heatmap.shape[1], 0.5 * heatmap.shape[0])
    )

    sns.heatmap(
        heatmap,
        cmap=cmap,
        norm=norm,
        linewidths=0.3,
        cbar=False,
        ax=ax,
    )

    # ---------------------------------------------------------
    # Overlay runner-up markers
    # ---------------------------------------------------------
    feature_to_x = {f: i for i, f in enumerate(heatmap.columns)}
    function_to_y = {f: i for i, f in enumerate(heatmap.index)}

    for _, r in df_plot.iterrows():

        if pd.isna(r["winner_marker"]):
            continue

        if r["feature_name"] not in feature_to_x:
            continue
        if r["function_id"] not in function_to_y:
            continue

        ax.scatter(
            feature_to_x[r["feature_name"]] + 0.5,
            function_to_y[r["function_id"]] + 0.5,
            marker=r["winner_marker"],
            s=60,
            facecolors="none",
            edgecolors="black",
            linewidths=1,
            zorder=10,
        )

    ax.set_xlabel("Feature")
    ax.set_ylabel("Function")

    # ---------------------------------------------------------
    # Legend (r-based colors)
    # ---------------------------------------------------------
    color_legend = ax.legend(
        handles=[
            Patch(color=r_colors[i], label=f"r = {r_values[i]}")
            for i in range(len(r_values))
        ],
        title="Winning reduction ratio",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.add_artist(color_legend)

    # second-best legend (methods)
    ax.legend(
        handles=[
            Line2D(
                [0], [0],
                marker=m,
                color="black",
                linestyle="None",
                markerfacecolor="none",
                markersize=8,
                label=ds,
            )
            for ds, m in dataset_to_marker.items()
        ],
        title="Best method",
        bbox_to_anchor=(1.02, 0.1),
        loc="upper left",
    )

    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    return fig, ax


def plot_parallel_function_overlay(
    df_rank: pd.DataFrame,
    method_order=None,
    plot_final_rank:bool=False,
):
    r"""
    Overlay FINAL rank and AVERAGE rank in one parallel plot (per function).

    Args
    --------------
        df_rank (pd.DataFrame): A DataFrame containing columns 'method', 'function_id', 'aggregated_feature_rank', 'rank_std_err', and optionally 'final_rank'.
        method_order (List[str], optional): A list specifying the order of methods to plot. If None, the order in df_rank will be used.
        plot_final_rank (bool, optional): Whether to plot the final rank as an overlay. Default is False.
    
    Returns
    --------------
        Tuple[plt.Figure, plt.Axes]: The matplotlib Figure and Axes objects containing the plot.
    """

    # --- Pivot average + std error (always needed)
    df_avg = df_rank.pivot(
        index="method",
        columns="function_id",
        values="aggregated_feature_rank",
    )

    df_std_error = df_rank.pivot(
        index="method",
        columns="function_id",
        values="rank_std_err",
    )

    # --- Optional: final rank
    if plot_final_rank:
        df_final = df_rank.pivot(
            index="method",
            columns="function_id",
            values="final_rank",
        )
        df_final = df_final.sort_index(axis=1)

    # --- Ensure consistent column order
    columns = sorted(df_avg.columns)
    df_avg = df_avg[columns]
    df_std_error = df_std_error[columns]

    if plot_final_rank:
        df_final = df_final[columns]

    # --- Method order handling
    if method_order is None:
        method_order = df_avg.index.tolist()

    df_avg = df_avg.reindex(method_order)
    df_std_error = df_std_error.reindex(method_order)

    if plot_final_rank:
        df_final = df_final.reindex(method_order)

    # --- Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.get_cmap("tab10").colors

    for i, method in enumerate(df_avg.index):

        if method not in df_avg.index:
            continue  # (extra safety, but usually unnecessary)

        color = colors[i % len(colors)]

        # --- Average rank
        ax.scatter(
            df_avg.columns,
            df_avg.loc[method],
            color=color,
            marker="o",
            label=method,

        )

        # --- Confidence interval
        #ax.fill_between(
        #    df_avg.columns,
        #    df_avg.loc[method] - 1.96 * df_std_error.loc[method],
        #    df_avg.loc[method] + 1.96 * df_std_error.loc[method],
        #    color=color,
        #    alpha=0.2,
        #)

        ax.errorbar(
            df_avg.columns,
            df_avg.loc[method],
            yerr=1.96 * df_std_error.loc[method],
            color=color,
            fmt="o",
            capsize=5,
            #label=method
        )

        # --- Final rank overlay (optional)
        if plot_final_rank:
            ax.plot(
                df_final.columns,
                df_final.loc[method],
                linestyle="--",
                marker="x",
                color=color,
                alpha=0.7,
            )

    # --- Formatting
    ax.invert_yaxis()
    ax.set_yticks(np.arange(1,11,1))
    ax.set_xticks(columns)
    ax.set_xlabel("Function ID")
    ax.set_ylabel("Rank")

    if plot_final_rank:
        ax.set_title("Final vs Average Ranking (Functions)")
    else:
        ax.set_title("Average Ranking (Functions)")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)

    return fig, ax


def plot_parallel_feature_overlay(df_rank_feat:pd.DataFrame, 
                                  method_order=None,
                                  feature_order=None,
                                  plot_final_rank:bool=False):
    r"""
    Overlay FINAL rank and AVERAGE rank in one parallel plot (per feature).

    Args
    --------------
        df_rank_feat (pd.DataFrame): A DataFrame containing columns 'method', 'feature_name', 'aggregated_function_rank', 'rank_std_err', and optionally 'final_rank'.
        method_order (List[str], optional): A list specifying the order of methods to plot. If None, the order in df_rank_feat will be used.
        feature_order (List[str], optional): A list specifying the order of features to plot. If None, the order in df_rank_feat will be used.
        plot_final_rank (bool, optional): Whether to plot the final rank as an overlay. Default is False.
    
    Returns
    --------------
        Tuple[plt.Figure, plt.Axes]: The matplotlib Figure and Axes objects containing the plot.
    """

    df_final = df_rank_feat.pivot(
        index="method",
        columns="feature_name",
        values="final_rank"
    )

    df_avg = df_rank_feat.pivot(
        index="method",
        columns="feature_name",
        values="aggregated_function_rank"
    )

    df_error = df_rank_feat.pivot(
        index="method",
        columns="feature_name",
        values="rank_std_err"
    )

    df_final = df_final.sort_index(axis=1)
    df_avg = df_avg[df_final.columns]
    df_error = df_error[df_final.columns]

    # Reindex ensures correct order
    df_final = df_final.reindex(method_order)
    df_avg   = df_avg.reindex(method_order)
    df_error = df_error.reindex(method_order)

    if feature_order is not None:
        df_final = df_final[feature_order]
        df_avg = df_avg[feature_order]
        df_error = df_error[feature_order]

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = plt.get_cmap("tab10").colors

    for i, method in enumerate(df_final.index):

        if method not in df_final.index:
            continue

        color = colors[i % len(colors)]

        # Average rank
        ax.plot(
            range(len(df_avg.columns)),
            df_avg.loc[method],
            color=color,
            linewidth=2,
            marker="o",
            label=method,
            linestyle="-",
        )

        ax.errorbar(
            range(len(df_avg.columns)),
            df_avg.loc[method],
            yerr=1.96 * df_error.loc[method],
            capsize=5,
            color=color,
            fmt="o",
        )


        if plot_final_rank:
            # Final rank
            ax.scatter(
                range(len(df_final.columns)),
                df_final.loc[method],
                linestyle="--",
                marker="x",
                color=color,
                alpha=0.7
            )

    ax.invert_yaxis()

    ax.set_yticks(np.arange(1,11,1))

    ax.set_xticks(range(len(df_final.columns)))
    ax.set_xticklabels(df_final.columns, rotation=90)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Rank")

    if plot_final_rank:
        ax.set_title("Final vs Average Ranking (Features)")
    else:
        ax.set_title("Average Ranking (Features)")

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)

    return fig, ax

def plot_critical_difference_diagram_function(
    avg_ranks,
    nemenyi_df: pd.DataFrame,
    alpha: float = 0.05,
    method_order=None,
    plot_title: str = "Critical Difference Diagram (Functions)",
) -> Tuple[plt.Figure, plt.Axes]:
    r"""
    Plot a critical difference diagram based on average ranks.

    Args
    --------------
        avg_ranks (pd.Series or pd.DataFrame):
            Either:
            - pd.Series: index=method, values=avg rank
            - pd.DataFrame with columns ['method', 'avg_rank']

        nemenyi_df (pd.DataFrame):
            Square matrix of p-values (methods x methods)

        alpha (float): significance level
        method_order (List[str], optional): desired plotting order

    Returns
    --------------
        Tuple[plt.Figure, plt.Axes]
    """

    # Start a vector of colors for the methods (will be used in CD diagram)
    colors = plt.get_cmap("tab10").colors

    # --- Convert avg_ranks to Series if needed
    if isinstance(avg_ranks, pd.DataFrame):
        if "method" in avg_ranks.columns:
            avg_ranks = avg_ranks.set_index("method").iloc[:, 0]
        else:
            raise ValueError("avg_ranks DataFrame must contain 'method' column")

    # --- Reorder if needed
    if method_order is not None:
        avg_ranks = avg_ranks.reindex(method_order)
        nemenyi_df = nemenyi_df.loc[method_order, method_order]
        # For the colors
        method_to_color = {method: colors[i % len(colors)] for i, method in enumerate(method_order)}
    else:
        method_to_color = {method: colors[i % len(colors)] for i, method in enumerate(avg_ranks.index)}
        
    

    # --- Sort by rank (important for CD diagram readability)
    avg_ranks = avg_ranks.sort_values()

    # --- Create plot
    fig, ax = plt.subplots(figsize=(10, 3))

    sp.critical_difference_diagram(
        avg_ranks,
        nemenyi_df,
        alpha=alpha,
        ax=ax,
        color_palette=[method_to_color[method] for method in avg_ranks.index]
    )

    # 🔥 Override colors AFTER plotting
    for text in ax.texts:
        method = text.get_text()
        if method in method_to_color:
            text.set_color(method_to_color[method])

    ax.set_title(plot_title)

    return fig, ax


def combine_wasserstein_results(
    list_of_dfs: List[pd.DataFrame],
    dataset_names: List[str],
) -> pd.DataFrame:
    r"""
    Combine multiple Wasserstein result dataframes into a single dataframe
    with a 'method' column.

    Args
    --------------
        list_of_dfs (List[pd.DataFrame]): A list of DataFrames, each containing columns ['function_id', 'instance_id', 'feature_name', 'wasserstein_distance'].
        dataset_names (List[str]): A list of names corresponding to each DataFrame in list_of_dfs, to be used in the 'method' column.   

    Returns
    --------------
        pd.DataFrame: A combined DataFrame with columns ['function_id', 'instance_id', 'feature_name', 'method', 'wasserstein_distance'].
    """
    if len(list_of_dfs) != len(dataset_names):
        raise ValueError("Length of list_of_dfs and dataset_names must match")

    combined = []
    for df_single, method_name in zip(list_of_dfs, dataset_names):
        tmp = df_single.copy()
        tmp["method"] = method_name
        combined.append(tmp)

    return pd.concat(combined, ignore_index=True)




def _aggregate_instances_per_feature(
    df: pd.DataFrame,
    agg: str = "median",
) -> pd.DataFrame:
    r"""
    Aggregate Wasserstein distances across instances for each
    (function_id, feature_name, method).

    Args
    --------------
        df (pd.DataFrame): Must contain
            ['function_id', 'instance_id', 'feature_name', 'method', 'wasserstein_distance'].
        agg (str): Aggregation across instances, either 'mean' or 'median'.

    Returns
    --------------
        pd.DataFrame: Columns
            ['function_id', 'feature_name', 'method', 'wasserstein_distance'].
    """
    assert agg in ["mean", "median"], "agg must be 'mean' or 'median'"

    out = (
        df.groupby(["function_id", "feature_name", "method"])["wasserstein_distance"]
        .agg(agg)
        .reset_index()
    )
    
    return out


def best_method_per_function_rank_based(
    df: pd.DataFrame,
    agg_instances: str = "median",
    agg_ranks: str = "mean",
) -> pd.DataFrame:
    r"""
    Determine the best method per function using rank aggregation across features.

    Args
    --------------
        df (pd.DataFrame): Must contain
            ['function_id', 'instance_id', 'feature_name', 'method', 'wasserstein_distance'].
        agg_instances (str): Aggregation across instances for ranking, 'mean' or 'median'.
        agg_ranks (str): Aggregation across features for final ranking, 'mean' or 'median'.
    
    Returns
    --------------
        pd.DataFrame: One row per (function, method), with columns
            ['function_id', 'method', 'aggregated_feature_rank', 'rank_std_dev', 'rank_std_err', 'final_rank']. 
    """

    assert agg_instances in ["mean", "median"], "agg_instances must be 'mean' or 'median'"
    assert agg_ranks in ["mean", "median"], "agg_ranks must be 'mean' or 'median'"

    # Step 1: aggregate over instances within each feature
    df_feat = _aggregate_instances_per_feature(df, agg=agg_instances)

    # Step 2: rank methods within each (function, feature)
    df_feat["feature_rank"] = (
        df_feat.groupby(["function_id", "feature_name"])["wasserstein_distance"]
        .rank(method="average", ascending=True)
    )

    # Step 3: aggregate ranks across features
    df_func = (
        df_feat.groupby(["function_id", "method"])["feature_rank"]
        .agg(agg_ranks)
        .reset_index(name="aggregated_feature_rank")
    )

    # Step 3_2: compute std of feature ranks
    df_std = (
        df_feat.groupby(["function_id", "method"])["feature_rank"]
        .std()
        .reset_index(name="rank_std_dev")
    )



    # Merge std into main dataframe
    df_func = df_func.merge(df_std, on=["function_id", "method"], how="left")

    # Step 3_3: Compute standard error of the mean rank across features

    # Count number of features per (function, method)
    df_count = (
        df_feat.groupby(["function_id", "method"])
        .size()
        .reset_index(name="n_features")
    )

    # Merge counts into df_func
    df_func = df_func.merge(df_count, on=["function_id", "method"], how="left")

    # Compute standard error
    df_func["rank_std_err"] = df_func["rank_std_dev"] / np.sqrt(df_func["n_features"]) 

    # Step 4: final ranking per function
    df_func["final_rank"] = (
        df_func.groupby("function_id")["aggregated_feature_rank"]
        .rank(method="average", ascending=True)
        .astype(int)
    )

    return df_func.sort_values(["function_id", "final_rank", "method"]).reset_index(drop=True)


def best_method_per_feature_rank_based(
    df: pd.DataFrame,
    agg_instances: str = "median",
    agg_ranks: str = "mean",
) -> pd.DataFrame:
    r"""
    Determine the best method per feature across functions using rank aggregation.

    Args
    --------------
        df (pd.DataFrame): Must contain
            ['function_id', 'instance_id', 'feature_name', 'method', 'wasserstein_distance'].
        agg_instances (str): Aggregation across instances for ranking, 'mean' or 'median'.
        agg_ranks (str): Aggregation across functions for final ranking, 'mean' or 'median'.
    
    Returns
    --------------
        pd.DataFrame: One row per (feature, method), with columns
            ['feature_name', 'method', 'aggregated_function_rank', 'rank_std_dev', 'rank_std_err', 'final_rank'].
    """

    assert agg_instances in ["mean", "median"], "agg_instances must be 'mean' or 'median'"
    assert agg_ranks in ["mean", "median"], "agg_ranks must be 'mean' or 'median'"

    # Step 1: aggregate over instances
    df_feat = _aggregate_instances_per_feature(df, agg=agg_instances)

    # Step 2: rank methods within each (function, feature)
    df_feat["feature_rank"] = (
        df_feat.groupby(["function_id", "feature_name"])["wasserstein_distance"]
        .rank(method="average", ascending=True)
    )

    # Step 3 + 5 combined: aggregate ranks across functions + compute stats
    df_feature = (
        df_feat.groupby(["feature_name", "method"])["feature_rank"]
        .agg(
            aggregated_function_rank=agg_ranks,
            rank_std_dev="std",
            n_functions="count",
        )
        .reset_index()
    )

    # Step 5_2: standard error
    df_feature["rank_std_err"] = (
        df_feature["rank_std_dev"] / np.sqrt(df_feature["n_functions"])
    )

    # Optional: handle NaNs (single function case)
    df_feature["rank_std_dev"] = df_feature["rank_std_dev"].fillna(0.0)
    df_feature["rank_std_err"] = df_feature["rank_std_err"].fillna(0.0)

    # Step 4: final ranking per feature
    df_feature["final_rank"] = (
        df_feature.groupby("feature_name")["aggregated_function_rank"]
        .rank(method="average", ascending=True)
        .astype(int)
    )

    return df_feature.sort_values(
        ["feature_name", "final_rank", "method"]
    ).reset_index(drop=True)


def significance_best_vs_second_per_function_feature(
    df: pd.DataFrame,
    agg_instances: str = "median",
    p_adjust: str = "holm",
) -> pd.DataFrame:
    r"""
    For each (function, feature), test whether the top-ranked method is significantly
    better than the second-ranked method using a paired one-sided Wilcoxon test
    across instances.

    Notes
    --------------
    - Ranking is based on aggregated performance over instances for each method.
    - The significance test itself uses paired instance-level distances.
    - Smaller Wasserstein distance is better.
    - alternative='less' tests whether best < second.

    Args
    --------------
        df (pd.DataFrame): Must contain
            ['function_id', 'instance_id', 'feature_name', 'method', 'wasserstein_distance'].
        agg_instances (str): Aggregation across instances for ranking, 'mean' or 'median'.
        p_adjust (str): Currently supports 'holm' or 'none'.

    Returns
    --------------
        pd.DataFrame: One row per (function, feature), with
            best_method, second_method, p_value, adjusted_p_value, significant.
    """
    assert agg_instances in ["mean", "median"], "agg_instances must be 'mean' or 'median'"
    assert p_adjust in ["holm", "none"], "p_adjust must be 'holm' or 'none'"

    # For ranking the methods
    df_rank_source = _aggregate_instances_per_feature(df, agg=agg_instances)

    results = []

    for (function_id, feature_name), grp_rank in df_rank_source.groupby(["function_id", "feature_name"]):
        grp_rank = grp_rank.sort_values("wasserstein_distance", ascending=True)

        if grp_rank["method"].nunique() < 2:
            continue

        best_method = grp_rank.iloc[0]["method"]
        second_method = grp_rank.iloc[1]["method"]

        grp_raw = df[
            (df["function_id"] == function_id) &
            (df["feature_name"] == feature_name) &
            (df["method"].isin([best_method, second_method]))
        ].copy()

        best_df = grp_raw[grp_raw["method"] == best_method][
            ["instance_id", "wasserstein_distance"]
        ].rename(columns={"wasserstein_distance": "wd_best"})

        second_df = grp_raw[grp_raw["method"] == second_method][
            ["instance_id", "wasserstein_distance"]
        ].rename(columns={"wasserstein_distance": "wd_second"})

        merged = best_df.merge(second_df, on="instance_id", how="inner")

        # Need at least 2 paired observations for Wilcoxon to be meaningful
        if len(merged) < 2:
            p_value = np.nan
            statistic = np.nan
        else:
            try:
                statistic, p_value = wilcoxon(
                    merged["wd_best"].values,
                    merged["wd_second"].values,
                    alternative="less",
                    zero_method="zsplit",
                    method="approx",
                )
            except ValueError:
                # Happens for degenerate cases, e.g. all paired differences zero
                statistic, p_value = np.nan, np.nan

        results.append({
            "function_id": function_id,
            "feature_name": feature_name,
            "best_method": best_method,
            "second_method": second_method,
            "best_score": grp_rank.iloc[0]["wasserstein_distance"],
            "second_score": grp_rank.iloc[1]["wasserstein_distance"],
            "n_pairs": len(merged),
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
        })

    out = pd.DataFrame(results)

    if out.empty:
        out["adjusted_p_value"] = []
        out["significant"] = []
        return out

    if p_adjust == "none":
        out["adjusted_p_value"] = out["p_value"]
    else:
        out["adjusted_p_value"] = holm_adjust_pvalues(out["p_value"].values)

    out["significant"] = out["adjusted_p_value"] < 0.05
    return out.sort_values(["function_id", "feature_name"]).reset_index(drop=True)


def holm_adjust_pvalues(pvalues: np.ndarray) -> np.ndarray:
    r"""
    Holm step-down p-value adjustment.
    NaN values are preserved.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan, dtype=float)

    valid_mask = ~np.isnan(pvalues)
    valid_p = pvalues[valid_mask]

    if len(valid_p) == 0:
        return adjusted

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    m = len(sorted_p)

    holm_vals = np.empty(m, dtype=float)
    for i, p in enumerate(sorted_p):
        holm_vals[i] = (m - i) * p

    # enforce monotonicity
    holm_vals = np.maximum.accumulate(holm_vals)
    holm_vals = np.clip(holm_vals, 0.0, 1.0)

    # put back original order
    unsorted = np.empty(m, dtype=float)
    unsorted[order] = holm_vals

    adjusted[valid_mask] = unsorted
    return adjusted


def friedman_test_per_feature(
    df: pd.DataFrame,
    agg_instances: str = "median",
) -> pd.DataFrame:
    r"""
    Run a Friedman test for each feature across functions.

    Blocks = functions
    Treatments = methods
    Response = aggregated instance-level Wasserstein distance for that feature

    Returns one row per feature.
    """
    assert agg_instances in ["mean", "median"], "agg_instances must be 'mean' or 'median'"

    df_feat = _aggregate_instances_per_feature(df, agg=agg_instances)

    results = []

    for feature_name, grp in df_feat.groupby("feature_name"):
        pivot = grp.pivot_table(
            index="function_id",
            columns="method",
            values="wasserstein_distance",
            aggfunc="first"
        )

        # keep only complete blocks
        pivot = pivot.dropna(axis=0, how="any")

        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            results.append({
                "feature_name": feature_name,
                "n_functions": pivot.shape[0],
                "n_methods": pivot.shape[1],
                "friedman_statistic": np.nan,
                "p_value": np.nan,
            })
            continue

        statistic, p_value = friedmanchisquare(*[pivot[c].values for c in pivot.columns])

        results.append({
            "feature_name": feature_name,
            "n_functions": pivot.shape[0],
            "n_methods": pivot.shape[1],
            "friedman_statistic": statistic,
            "p_value": p_value,
        })

    return pd.DataFrame(results).sort_values("feature_name").reset_index(drop=True)


def nemenyi_test_from_rank_df(
    df_rank: pd.DataFrame,
    index_cols: List[str] = ["function_id"],
    value_col: str = "aggregated_feature_rank",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, float, float]:
    r"""
    Perform Friedman + Nemenyi test from a rank DataFrame.

    Supports:
    - function-level (default)
    - function × feature-level (global)

    Args
    --------------
        df_rank : DataFrame containing at least
            ['method', value_col] + index_cols
        index_cols : list of columns defining datasets (blocks)
            e.g. ['function_id'] or ['function_id', 'feature_name']
        value_col : column containing rank values

    Returns
    --------------
        df_matrix : pivot table (datasets × methods)
        avg_ranks : average rank per method
        nemenyi_df : pairwise p-values
        stat : Friedman statistic
        p : Friedman p-value
    """

    # --- Pivot: rows = datasets, cols = methods
    df_matrix = df_rank.pivot_table(
        index=index_cols,
        columns="method",
        values=value_col,
        aggfunc="mean",   # safe even if duplicates
    )

    # --- Drop incomplete rows
    df_matrix = df_matrix.dropna(axis=0, how="any")

    if df_matrix.shape[0] < 2 or df_matrix.shape[1] < 2:
        raise ValueError("Not enough data for Friedman test")

    # --- Friedman test
    stat, p = friedmanchisquare(*[df_matrix[c].values for c in df_matrix.columns])

    # --- Nemenyi test
    nemenyi = sp.posthoc_nemenyi_friedman(df_matrix.values)
    nemenyi.index = df_matrix.columns
    nemenyi.columns = df_matrix.columns

    # --- Average ranks
    avg_ranks = df_matrix.mean(axis=0).sort_values()

    return df_matrix, avg_ranks, nemenyi, stat, p

def nemenyi_grouping(avg_ranks: pd.Series, 
                     nemenyi_df: pd.DataFrame, 
                     alpha=0.05)->Dict[str, str]:
    r"""
    Assign group letters based on Nemenyi test.
    Methods not significantly different share a group.

    Args
    --------------
    - avg_ranks: Series of average ranks, indexed by method.
    - nemenyi_df: DataFrame of Nemenyi p-values, indexed and columned
        by method.
    - alpha: significance level for grouping.

    Returns
    --------------
    - groups: dict mapping method -> group letter (e.g. 'a', 'b', ...)
    """

    methods = avg_ranks.index.tolist()
    groups = {}
    current_group = 'a'

    for i, m1 in enumerate(methods):
        if m1 in groups:
            continue

        groups[m1] = current_group

        for m2 in methods[i+1:]:
            if nemenyi_df.loc[m1, m2] >= alpha:
                groups[m2] = current_group

        current_group = chr(ord(current_group) + 1)

    return groups

def print_nemenyi_summary(avg_ranks, groups):
    r"""
    Print clean ranking + grouping table.
    """

    print("\n=== Average ranks with significance groups ===")
    for method in avg_ranks.index:
        print(f"{method:40s}  rank={avg_ranks[method]:.3f}   group={groups[method]}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:

    # Load all datasets
    datasets = load_all_datasets()

    # Get list of all feature names from a reference dataset (e.g. reduced with r=0.5)
    reference_df = datasets[("reduced", 200, 0.5)]
    all_feature_names = [
                            col for col in reference_df.columns
                            if col not in EXCLUDED_COLUMNS
    ]

    # Get the Wasserstein distances between full datasets of different sizes
    df_wasserstein_full = compute_wasserstein_distance(
        datasets[("full", 2000, None)],
        datasets[("full", 200, None)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )


    df_wasserstein_slices_0_05_0, df_wasserstein_slices_0_05_gen = compute_wasserstein_distance_slices(
        datasets[("full", 2000, None)],
        datasets[("slices", 200, 0.5)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    df_wasserstein_slices_0_025_0, df_wasserstein_slices_0_025_gen = compute_wasserstein_distance_slices(
        datasets[("full", 2000, None)],
        datasets[("slices", 200, 0.25)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    df_wasserstein_slices_0_01_0, df_wasserstein_slices_0_01_gen = compute_wasserstein_distance_slices(
        datasets[("full", 2000, None)],
        datasets[("slices", 200, 0.1)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    df_wasserstein_slices_all_in_0_05_0, df_wasserstein_slices_all_in_0_05_gen = compute_wasserstein_distance_slices(
        datasets[("full", 2000, None)],
        datasets[("slices_all_in", 200, 0.5)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    df_wasserstein_slices_all_in_0_025_0, df_wasserstein_slices_all_in_0_025_gen = compute_wasserstein_distance_slices(
        datasets[("full", 2000, None)],
        datasets[("slices_all_in", 200, 0.25)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    df_wasserstein_slices_all_in_0_01_0, df_wasserstein_slices_all_in_0_01_gen = compute_wasserstein_distance_slices(
        datasets[("full", 2000, None)],
        datasets[("slices_all_in", 200, 0.1)],
        all_feature_names,
        FUNCTION_IDS,
        INSTANCE_IDS
    )

    method_order = [
        "Full/ELA$_{\\mathrm{A}}$",
        "Sliced/ELA$_{\\mathrm{A}},r=0.5$",
        "Sliced/ELA$_{\\mathrm{A}},r=0.25$",
        "Sliced/ELA$_{\\mathrm{A}},r=0.1$",
        "All\_in/ELA$_{\\mathrm{A}},r=0.5$",
        "All\_in/ELA$_{\\mathrm{A}},r=0.25$",
        "All\_in/ELA$_{\\mathrm{A}},r=0.1$",
        "All\_in/ELA$_{\\mathrm{R}},r=0.5$",
        "All\_in/ELA$_{\\mathrm{R}},r=0.25$",
        "All\_in/ELA$_{\\mathrm{R}},r=0.1$",
    ]

    df_all_methods = combine_wasserstein_results(
    [
        df_wasserstein_full,
        df_wasserstein_slices_0_05_0,
        df_wasserstein_slices_0_025_0,
        df_wasserstein_slices_0_01_0,
        df_wasserstein_slices_all_in_0_05_0,
        df_wasserstein_slices_all_in_0_025_0,
        df_wasserstein_slices_all_in_0_01_0,
        df_wasserstein_slices_all_in_0_05_gen,
        df_wasserstein_slices_all_in_0_025_gen,
        df_wasserstein_slices_all_in_0_01_gen,
    ],
    method_order
    )
    
    # 1) Best method per function overall, using rank aggregation across features
    df_best_per_function = best_method_per_function_rank_based(
        df_all_methods,
        agg_instances="median",
        agg_ranks="mean",
    )

    print("\nBest method per function overall:")
    print(df_best_per_function.head(30))

    # ------------------------------------------------------------------
    # GLOBAL RANKING + NEMENYI TEST
    # ------------------------------------------------------------------

    df_matrix_1, avg_ranks_1, nemenyi_df_1, friedman_stat, friedman_p = (
        nemenyi_test_from_rank_df(df_best_per_function,
                                  index_cols=["function_id"])
    )

    print("\n=== Friedman test ===")
    print(f"statistic = {friedman_stat:.4f}, p-value = {friedman_p:.4e}")

    print("\n=== Nemenyi p-values ===")
    print(nemenyi_df_1.round(4))

    # Compute grouping
    groups_1 = nemenyi_grouping(avg_ranks_1, nemenyi_df_1, alpha=0.05)

    # Print clean summary
    print_nemenyi_summary(avg_ranks_1, groups_1)

    # Save results
    nemenyi_df_1.to_csv(
        SAVE_FIGURE_DIRECTORY / f"nemenyi_matrix_per_function_mode_{MODE}.csv"
    )

    avg_ranks_1.to_csv(
        SAVE_FIGURE_DIRECTORY / f"average_ranks_per_function_mode_{MODE}.csv",
        header=["avg_rank"]
    )

    # 2) Best method per feature across functions
    df_best_per_feature = best_method_per_feature_rank_based(
        df_all_methods,
        agg_instances="median",
        agg_ranks="mean",
    )

    print("\nBest method per feature:")
    print(df_best_per_feature.head(30))

    # ------------------------------------------------------------------
    # GLOBAL RANKING + NEMENYI TEST
    # ------------------------------------------------------------------

    df_feat = _aggregate_instances_per_feature(df_all_methods, agg="median")

    df_feat["feature_rank"] = (
        df_feat.groupby(["function_id", "feature_name"])["wasserstein_distance"]
        .rank(method="average", ascending=True)
    )


    df_matrix_2, avg_ranks_2, nemenyi_df_2, friedman_stat, friedman_p = (
    nemenyi_test_from_rank_df(
        df_feat,
        index_cols=["function_id", "feature_name"],
        value_col="feature_rank",   # 🔥 THIS LINE FIXES YOUR ERROR
    )
    )

    print("\n=== Friedman test ===")
    print(f"statistic = {friedman_stat:.4f}, p-value = {friedman_p:.4e}")

    print("\n=== Nemenyi p-values ===")
    print(nemenyi_df_2.round(4))

    # Compute grouping
    groups_2 = nemenyi_grouping(avg_ranks_2, nemenyi_df_2, alpha=0.05)

    # Print clean summary
    print_nemenyi_summary(avg_ranks_2, groups_2)

    # Save results
    nemenyi_df_2.to_csv(
        SAVE_FIGURE_DIRECTORY / f"nemenyi_matrix_per_feature_mode_{MODE}.csv"
    )

    avg_ranks_2.to_csv(
        SAVE_FIGURE_DIRECTORY / f"average_ranks_per_feature_mode_{MODE}.csv",
        header=["avg_rank"]
    )

    # 3) Significance: best vs second-best for each function-feature pair
    df_significance = significance_best_vs_second_per_function_feature(
        df_all_methods,
        agg_instances="median",
        p_adjust=P_ADJUST,
    )

    print("\nSignificance results:")
    print(df_significance.head(30))

    # 4) Friedman test per feature across functions
    df_friedman = friedman_test_per_feature(
        df_all_methods,
        agg_instances="median",
    )

    print("\nFriedman per feature:")
    print(df_friedman.head(30))


    SAVE_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Save results
    df_best_per_function.to_csv(SAVE_FIGURE_DIRECTORY / f"best_method_per_function_mode_{MODE}.csv", index=False)
    df_best_per_feature.to_csv(SAVE_FIGURE_DIRECTORY / f"best_method_per_feature_mode_{MODE}.csv", index=False)
    df_significance.to_csv(SAVE_FIGURE_DIRECTORY / f"significance_best_vs_second_mode_{MODE}.csv", index=False)
    df_friedman.to_csv(SAVE_FIGURE_DIRECTORY / f"friedman_per_feature_mode_{MODE}.csv", index=False)

    fig, ax = plot_parallel_feature_overlay(df_best_per_feature, method_order, feature_order=all_feature_names)
    fig.savefig(SAVE_FIGURE_DIRECTORY / f"wasserstein_ranking_parallel_feature_{MODE}.pdf", dpi=300, bbox_inches="tight")

    plt.close(fig)

    fig, ax = plot_parallel_function_overlay(df_best_per_function, method_order)
    fig.savefig(SAVE_FIGURE_DIRECTORY / f"wasserstein_ranking_parallel_function_{MODE}.pdf", dpi=300, bbox_inches="tight")

    plt.close(fig)

    fig, ax = plot_critical_difference_diagram_function(avg_ranks_1, 
                                                        nemenyi_df_1, 
                                                        alpha=0.05, 
                                                        method_order=method_order)
    
    fig.savefig(SAVE_FIGURE_DIRECTORY / f"cd_diagram_mode_{MODE}_function.pdf",dpi=300, 
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plot_critical_difference_diagram_function(avg_ranks_2, 
                                                        nemenyi_df_2, 
                                                        alpha=0.05, 
                                                        method_order=method_order,
                                                        plot_title="Critical Difference Diagram (Features)")
    
    fig.savefig(SAVE_FIGURE_DIRECTORY / f"cd_diagram_mode_{MODE}_feature.pdf",dpi=300, 
                bbox_inches="tight")
    plt.close(fig)
    

    # The heatmap of best methods per function-feature combination is quite large, 
    # so we save it as a separate figure.
    fig, ax = heatmap_wasserstein_rankings_2(
        df_all_methods,
        FUNCTION_IDS,
        all_feature_names,
        agg="median",
        significance_df=df_significance,
    )

    fig.savefig(SAVE_FIGURE_DIRECTORY / f"wasserstein_ranking_heatmap_mode_{MODE}.pdf", dpi=300)

    plt.close(fig)

    # The heatmap of best methods per function-feature combination is quite large, 
    # so we save it as a separate figure.
    fig, ax = heatmap_wasserstein_rankings_3(
        df_all_methods,
        FUNCTION_IDS,
        all_feature_names,
        agg="median",
    )

    fig.savefig(SAVE_FIGURE_DIRECTORY / f"wasserstein_ranking_heatmap_mode_{MODE}_r.pdf", dpi=300)

    plt.close(fig)


   

if __name__ == "__main__":
    main()