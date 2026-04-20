import pandas as pd
import seaborn as sns
import numpy as np
import os, sys
from pathlib import Path
import matplotlib

from typing import Tuple, List


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
SAVE_FIGURE_DIRECTORY = ROOT_DIRECTORY.joinpath("figures_barplots_slices_comparison_full")

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

def combine_differences_results(
        list_of_dfs: List[pd.DataFrame],
        dataset_names: List[str]) -> pd.DataFrame:
    """
    Combine multiple difference result dataframes into a single LONG dataframe
    with columns:
        function_idx, instance_idx, method, feature, ratio
    """

    if len(list_of_dfs) != len(dataset_names):
        raise ValueError("Length of list_of_dfs and dataset_names must match")

    # ---------------------------------------------------------
    # Concatenate with method column (no explicit loop needed)
    # ---------------------------------------------------------
    df = pd.concat(
        (d.assign(method=name) for d, name in zip(list_of_dfs, dataset_names)),
        ignore_index=True
    )

    # ---------------------------------------------------------
    # Identify ratio columns (fast vectorized)
    # ---------------------------------------------------------
    ratio_cols = df.columns[df.columns.str.startswith("ratio_")]

    # ---------------------------------------------------------
    # Melt → features become rows
    # ---------------------------------------------------------
    df_long = df.melt(
        id_vars=["function_idx", "instance_idx", "method"],
        value_vars=ratio_cols,
        var_name="feature",
        value_name="ratio",
    )

    # ---------------------------------------------------------
    # Clean feature names (vectorized, once)
    # ---------------------------------------------------------
    df_long["feature"] = df_long["feature"].str.removeprefix("ratio_")

    return df_long



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
# Problem Differences
# -----------------------------------------------------------------------------


def compute_differences_full(
    df_full_2000:pd.DataFrame,
    df_full_200:pd.DataFrame,
    feature_name_list:List[str],
    function_id_list:List[int]=FUNCTION_IDS,
    instance_id_list:List[int]=INSTANCE_IDS,
    agg:str="median",
    ) -> pd.DataFrame:

    r"""
    Computes a relative difference between two full feature_dataset of different dataset sizes.

    Args
    --------------
    - df_full_2000 (pd.DataFrame): The full dataset DataFrame computed in ambient space with 2000 samples.
    - df_full_200 (pd.DataFrame): The full dataset DataFrame computed in ambient space with 200 samples.
    - feature_name_list (List[str]): List of feature names to consider.
    - function_id_list (List[int]): List of function IDs to filter the datasets.
    - instance_id_list (List[int]): List of instance IDs to filter the datasets.
    - agg (str): Aggregation method to use ("mean" or "median").
    
    Returns
    --------------
    pd.DataFrame: A DataFrame containing the relative differences for each feature.
    """

    df_full_2000_filtered = df_full_2000[
    (df_full_2000["function_idx"].isin(function_id_list))
    & (df_full_2000["instance_idx"].isin(instance_id_list))
    ].copy()

    df_full_200_filtered = df_full_200[
    (df_full_200["function_idx"].isin(function_id_list))
    & (df_full_200["instance_idx"].isin(instance_id_list))
    ].copy()

    # For each tuple feature, function, instance, compute a mean which is a reference
    df_full_2000_agg = (
    df_full_2000_filtered
    .groupby(["function_idx", "instance_idx"])[feature_name_list]
    .agg(agg)
    .reset_index()
    )

    # Now for each match in function id and instance from the full 2000,
    # compute a ratio difference with the full 200

    df_merged = pd.merge(
    df_full_2000_agg,
    df_full_200_filtered,
    on=["function_idx", "instance_idx"],
    suffixes=("_2000_ref", "_200"),
    )

    for feature_name in feature_name_list:
        df_merged[f"ratio_{feature_name}"] = (
       (df_merged[f"{feature_name}_200"]-df_merged[f"{feature_name}_2000_ref"])/(np.abs(df_merged[f"{feature_name}_2000_ref"]) + EPSILON)
        )
    
    # Delete unneeded columns
    columns_to_delete = []
    for feature_name in feature_name_list:
        columns_to_delete.append(f"{feature_name}_2000_ref")
        columns_to_delete.append(f"{feature_name}_200")

    df_merged.drop(columns=columns_to_delete, inplace=True)
    
    return df_merged

def compute_differences_in_reduced(
    df_full_2000:pd.DataFrame,
    df_reduced_2000:pd.DataFrame,
    feature_name_list:List[str],
    function_id_list:List[int]=FUNCTION_IDS,
    instance_id_list:List[int]=INSTANCE_IDS,
    agg:str="median",
    ) -> pd.DataFrame:

    r"""
    Computes a relative difference between full feature_dataset and reduced feature_dataset of same dataset size.

    Args
    --------------
    - df_full_2000 (pd.DataFrame): The full dataset DataFrame computed in ambient space with 2000 samples.
    - df_reduced_2000 (pd.DataFrame): The reduced dataset DataFrame computed in reduced space with 2000 samples.
    - feature_name_list (List[str]): List of feature names to consider.
    - function_id_list (List[int]): List of function IDs to filter the datasets.
    - instance_id_list (List[int]): List of instance IDs to filter the datasets.
    - agg (str): Aggregation method to use ("mean" or "median").

    Returns
    --------------
    pd.DataFrame: A DataFrame containing the relative differences for each feature.
    """

    df_full_2000_filtered = df_full_2000[
    (df_full_2000["function_idx"].isin(function_id_list))
    & (df_full_2000["instance_idx"].isin(instance_id_list))
    ].copy()

    df_reduced_2000_filtered = df_reduced_2000[
    (df_reduced_2000["function_idx"].isin(function_id_list))
    & (df_reduced_2000["instance_idx"].isin(instance_id_list))
    ].copy()

    # For each combination of feature, function, instance, seed lhs, compute an aggregate for each embedding seed
    df_reduced_2000_agg = (
    df_reduced_2000_filtered
    .groupby(["function_idx", "instance_idx","seed_lhs"])[feature_name_list]
    .agg(agg)
    .reset_index()
    )

    # For each combination of feature, function, instance, compute an aggregate for each 
    df_full_2000_agg = (
    df_full_2000_filtered
    .groupby(["function_idx", "instance_idx"])[feature_name_list]
    .agg(agg)
    .reset_index()
    )

    # Now for each match in function id and instance and seed from the reduced,
    # compute a ratio difference with the full 2000
    df_merged = pd.merge(
    df_reduced_2000_agg,
    df_full_2000_agg,
    on=["function_idx", "instance_idx"],
    suffixes=("_reduced_2000","_full_2000"),
    )

    # COmpute the ratio differences
    for feature_name in feature_name_list:
        df_merged[f"ratio_{feature_name}"] = (
       (df_merged[f"{feature_name}_reduced_2000"]-df_merged[f"{feature_name}_full_2000"])/(np.abs(df_merged[f"{feature_name}_full_2000"] ) + EPSILON )
        )
    
    # Delete unneeded columns
    columns_to_delete = []
    for feature_name in feature_name_list:
        columns_to_delete.append(f"{feature_name}_reduced_2000")
        columns_to_delete.append(f"{feature_name}_full_2000")
    
    df_merged.drop(columns=columns_to_delete, inplace=True)

    return df_merged


def compute_differences_in_slices_0(
    df_full_2000:pd.DataFrame,
    df_slices_200:pd.DataFrame,
    feature_name_list:List[str],
    function_id_list:List[int]=FUNCTION_IDS,
    instance_id_list:List[int]=INSTANCE_IDS,
    agg:str="median",
    ) -> pd.DataFrame:

    r"""
    Computes a relative difference between full feature_dataset and slices dataset for slice_id=0.

    Args
    --------------
    - df_full_2000 (pd.DataFrame): The full dataset DataFrame computed in ambient space with 2000 samples.
    - df_slices_200 (pd.DataFrame): The reduced dataset DataFrame computed in reduced space with 200 samples using slices.
    - feature_name_list (List[str]): List of feature names to consider.
    - function_id_list (List[int]): List of function IDs to filter the datasets.
    - instance_id_list (List[int]): List of instance IDs to filter the datasets.
    - agg (str): Aggregation method to use ("mean" or "median").

    Returns
    --------------
    pd.DataFrame: A DataFrame containing the relative differences for each feature.
    """

    df_full_2000_filtered = df_full_2000[
    (df_full_2000["function_idx"].isin(function_id_list))
    & (df_full_2000["instance_idx"].isin(instance_id_list))
    ].copy()

    df_slices_200_filtered = df_slices_200[
    (df_slices_200["function_idx"].isin(function_id_list))
    & (df_slices_200["instance_idx"].isin(instance_id_list))
    & (df_slices_200["slice_id"]==0)
    ].copy()

    # For each combination of feature, function, instance, group_id, compute an aggregate for each embedding seed
    df_slices_200_agg = (
    df_slices_200_filtered
    .groupby(["function_idx", "instance_idx","group_id"])[feature_name_list]
    .agg(agg)
    .reset_index()
    )

    # For each combination of feature, function, instance, compute an aggregate for each 
    df_full_2000_agg = (
    df_full_2000_filtered
    .groupby(["function_idx", "instance_idx"])[feature_name_list]
    .agg(agg)
    .reset_index()
    )

    # Now for each match in function id and instance and seed from the reduced,
    # compute a ratio difference with the full 2000
    df_merged = pd.merge(
    df_slices_200_agg,
    df_full_2000_agg,
    on=["function_idx", "instance_idx"],
    suffixes=("_slices_200","_full_2000"),
    )

    # COmpute the ratio differences
    for feature_name in feature_name_list:
        df_merged[f"ratio_{feature_name}"] = (
       (df_merged[f"{feature_name}_slices_200"]-df_merged[f"{feature_name}_full_2000"])/(np.abs(df_merged[f"{feature_name}_full_2000"] ) + EPSILON )
        )
    
    # Delete unneeded columns
    columns_to_delete = []
    for feature_name in feature_name_list:
        columns_to_delete.append(f"{feature_name}_slices_200")
        columns_to_delete.append(f"{feature_name}_full_2000")
    
    df_merged.drop(columns=columns_to_delete, inplace=True)

    return df_merged

def compute_differences_in_slices_general(
    df_full_2000:pd.DataFrame,
    df_slices_200:pd.DataFrame,
    feature_name_list:List[str],
    function_id_list:List[int]=FUNCTION_IDS,
    instance_id_list:List[int]=INSTANCE_IDS,
    agg:str="median",
    ) -> pd.DataFrame:

    r"""
    Computes a relative difference between full feature_dataset and slices dataset for slice_id!=0.

    Args
    --------------
    - df_full_2000 (pd.DataFrame): The full dataset DataFrame computed in ambient space with 2000 samples.
    - df_slices_200 (pd.DataFrame): The reduced dataset DataFrame computed in reduced space with 200 samples using slices.
    - feature_name_list (List[str]): List of feature names to consider.
    - function_id_list (List[int]): List of function IDs to filter the datasets.
    - instance_id_list (List[int]): List of instance IDs to filter the datasets.
    - agg (str): Aggregation method to use ("mean" or "median").

    Returns
    --------------
    pd.DataFrame: A DataFrame containing the relative differences for each feature.
    """

    df_full_2000_filtered = df_full_2000[
    (df_full_2000["function_idx"].isin(function_id_list))
    & (df_full_2000["instance_idx"].isin(instance_id_list))
    ].copy()

    df_slices_200_filtered = df_slices_200[
    (df_slices_200["function_idx"].isin(function_id_list))
    & (df_slices_200["instance_idx"].isin(instance_id_list))
    & (df_slices_200["slice_id"]!=0)
    ].copy()

    # For each combination of feature, function, instance, group_id, compute an aggregate for each embedding seed
    df_slices_200_agg = (
    df_slices_200_filtered
    .groupby(["function_idx", "instance_idx","group_id", "slice_id"])[feature_name_list]
    .agg(agg)
    .reset_index()
    )

    # For each combination of feature, function, instance, compute an aggregate for each 
    df_full_2000_agg = (
    df_full_2000_filtered
    .groupby(["function_idx", "instance_idx"])[feature_name_list]
    .agg(agg)
    .reset_index()
    )

    # Now for each match in function id and instance and seed from the reduced,
    # compute a ratio difference with the full 2000
    df_merged = pd.merge(
    df_slices_200_agg,
    df_full_2000_agg,
    on=["function_idx", "instance_idx"],
    suffixes=("_slices_200","_full_2000"),
    )

    # COmpute the ratio differences
    for feature_name in feature_name_list:
        df_merged[f"ratio_{feature_name}"] = (
       (df_merged[f"{feature_name}_slices_200"]-df_merged[f"{feature_name}_full_2000"])/(np.abs(df_merged[f"{feature_name}_full_2000"] ) + EPSILON )
        )
    
    # Delete unneeded columns
    columns_to_delete = []
    for feature_name in feature_name_list:
        columns_to_delete.append(f"{feature_name}_slices_200")
        columns_to_delete.append(f"{feature_name}_full_2000")
    
    df_merged.drop(columns=columns_to_delete, inplace=True)

    return df_merged


def box_plots_of_differences(
    df_differences_full: pd.DataFrame,
    df_differences_reduced: pd.DataFrame,
    feature_name_list: List[str],
    function_id: int,
    instance_id_list: List[int] = INSTANCE_IDS,
) -> Tuple[plt.Figure, plt.Axes]:

    r"""
    Create box plots to visualize the differences in feature values between
    the full dataset and the reduced dataset.
    """

    # Columns to plot
    ratio_columns = [f"ratio_{f}" for f in feature_name_list]

    # --- Filter ---
    df_full = df_differences_full[
        (df_differences_full["function_idx"] == function_id)
        & (df_differences_full["instance_idx"].isin(instance_id_list))
    ].copy()

    df_reduced = df_differences_reduced[
        (df_differences_reduced["function_idx"] == function_id)
        & (df_differences_reduced["instance_idx"].isin(instance_id_list))
    ].copy()

    # --- Melt to long format ---
    df_full_melted = df_full.melt(
        id_vars=["function_idx", "instance_idx"],
        value_vars=ratio_columns,
        var_name="feature",
        value_name="ratio",
    )
    df_full_melted["dataset"] = "Full (200 vs 2000)"

    df_reduced_melted = df_reduced.melt(
        id_vars=["function_idx", "instance_idx"],
        value_vars=ratio_columns,
        var_name="feature",
        value_name="ratio",
    )
    df_reduced_melted["dataset"] = "Reduced 200 vs full 2000"

    # Clean feature names (remove "ratio_")
    for df_ in (df_full_melted, df_reduced_melted):
        df_["feature"] = df_["feature"].str.replace("ratio_", "", regex=False)

    # --- Combine ---
    df_plot = pd.concat([df_full_melted, df_reduced_melted], ignore_index=True)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(1.5 * len(feature_name_list), 6))

    sns.boxplot(
        data=df_plot,
        x="feature",
        y="ratio",
        hue="dataset",
        ax=ax,
    )

    ax.set_ylim(ymax=-2, ymin=2)

    ax.set_title(f"Relative differences for function {function_id}")
    ax.set_ylabel("Relative difference")
    ax.set_xlabel("Feature")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()

    return fig, ax


def box_plots_of_differences_global(
    df_differences_full: pd.DataFrame,
    df_differences_reduced: pd.DataFrame,
    df_differences_slices_0: pd.DataFrame,
    df_differences_slices_gen: pd.DataFrame,
    feature_name: str,
    function_id: int,
    instance_id_list: List[int] = INSTANCE_IDS,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create box plots of relative differences for a single feature
    and a single function across different dataset variants.
    """

    ratio_col = f"ratio_{feature_name}"

    # --- Helper ---
    def prepare(df: pd.DataFrame, label: str) -> pd.DataFrame:
        out = df.loc[
            (df["function_idx"] == function_id)
            & (df["instance_idx"].isin(instance_id_list)),
            ["instance_idx", ratio_col],
        ].copy()

        out["dataset"] = label
        out["feature"] = feature_name
        out.rename(columns={ratio_col: "ratio"}, inplace=True)
        return out

    # --- Prepare datasets ---
    df_plot = pd.concat(
        [
            prepare(df_differences_full, "Full (200 vs 2000)"),
            prepare(df_differences_reduced, "Reduced 200 vs full 2000"),
            prepare(df_differences_slices_0, "Sliced 200 vs full 2000"),
            prepare(df_differences_slices_gen, "Sliced gen vs full 2000"),
        ],
        ignore_index=True,
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(4, 6))

    sns.boxplot(
        data=df_plot,
        x="feature",
        y="ratio",
        hue="dataset",
        ax=ax,
    )

    #ax.set_ylim(-2, 2)
    ax.set_title(f"Relative differences – function {function_id}")
    ax.set_ylabel("Relative difference")
    ax.set_xlabel("Feature")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    return fig, ax


def violin_plots_of_differences_global(
    df_differences_full: pd.DataFrame,
    df_differences_reduced: pd.DataFrame,
    df_differences_slices_0: pd.DataFrame,
    df_differences_slices_gen: pd.DataFrame,
    feature_name: str,
    function_id: int,
    instance_id_list: List[int] = INSTANCE_IDS,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create box plots of relative differences for a single feature
    and a single function across different dataset variants.
    """

    ratio_col = f"ratio_{feature_name}"

    # --- Helper ---
    def prepare(df: pd.DataFrame, label: str) -> pd.DataFrame:
        out = df.loc[
            (df["function_idx"] == function_id)
            & (df["instance_idx"].isin(instance_id_list)),
            ["instance_idx", ratio_col],
        ].copy()

        out["dataset"] = label
        out["feature"] = feature_name
        out.rename(columns={ratio_col: "ratio"}, inplace=True)
        return out

    # --- Prepare datasets ---
    df_plot = pd.concat(
        [
            prepare(df_differences_full, "Full (200 vs 2000)"),
            prepare(df_differences_reduced, "Reduced 200 vs full 2000"),
            prepare(df_differences_slices_0, "Sliced 200 vs full 2000"),
            prepare(df_differences_slices_gen, "Sliced gen vs full 2000"),
        ],
        ignore_index=True,
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(4, 6))

    sns.violinplot(
        data=df_plot,
        x="feature",
        y="ratio",
        hue="dataset",
        ax=ax,
    )

    #ax.set_ylim(-2, 2)
    ax.set_title(f"Relative differences – function {function_id}")
    ax.set_ylabel("Relative difference")
    ax.set_xlabel("Feature")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    return fig, ax


def violin_plots_of_differences_global_2(
        df_list: pd.DataFrame,
        feature_name: str,
        function_id: int,
        instance_id_list: List[int] = INSTANCE_IDS,
    ) -> Tuple[plt.Figure, plt.Axes]:

    r"""
    Create violin plots of relative differences for a single feature
    and a single function across different dataset variants.

    Args
    --------------
    - df_list (pd.DataFrame): A DataFrame containing the differences for the dataset variant.
    - feature_name (str): The name of the feature to plot.
    - function_id (int): The function ID to filter the datasets.
    - instance_id_list (List[int]): A list of instance IDs to filter the datasets.

    Returns
    --------------
    - Tuple[plt.Figure, plt.Axes]: The figure and axes objects of the created plot.
    """

    # --- Helper ---
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        out = df.loc[
            (df["function_idx"] == function_id)
            & (df["instance_idx"].isin(instance_id_list)) &
            (df["feature"] == feature_name),
        ].copy()

        return out
    
    # --- Prepare datasets ---
    df_plot = prepare(df_list)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot the horizontal line at y=0
    ax.axhline(0, color="red", linestyle="--", linewidth=0.5, alpha=0.7)

    sns.violinplot(
        data=df_plot,
        x="feature",
        y="ratio",
        hue="method",
        ax=ax,
    )

    ax.set_title(f"Relative differences – f{function_id}")
    ax.set_ylabel("Relative difference")
    ax.set_xlabel("Feature")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

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






# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    datasets = load_all_datasets()


    reference_df = datasets[("reduced", 200, 0.5)]
    all_feature_names = [
    col for col in reference_df.columns
    if col not in EXCLUDED_COLUMNS
    ]

    # Get the differences between full datasets of different sizes
    df_differences_full = compute_differences_full(
        datasets[("full", 2000, None)],
        datasets[("full", 200, None)],
        all_feature_names,
        agg="median"
    )

    # # Get the differences between full and reduced datasets
    # df_differences_reduced_05 = compute_differences_in_reduced(
    #     datasets[("full", 2000, None)],
    #     #datasets[("reduced", 200, 0.5)],
    #     datasets[("oneshot", 200, 0.5)],
    #     all_feature_names,
    #     agg="median"
    # )

    # # Get the differences between full and reduced datasets
    # df_differences_reduced_025 = compute_differences_in_reduced(
    #     datasets[("full", 2000, None)],
    #     #datasets[("reduced", 200, 0.5)],
    #     datasets[("oneshot", 200, 0.25)],
    #     all_feature_names,
    #     agg="median"
    # )

    # # Get the differences between full and reduced datasets
    # df_differences_reduced_01 = compute_differences_in_reduced(
    #     datasets[("full", 2000, None)],
    #     #datasets[("reduced", 200, 0.5)],
    #     datasets[("oneshot", 200, 0.1)],
    #     all_feature_names,
    #     agg="median"
    # )

    df_differences_slices_0_05 = compute_differences_in_slices_0(
        datasets[("full", 2000, None)],
        datasets[("slices", 200, 0.5)],
        all_feature_names,
        agg="median"
    )

    # df_differences_slices_gen_05 = compute_differences_in_slices_general(
    #     datasets[("full", 2000, None)],
    #     datasets[("slices", 200, 0.5)],
    #     all_feature_names,
    #     agg="median"
    # )

    # Get the differences between full and slices datasets
    df_differences_slices_0_025 = compute_differences_in_slices_0(
        datasets[("full", 2000, None)],
        datasets[("slices", 200, 0.25)],
        all_feature_names,
        agg="median"
    )

    # df_differences_slices_gen_025 = compute_differences_in_slices_general(
    #     datasets[("full", 2000, None)],
    #     datasets[("slices", 200, 0.25)],
    #     all_feature_names,
    #     agg="median"
    # )

    # Get the differences between full and slices datasets
    df_differences_slices_0_01 = compute_differences_in_slices_0(
        datasets[("full", 2000, None)],
        datasets[("slices", 200, 0.1)],
        all_feature_names,
        agg="median"
    )

    # df_differences_slices_gen_01 = compute_differences_in_slices_general(
    #     datasets[("full", 2000, None)],
    #     datasets[("slices", 200, 0.1)],
    #     all_feature_names,
    #     agg="median"
    # )

    df_differences_slices_all_in_0_05 = compute_differences_in_slices_0(
        datasets[("full", 2000, None)],
        datasets[("slices_all_in", 200, 0.5)],
        all_feature_names,
        agg="median"
    )

    df_differences_slices_all_in_gen_0_05 = compute_differences_in_slices_general(
        datasets[("full", 2000, None)],
        datasets[("slices_all_in", 200, 0.5)],
        all_feature_names,
        agg="median"
    )

    df_differences_slices_all_in_0_025 = compute_differences_in_slices_0(
        datasets[("full", 2000, None)],
        datasets[("slices_all_in", 200, 0.25)],
        all_feature_names,
        agg="median"
    )

    df_differences_slices_all_in_gen_0_025 = compute_differences_in_slices_general(
        datasets[("full", 2000, None)],
        datasets[("slices_all_in", 200, 0.25)],
        all_feature_names,
        agg="median"
    )

    df_differences_slices_all_in_0_01 = compute_differences_in_slices_0(
        datasets[("full", 2000, None)],
        datasets[("slices_all_in", 200, 0.1)],
        all_feature_names,
        agg="median"
    )

    df_differences_slices_all_in_gen_0_01 = compute_differences_in_slices_general(
        datasets[("full", 2000, None)],
        datasets[("slices_all_in", 200, 0.1)],
        all_feature_names,
        agg="median"
    )

    # Make a combined dataset for plotting
    df_differences_list = combine_differences_results(
        [df_differences_full,
        df_differences_slices_0_05,
        df_differences_slices_0_025,
        df_differences_slices_0_01,
        df_differences_slices_all_in_0_05,
        df_differences_slices_all_in_0_025,
        df_differences_slices_all_in_0_01,
        df_differences_slices_all_in_gen_0_05,
        df_differences_slices_all_in_gen_0_025,
        df_differences_slices_all_in_gen_0_01],
        ["Full/ELA$_{\\mathrm{A}}$",
        "Sliced/ELA$_{\\mathrm{A}},r=0.5$",
        "Sliced/ELA$_{\\mathrm{A}},r=0.25$",
        "Sliced/ELA$_{\\mathrm{A}},r=0.1$",
        "All\_in/ELA$_{\\mathrm{A}},r=0.5$",
        "All\_in/ELA$_{\\mathrm{A}},r=0.25$",
        "All\_in/ELA$_{\\mathrm{A}},r=0.1$",
        "All\_in/ELA$_{\\mathrm{R}},r=0.5$",
        "All\_in/ELA$_{\\mathrm{R}},r=0.25$",
        "All\_in/ELA$_{\\mathrm{R}},r=0.1$"]
    )

    # Delete the individual datasets to save memory
    del df_differences_full, df_differences_slices_0_05, df_differences_slices_0_025
    del df_differences_slices_0_01, df_differences_slices_all_in_0_05, df_differences_slices_all_in_gen_0_05
    del df_differences_slices_all_in_0_025, df_differences_slices_all_in_gen_0_025, df_differences_slices_all_in_0_01, df_differences_slices_all_in_gen_0_01



    #fig, ax = box_plots_of_differences(
    #        df_differences_full,
    #        df_differences_reduced_05,
    #        all_feature_names,
    #        function_id =20,
    #    )
    

    #plt.show()

    # Get the plot

    for function_id in FUNCTION_IDS:
        for feature_name in all_feature_names:
            fig, ax = violin_plots_of_differences_global_2(
                df_differences_list,
                feature_name,
                function_id,
            )

            figure_path = (
            SAVE_FIGURE_DIRECTORY
            / f"function_id_{function_id}"
            / f"feature_{feature_name}"
            )

            figure_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(
            figure_path / "violin_plot_comparison_all_variants.pdf",
            dpi=300,
            bbox_inches="tight",
            )

            plt.close(fig)

    # for function_id in FUNCTION_IDS:
    #     fig, ax = box_plots_of_differences(
    #         df_differences_full,
    #         df_differences_reduced_05,
    #         all_feature_names,
    #         function_id =function_id,
    #     )

    #     figure_path = (
    #     SAVE_FIGURE_DIRECTORY
    #     / f"function_id_{function_id}"
    #     / f"comparison_full_vs_reduced"
    #     )

    #     figure_path.mkdir(parents=True, exist_ok=True)
    #     fig.savefig(
    #     figure_path / "box_plot_comparison_full_vs_reduced.pdf",
    #     dpi=300,
    #     #bbox_inches="tight",
    #     )

    #     plt.close(fig)



    a = 1


    # for function_id in FUNCTION_IDS:
    #     print(f"Processing function ID: {function_id}")


    #     for feature_name in all_feature_names:
    #         print(f"Processing feature: {feature_name}")

    #     for n_samples in DATA_SIZES:
    #         for ratio in REDUCTION_RATIOS:
    #             print(f"Processing feature: {feature_name}")

    #             df_full = datasets[("full", n_samples, None)]
    #             df_reduced = datasets[("reduced", n_samples, ratio)]

    #             if n_samples == 200 and ratio == 0.25:
    #                 continue

    #             plot_color = "skyblue"


    #             #fig, ax = plot_violin_plots_biased_3_v2(
    #             #df_full,
    #             #df_reduced,
    #             #feature_name_list=all_feature_names,
    #             #reduction_ratio=ratio,
    #             #data_size=n_samples,
    #             #function_id=function_id,
    #             #instance_id_list=INSTANCE_IDS,
    #             #show_fig=False,
    #             #fig_size=(16, 3),
    #             #plot_color=plot_color,
    #             #)


    #             if fig is None:
    #                 continue


    #             figure_path = (
    #             SAVE_FIGURE_DIRECTORY
    #             / f"function_id_{function_id}"
    #             / f"reduction_ratio_{ratio}"
    #             / f"n_samples_{n_samples}"
    #             )
    #             figure_path.mkdir(parents=True, exist_ok=True)


    #             fig.savefig(
    #             figure_path / "violin_plot_corrected.pdf",
    #             dpi=300,
    #             bbox_inches="tight",
    #             )
    #             plt.close(fig)


if __name__ == "__main__":
    main()