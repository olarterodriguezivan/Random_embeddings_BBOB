#!/usr/bin/env python3
"""
Parallel loader for full-space CSV feature files.

This is the no-slices version of parallel_loader_slices.py.

Expected output structure from the full-space sampler:
    sampling_outputs_20D_fullspace/f3/iid_7/group1/full.csv

Main changes relative to the slice loader:
- only loads full.csv files
- removes slice_id metadata
- uses reduction_ratio = 1.0
- parses dimensions from folder names like sampling_outputs_20D_fullspace
- keeps the same chunked multiprocessing + Parquet streaming logic
"""

import gc
import multiprocessing as mp
from itertools import islice
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ============================================================
# USER CONFIGURATION
# ============================================================


N_WORKERS = 2                 # Safe on 16 GB RAM
CHUNK_SIZE = 10_000           # Files per chunk
N_SAMPLES = 1000                # For metadata only, not used in loading
EMBEDDED_D = 2                # For metadata only, not used in loading
OUTPUT_FILE = f"fullspace_{N_SAMPLES}_sub_{EMBEDDED_D}D.parquet"


# Metadata dtypes
META_DTYPES = {
    "dimension": "int16",
    "group_id": "int32",
    "n_samples": "int32",
    "function_idx": "int16",
    "instance_idx": "int32",
    "reduction_ratio": "float32",
    "embedded_dimension": "int16",
}

# ============================================================
# GLOBALS initialized in workers
# ============================================================

FEATURE_DTYPES = None

# ============================================================
# METADATA PARSING
# ============================================================

def _parse_dimension_from_sampling_folder(folder_name: str) -> int:
    """
    Extract D from folder names such as:
        sampling_outputs_20D_fullspace
        sampling_outputs_20D
        sampling_outputs_20D_2D

    For full-space data, only the ambient/full dimension matters.
    """
    for token in folder_name.split("_"):
        if token.endswith("D"):
            return int(token.removesuffix("D"))
    raise ValueError(f"Could not parse dimension from folder name: {folder_name}")


def extract_meta_data_from_fullspace_feature_file_path(
    file_path: Union[str, Path],
    *,
    n_samples: int = N_SAMPLES,
    embedded_dimension: int = EMBEDDED_D,
) -> Dict[str, int | float]:
    """
    Extract numeric metadata from a full-space feature CSV path.

    Expected path structure:
        sampling_outputs_20D_fullspace/f{fid}/iid_{iid}/group{group_id}/full.csv

    Example:
        sampling_outputs_20D_fullspace/f3/iid_7/group1/full.csv
    """
    if not isinstance(file_path, (str, Path)):
        raise TypeError("file_path must be a string or Path object")

    file_path = Path(file_path)

    if file_path.name != "full.csv":
        raise ValueError(f"Expected only full.csv files, got: {file_path.name}")

    metadata: Dict[str, int | float] = {}

    # Path pieces from right to left:
    #   ... / sampling_outputs_20D_fullspace / f3 / iid_7 / group1 / full.csv
    group_part = file_path.parts[-2]
    instance_part = file_path.parts[-3]
    function_part = file_path.parts[-4]
    sampling_part = file_path.parts[-5]

    try:
        metadata["group_id"] = int(group_part.removeprefix("group"))
        metadata["instance_idx"] = int(instance_part.removeprefix("iid_"))
        metadata["function_idx"] = int(function_part.removeprefix("f"))
    except ValueError as exc:
        raise ValueError(
            f"Invalid path segments: {function_part} / {instance_part} / {group_part}"
        ) from exc

    metadata["dimension"] = _parse_dimension_from_sampling_folder(sampling_part)
    metadata["reduction_ratio"] = 1.0
    metadata["n_samples"] = n_samples
    metadata["embedded_dimension"] = embedded_dimension

    return metadata


# ============================================================
# CSV LOADING
# ============================================================

def load_fullspace(fp):
    """
    Load one full-space CSV and inject typed metadata.
    Executed inside worker processes.
    """
    df = pd.read_csv(
        fp,
        dtype=FEATURE_DTYPES,
        engine="c",
        low_memory=False,
    )

    meta = extract_meta_data_from_fullspace_feature_file_path(fp)

    for k, v in meta.items():
        df[k] = pd.Series(v, dtype=META_DTYPES[k], index=df.index)

    return df


# ============================================================
# MULTIPROCESSING UTILITIES
# ============================================================

def init_worker(feature_dtypes):
    global FEATURE_DTYPES
    FEATURE_DTYPES = feature_dtypes


def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


def load_chunk_parallel(file_chunk, feature_dtypes):
    with mp.Pool(
        processes=N_WORKERS,
        initializer=init_worker,
        initargs=(feature_dtypes,),
    ) as pool:
        dfs = pool.map(load_fullspace, file_chunk)

    return pd.concat(dfs, ignore_index=True, copy=False)


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_complete_schema(example_files):
    all_cols = set()

    for f in example_files:
        cols = pd.read_csv(f, nrows=0).columns
        all_cols.update(cols)

    # ELA feature columns are numeric. Adjust here if any feature needs another dtype.
    feature_dtypes = {c: "float32" for c in sorted(all_cols)}
    return feature_dtypes


def main(fullspace_files):
    fullspace_files = [Path(fp) for fp in fullspace_files]

    if not fullspace_files:
        raise ValueError("No full-space CSV files provided.")

    # Safety filter: this loader is for the no-slices outputs only.
    fullspace_files = [fp for fp in fullspace_files if fp.name == "full.csv"]

    if not fullspace_files:
        raise ValueError("No full.csv files found after filtering.")

    feature_dtypes = build_complete_schema(fullspace_files)

    # Features first, metadata last.
    full_columns = list(feature_dtypes.keys()) + list(META_DTYPES.keys())

    writer = None
    arrow_schema = None

    try:
        for i, file_chunk in enumerate(chunked(fullspace_files, CHUNK_SIZE), start=1):
            print(f"[Chunk {i}] Loading {len(file_chunk)} full-space files...")

            chunk_df = load_chunk_parallel(file_chunk, feature_dtypes)

            # Ensure full schema, including missing feature columns.
            chunk_df = chunk_df.reindex(columns=full_columns)

            # Enforce feature dtypes.
            for col, dtype in feature_dtypes.items():
                if col in chunk_df.columns:
                    chunk_df[col] = chunk_df[col].astype(dtype, copy=False)

            # Enforce metadata dtypes.
            for col, dtype in META_DTYPES.items():
                if col in chunk_df.columns:
                    chunk_df[col] = chunk_df[col].astype(dtype, copy=False)

            table = pa.Table.from_pandas(chunk_df, preserve_index=False)

            if writer is None:
                arrow_schema = table.schema
                writer = pq.ParquetWriter(OUTPUT_FILE, arrow_schema)

            if table.schema != arrow_schema:
                print("❌ SCHEMA MISMATCH DETECTED")
                print("Expected schema:")
                print(arrow_schema)
                print("Current schema:")
                print(table.schema)
                raise RuntimeError("Chunk schema does not match initial schema.")

            writer.write_table(table)

            del chunk_df, table
            gc.collect()

    finally:
        if writer is not None:
            writer.close()

    print(f"✔ Finished successfully: {OUTPUT_FILE}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    full_path = Path(r"C:\Users\iolar\OneDrive - Universiteit Leiden\Documents\dimensionality_reduction repo\ela_dim_red\sampling_outputs_2D_fullspace_mult_500_groups40_seed43_normalizedTrue_True").resolve()

    
    fullspace_files = sorted(full_path.rglob("full.csv"))

    main(fullspace_files)
