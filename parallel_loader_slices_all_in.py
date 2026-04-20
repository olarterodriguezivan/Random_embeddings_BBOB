#!/usr/bin/env python3
"""
Parallel loader for reduced CSV feature files.

- Parallelizes CSV I/O + parsing
- Uses fixed dtypes
- Injects typed metadata
- Concatenates safely in chunks
- Streams output to Parquet

Designed for ~1M tiny CSVs on a 16 GB machine.
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

N_WORKERS = 5                 # Safe on 16 GB RAM
CHUNK_SIZE = 10_000           # Files per chunk
OUTPUT_FILE = "slices_200_all_in_0.5.parquet"

# Metadata dtypes (must be defined by you)
META_DTYPES = {
    "dimension": "int16",
    "group_id": "int32",
    "n_samples": "int32",
    "function_idx": "int16",
    "instance_idx": "int32",
    "reduction_ratio": "float32",
    "slice_id": "int32",
}
# ============================================================
# GLOBALS (initialized in main / workers)
# ============================================================

FEATURE_DTYPES = None

# ============================================================
# USER FUNCTIONS (you provide implementations)
# ============================================================

def extract_meta_data_from_reduced_feature_file_path(
    file_path: Union[str, Path],
    *,
    n_samples: int = 200,
) -> Dict[str, int | float]:
    """
    Extract key-value numeric metadata from a reduced feature file path.

    Expected path structure (example):
        sampling_XXX_10D_2D/function3/group1_instance7/slice4.csv
        sampling_XXX_10D_2D/function3/group1_instance7/full.csv
    """

    if not isinstance(file_path, (str, Path)):
        raise TypeError("file_path must be a string or Path object")

    file_path = Path(file_path)

    metadata: Dict[str, int | float] = {}

    # ----------------------------
    # Slice ID
    # ----------------------------
    file_name = file_path.name
    if file_name == "full.csv":
        metadata["slice_id"] = 0
    else:
        try:
            metadata["slice_id"] = int(
                file_name.replace("slice", "").replace(".csv", "")
            )
        except ValueError as exc:
            raise ValueError(f"Invalid slice file name: {file_name}") from exc

    # ----------------------------
    # Group ID & Instance Index
    # ----------------------------
    group_part = file_path.parts[-2]  # e.g. "group1_instance7"
    instance_part = file_path.parts[-3]  # e.g. "group1_instance7"

    try:
        
        metadata["group_id"] = int(group_part.removeprefix("group"))
        metadata["instance_idx"] = int(instance_part.removeprefix("iid_"))
    except Exception as exc:
        raise ValueError(
            f"Invalid group/instance segment: {group_part} / {instance_part}"
        ) from exc

    # ----------------------------
    # Function Index
    # ----------------------------
    function_part = file_path.parts[-4]  # e.g. "function3"
    try:
        metadata["function_idx"] = int(function_part.removeprefix("f"))
    except ValueError as exc:
        raise ValueError(f"Invalid function segment: {function_part}") from exc

    # ----------------------------
    # Sampling / Dimensions
    # ----------------------------
    sampling_part = file_path.parts[-5]
    splits = sampling_part.split("_")

    try:
        ambient_dimension = int(splits[-2].removesuffix("D"))
        reduced_dimension = int(splits[-1].removesuffix("D"))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid sampling segment: {sampling_part}") from exc

    metadata["dimension"] = ambient_dimension
    metadata["reduction_ratio"] = reduced_dimension / ambient_dimension
    metadata["n_samples"] = n_samples

    return metadata


def load_reduced(fp):
    """
    Load one reduced CSV and inject typed metadata.
    Executed inside worker processes.
    """
    df = pd.read_csv(
        fp,
        dtype=FEATURE_DTYPES,
        engine="c",
        low_memory=False
    )

    meta = extract_meta_data_from_reduced_feature_file_path(fp)

    for k, v in meta.items():
        df[k] = pd.Series(v, dtype=META_DTYPES[k], index=df.index)

    return df

# ============================================================
# MULTIPROCESSING UTILITIES
# ============================================================

def init_worker(feature_dtypes):
    """
    Initialize worker globals (runs once per process).
    """
    global FEATURE_DTYPES
    FEATURE_DTYPES = feature_dtypes


def chunked(iterable, size):
    """
    Yield fixed-size chunks from iterable.
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


def load_chunk_parallel(file_chunk, feature_dtypes):
    """
    Load a chunk of files in parallel and concat locally.
    """
    with mp.Pool(
        processes=N_WORKERS,
        initializer=init_worker,
        initargs=(feature_dtypes,)
    ) as pool:
        dfs = pool.map(load_reduced, file_chunk)

    return pd.concat(dfs, ignore_index=True, copy=False)

# ============================================================
# MAIN PIPELINE
# ============================================================

def build_complete_schema(example_files):
    all_cols = set()

    for f in example_files:
        cols = pd.read_csv(f, nrows=0).columns
        all_cols.update(cols)

    feature_dtypes = {c: "float32" for c in sorted(all_cols)}  # or your true types

    return feature_dtypes


def main(reduced_files):
    reduced_files = list(map(Path, reduced_files))

    if not reduced_files:
        raise ValueError("No input files provided.")

    # --------------------------------------------------------
    # Build COMPLETE feature schema ONCE
    # --------------------------------------------------------
    feature_dtypes = build_complete_schema(reduced_files)

    # Full column order (features first, then metadata)
    full_columns = list(feature_dtypes.keys()) + list(META_DTYPES.keys())

    writer = None
    arrow_schema = None

    for i, file_chunk in enumerate(chunked(reduced_files, CHUNK_SIZE), start=1):
        print(f"[Chunk {i}] Loading {len(file_chunk)} files...")

        chunk_df = load_chunk_parallel(file_chunk, feature_dtypes)

        # ----------------------------------------------------
        # Ensure full schema (add missing columns)
        # ----------------------------------------------------
        chunk_df = chunk_df.reindex(columns=full_columns)

        # ----------------------------------------------------
        # Enforce feature dtypes
        # ----------------------------------------------------
        for col, dtype in feature_dtypes.items():
            if col in chunk_df.columns:
                chunk_df[col] = chunk_df[col].astype(dtype, copy=False)

        # ----------------------------------------------------
        # Enforce metadata dtypes
        # ----------------------------------------------------
        for col, dtype in META_DTYPES.items():
            if col in chunk_df.columns:
                chunk_df[col] = chunk_df[col].astype(dtype, copy=False)

        # ----------------------------------------------------
        # Convert to Arrow table
        # ----------------------------------------------------
        table = pa.Table.from_pandas(
            chunk_df,
            preserve_index=False
        )

        # ----------------------------------------------------
        # Initialize writer once with locked schema
        # ----------------------------------------------------
        if writer is None:
            arrow_schema = table.schema
            writer = pq.ParquetWriter(OUTPUT_FILE, arrow_schema)

        # ----------------------------------------------------
        # Validate schema consistency
        # ----------------------------------------------------
        if table.schema != arrow_schema:
            print("❌ SCHEMA MISMATCH DETECTED")
            print("Expected schema:")
            print(arrow_schema)
            print("Current schema:")
            print(table.schema)
            raise RuntimeError("Chunk schema does not match initial schema.")

        writer.write_table(table)

        del chunk_df
        gc.collect()

    # --------------------------------------------------------
    # Close writer AFTER loop
    # --------------------------------------------------------
    if writer is not None:
        writer.close()

    print("✔ Finished successfully.")
# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Example usage:
    # reduced_files = sorted(Path("data").glob("*.csv"))
    #reduced_files = [...]

    ela_features_reduced_path = Path("sampling_outputs_all_in_20D_10D").resolve()
    slices_files = sorted(ela_features_reduced_path.rglob("*.csv"))

    main(slices_files)