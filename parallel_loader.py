#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunked, memory-safe version of plotting_partial.ipynb pipeline.

- Loads CSV files in chunks
- Extracts metadata from file paths
- Appends metadata to each chunk
- Writes a single CSV + Parquet dataset incrementally
"""

from pathlib import Path
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("./ela_features_2").resolve()

OUTPUT_PREFIX = "complete_data_generated"

CHUNKSIZE = 100_000  # adjust based on RAM / IO

# ============================================================
# END CONFIG
# ============================================================


def extract_meta_data_from_complete_feature_file_path(file_path: Path) -> dict:
    """
    Extract metadata from directory structure.

    Expected hierarchy:
        dimension_x/
          seed_y/
            n_samples_z/
              function_k/
                instance_i/
                  file.csv
    """
    metadata = {}

    parts = file_path.parts

    metadata["instance_idx"] = int(parts[-2].split("_")[-1])
    metadata["function_idx"] = int(parts[-3].split("_")[-1])
    metadata["n_samples"] = int(parts[-4].split("_")[-1])
    metadata["seed_lhs"] = int(parts[-5].split("_")[-1])
    metadata["dimension"] = int(parts[-6].split("_")[-1])

    return metadata


def main():
    print("=" * 70)
    print("Building complete_data (chunked)")
    print("=" * 70)

    csv_files = sorted(DATA_DIR.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}")

    csv_out = Path(f"{OUTPUT_PREFIX}.csv")
    parquet_out = Path(f"{OUTPUT_PREFIX}.parquet")

    # Remove old outputs if they exist
    csv_out.unlink(missing_ok=True)
    parquet_out.unlink(missing_ok=True)

    first_chunk = True
    parquet_chunks = []

    for f in csv_files:
        print(f"Processing file: {f}")

        file_metadata = extract_meta_data_from_complete_feature_file_path(f)

        for chunk in pd.read_csv(f, chunksize=CHUNKSIZE):
            # ----------------------------------------------
            # Metadata injection (identical to notebook)
            # ----------------------------------------------
            for key, value in file_metadata.items():
                chunk[key] = value

            chunk["source_file"] = f.name

            # ----------------------------------------------
            # CSV: stream append
            # ----------------------------------------------
            chunk.to_csv(
                csv_out,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )

            # ----------------------------------------------
            # Parquet: collect chunks (fast + schema-safe)
            # ----------------------------------------------
            parquet_chunks.append(chunk)

            first_chunk = False

    # Write parquet once (much faster & safer)
    print("Writing Parquet file...")
    complete_data = pd.concat(parquet_chunks, ignore_index=True)
    complete_data.to_parquet(parquet_out, index=False)

    print("Done ✔")
    print(f"CSV     : {csv_out}")
    print(f"Parquet : {parquet_out}")


if __name__ == "__main__":
    main()
