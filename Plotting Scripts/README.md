# Plotting Scripts

This directory contains Python scripts for analysing and visualising **ELA (Exploratory Landscape Analysis) feature distributions** produced by the various sampling strategies implemented in this project (full-space sampling, random-embedding / one-shot reduction, sliced embeddings, and all-in embeddings). All scripts compare feature distributions against the high-density reference dataset (2000-sample full-space ELA features) and are formatted for publication-quality PDF output via LaTeX/pgf.

---

## File Overview

| Script | Purpose | Output directory |
|--------|---------|-----------------|
| `shifts_computations_with_slices.py` | Relative-shift (ratio) analysis with violin plots | `figures_barplots_slices_comparison_full/` |
| `wasserstein_1_distances_slices_stats.py` | Wasserstein-1 distance analysis + full statistical pipeline | `tables_wasserstein_2_distances_slices_stats/` |
| `wasserstein_1_distances_with_slices_refactored.py` | Refactored Wasserstein-1 violin plots (subset of methods) | `figures_wasserstein_refactored/` |

---

## Dependencies

All scripts share the same runtime dependencies as the rest of the project.  Install them from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

Additional packages used specifically by `wasserstein_1_distances_slices_stats.py`:

| Package | Purpose |
|---------|---------|
| `scikit-posthocs` | Nemenyi post-hoc test |
| `scipy` | Wasserstein distance, Wilcoxon test, Friedman test |

---

## Expected Input Data

All scripts expect pre-generated dataset files to be present **in the same directory** as the scripts (i.e. inside `Plotting Scripts/`). These files are produced by the pipeline in the repository root (see the main `README.md`).

| File pattern | Description |
|---|---|
| `complete_data_2.csv` | Full-space ELA features, 200 samples |
| `complete_data_generated.csv` | Full-space ELA features, 2000 samples |
| `reduced_1_200_{r}.parquet` | Reduced-space ELA features, 200 samples, ratio `r` ∈ {0.25, 0.5} |
| `reduced_2_2000_{r}.parquet` | Reduced-space ELA features, 2000 samples, ratio `r` ∈ {0.25, 0.5} |
| `reduced_oneshot_3_{n}_{r}.parquet` | One-shot embedding ELA features, size `n`, ratio `r` ∈ {0.1, 0.25, 0.5} |
| `slices_{n}_{r}.parquet` | Sliced embedding ELA features, size `n` = 200, ratio `r` ∈ {0.1, 0.25, 0.5} |
| `slices_{n}_all_in_{r}.parquet` | All-in embedding ELA features, size `n` = 200, ratio `r` ∈ {0.1, 0.25, 0.5} |

> **Note:** Slice datasets are only available for `n = 200`.

### Seed ranges

| Dataset size | LHS seeds considered |
|---|---|
| 200 | 1001 – 1040 |
| 2000 | 2001 – 2040 |

---

## Script Details

### `shifts_computations_with_slices.py`

**What it does:**

Computes per-feature, per-function **relative differences** (ratio shifts) between the 2000-sample reference and each alternative sampling strategy:

```
ratio = (feature_alt − feature_ref) / (|feature_ref| + ε)
```

The following method variants are compared:

| Method label | Description |
|---|---|
| `Full/ELA_A` | Full-space 200 vs 2000 samples |
| `Sliced/ELA_A, r=…` | Sliced embedding, slice 0 (aligned), ratio r |
| `All_in/ELA_A, r=…` | All-in embedding, slice 0 (aligned), ratio r |
| `All_in/ELA_R, r=…` | All-in embedding, remaining slices (random), ratio r |

For every (function, feature) pair a violin plot is saved showing the distribution of ratio shifts across all instances and seeds.

**Run:**

```bash
python "shifts_computations_with_slices.py"
```

**Output structure:**

```
figures_barplots_slices_comparison_full/
  function_id_{f}/
    feature_{feat}/
      violin_plot_comparison_all_variants.pdf
```

---

### `wasserstein_1_distances_slices_stats.py`

**What it does:**

The most comprehensive analysis script. It:

1. Computes **Wasserstein-1 distances** between the ELA feature distributions of each method and the 2000-sample reference, for all 24 BBOB functions, 15 instances, and all ELA features.
2. Performs **rank-based aggregation** of distances across features (per function) and across functions (per feature).
3. Runs a global **Friedman test** to check whether method rankings differ significantly.
4. Runs a **Nemenyi post-hoc test** to identify groups of statistically equivalent methods.
5. Runs **pairwise Wilcoxon tests** (best vs second-best per function–feature pair) with optional Holm p-value adjustment.
6. Produces **parallel coordinate plots** (average rank per function and per feature) and **Critical Difference diagrams**.
7. Produces a **colour-coded heatmap** of best/second-best methods across the function × feature grid.

A `MODE` constant (top of file) controls whether PCA features are included (`MODE=1`) or excluded (`MODE=2`).

**Run:**

```bash
python "wasserstein_1_distances_slices_stats.py"
```

**Output structure:**

```
tables_wasserstein_2_distances_slices_stats/
  best_method_per_function_mode_{MODE}.csv
  best_method_per_feature_mode_{MODE}.csv
  significance_best_vs_second_mode_{MODE}.csv
  friedman_per_feature_mode_{MODE}.csv
  nemenyi_matrix_per_function_mode_{MODE}.csv
  nemenyi_matrix_per_feature_mode_{MODE}.csv
  average_ranks_per_function_mode_{MODE}.csv
  average_ranks_per_feature_mode_{MODE}.csv
  wasserstein_ranking_parallel_function_{MODE}.pdf
  wasserstein_ranking_parallel_feature_{MODE}.pdf
  wasserstein_ranking_heatmap_mode_{MODE}.pdf
  wasserstein_ranking_heatmap_mode_{MODE}_r.pdf
  cd_diagram_mode_{MODE}_function.pdf
  cd_diagram_mode_{MODE}_feature.pdf
```

---

### `wasserstein_1_distances_with_slices_refactored.py`

**What it does:**

A cleaner, self-contained refactoring of the Wasserstein-1 distance computation. It loads the same datasets, computes per-(function, instance, feature) Wasserstein-1 distances between each method and the 2000-sample reference, and saves **one violin plot per (function, feature) pair** showing the distance distributions across instances for all methods.

Methods compared (configurable via `PLOT_METHOD_LABELS`):

| Key | Label |
|---|---|
| `full` | `Full/ELA_A` |
| `sliced_0_{r}` | `Sliced/ELA_A, r={r}` |
| `all_in_0_{r}` | `All_in/ELA_A, r={r}` |
| `all_in_gen_{r}` | `All_in/ELA_R, r={r}` |

**Run:**

```bash
python "wasserstein_1_distances_with_slices_refactored.py"
```

**Output structure:**

```
figures_wasserstein_refactored/
  function_id_{f}/
    feature_{feat}/
      violin_wasserstein.pdf
```

---

## Common Configuration Constants

All three scripts share the following top-level constants (edit in the script files to change behaviour):

| Constant | Default | Meaning |
|---|---|---|
| `FUNCTION_IDS` | 1 – 24 | BBOB functions to analyse |
| `INSTANCE_IDS` | 0 – 14 | BBOB instances to analyse |
| `DATA_SIZES` | [200, 2000] | Sample-count variants |
| `REDUCTION_RATIOS` | [0.5, 0.25, 0.1] | Embedding reduction ratios |
| `EPSILON` | 1e-9 | Numerical stability constant |
| `EXCLUDED_COLUMNS` | metadata + (optionally PCA) columns | Columns not treated as ELA features |

Plot rendering uses LaTeX fonts (`pdflatex`) for publication-ready output. If LaTeX is not installed, remove or comment out the `plt.rcParams.update(...)` block at the top of each script.
