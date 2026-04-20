# BBOB Data Generation Pipeline

This repository implements a **three-step pipeline** for generating, processing, and using datasets derived from **BBOB (Black-Box Optimization Benchmark) functions**.

This README currently documents **Step 1**, which is responsible for **data generation via Design of Experiments (DoE)**.  
Steps 2 and 3 will build on the data generated here.

---

## Step 1 — Data Generation (`sampler.py`)

The first step of the pipeline samples BBOB benchmark functions using different DoE strategies and logs the resulting datasets to disk.

This step is implemented in `sampler.py`

---

## Purpose

`sampler.py` generates datasets by:

1. Selecting a BBOB function (problem ID, dimension, instance),
2. Sampling points in the decision space using a chosen DoE method,
3. Scaling samples to the true BBOB bounds,
4. Evaluating the BBOB function at those points,
5. Logging inputs and function values using IOH's `Analyzer`.

The script is designed to be executed from the command line and produces structured, reproducible datasets.

---

## Dependencies

The following Python packages are required:

- `numpy`
- `scipy`
- `ioh`
- `pflacco`
- `scikit-learn`

Install them with:
```
pip install numpy scipy ioh pflacco scikit-learn
```

---

## Basic Usage

Run the sampler from the repository root:
```
python sampler.py [OPTIONS]
```

Minimal example:
```
python sampler.py --problem-id 1 --dimension 5 --sampler lhs
```

This command:
- Selects **BBOB problem 1**
- Uses **5 dimensions**
- Applies **Latin Hypercube Sampling**
- Generates `dimension × multiplier` samples

---

## Command-Line Arguments

### BBOB Problem Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--problem-id` | BBOB function ID (1–24) | `1` |
| `--dimension` | Problem dimensionality | `2` |
| `--instance` | BBOB instance ID (1–15) | `1` |

---

### Sampling Configuration

| Argument | Description | Default |
|----------|-------------|---------|
| `--sampler` | Sampling method: `monte-carlo`, `lhs`, `sobol`, `halton` | `lhs` |
| `--multiplier` | Number of samples = `dimension × multiplier` | `25` |
| `--random-seed` | Random seed for reproducibility | `42` |

---

### Sampler-Specific Options

These options affect LHS, Sobol, and Halton samplers.

| Argument | Description | Default |
|----------|-------------|---------|
| `--quasi-random-criterion` | Optimization criterion (`random-cd`, `lloyd`) | `random-cd` |
| `--lhs-strength` | Strength of LHS design (1 or 2) | `1` |

**Note:**  
Sobol sampling automatically rounds the number of samples **up to the nearest power of two**, as required by the Sobol sequence construction.

---

## Supported Sampling Methods

- **Monte Carlo**  
  Uniform random sampling over the unit hypercube.

- **Latin Hypercube Sampling (LHS)**  
  Stratified sampling using `scipy.stats.qmc.LatinHypercube`, with optional optimization.

- **Sobol Sequences**  
  Low-discrepancy quasi-random sequences with scrambling and optimization.

- **Halton Sequences**  
  Deterministic low-discrepancy sequences with optional scrambling.

All sampling methods initially generate points in `[0, 1]^d`, which are then scaled to the BBOB problem bounds.

---

## Output Structure

Results are logged automatically using IOH's `Analyzer`.

The output directory structure is:
```
data/
└── <problem_id>/
```
