# Prediction-Powered Inference on California Housing

A small notebook experiment that compares Prediction-Powered Inference (PPI) confidence intervals against classical confidence intervals when estimating a mean.

## What it does
Estimates the mean median-house-value of the scikit-learn California Housing dataset using only a small number of labeled examples plus model predictions on the rest, and checks how PPI intervals compare to classical intervals as the labeled sample size grows.

## Approach (from `ppi_test_on_fetch_california_housing.ipynb`)
1. Load `fetch_california_housing` from scikit-learn.
2. Split 30% train / 70% inference (`random_state=42`).
3. Train a `RandomForestRegressor` (100 trees) on the train split and predict on the inference split.
4. Treat the inference split as the pool: for 10 labeled-sample sizes (`np.linspace(50, 500, 10)`) and 100 random trials each (alpha = 0.05), compute
   - the PPI interval with `ppi_py.ppi_mean_ci`
   - the classical interval with `ppi_py.classical_mean_ci`
5. Collect results in a DataFrame and plot with `make_plots` (imported from a `utils` module, saved to `./plots/california_housing.pdf`).

`output.png` is the saved result figure from this run.

## Tech stack
Python, NumPy, pandas, scikit-learn, `ppi_py` (prediction-powered inference), tqdm, Jupyter.

## Structure
```
ppi_test_on_fetch_california_housing.ipynb   # experiment
output.png                                    # result figure
```

## Setup
No requirements file is included. Install the imports used: `numpy pandas scikit-learn tqdm ppi_python jupyter`. The notebook also does `from utils import *`, which comes from the upstream PPI examples repo and is **not** in this repo, so you must add a `utils.py` providing `make_plots` before it will run.

## Limitations
- Missing `utils.py` and requirements file; not runnable as-is.
- Numeric results are not printed in the notebook; the evidence is the figure only.
- Single dataset and single model; an exploratory test, not a library.
