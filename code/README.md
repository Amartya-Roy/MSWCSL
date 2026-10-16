# Code

This directory contains the full implementation of MSWCSL and the benchmark
harness used in the paper.

## Directory layout

```
code/
├── mswcsl/          MSWCSL algorithm package
├── mswcsl_bench/    Benchmark harness: dataset loaders, comparator adapters,
│                    scoring, table/figure builders
├── scripts/         Entry-point scripts for running and summarising experiments
└── requirements.txt Tested package versions
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt
```

The packages `mswcsl` and `mswcsl_bench` are loaded from `code/` via the
scripts' `sys.path`; no `pip install -e` step is required.

## Data

Set the environment variable `MSWCSL_DATA_ROOT` to the directory that contains
the unpacked benchmark datasets before running any script that uses real data:

```bash
export MSWCSL_DATA_ROOT=/path/to/data/unpacked/mswcsl_data
```

See `data/README.md` for acquisition instructions for each dataset.

## Running experiments

All scripts are in `code/scripts/`. Run them from the repository root:

```bash
# Independent validation (persistence + cancellation)
python code/scripts/run_independent_validation.py

# Selection-aware study
python code/scripts/run_selection_aware.py

# CauseMe-style baselines
python code/scripts/run_causeme_baselines.py

# Review diagnostics
python code/scripts/run_review_diagnostics.py
```

## Rebuilding paper figures and tables

```bash
python code/scripts/build_final_figures.py    # figures/benchmark_regimes.pdf etc.
python code/scripts/build_native_tables.py    # tex/tables/benchmark_main_*.tex
python code/scripts/build_finance_analysis.py # tex/tables/finance_failure.tex
```

## Integrity audit

```bash
python code/scripts/audit_final_pass.py
```

Prints a JSON report verifying that figure data, table numerics, fMRI F1
values, and PDF page count match the archived snapshot.
