# Fold Pipeline Runner

This folder contains a standalone modular runner for 5x5 fold workflows:

1. Convert train/test fold CSV files into training property txt files.
2. Train CVAE on train fold and validate on test fold (no internal random split).
3. Sample molecules from the trained checkpoint.
4. Run analysis/statistics pipeline on generated molecules.

No fold mixing:
- Train fold is always loaded from `combination_1000.../fold_iteration_k.csv`.
- Test fold is always loaded from `combination_500.../fold_iteration_k.csv`.
- `train_labels.py` runs in external-split mode (`data.test_prop_file`) so internal random splitting is bypassed.

## Files

- `run_fold_pipeline.py`: top-level orchestrator.
- `fold_data.py`: fold discovery + CSV-to-prop conversion utilities.
- `sampling_pipeline.py`: checkpoint restore + molecule sampling + quality summary.
- `fold_pipeline_config.example.json`: example configuration.

## Usage

From workspace root:

```powershell
python fold_pipeline/run_fold_pipeline.py --config fold_pipeline/fold_pipeline_config.example.json
```

Run a single fold:

```powershell
python fold_pipeline/run_fold_pipeline.py --config fold_pipeline/fold_pipeline_config.example.json --only-fold 0
```

## Output Layout

For each fold, artifacts are grouped in one folder:

- `.../fold_<k>/data/` (converted prop txt + data manifest)
- `.../fold_<k>/training/` (checkpoints + history + training_config)
- `.../fold_<k>/sampling/` (generated CSV + quality summary + sampling debug)
- `.../fold_<k>/analysis/` (analysis outputs + per-fold analysis config)
- `.../fold_<k>/logs/` (train/analysis subprocess logs)
- `.../fold_<k>/fold_manifest.json`

A global manifest is saved at:

- `.../global_manifest.json`

## Logging

- Training and analysis subprocess logs are streamed live to console.
- The same output is also written to per-fold log files:
	- `.../fold_<k>/logs/train.log`
	- `.../fold_<k>/logs/analysis.log`
