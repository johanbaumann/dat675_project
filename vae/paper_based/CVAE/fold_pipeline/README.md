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

The fold runner supports a split layout so big training checkpoints stay under `save/` (gitignored) while everything else stays outside:

**Artifacts (typically outside `save/`)** — under `artifacts_output_root/fold_<k>/`:

- `data/` (converted prop txt + data manifest)
- `sampling/` (generated CSV + quality summary)
- `analysis/` (analysis outputs + per-fold analysis config)
- `logs/` (train/analysis subprocess logs)
- `fold_manifest.json`

**Training (typically under `save/`)** — under `training_output_root/fold_<k>/`:

- `train_config.json` (the exact config used to train that fold)
- `training/` (checkpoints + history + training_config)

A global manifest is saved under `artifacts_output_root/`:

- `global_manifest.json`

## Logging

- Training and analysis subprocess logs are streamed live to console.
- The same output is also written to per-fold log files:
	- `.../fold_<k>/logs/train.log`
	- `.../fold_<k>/logs/analysis.log`

## Stage Toggles

The pipeline can toggle major stages directly from config:

- `train.enabled`: run or skip training per fold.
- `sampling.enabled`: run or skip sampling per fold.
- `analysis.enabled`: run or skip analysis per fold.

Notes:

- If `analysis.enabled=true`, sampling must also be enabled (analysis consumes generated CSV output).
- Skipping training is supported when checkpoints already exist in each fold's `training/` directory.

## Presets

The runner supports config presets for one-switch behavior:

- `pipeline_preset`: name of the preset to apply.
- `presets`: mapping of preset name -> config overrides.

Example included in `fold_pipeline_config.example.json`:

- `quiet_pipeline`: keeps training and sampling enabled, suppresses RDKit parse spam, enables test-scaffold exclusion, and disables analysis.

Set `pipeline_preset` to `null` (or remove it) to run raw top-level settings without preset overrides.

## Sampling Noise + Scaffold Controls

Sampling config supports two quality-of-life controls:

- `sampling.suppress_rdkit_parse_errors` (default `true`): hides noisy RDKit parse error spam while preserving all validity/quality counters.
- `sampling.exclude_test_scaffolds` (default `false`): rejects generated molecules whose Murcko scaffold appears in the test fold.

When scaffold exclusion is enabled, the fold runner automatically uses that fold's test CSV as scaffold source (unless `sampling.test_scaffold_csv` is set explicitly).

## Training-Distribution Sampling (Fold Pipeline)

Fold sampling supports the same training-distribution mode as `sample_labels.py`:

- `sampling.run_training_dist`: when true, each sampled molecule uses a per-sample target drawn from training property statistics.
- `sampling.training_dist_std_scale`
- `sampling.training_dist_clip_n_std`
- `sampling.training_dist_seed`

In this mode, generated target columns vary per molecule (for example `target_pIC50`) rather than being one fixed value for all rows.

## Generated CSV Column Control

Sampling supports explicit output schema control via:

- `sampling.generated_outputs`: list of output columns to keep in `generated.csv`.

Example:

- `"generated_outputs": ["smiles", "pred_pIC50"]`

Notes:

- Column names must exist at runtime; invalid names raise a clear error.
- Predicted-column naming is now sourced from property metadata sidecars produced during fold conversion, so one-property BACE runs use `pred_pIC50` (not generic `pred_prop_0`).

Additional save toggles:

- `sampling.save_generated_csv`: write/skip generated CSV.
- `sampling.save_quality_summary`: write/skip quality summary CSV.

If analysis is enabled, `sampling.save_generated_csv` must be true.
