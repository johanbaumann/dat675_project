# CV Fold-Iteration Pipeline

This pipeline now runs **normal CV iterations from one fold directory**.

For each CV iteration:

1. One fold file is used as validation.
2. All remaining fold files are merged as training data.
3. One model is trained.
4. 10k molecules (or configured `num_unique`) are sampled.
5. Optional analysis runs.

## Analysis-only mode (no resampling)

You can now run with:

- `train.enabled=false`
- `sampling.enabled=false`
- `analysis.enabled=true`

In this mode, the pipeline reuses existing per-fold files from:

- `artifacts_output_root/cv_iteration_<k>/generated/generated.csv`
- `artifacts_output_root/cv_iteration_<k>/generated/quality_summary.csv`

Behavior in analysis-only mode:

- Per-fold analysis still runs for each discovered CV iteration.
- A hard fail is triggered if `generated.csv` is missing or has zero data rows.
- Missing `quality_summary.csv` does not stop per-fold analysis, but that fold is skipped for cross-fold V.U.N aggregation.
- The runner writes a cross-fold aggregate file:
  - `artifacts_output_root/cross_fold_analysis_summary.json`

Per-iteration analysis summary now also includes V.U.N:

- `artifacts_output_root/cv_iteration_<k>/analysis/analysis_summary.json`
- Field: `summary.vun`

`summary.vun` source priority:

1. `generated/quality_summary.csv` (preferred, count-based V.U.N matching sampling stats)
2. Fallback computed from loaded analysis data (`train` + `generated`) when quality summary is missing

Included keys:

- `source`
- `quality_summary_csv_path`
- `quality_run_scope`
- `quality_counts`
- `validity`, `uniqueness`, `novelty`, `acceptance_rate`
- `valid_count`, `unique_count`, `novel_count`

Cross-fold aggregate includes:

- V.U.N metrics aggregated from quality-summary counts (`validity`, `uniqueness`, `novelty`, `acceptance_rate`)
- Diversity aggregated from per-fold analysis summaries (same definition as analysis pipeline: `1 - mean_tanimoto_all_pairs`)
- Per-fold metric snapshot table
- Combined CV metric plots under `artifacts_output_root/cv_combo/`:
  - `cv_combo_metrics_std.png` (4 subplots: validity, uniqueness, novelty, diversity with std bars)
  - `cv_combo_metrics_boxplots.png` (4 subplots: one boxplot per metric across folds)
  - `cv_combo_metrics_stats.json` (mean/std/count + per-fold values)

## CV combo-only mode

You can run only the cross-fold combo plots (no fold discovery, no train/sampling/analysis) by setting:

- `cv_combo.enabled=true`
- `cv_combo.only=true`

Optional:

- `cv_combo.cross_fold_summary_path`: explicit path to `cross_fold_analysis_summary.json`.
  - If omitted, defaults to `artifacts_output_root/cross_fold_analysis_summary.json`.

In combo-only mode, the runner writes/updates:

- `artifacts_output_root/cv_combo/cv_combo_metrics_std.png`
- `artifacts_output_root/cv_combo/cv_combo_metrics_boxplots.png`
- `artifacts_output_root/cv_combo/cv_combo_metrics_stats.json`
- `artifacts_output_root/global_manifest.json` (with combo output paths)

## Distribution plot behavior (analysis)

Per-fold analysis now writes a single high-contrast property distribution plot:

- `property_distribution.png` shows only target-property distributions
  (for BACE, this is `pIC50`) for train/validation/generated.
- Generated is drawn semi-transparent with an outline.
- Train and validation are drawn as high-contrast line histograms so they remain visible when generated dominates.

## Iteration behavior

If you have `fold_0.csv ... fold_4.csv`, the pipeline creates 5 iterations:

- Iteration 0: validation=`fold_0`, training=`fold_1, fold_2, fold_3, fold_4`
- Iteration 1: validation=`fold_1`, training=`fold_0, fold_2, fold_3, fold_4`
- ... and so on.

At startup of each iteration, the runner prints this assignment in an easy-to-read list.

## Files

- `run_fold_pipeline.py`: CV iteration orchestrator.
- `fold_data.py`: discovers fold CSVs and builds train/validation prop files per iteration.
- `sampling_pipeline.py`: sampling + quality stats + scaffold exclusion.
- `fold_pipeline_config.example.json`: simplified reference config.

Internal helper naming is iteration-first:

- `*_for_iteration` (not `*_for_fold`)
- manifests and output folders use `cv_iteration_<k>`

## Run

From workspace root:

```powershell
python fold_pipeline/run_fold_pipeline.py --config fold_pipeline/fold_pipeline_config.example.json
```

Run one iteration only:

```powershell
python fold_pipeline/run_fold_pipeline.py --config fold_pipeline/fold_pipeline_config.example.json --only-fold 0
```

## Config (simplified)

Top-level fold input now uses one path only:

- `train_validation_folds_dir`: folder containing all fold CSV files.
- `fold_glob`: filename pattern (for example `fold_*.csv`).

Other key paths:

- `training_output_root`: checkpoints/history output.
- `artifacts_output_root`: per-iteration data/sampling/logs/manifests.

Combo-only keys:

- `cv_combo.enabled`: enable combo plotting.
- `cv_combo.only`: run combo-only mode and skip per-fold pipeline.
- `cv_combo.cross_fold_summary_path`: optional override path for the summary JSON.

## Heldout usage and scaffold exclusion

Heldout set is used **only for scaffold exclusion**.

Key options:

- `sampling.exclude_training`: rejects molecules seen in training data.
- `sampling.exclude_validation_scaffolds`: rejects scaffold overlap with current validation fold.
- `sampling.exclude_heldout_scaffolds`: rejects scaffold overlap with heldout test set.
- `sampling.heldout_smiles_csv`: heldout CSV path.
- `sampling.validation_smiles_column`, `sampling.heldout_smiles_column`: SMILES column names.
- `sampling.scaffold_make_generic`:
  - `true` => generic Murcko scaffold
  - `false` => specific Murcko scaffold

## Important debug output

Per iteration, logs print:

- iteration index,
- validation fold name,
- all training fold names,
- training and validation row counts,
- scaffold exclusion sources and blocked scaffold counts,
- periodic rejection counters during sampling,
- per-iteration manifest location.

## Output layout

Per iteration under `artifacts_output_root/cv_iteration_<k>/`:

- `data/`
- `sampling/`
- `analysis/` (if enabled)
- `logs/`
- `iteration_manifest.json`

Training artifacts under `training_output_root/cv_iteration_<k>/training/`.

Global run manifest:

- `artifacts_output_root/global_manifest.json`

Additional aggregate (when `analysis.enabled=true` and `cv_combo.enabled=true`):

- `artifacts_output_root/cross_fold_analysis_summary.json`
- `artifacts_output_root/cv_combo/cv_combo_metrics_std.png`
- `artifacts_output_root/cv_combo/cv_combo_metrics_boxplots.png`
- `artifacts_output_root/cv_combo/cv_combo_metrics_stats.json`
