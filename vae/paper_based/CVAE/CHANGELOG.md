# Changelog

All notable changes to this project are documented in this file.

## Unreleased

### Fixed

- `analysis_modules/config.py`: `load_analysis_config_from_file(...)` now ignores unknown config override keys before building `AnalysisConfig`, preventing crashes like `unexpected keyword argument 'print_vocab_size'` when runner-only toggles are present.

- `sample.py` / `debug_sampling.py`: Sampling now prefers the `model_config` embedded in the checkpoint (`.pt`) payload. This prevents a subtle failure mode where `save/training_config.json` gets overwritten by later training runs (often Transformer experiments), causing sampling to use the wrong `prop_file` / `seq_length` / `num_prop` and especially the wrong `prop_norm_mean/std` (=> invalid SMILES and/or near-zero acceptance).
- `model_labels.py`: Fixed a wiring bug where `include_condition_in_label_head=True` built a `(z, c)` label head but prediction still used `z` only.
- `sample.py`: Default decoding is stochastic again (`do_sample=True`). During Transformer refactors the default was set to greedy decoding, which commonly collapses to a single repeated molecule ("not unique") and can also reduce validity.
- `sample.py` / `utils.py`: Novelty and duplicate checks now use the same canonicalization pipeline for both generated molecules and training-set molecules, avoiding mismatches from inconsistent canonical forms.
- `utils.py`: Cleanup/standardization during filtering is now strict: if a requested cleanup step (salt stripping / uncharging / tautomer canonicalization) fails, the molecule is discarded instead of silently passing.
- `sample.py` / `sample_labels.py` / `utils_labels.py`: Cleanup-step failures are now counted explicitly as `discarded_cleanup` (instead of being lumped into `invalid_or_empty`), and quality summaries include the corresponding count and rate.
- `sample.py` / `debug_sampling.py` / `sweep_sampling.py`: Sampling startup no longer calls full `load_data(...)` just to get charset/vocab/num_prop. Scripts now use a lightweight metadata loader with cache, avoiding long startup stalls on very large property files.

### Added

- `fold_pipeline/` module: added a standalone cross-validation orchestrator (`run_fold_pipeline.py`) with modular helpers for fold discovery/conversion (`fold_data.py`) and sampling (`sampling_pipeline.py`). The runner executes train -> sample -> analysis per fold and stores all artifacts under `fold_<k>/`.

- `train_labels.py`: added runtime config override support via `--config-json` (and `TRAIN_LABELS_CONFIG_JSON` env fallback) to enable non-interactive orchestration without removing existing in-file config workflow.

- `train_labels.py`: added external split mode support through `data.test_prop_file`, including aligned tokenization for separate train/test files via shared vocabulary, preventing accidental fold mixing when scaffold-split files are provided.

- `sample_labels.py` / `debug_sampling.py` / `sweep_sampling.py`: Sampling startup now prints resolved sampling metadata including `vocab_size` (plus inferred `num_prop` and `prop_file`) so runs show the active vocabulary immediately.
- `run_viz_pipeline.py`: Added an optional, backward-compatible sampling-metadata print path; when `overrides.prop_file` and `overrides.seq_length` are present in the analysis config JSON, the runner prints inferred `vocab_size` before analysis starts.
- `run_viz_pipeline.py` / `analysis_run_config.json`: Added `overrides.print_vocab_size` toggle to enable/disable vocab-size printing without affecting analysis pipeline execution.
- `utils_labels.py` / `sample_labels.py`: Added output naming control `output.auto_quality_summary_filename`; when set to `False` and `quality_summary_filename=None`, sampling no longer auto-creates `<result>_quality_summary.csv`.

- `analysis_modules/` package and `run_viz_pipeline.py`: Added reusable Python modules that mirror the major `viz.ipynb` analysis flow (data loading, canonicalization, validity flags, Tanimoto-to-reference, diversity score, scaffold overlap, and CSV/JSON outputs) while keeping the notebook unchanged.
- `analysis_modules/config.py`: Added starter profile configs for both `zinc_logp` and `bace_pic50_10k` with explicit path knobs (`train_folder`, `train_data_path`, `generated_data_path`, `output_dir`) so runs can be re-pointed quickly.
- `run_viz_pipeline.py`: Added CLI/runtime overrides for profile paths and row caps so the same module pipeline works for the generated BACE pIC50 dataset and 10k generated-molecule files.
- `analysis_run_config.json`: Added a file-based starter run config so the whole analysis pipeline can be executed from config without passing per-run path arguments.
- `analysis_modules/config.py` / `analysis_modules/pipeline.py` / `analysis_run_config.json`: Added config-driven analysis debug mode (`debug`) that prints stage progress, resolved target/prediction columns, validity/similarity progress, and every written analysis artifact path.

- `train.py` / `utils.py`: Added run-folder controls directly in the training config (`training.run_name`, `training.use_run_subdir`). Training now resolves an effective run save path via `build_train_run_save_dir(...)`, so each run can write to its own subdirectory under `training.save_dir` without overwriting other runs.
- `utils.py`: Added `resolve_checkpoint_path(...)` to resolve checkpoints either from an explicit file path or from a run directory (preferring `model_best.ckpt-*.pt`, then falling back to newest `.pt`).
- `train_labels.py` / `model_labels.py`: Optional auxiliary label head that predicts selected properties from latent `z` (default: LogP-only for 2-prop `[MW, LogP]` setup).
- `train_labels.py` / `model_labels.py`: Added `include_condition_in_label_head` option so the label head can predict labels from `(z, c)` (latent + conditioning) instead of `z` only.
- `train_labels.py` / `sample_labels.py`: Added `label_targets_use_raw_scale` (saved as `label_target_scale`) so the label head can be trained/evaluated in raw property units instead of normalized units.
- `sample_labels.py`: Sampling output can include denormalized predicted label columns (e.g. `pred_LogP`) and a direct comparison column `pred_LogP_minus_rdkit_LogP`.
- `sample_labels.py`: Added `training_dist` sampling mode (`run_training_dist`) to sample conditioning targets from an approximate training-property distribution (Gaussian fit via saved `prop_norm_mean/std`). This keeps conditioning near the training manifold, which often improves label-head calibration around typical training values.

### Changed

- `fold_pipeline/run_fold_pipeline.py`: subprocess execution now streams logs live to console while simultaneously writing to per-fold log files (`logs/train.log`, `logs/analysis.log`) so epoch/loss diagnostics are visible during pipeline runs.

- `fold_pipeline/fold_pipeline_config.example.json`: removed `data.train_ratio` from the example fold config to avoid confusion in external split mode; fold pipeline now explicitly prints that train/test come from pre-split fold files and that random split is bypassed.

- `train_labels.py`: when `data.test_prop_file` is set, startup now logs that `train_ratio` is ignored and external split files are used directly.

- `run_viz_pipeline.py`: Runner now primarily loads settings from a JSON config file (`--config`, default `analysis_run_config.json`) instead of many path/column CLI arguments.
- `analysis_modules/pipeline.py`: Extended coverage to include notebook-equivalent train-loss plotting, Tanimoto histogram, chemical-space PCA+t-SNE, and descriptor-space PCA+t-SNE in addition to similarity/distribution/scaffold outputs.
- `analysis_modules/pipeline.py` / `analysis_modules/config.py`: Added notebook-parity scaffold grid outputs for top train and generated scaffolds (plus novel generated scaffolds), toggle-controlled via config.
- `analysis_modules/pipeline.py`: Added explicit scaffold stats artifact (`scaffold_stats.json`) with unique scaffold counts for train vs generated sets, overlap, and novel scaffold counts.
- `analysis_modules/pipeline.py`: Added prediction-error scatter plot (`absolute error vs ground truth`) and summary metrics (MSE/MAE/median/std) when target/predicted property columns are available.
- `analysis_modules/pipeline.py`: Corrected distribution plotting semantics so histogram y-axis is explicit `Count`, generated-property x-axis uses predicted property when available (e.g., `pred_pIC50`), and MW comparison now includes computed train MW fallback from SMILES when MW is not present in train file.
- `analysis_modules/pipeline.py` / `analysis_modules/config.py`: Added MW distribution-difference artifact (`mw_distribution_diff_train_minus_generated.png`) showing per-bin count delta (`train - generated`).

- `sample.py`: Runtime model config now supports `run_dir` + `checkpoint_glob` (in addition to `save_file`) and resolves checkpoint paths through shared utility logic.
- `debug_sampling.py` / `sweep_sampling.py`: Updated defaults to support run-folder based checkpoint selection, matching training output layout.
- `sample.py`: Added sweep-level quality reporting for the whole generated sweep (`WHOLE GENERATED SWEEP`) with V/U/N and detailed counters aggregated across all property pairs.
- `sample.py`: Added per-sweep-pair statistics export fields for downstream heatmaps, including acceptance and filtering counters.
- `sample.py`: Sampling now also persists a run-level quality summary CSV (default: `<result_filename>_quality_summary.csv`) containing aggregated V/U/N/Acceptance plus detailed not-ok breakdown counters and rates (`not_ok_count/rate`, `invalid_or_empty_rate`, `in_training_rate`, `duplicate_rate`, `rejected_by_filter_rate`) so sweep-level totals no longer need to be recomputed from per-pair rows.
- `sample.py`: Added explicit runtime canonicalization logging and counters in quality stats (`salt_stripped`, `tautomer_canonicalized`).
- `utils.py`: Added robust canonicalization helper for filtering/novelty (`canonicalize_for_filtering(...)`) with configurable salt stripping, decharge, and optional tautomer canonicalization.
- `train_labels.py`: Added an active, smaller default Transformer config block (while keeping the previous larger example block commented out) for easier baseline runs.
- `train_labels.py` / `utils.py`: Added configurable train-only SMILES augmentation via `data.smiles_augmentation_duplicates` (number of randomized SMILES variants generated per original training molecule).
- `train_labels.py`: Augmentation metadata is now persisted in `training_config.json` (`smiles_augmentation_duplicates`, train set size before/after augmentation) for run reproducibility.
- `utils.py`: Added persistence helpers: uncompressed `save_pickle(...)` / `load_pickle(...)` for optional outputs, and gzip-pickle helpers (`save_pickle_gz(...)` / `load_pickle_gz(...)`) reserved for internal caches.
- `utils.py`: Added `load_sampling_metadata(...)` for fast, cached extraction of charset/vocab/num_prop from large property files without constructing full training tensors.


- `sample_labels.py`: Added an explicit re-canonicalization pass of generated outputs right before descriptor evaluation/export to keep evaluation rows canonicalized and aligned with optional payload columns.
- `sample.py` / `sample_labels.py`: Optional molecule artifact outputs now use plain pickle (`.pkl`) instead of gzip pickles. Gzip pickles remain in use for internal caches only.

### Changed

- `sample.py`: Runtime sampling configuration is now nested by concern (`model`, `generation`, `sampling`, `filters`, `cleanup`, `sweep`, `output`) and composed into flat runtime keys internally for compatibility.
- `sample.py`: Cleanup controls (`strip_salts`, `decharge`, `canonicalize_tautomer`) are now first-class under `runtime_config['cleanup']` (with backward-compatible fallback if older configs still put them under `filters`).
- `utils.py`: Training-set canonical cache naming now includes canonicalization mode flags to prevent stale cache reuse when cleanup options change.
- `utils.py`: Canonicalization path was optimized for speed by default (parse/canonicalize + optional decharge; tautomer canonicalization remains optional due to runtime cost).
- `utils.py`: Removed multiple legacy, unreferenced helper functions from earlier workflows to reduce maintenance surface and keep utility scope focused on active training/sampling paths.

### Changed

- `train.py`: Restored LSTM-friendly default training hyperparameters (Adam, `lr=1e-4`, `weight_decay=0`, AMP off, grad-clip back to 1.0). The Transformer preset still opts into AdamW/AMP/KL warmup when explicitly selected.
- `train.py`: Checkpointing logic is now explicit and split by purpose: a single rolling best checkpoint (`model_best.ckpt-<best_epoch>.pt`, replacing previous `*best*` checkpoints on each improvement), periodic current checkpoints every `save_every` epochs (`model_<epoch>_periodic.ckpt-<epoch>.pt`), a final current checkpoint when the last epoch is reached (`model_<epoch>_final.ckpt-<epoch>.pt`), and an early-stop current checkpoint (`model_<epoch>_early_stop.ckpt-<epoch>.pt`).

### Changed (Config & Presets)

- `train.py` / `utils.py`: Training config now supports grouped sections (`data`, `model`, `transformer`, `optimization`, `training`, `scheduler`, `kl`, `diagnostics`) while remaining backward compatible with legacy flat keys.
- `train.py`: Simplified default config and aligned KL annealing defaults to safer startup values (`start_beta=0.01`, `hold_epochs=0`, `warmup_epochs=50`).
- `train.py`: `stable_transformer` preset now keeps AMP enabled and applies the safer KL schedule.

### Fixed (Transformer Stability)

- `model.py`: Transformer decoder path no longer passes `tgt_key_padding_mask`, avoiding all-masked-query NaN behavior.
- `model.py`: Under AMP, Transformer encoder/decoder attention blocks now execute in fp32 via selective autocast disable, reducing fp16/bf16 attention-softmax instability.

### Added

- `model.py`: Added dual architecture support in `CVAE` via `model_mode` (`lstm` or `transformer`) with shared training/sampling API.
- `model.py`: Added Transformer-specific components (`PositionalEncoding`, encoder/decoder projections, causal masking, and padding masks) while preserving the original LSTM path.
- `train.py`: Added CLI config keys for `model_mode` and Transformer hyperparameters (`transformer_heads`, `transformer_ff_size`, `transformer_dropout`).
- `utils.py`: Added modular config helpers: `build_train_config(...)`, `get_model_config(...)`, `save_json(...)`, `load_json(...)`, `save_training_config(...)`, `infer_training_config_path(...)`.
- `train.py`: Added automatic save of `save/training_config.json` containing essential model recreation config.
- `utils.py`: Added centralized train config defaults + composition helper (`compose_train_config(...)`) supporting defaults + JSON config + CLI overrides.
- `train.py`: Added `--config_file` support so training can run directly from a JSON config.
- `utils.py`: Added reusable utility helpers `ensure_dir(...)` and `split_train_test(...)` used by training flow.
- `utils.py`: Added `compose_train_config_from_dict(...)` for validating/normalizing a direct in-file config dictionary.
- `train.py`: Added in-file config controls for `optimizer` (`adam`/`adamw`), optional `ReduceLROnPlateau`, and robust early stopping parameters.
- `train.py` / `utils.py`: Added mixed-precision config keys `use_amp` and `amp_dtype` (`float16`/`bfloat16`) to defaults and normalization paths.
- `model.py`: Added AMP training support with CUDA autocast and GradScaler integration for stable mixed-precision updates.
- `sample.py`: Added `--num_unique` (and `--max_batches` safety cap) to keep generating batches until a target number of **unique, valid** molecules is collected.
- `sample.py`: Added generation quality reporting with total generated count, accepted count, not-ok share, and breakdown (`invalid_or_empty`, `in_training`, `duplicate`).
- `utils.py`: Added `load_training_canonical_smiles(...)` and `collect_new_unique_from_raw(...)` helper utilities.
- `sample.py`: Added config flag `exclude_training` to enable/disable filtering out molecules present in training data.
- `train.py`: Added configurable KL annealing controls (`kl_anneal_enabled`, `kl_anneal_start_beta`, `kl_anneal_max_beta`, `kl_anneal_hold_epochs`, `kl_anneal_warmup_epochs`) and per-epoch diagnostics logging.
- `model.py`: Added optional detailed batch metrics from `train_batch(...)` / `test_batch(...)` (reconstruction loss, KL loss, latent stats, gradient norm) used for stability debugging.
- `train.py`: Added one-click training preset switch via `training_preset`, including `stable_transformer` mode that auto-applies safer anti-divergence settings.

### Changed (Workflow & Checkpointing)

- `sample.py`: Model initialization now loads training/model hyperparameters from `training_config.json` (in checkpoint folder by default), so manual retyping of architecture config is no longer required.
- `sample.py`: Added clean runtime override behavior for `batch_size`, `prop_file`, `seq_length`, `mean`, and `stddev` on top of loaded training config.
- `train.py`: Uses one args-driven config source (instead of a separate hardcoded config dict) and passes normalized config to `CVAE`.
- `train.py`: Parser defaults are now `None` for model/training params to cleanly allow JSON config + selective CLI overrides.
- `train.py`: Primary training workflow now uses a single editable `config` dictionary inside the file (no external JSON or CLI arguments required).
- `train.py`: Early stopping now uses robust best-loss tracking (`best_epoch`, `epochs_without_improvement`, `min_delta`) and optional best-weight restore before final save.
- `train.py`: Training history now also logs learning rate per epoch (`lr`).
- `model.py`: Checkpoint payloads now also persist GradScaler state for AMP-enabled training resume.
- `model.py`: `CVAE.save(...)` now stores optional `model_config` metadata in checkpoint payload.
- `sample.py`: Refactored generation flow into small helper functions and clarified comments.
- `sample.py`: Removed command-line argument parsing and switched to a config-only workflow (single editable `config` block).
- `sample.py`: Now excludes molecules present in the training/property file from accepted generated results.
- `README.md`: Rewritten usage guide with dedicated sections for LSTM/Transformer training commands, JSON config workflow, and a clear "differences from original paper" explanation.
- `README.md`: Training instructions now emphasize in-file `train.py` config editing as the default/primary workflow.
- `train.py`: Training now uses shuffled per-epoch batching without replacement instead of random sampling with replacement, reducing unstable repeated updates.
- `train.py`: Default debug-safe Transformer training settings now start with `use_amp=False` and `weight_decay=0.0` while keeping both features configurable.
- `model.py`: AdamW optimizer now uses parameter groups so weight decay is not applied to biases, norm parameters, embeddings, and VAE posterior heads (`out_mean`, `out_log_sigma`).
- `model.py`: ELBO now supports a KL weight (`beta`) for warm-up/annealing while preserving old behavior when `beta=1.0`.

### Fixed (Compatibility & Reliability)

- `model.py`: In `transformer` mode, token embedding width now uses `latent_size` (instead of `unit_size`), so embedding can be significantly smaller while keeping Transformer internal width controlled by `unit_size`.
- `utils.py`: `compose_train_config_from_dict(...)` and `compose_train_config(...)` now preserve `num_prop=None` during initial config normalization, preventing a startup crash (`int(None)`) before `train.py` infers `num_prop` from the property file.
- `model.py`: `CVAE.sample()` now stops decoding early once EOS (`'E'`) has been generated for all sequences in the batch, instead of always running the full `seq_length` loop. EOS index is inferred from the known vocab construction (`E = vocab_size - 2`).
- `model.py`: Transformer decoding now uses an explicit boolean causal `tgt_mask` so mask dtypes are aligned with key padding masks and runtime behavior is stable across current PyTorch versions.
- `model.py`: Checkpoint restore now uses explicit `weights_only` handling when supported by current PyTorch to avoid future-warning-prone implicit load behavior.
- `utils.py`: Fixed latent utility issues discovered during validation (`load_dataset` now imports `h5py` locally, `from_one_hot_array` now returns `Optional[int]`, and one-hot/inchi helper typing edge cases were hardened).
- `model.py`: Replaced fixed Adam optimizer with configurable optimizer selection (`adam`/`adamw`) and weight decay support.
- `train.py` / `model.py`: Addressed Transformer divergence pattern (epoch-1 loss explosion) by adding KL warm-up, safer optimizer decay behavior, and explicit diagnostics to identify exploding terms.
- Compatibility: LSTM path and existing APIs remain functional; stability additions are opt-in via config or backward-compatible defaults.

### Tested

- Smoke trained `lstm` mode for 1 epoch and confirmed artifact save to `save/smoke_lstm/` (`training_config.json`, checkpoint, history).
- Smoke trained `transformer` mode for 1 epoch and confirmed artifact save to `save/smoke_transformer/` (`training_config.json`, checkpoint, history).
- Verified end-to-end restore + sample for both checkpoints (`model_.ckpt-0.pt`) by loading saved config and generating sample outputs.
