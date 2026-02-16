# Changelog

All notable changes to this project are documented in this file.

## Unreleased

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

### Changed

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

### Fixed

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
