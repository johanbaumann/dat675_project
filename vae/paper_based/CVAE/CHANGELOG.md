# Changelog

All notable changes to this project are documented in this file.

## Unreleased

### Added
- `sample.py`: Added `--num_unique` (and `--max_batches` safety cap) to keep generating batches until a target number of **unique, valid** molecules is collected.
- `sample.py`: Added generation quality reporting with total generated count, accepted count, not-ok share, and breakdown (`invalid_or_empty`, `in_training`, `duplicate`).
- `utils.py`: Added `load_training_canonical_smiles(...)` and `collect_new_unique_from_raw(...)` helper utilities.
- `sample.py`: Added config flag `exclude_training` to enable/disable filtering out molecules present in training data.

### Changed
- `sample.py`: Refactored generation flow into small helper functions and clarified comments.
- `sample.py`: Removed command-line argument parsing and switched to a config-only workflow (single editable `config` block).
- `sample.py`: Now excludes molecules present in the training/property file from accepted generated results.

### Fixed
- `model.py`: `CVAE.sample()` now stops decoding early once EOS (`'E'`) has been generated for all sequences in the batch, instead of always running the full `seq_length` loop. EOS index is inferred from the known vocab construction (`E = vocab_size - 2`).
