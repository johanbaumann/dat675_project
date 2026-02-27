from __future__ import annotations

import argparse
import json
import os

from analysis_modules import load_analysis_config_from_file, run_analysis_pipeline
from utils import load_sampling_metadata


DEFAULT_CONFIG_PATH = 'analysis_run_config.json'


def _coerce_bool(value, default: bool = True) -> bool:
    """Coerce common bool-like config values while staying backward compatible."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {'1', 'true', 'yes', 'on'}:
            return True
        if norm in {'0', 'false', 'no', 'off'}:
            return False
    return bool(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run module-based analysis from a JSON config file.')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(
            f'Config file not found: {args.config}. '\
            f'Create it (example: {DEFAULT_CONFIG_PATH}) and re-run.'
        )

    # Optional backwards-compatible startup diagnostics.
    # Toggle with:
    #   - overrides.print_vocab_size: true/false
    # Default is True so existing behavior remains unchanged.
    # If enabled and config JSON contains sampling hints under overrides
    #   - overrides.prop_file
    #   - overrides.seq_length
    # then print the vocabulary size inferred from that file.
    # This does not change analysis behavior and is skipped silently if missing.
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            raw_cfg = json.load(f)
        overrides = raw_cfg.get('overrides', {}) if isinstance(raw_cfg, dict) else {}
        print_vocab_size = _coerce_bool(overrides.get('print_vocab_size', True), default=True)
        prop_file = overrides.get('prop_file')
        seq_length = overrides.get('seq_length')
        if print_vocab_size and prop_file is not None and seq_length is not None:
            charset, _, inferred_num_prop = load_sampling_metadata(str(prop_file), int(seq_length))
            print(
                f"[analysis] optional sampling metadata: vocab_size={len(charset)}, "
                f"num_prop={int(inferred_num_prop)}, prop_file={prop_file}"
            )
    except Exception:
        # Keep runner robust: analysis should run even if optional metadata lookup fails.
        pass

    cfg = load_analysis_config_from_file(args.config)
    if bool(cfg.debug):
        print('[analysis:debug] Debug mode enabled from config.')
    summary = run_analysis_pipeline(cfg)

    print(80 * '=')
    print(f'Analysis pipeline finished using config: {args.config}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
