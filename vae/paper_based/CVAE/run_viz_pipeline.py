from __future__ import annotations

import argparse
import json
import os

from analysis_modules import load_analysis_config_from_file, run_analysis_pipeline


DEFAULT_CONFIG_PATH = 'analysis_run_config.json'


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

    cfg = load_analysis_config_from_file(args.config)
    summary = run_analysis_pipeline(cfg)

    print(80 * '=')
    print(f'Analysis pipeline finished using config: {args.config}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
