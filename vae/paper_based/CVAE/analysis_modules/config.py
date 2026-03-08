from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AnalysisConfig:
    profile_name: str
    train_folder: str
    train_data_path: str
    validation_data_path: Optional[str]
    generated_data_path: str
    output_dir: str

    smiles_column: str = 'smiles'
    train_sep: Optional[str] = None
    validation_sep: Optional[str] = None
    generated_sep: str = ','

    target_property_column: Optional[str] = None
    predicted_property_column: Optional[str] = None

    train_max: int = 50_000
    validation_max: int = 50_000
    generated_max: int = 50_000
    random_seed: int = 42

    tanimoto_radius: int = 2
    tanimoto_n_bits: int = 2048
    tanimoto_cache_filename: str = 'tanimoto_cache.bin'
    tanimoto_flush_every: int = 1000

    save_distribution_plot: bool = True
    save_scaffold_plot: bool = True
    save_scaffold_grids: bool = True
    run_prediction_error_plot: bool = True

    distribution_plot_filename: str = 'property_distribution.png'
    mw_distribution_diff_plot_filename: str = 'mw_distribution_diff_train_minus_generated.png'
    scaffold_plot_filename: str = 'scaffold_distribution_train_vs_gen.png'
    train_scaffold_grid_filename: str = 'train_top_scaffolds_grid.png'
    generated_scaffold_grid_filename: str = 'generated_top_scaffolds_grid.png'
    novel_scaffold_grid_filename: str = 'novel_top_scaffolds_grid.png'
    scaffold_stats_filename: str = 'scaffold_stats.json'
    train_loss_plot_filename: str = 'train_vs_test_loss.png'
    tanimoto_histogram_filename: str = 'tanimoto_similarity_distribution.png'
    prediction_error_plot_filename: str = 'prediction_error_vs_ground_truth.png'
    processed_csv_filename: str = 'generated_subset_with_similarity.csv'
    summary_json_filename: str = 'analysis_summary.json'

    run_train_loss_plot: bool = True
    run_tanimoto_histogram: bool = True
    run_chemical_space: bool = True
    run_descriptor_space: bool = True
    debug: bool = False

    embedding_train_sample: int = 10_000
    embedding_validation_sample: int = 10_000
    embedding_generated_sample: int = 10_000

    chemical_tsne_perplexity: float = 30.0
    chemical_pca_pre_dim: int = 50
    descriptor_tsne_perplexity: float = 30.0
    descriptor_pca_pre_dim: int = 0
    embedding_point_size: int = 6
    embedding_alpha: float = 0.5
    scaffold_grid_top_n: int = 24
    scaffold_grid_n_cols: int = 6

    descriptor_names: tuple[str, ...] = (
        'HeavyAtomCount',
        'RingCount',
        'MolLogP',
        'MolWt',
        'NumHAcceptors',
        'NumHDonors',
    )

    chemical_pca_plot_filename: str = 'pca2_train_vs_generated.png'
    chemical_tsne_plot_filename: str = 'tsne_train_vs_generated.png'
    chemical_tsne_tanimoto_plot_filename: str = 'tsne_generated_colored_by_tanimoto.png'
    descriptor_pca_plot_filename: str = 'pca2_descriptors_train_vs_generated.png'
    descriptor_tsne_plot_filename: str = 'tsne_descriptors_train_vs_generated.png'
    descriptor_tsne_tanimoto_plot_filename: str = 'tsne_descriptors_generated_colored_by_tanimoto.png'


DEFAULT_ZINC_PROFILE = AnalysisConfig(
    profile_name='zinc_logp',
    train_folder='save/run_20260224_205844',
    train_data_path='250k_zinc_clean.txt',
    validation_data_path=None,
    generated_data_path='train_dist_temp_transformer_300k_test.txt',
    output_dir='save/run_20260224_205844/analysis',
    target_property_column='LogP',
    predicted_property_column='pred_LogP',
)


DEFAULT_BACE_PROFILE = AnalysisConfig(
    profile_name='bace_pic50_10k',
    train_folder='save/run_20260226_095012',
    train_data_path='bace_pic50.txt',
    validation_data_path=None,
    generated_data_path='10k_bace_test.txt',
    output_dir='save/run_20260226_095012/analysis_bace_10k',
    target_property_column='pIC50',
    predicted_property_column='pred_pIC50',
)


def build_profile_config(profile: str = 'bace_pic50_10k', **overrides) -> AnalysisConfig:
    normalized = str(profile).strip().lower()
    if normalized in ('zinc', 'zinc_logp'):
        base = DEFAULT_ZINC_PROFILE
    elif normalized in ('bace', 'bace_pic50', 'bace_pic50_10k'):
        base = DEFAULT_BACE_PROFILE
    else:
        raise ValueError("profile must be one of: 'zinc_logp', 'bace_pic50_10k'")

    payload = dict(base.__dict__)
    payload.update(overrides)
    return AnalysisConfig(**payload)


def load_analysis_config_from_file(path: str) -> AnalysisConfig:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError('Config JSON must be an object.')

    profile = str(payload.get('profile', 'bace_pic50_10k'))
    overrides = payload.get('overrides', {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError('Config key "overrides" must be an object when provided.')

    for key, value in payload.items():
        if key not in ('profile', 'overrides'):
            overrides[key] = value

    if 'descriptor_names' in overrides and isinstance(overrides['descriptor_names'], list):
        overrides['descriptor_names'] = tuple(str(x) for x in overrides['descriptor_names'])

    # Backward/forward compatibility: allow extra keys in config JSON (for runner
    # toggles or future extensions) without breaking AnalysisConfig construction.
    valid_keys = set(AnalysisConfig.__dataclass_fields__.keys())
    overrides = {k: v for k, v in overrides.items() if k in valid_keys}

    return build_profile_config(profile=profile, **overrides)
