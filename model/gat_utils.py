import copy
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== target scaling ====================
# Optional per-fold target standardization. If enabled, we fit mean/std on the
# training fold only, train the model to predict standardized targets, and invert
# predictions back to the original target scale for metrics and reporting.


def _get_target_standardization_config(config: dict[str, Any]) -> dict[str, Any]:
	target_cfg = config.get("training", {}).get("target_standardization", {})
	if not isinstance(target_cfg, dict):
		target_cfg = {}
	return {
		"enabled": bool(target_cfg.get("enabled", False)),
		"epsilon": float(target_cfg.get("epsilon", 1e-8)),
	}


def fit_target_standardizer(
	train_targets: np.ndarray,
	*,
	epsilon: float = 1e-8,
) -> tuple[float, float]:
	"""Fit (mean, std) on the training fold targets.

	Uses population std (ddof=0) and guards against near-zero std.
	"""
	mean = float(np.mean(train_targets))
	std = float(np.std(train_targets))
	if not np.isfinite(std) or std < epsilon:
		std = 1.0
	return mean, std


def standardize_targets_tensor(
	y: torch.Tensor,
	*,
	mean: torch.Tensor,
	std: torch.Tensor,
) -> torch.Tensor:
	return (y - mean) / std


def invert_standardization_tensor(
	y_scaled: torch.Tensor,
	*,
	mean: torch.Tensor,
	std: torch.Tensor,
) -> torch.Tensor:
	return y_scaled * std + mean


def build_target_standardizer(
	train_targets: np.ndarray,
	*,
	epsilon: float = 1e-8,
) -> dict[str, Any]:
	mean, std = fit_target_standardizer(train_targets, epsilon=epsilon)
	return build_target_standardizer_from_stats(mean, std)


def build_target_standardizer_from_stats(
	mean: float,
	std: float,
) -> dict[str, Any]:
	mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device)
	std_tensor = torch.tensor(std, dtype=torch.float32, device=device)
	return {
		"mean": mean,
		"std": std,
		"mean_tensor": mean_tensor,
		"std_tensor": std_tensor,
	}


def standardize_batch_targets(
	y_true: torch.Tensor,
	*,
	target_standardizer: dict[str, Any] | None,
) -> torch.Tensor:
	if target_standardizer is None:
		return y_true
	return standardize_targets_tensor(
		y_true,
		mean=target_standardizer["mean_tensor"],
		std=target_standardizer["std_tensor"],
	)


def invert_standardized_predictions(
	y_pred: torch.Tensor,
	*,
	target_standardizer: dict[str, Any] | None,
) -> torch.Tensor:
	if target_standardizer is None:
		return y_pred
	return invert_standardization_tensor(
		y_pred,
		mean=target_standardizer["mean_tensor"],
		std=target_standardizer["std_tensor"],
	)


# ==================== feature spaces ====================
# Categorical features can be encoded as one_hot (+other) or index (+unknown index)
# depending on CONFIG["features"]["categorical_encoding_mode"].
HYBRIDIZATION_TYPES = [
	("sp", Chem.rdchem.HybridizationType.SP),
	("sp2", Chem.rdchem.HybridizationType.SP2),
	("sp3", Chem.rdchem.HybridizationType.SP3),
	("sp3d", Chem.rdchem.HybridizationType.SP3D),
	("sp3d2", Chem.rdchem.HybridizationType.SP3D2),
]

CHIRAL_TYPES = [
	("chi_unspecified", Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
	("chi_cw", Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW),
	("chi_ccw", Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW),
	("chi_other", Chem.rdchem.ChiralType.CHI_OTHER),
]

BOND_TYPE_VALUES = [
	("single", Chem.rdchem.BondType.SINGLE),
	("double", Chem.rdchem.BondType.DOUBLE),
	("triple", Chem.rdchem.BondType.TRIPLE),
	("aromatic", Chem.rdchem.BondType.AROMATIC),
]

BOND_STEREO_TYPES = [
	("none", Chem.rdchem.BondStereo.STEREONONE),
	("any", Chem.rdchem.BondStereo.STEREOANY),
	("z", Chem.rdchem.BondStereo.STEREOZ),
	("e", Chem.rdchem.BondStereo.STEREOE),
	("cis", Chem.rdchem.BondStereo.STEREOCIS),
	("trans", Chem.rdchem.BondStereo.STEREOTRANS),
]


ATOM_BASE_FEATURE_NAMES = [
	"atomic_num",
	"degree",
	"total_num_hs",
	"formal_charge",
	"implicit_valence",
	"mass_scaled",
	"is_aromatic",
	"is_in_ring",
]

# Descriptor keys for config-driven feature selection.
ALLOWED_ATOM_DESCRIPTORS = set(
	ATOM_BASE_FEATURE_NAMES + ["hybridization", "chirality"]
)
ALLOWED_BOND_DESCRIPTORS = {
	"bond_type",
	"bond_order",
	"is_conjugated",
	"is_in_ring",
	"bond_stereo",
}

DEFAULT_ATOM_DESCRIPTORS = [
	"atomic_num",
	"degree",
	"total_num_hs",
	"formal_charge",
	"implicit_valence",
	"mass_scaled",
	"is_aromatic",
	"is_in_ring",
	"hybridization",
	"chirality",
]

DEFAULT_BOND_DESCRIPTORS = [
	"bond_type",
	"bond_order",
	"is_conjugated",
	"is_in_ring",
	"bond_stereo",
]

HYBRIDIZATION_VALUES = [v for _, v in HYBRIDIZATION_TYPES]
CHIRAL_VALUES = [v for _, v in CHIRAL_TYPES]
BOND_TYPE_CATEGORY_VALUES = [v for _, v in BOND_TYPE_VALUES]
BOND_STEREO_VALUES = [v for _, v in BOND_STEREO_TYPES]


def set_seed(seed: int) -> None:
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _normalize_descriptor_list(value, *, allowed: set[str], default: list[str], label: str) -> list[str]:
	if value is None:
		value = default
	if isinstance(value, dict):
		# Allow {name: bool} for convenience.
		value = [k for k, v in value.items() if bool(v)]
	if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
		raise ValueError(f"CONFIG['features']['{label}'] must be a list[str] or dict[str, bool].")
	value = [str(x) for x in value]
	unknown = sorted(set(value) - allowed)
	if unknown:
		raise ValueError(
			f"Unknown {label} entries: {unknown}. Allowed: {sorted(allowed)}"
		)
	# Preserve order but drop duplicates.
	seen = set()
	ordered = []
	for name in value:
		if name not in seen:
			seen.add(name)
			ordered.append(name)
	if len(ordered) == 0:
		raise ValueError(f"CONFIG['features']['{label}'] must enable at least one descriptor.")
	return ordered


def build_feature_names(
	encoding_mode: str,
	*,
	atom_descriptors: list[str],
	bond_descriptors: list[str],
):
	atom_feature_names: list[str] = []
	edge_feature_names: list[str] = []

	# Atom (node) descriptors
	for desc in atom_descriptors:
		if desc in ATOM_BASE_FEATURE_NAMES:
			atom_feature_names.append(desc)
		elif desc == "hybridization":
			if encoding_mode == "one_hot":
				atom_feature_names += [
					f"hybridization_{name}" for name, _ in HYBRIDIZATION_TYPES
				] + ["hybridization_other"]
			else:
				atom_feature_names.append("hybridization_index")
		elif desc == "chirality":
			if encoding_mode == "one_hot":
				atom_feature_names += [
					f"chirality_{name}" for name, _ in CHIRAL_TYPES
				] + ["chirality_other"]
			else:
				atom_feature_names.append("chirality_index")

	# Bond (edge) descriptors
	for desc in bond_descriptors:
		if desc == "bond_type":
			if encoding_mode == "one_hot":
				edge_feature_names += [
					f"bond_type_{name}" for name, _ in BOND_TYPE_VALUES
				] + ["bond_type_other"]
			else:
				edge_feature_names.append("bond_type_index")
		elif desc == "bond_order":
			edge_feature_names.append("bond_order")
		elif desc == "is_conjugated":
			edge_feature_names.append("is_conjugated")
		elif desc == "is_in_ring":
			edge_feature_names.append("is_in_ring")
		elif desc == "bond_stereo":
			if encoding_mode == "one_hot":
				edge_feature_names += [
					f"bond_stereo_{name}" for name, _ in BOND_STEREO_TYPES
				] + ["bond_stereo_other"]
			else:
				edge_feature_names.append("bond_stereo_index")

	return atom_feature_names, edge_feature_names


def prepare_feature_config(config: dict[str, Any]) -> dict[str, Any]:
	featurizer = str(config["features"].get("featurizer", "custom")).lower().strip()
	if featurizer not in {"custom", "chemprop_simple"}:
		raise ValueError(
			"CONFIG['features']['featurizer'] must be 'custom' or 'chemprop_simple'."
		)

	if featurizer == "chemprop_simple":
		node_dim, edge_dim = _probe_chemprop_dims()
		config["features"]["atom_feature_dim"] = node_dim
		config["features"]["edge_feature_dim"] = edge_dim
		return {
			"featurizer": "chemprop_simple",
			"categorical_encoding_mode": "n/a",
			"atom_feature_names": [],
			"edge_feature_names": [],
			"atom_feature_dim": node_dim,
			"edge_feature_dim": edge_dim,
			"atom_descriptors": [],
			"bond_descriptors": [],
		}

	encoding_mode = str(
		config["features"].get("categorical_encoding_mode", "one_hot")
	).lower()
	if encoding_mode not in {"one_hot", "index"}:
		raise ValueError(
			"CONFIG['features']['categorical_encoding_mode'] must be 'one_hot' or 'index'."
		)

	atom_descriptors = _normalize_descriptor_list(
		config["features"].get("atom_descriptors"),
		allowed=ALLOWED_ATOM_DESCRIPTORS,
		default=DEFAULT_ATOM_DESCRIPTORS,
		label="atom_descriptors",
	)
	bond_descriptors = _normalize_descriptor_list(
		config["features"].get("bond_descriptors"),
		allowed=ALLOWED_BOND_DESCRIPTORS,
		default=DEFAULT_BOND_DESCRIPTORS,
		label="bond_descriptors",
	)

	atom_feature_names, edge_feature_names = build_feature_names(
		encoding_mode,
		atom_descriptors=atom_descriptors,
		bond_descriptors=bond_descriptors,
	)
	atom_feature_dim = len(atom_feature_names)
	edge_feature_dim = len(edge_feature_names)

	config["features"]["atom_feature_dim"] = atom_feature_dim
	config["features"]["edge_feature_dim"] = edge_feature_dim
	config["features"]["atom_descriptors"] = atom_descriptors
	config["features"]["bond_descriptors"] = bond_descriptors

	return {
		"featurizer": "custom",
		"categorical_encoding_mode": encoding_mode,
		"atom_feature_names": atom_feature_names,
		"edge_feature_names": edge_feature_names,
		"atom_feature_dim": atom_feature_dim,
		"edge_feature_dim": edge_feature_dim,
		"atom_descriptors": atom_descriptors,
		"bond_descriptors": bond_descriptors,
	}


def print_feature_summary(config: dict[str, Any], feature_context: dict[str, Any]) -> None:
	print(f"Current device: {device}")
	print("\nFeature summary:")
	featurizer = feature_context.get("featurizer", "custom")
	print(f"  Featurizer: {featurizer}")
	print(f"  Atom feature dim: {feature_context['atom_feature_dim']}")
	print(f"  Edge feature dim: {feature_context['edge_feature_dim']}")
	if featurizer == "custom":
		print(
			f"  Categorical encoding mode: "
			f"{feature_context['categorical_encoding_mode']}"
		)
		print(f"  Atom descriptors: {feature_context['atom_descriptors']}")
		print(f"  Bond descriptors: {feature_context['bond_descriptors']}")
		print(f"  Atom features: {feature_context['atom_feature_names']}")
		print(f"  Edge features: {feature_context['edge_feature_names']}")
	print(f"  Residual connections: {config['model'].get('residual', True)}")
	print(
		f"  Conv normalization: "
		f"{str(config['model'].get('normalization', 'layernorm')).lower()}"
	)
	print(f"  FFNN hidden layers: {config['model']['ffnn_hidden_layers']}")


def one_hot_with_unknown(value, allowed_values):
	encoded = [1 if value == allowed else 0 for allowed in allowed_values]
	encoded.append(1 if value not in allowed_values else 0)
	return encoded


def index_with_unknown(value, allowed_values):
	for i, allowed in enumerate(allowed_values):
		if value == allowed:
			return float(i)
	return float(len(allowed_values))


def _validate_feature_vector_length(
	feature_values: list[float],
	*,
	expected_dim: int,
	feature_label: str,
	molecule_smiles: str,
) -> None:
	if len(feature_values) != expected_dim:
		raise ValueError(
			f"{feature_label} feature length mismatch for SMILES '{molecule_smiles}': "
			f"expected {expected_dim}, got {len(feature_values)}. "
			f"Check CONFIG['features'] descriptor settings."
		)


# ==================== chemprop featurizer (optional alternative) ====================
_chemprop_featurizer_instance = None


def _get_chemprop_featurizer():
	global _chemprop_featurizer_instance
	if _chemprop_featurizer_instance is None:
		from chemprop import featurizers as _chemprop_featurizers
		_chemprop_featurizer_instance = _chemprop_featurizers.SimpleMoleculeMolGraphFeaturizer()
	return _chemprop_featurizer_instance


def _probe_chemprop_dims() -> tuple[int, int]:
	"""Return (atom_feature_dim, edge_feature_dim) for chemprop's SimpleMoleculeMolGraphFeaturizer."""
	test_mol = Chem.MolFromSmiles("CC")  # two atoms, one bond -> two directed edges
	mol_graph = _get_chemprop_featurizer()(test_mol)
	return int(mol_graph.V.shape[1]), int(mol_graph.E.shape[1])


def _smiles_to_graph_chemprop(
	smiles: str,
	target: float,
	feature_context: dict[str, Any],
) -> Data | None:
	"""Convert SMILES to a PyG Data object using chemprop's SimpleMoleculeMolGraphFeaturizer."""
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return None
	mol_graph = _get_chemprop_featurizer()(mol)
	x = torch.tensor(mol_graph.V, dtype=torch.float)
	edge_dim = feature_context["edge_feature_dim"]
	if mol_graph.E.shape[0] > 0:
		ei = torch.tensor(mol_graph.edge_index, dtype=torch.long)
		# chemprop stores edge_index as [2, num_edges]; guard against [num_edges, 2] in older versions.
		if ei.shape[0] != 2:
			ei = ei.T.contiguous()
		edge_attr = torch.tensor(mol_graph.E, dtype=torch.float)
	else:
		ei = torch.empty((2, 0), dtype=torch.long)
		edge_attr = torch.empty((0, edge_dim), dtype=torch.float)
	y = torch.tensor([[float(target)]], dtype=torch.float)
	return Data(x=x, edge_index=ei, edge_attr=edge_attr, y=y)


# ==================== featurization ====================
# Atom features contain physically meaningful numeric values + one-hot categorical tags.
def get_atom_features(
	atom: Chem.rdchem.Atom,
	config: dict[str, Any],
	feature_context: dict[str, Any],
) -> list[float]:
	atom_descriptors = feature_context["atom_descriptors"]
	encoding_mode = feature_context["categorical_encoding_mode"]
	features: list[float] = []

	for desc in atom_descriptors:
		if desc == "atomic_num":
			features.append(float(atom.GetAtomicNum()))
		elif desc == "degree":
			features.append(float(atom.GetDegree()))
		elif desc == "total_num_hs":
			features.append(float(atom.GetTotalNumHs()))
		elif desc == "formal_charge":
			features.append(float(atom.GetFormalCharge()))
		elif desc == "implicit_valence":
			features.append(float(atom.GetValence(Chem.ValenceType.IMPLICIT)))
		elif desc == "mass_scaled":
			features.append(
				float(atom.GetMass()) / config["features"]["scale_atomic_mass_by"]
			)
		elif desc == "is_aromatic":
			features.append(float(atom.GetIsAromatic()))
		elif desc == "is_in_ring":
			features.append(float(atom.IsInRing()))
		elif desc == "hybridization":
			if encoding_mode == "one_hot":
				features += one_hot_with_unknown(
					atom.GetHybridization(), HYBRIDIZATION_VALUES
				)
			else:
				features.append(
					index_with_unknown(atom.GetHybridization(), HYBRIDIZATION_VALUES)
				)
		elif desc == "chirality":
			if encoding_mode == "one_hot":
				features += one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_VALUES)
			else:
				features.append(index_with_unknown(atom.GetChiralTag(), CHIRAL_VALUES))

	return features


# Bond features also use one-hot categorical fields (+unknown) for stability.
def get_bond_features(
	bond: Chem.rdchem.Bond,
	config: dict[str, Any],
	feature_context: dict[str, Any],
) -> list[float]:
	bond_descriptors = feature_context["bond_descriptors"]
	encoding_mode = feature_context["categorical_encoding_mode"]
	features: list[float] = []

	for desc in bond_descriptors:
		if desc == "bond_type":
			if encoding_mode == "one_hot":
				features += one_hot_with_unknown(
					bond.GetBondType(), BOND_TYPE_CATEGORY_VALUES
				)
			else:
				features.append(
					index_with_unknown(bond.GetBondType(), BOND_TYPE_CATEGORY_VALUES)
				)
		elif desc == "bond_order":
			features.append(float(bond.GetBondTypeAsDouble()))
		elif desc == "is_conjugated":
			features.append(float(bond.GetIsConjugated()))
		elif desc == "is_in_ring":
			features.append(float(bond.IsInRing()))
		elif desc == "bond_stereo":
			if encoding_mode == "one_hot":
				features += one_hot_with_unknown(bond.GetStereo(), BOND_STEREO_VALUES)
			else:
				features.append(index_with_unknown(bond.GetStereo(), BOND_STEREO_VALUES))

	return features


def smiles_to_graph(
	smiles: str,
	target: float,
	config: dict[str, Any],
	feature_context: dict[str, Any],
) -> Data | None:
	if feature_context.get("featurizer") == "chemprop_simple":
		return _smiles_to_graph_chemprop(smiles, target, feature_context)
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return None

	node_features = [
		get_atom_features(atom, config, feature_context) for atom in mol.GetAtoms()
	]
	for atom_feature_values in node_features:
		_validate_feature_vector_length(
			atom_feature_values,
			expected_dim=feature_context["atom_feature_dim"],
			feature_label="Atom",
			molecule_smiles=smiles,
		)
	x = torch.tensor(node_features, dtype=torch.float)

	edge_indices = []
	edge_attrs = []
	for bond in mol.GetBonds():
		i = bond.GetBeginAtomIdx()
		j = bond.GetEndAtomIdx()
		e_feat = get_bond_features(bond, config, feature_context)
		_validate_feature_vector_length(
			e_feat,
			expected_dim=feature_context["edge_feature_dim"],
			feature_label="Bond",
			molecule_smiles=smiles,
		)

		edge_indices += [[i, j], [j, i]]
		edge_attrs += [e_feat, e_feat]

	if edge_indices:
		edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
		edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
	else:
		edge_index = torch.empty((2, 0), dtype=torch.long)
		edge_attr = torch.empty(
			(0, feature_context["edge_feature_dim"]), dtype=torch.float
		)

	y = torch.tensor([[float(target)]], dtype=torch.float)
	return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ==================== data loading ====================
def get_synthetic_cv_config(config: dict[str, Any]) -> dict[str, Any]:
	synthetic_cfg = config.get("data", {}).get("synthetic_cv", {})
	if not isinstance(synthetic_cfg, dict):
		synthetic_cfg = {}
	keep_percentile = float(synthetic_cfg.get("keep_percentile", 1.0))
	if keep_percentile <= 0.0 or keep_percentile > 1.0:
		raise ValueError(
			"CONFIG['data']['synthetic_cv']['keep_percentile'] must be in (0, 1]."
		)
	row_keep_raw = synthetic_cfg.get("row_keep_fraction", 1.0)
	row_keep_fraction = None
	if row_keep_raw is not None:
		row_keep_fraction = float(row_keep_raw)
		if row_keep_fraction <= 0.0 or row_keep_fraction > 1.0:
			raise ValueError(
				"CONFIG['data']['synthetic_cv']['row_keep_fraction'] must be in (0, 1] "
				"or None."
			)

	ratio_raw = synthetic_cfg.get("max_train_synth_to_real_ratio", None)
	max_train_synth_to_real_ratio = None
	if ratio_raw is not None:
		max_train_synth_to_real_ratio = float(ratio_raw)
		if max_train_synth_to_real_ratio <= 0.0:
			raise ValueError(
				"CONFIG['data']['synthetic_cv']['max_train_synth_to_real_ratio'] "
				"must be > 0 when provided."
			)

	label_source = str(synthetic_cfg.get("label_source", "pred")).lower().strip()
	if label_source not in {"pred", "target"}:
		raise ValueError(
			"CONFIG['data']['synthetic_cv']['label_source'] must be 'pred' or 'target'."
		)
	train_selection = str(
		synthetic_cfg.get("train_selection", "matching_fold")
	).lower().strip()
	if train_selection not in {"matching_fold", "all", "none"}:
		raise ValueError(
			"CONFIG['data']['synthetic_cv']['train_selection'] must be one of "
			"{'matching_fold', 'all', 'none'}."
		)
	validation_selection = str(
		synthetic_cfg.get("validation_selection", "all_except_train_and_fold")
	).lower().strip()
	if validation_selection not in {
		"all_except_train_and_fold",
		"single_next_non_train",
		"none",
	}:
		raise ValueError(
			"CONFIG['data']['synthetic_cv']['validation_selection'] must be one of "
			"{'all_except_train_and_fold', 'single_next_non_train', 'none'}."
		)
	return {
		"include_in_training": bool(synthetic_cfg.get("include_in_training", True)),
		"include_in_validation": bool(synthetic_cfg.get("include_in_validation", False)),
		"keep_percentile": keep_percentile,
		"row_keep_fraction": row_keep_fraction,
		"max_train_synth_to_real_ratio": max_train_synth_to_real_ratio,
		"label_source": label_source,
		"train_selection": train_selection,
		"validation_selection": validation_selection,
	}


def get_synthetic_pretraining_config(config: dict[str, Any]) -> dict[str, Any]:
	pre_cfg = config.get("training", {}).get("synthetic_pretraining", {})
	if not isinstance(pre_cfg, dict):
		pre_cfg = {}
	epochs = int(pre_cfg.get("epochs", 0))
	if epochs < 0:
		raise ValueError("CONFIG['training']['synthetic_pretraining']['epochs'] must be >= 0.")

	min_synthetic_graphs = int(pre_cfg.get("min_synthetic_graphs", 32))
	if min_synthetic_graphs < 1:
		raise ValueError(
			"CONFIG['training']['synthetic_pretraining']['min_synthetic_graphs'] must be >= 1."
		)

	return {
		"enabled": bool(pre_cfg.get("enabled", False)),
		"epochs": epochs,
		"learning_rate": pre_cfg.get("learning_rate", None),
		"weight_decay": pre_cfg.get("weight_decay", None),
		"grad_clip_norm": pre_cfg.get("grad_clip_norm", None),
		"batch_size": int(pre_cfg.get("batch_size", config["data"]["batch_size"])),
		"shuffle": bool(pre_cfg.get("shuffle", True)),
		"min_synthetic_graphs": min_synthetic_graphs,
		"use_target_standardization": bool(pre_cfg.get("use_target_standardization", False)),
	}


def _extract_synthetic_iteration(path_obj: Path) -> int | None:
	match = re.match(r"^synthetic_data_iteration_(\d+)\.csv$", path_obj.name)
	if not match:
		return None
	return int(match.group(1))


def _get_available_synthetic_files(dir_path: Path) -> dict[int, Path]:
	files: dict[int, Path] = {}
	for path_obj in sorted(dir_path.glob("synthetic_data_iteration_*.csv")):
		iter_idx = _extract_synthetic_iteration(path_obj)
		if iter_idx is None:
			continue
		files[iter_idx] = path_obj
	return files


def _select_synthetic_indices(
	*,
	selection: str,
	val_idx: int,
	available_synth: dict[int, Path],
	context_label: str,
) -> set[int]:
	if selection == "matching_fold":
		if val_idx in available_synth:
			return {val_idx}
		print(f"  No matching synthetic file found for {context_label} this fold.")
		return set()
	if selection == "all":
		return set(available_synth.keys())
	return set()


def _resolve_optional(
	value: Any,
	fallback: Any,
):
	if value is None:
		return fallback
	return value


def get_fold_files(
	target_dir: str,
	val_idx: int,
	config: dict[str, Any],
	total_folds: int = 5,
):
	dir_path = Path(target_dir)
	val_file = dir_path / f"fold_{val_idx}.csv"

	policy = get_synthetic_cv_config(config)
	available_synth = _get_available_synthetic_files(dir_path)
	train_fold_indices = [i for i in range(total_folds) if i != val_idx]

	train_files = [dir_path / f"fold_{i}.csv" for i in train_fold_indices]
	val_files = [val_file]
	train_synth_indices: set[int] = set()

	if policy["include_in_training"]:
		train_synth_indices = _select_synthetic_indices(
			selection=policy["train_selection"],
			val_idx=val_idx,
			available_synth=available_synth,
			context_label="training",
		)
		if train_synth_indices:
			train_synth_files = [available_synth[i] for i in sorted(train_synth_indices)]
			train_files.extend(train_synth_files)
			print(
				"  Added training synthetic file(s): "
				+ ", ".join(path.name for path in train_synth_files)
			)
		else:
			print("  No synthetic files selected for training.")
	else:
		print("  Synthetic data excluded from training by config.")

	if policy["include_in_validation"]:
		validation_selection = policy["validation_selection"]
		val_synth_candidates = [
			i
			for i in sorted(available_synth.keys())
			if i != val_idx and i not in train_synth_indices
		]

		if validation_selection == "single_next_non_train" and val_synth_candidates:
			# Deterministic rotation for reproducibility.
			chosen_idx = val_synth_candidates[val_idx % len(val_synth_candidates)]
			val_synth_candidates = [chosen_idx]
		elif validation_selection == "none":
			val_synth_candidates = []

		if val_synth_candidates:
			val_synth_files = [available_synth[i] for i in val_synth_candidates]
			val_files.extend(val_synth_files)
			print(
				"  Added validation synthetic file(s) (no overlap with train/current fold): "
				+ ", ".join(path.name for path in val_synth_files)
			)
		else:
			print(
				"  No validation synthetic files selected "
				"(disjoint-set rule may have excluded all candidates)."
			)
	else:
		print("  Synthetic data excluded from validation by config.")

	return train_files, val_files


def get_synthetic_pretraining_files(
	target_dir: str,
	val_idx: int,
	config: dict[str, Any],
) -> list[Path]:
	"""Select synthetic CSV files for pretraining, independent of finetune train inclusion.

	This lets users pretrain on synthetic data even when
	CONFIG['data']['synthetic_cv']['include_in_training'] is False.
	"""
	pre_cfg = get_synthetic_pretraining_config(config)
	if not pre_cfg["enabled"]:
		return []

	policy = get_synthetic_cv_config(config)
	available_synth = _get_available_synthetic_files(Path(target_dir))
	if not available_synth:
		print("  No synthetic files found for pretraining.")
		return []

	selected_indices = _select_synthetic_indices(
		selection=policy["train_selection"],
		val_idx=val_idx,
		available_synth=available_synth,
		context_label="pretraining",
	)
	selected_files = [available_synth[i] for i in sorted(selected_indices)]
	if selected_files:
		print(
			"  Selected synthetic pretraining file(s): "
			+ ", ".join(path.name for path in selected_files)
		)
	else:
		print("  No synthetic files selected for pretraining.")

	return selected_files


def choose_target_column(df: pd.DataFrame, file_path: Path, config: dict[str, Any]) -> str:
	is_synthetic = file_path.name.startswith("synthetic_data_iteration_")

	if is_synthetic:
		synth_policy = get_synthetic_cv_config(config)
		label_source = synth_policy["label_source"]
		if label_source == "pred":
			preferred = [
				config["data"]["synthetic_target_column"],
				*config["data"]["fallback_target_columns"],
				config["data"]["real_target_column"],
			]
		else:
			preferred = [
				"target_pIC50",
				config["data"]["real_target_column"],
				*config["data"]["fallback_target_columns"],
				config["data"]["synthetic_target_column"],
			]
	else:
		preferred = [
			config["data"]["real_target_column"],
			*config["data"]["fallback_target_columns"],
		]

	for col in preferred:
		if col in df.columns:
			return col

	raise ValueError(
		f"No valid target column found in {file_path}. Tried: {preferred}"
	)


def _resolve_synthetic_filter_column(config: dict[str, Any]) -> str:
	synth_policy = get_synthetic_cv_config(config)
	if synth_policy["label_source"] == "pred":
		return config["data"]["synthetic_target_column"]
	return "target_pIC50"


def _filter_synthetic_rows(df: pd.DataFrame, path_obj: Path, config: dict[str, Any]) -> pd.DataFrame:
	synth_policy = get_synthetic_cv_config(config)
	keep_percentile = float(synth_policy["keep_percentile"])
	if keep_percentile >= 1.0:
		filtered = df
	else:
		filter_col = _resolve_synthetic_filter_column(config)
		if filter_col not in df.columns:
			raise ValueError(
				f"Synthetic filtering column '{filter_col}' is missing in {path_obj}."
			)

		series = pd.to_numeric(df[filter_col], errors="coerce")
		valid_mask = series.notna()
		valid_series = series[valid_mask]
		if valid_series.empty:
			raise ValueError(
				f"No numeric values available in synthetic filter column '{filter_col}' for {path_obj}."
			)

		tail = (1.0 - keep_percentile) / 2.0
		lower = float(valid_series.quantile(tail))
		upper = float(valid_series.quantile(1.0 - tail))

		keep_mask = valid_mask & series.ge(lower) & series.le(upper)
		filtered = df.loc[keep_mask].copy()

		kept_count = int(len(filtered))
		total_count = int(len(df))
		if kept_count == 0:
			raise ValueError(
				f"Synthetic percentile filtering removed all rows in {path_obj}. "
				f"keep_percentile={keep_percentile}"
			)

		kept_values = pd.to_numeric(filtered[filter_col], errors="coerce").dropna()
		kept_mean = float(kept_values.mean()) if not kept_values.empty else float("nan")
		kept_std = float(kept_values.std(ddof=0)) if not kept_values.empty else float("nan")

		print(
			f"  Synthetic filter [{path_obj.name}] by '{filter_col}': "
			f"kept {kept_count}/{total_count} rows "
			f"({kept_count / max(total_count, 1):.1%}) | "
			f"mean={kept_mean:.4f} std={kept_std:.4f}"
		)

	row_keep_fraction = synth_policy["row_keep_fraction"]
	if row_keep_fraction is not None and row_keep_fraction < 1.0:
		n_rows = len(filtered)
		keep_n = max(1, int(np.floor(n_rows * row_keep_fraction)))
		if keep_n < n_rows:
			# Deterministic row sampling to keep runs reproducible.
			filtered = filtered.sample(n=keep_n, random_state=42).copy()
			print(
				f"  Synthetic row subsample [{path_obj.name}]: "
				f"kept {keep_n}/{n_rows} rows ({row_keep_fraction:.1%})."
			)

	return filtered


def load_dataset(file_paths, config: dict[str, Any], feature_context: dict[str, Any]):
	if not isinstance(file_paths, list):
		file_paths = [file_paths]

	data_list = []
	for file_path in file_paths:
		path_obj = Path(file_path)
		df = pd.read_csv(path_obj)
		if "smiles" not in df.columns:
			raise ValueError(f"Missing required 'smiles' column in {path_obj}")

		is_synthetic = path_obj.name.startswith("synthetic_data_iteration_")
		if is_synthetic:
			df = _filter_synthetic_rows(df, path_obj, config)

		target_col = choose_target_column(df, path_obj, config)

		for _, row in df.iterrows():
			smiles = row["smiles"]
			target = row[target_col]

			if pd.isna(smiles) or pd.isna(target):
				continue

			graph = smiles_to_graph(str(smiles), float(target), config, feature_context)
			if graph is not None:
				graph.is_synthetic = bool(is_synthetic)
				data_list.append(graph)

	return data_list


def get_targets_from_graphs(data_list):
	return np.array([float(data.y.item()) for data in data_list], dtype=np.float32)


def baseline_rmse(train_targets: np.ndarray, val_targets: np.ndarray) -> float:
	train_mean = float(np.mean(train_targets))
	preds = np.full_like(val_targets, train_mean, dtype=np.float32)
	return float(np.sqrt(mean_squared_error(val_targets, preds)))


def safe_nanmean(values) -> float:
	return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def print_fold_data_summary(val_idx: int, train_data, val_data) -> None:
	train_targets = get_targets_from_graphs(train_data)
	val_targets = get_targets_from_graphs(val_data)
	fold_baseline_rmse = baseline_rmse(train_targets, val_targets)
	real_train, synthetic_train = split_real_and_synthetic(train_data)

	summary = (
		f"[Fold {val_idx}] train_n={len(train_data)} val_n={len(val_data)} "
		f"train_mean={np.mean(train_targets):.4f} val_mean={np.mean(val_targets):.4f} "
		f"baseline_rmse={fold_baseline_rmse:.4f}"
	)
	if len(synthetic_train) > 0:
		summary += (
			f" | train_original_n={len(real_train)} "
			f"train_synthetic_n={len(synthetic_train)}"
		)
	print(summary)


def split_real_and_synthetic(data_list: list[Data]) -> tuple[list[Data], list[Data]]:
	real_data = []
	synthetic_data = []
	for graph in data_list:
		if bool(getattr(graph, "is_synthetic", False)):
			synthetic_data.append(graph)
		else:
			real_data.append(graph)
	return real_data, synthetic_data


def cap_synthetic_train_ratio(
	train_data: list[Data],
	*,
	max_ratio: float | None,
	seed: int,
	fold_idx: int,
) -> list[Data]:
	if max_ratio is None:
		return train_data

	real_data, synthetic_data = split_real_and_synthetic(train_data)
	if not real_data or not synthetic_data:
		return train_data

	max_synth = int(np.floor(len(real_data) * max_ratio))
	if len(synthetic_data) <= max_synth:
		print(
			f"[Fold {fold_idx}] synthetic:real cap inactive "
			f"(ratio={len(synthetic_data) / max(len(real_data), 1):.3f} <= {max_ratio:.3f})."
		)
		return train_data

	rng = np.random.default_rng(seed + fold_idx)
	selected_indices = rng.choice(len(synthetic_data), size=max_synth, replace=False)
	selected_synth = [synthetic_data[int(i)] for i in selected_indices]

	rebalanced = real_data + selected_synth
	rng.shuffle(rebalanced)
	print(
		f"[Fold {fold_idx}] applied synthetic:real cap={max_ratio:.3f} | "
		f"real={len(real_data)} synthetic={len(selected_synth)} "
		f"(was {len(synthetic_data)} synthetic)."
	)
	return rebalanced


def run_synthetic_pretraining_stage(
	model: torch.nn.Module,
	synthetic_data: list[Data],
	*,
	config: dict[str, Any],
	fold_idx: int,
	target_std_cfg: dict[str, Any],
) -> None:
	pre_cfg = get_synthetic_pretraining_config(config)
	if not pre_cfg["enabled"] or pre_cfg["epochs"] <= 0:
		return

	if len(synthetic_data) < pre_cfg["min_synthetic_graphs"]:
		print(
			f"[Fold {fold_idx}] skipping synthetic pretraining: "
			f"only {len(synthetic_data)} synthetic graphs "
			f"(< min_synthetic_graphs={pre_cfg['min_synthetic_graphs']})."
		)
		return

	learning_rate = _resolve_optional(
		pre_cfg["learning_rate"],
		config["optimization"]["learning_rate"],
	)
	weight_decay = _resolve_optional(
		pre_cfg["weight_decay"],
		config["optimization"]["weight_decay"],
	)
	grad_clip_norm = _resolve_optional(
		pre_cfg["grad_clip_norm"],
		config["optimization"].get("grad_clip_norm", 0.0),
	)

	target_standardizer = None
	if pre_cfg["use_target_standardization"]:
		synth_targets = get_targets_from_graphs(synthetic_data)
		target_standardizer = build_target_standardizer(
			synth_targets,
			epsilon=target_std_cfg["epsilon"],
		)

	synth_loader = DataLoader(
		synthetic_data,
		batch_size=pre_cfg["batch_size"],
		shuffle=pre_cfg["shuffle"],
	)
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=float(learning_rate),
		weight_decay=float(weight_decay),
	)
	criterion = torch.nn.MSELoss()

	print(
		f"[Fold {fold_idx}] synthetic pretraining: "
		f"epochs={pre_cfg['epochs']} n={len(synthetic_data)} "
		f"lr={float(learning_rate):.2e} wd={float(weight_decay):.2e}"
	)
	for epoch in range(1, pre_cfg["epochs"] + 1):
		model.train()
		epoch_loss = 0.0
		for batch in synth_loader:
			batch = batch.to(device)
			optimizer.zero_grad()
			out = model(batch).view(-1)
			y_true = standardize_batch_targets(
				batch.y.view(-1),
				target_standardizer=target_standardizer,
			)
			loss = criterion(out, y_true)
			loss.backward()
			if grad_clip_norm and float(grad_clip_norm) > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
			optimizer.step()
			epoch_loss += loss.item() * batch.num_graphs

		epoch_loss /= max(len(synthetic_data), 1)
		print_every = max(1, pre_cfg["epochs"] // 5)
		if epoch % print_every == 0 or epoch == pre_cfg["epochs"]:
			print(
				f"[Fold {fold_idx}][Synthetic pretrain][Epoch {epoch:03d}] "
				f"train_mse={'(scaled) ' if target_standardizer is not None else ''}{epoch_loss:.4f}"
			)


# ==================== training helpers ====================
def evaluate(
	model: torch.nn.Module,
	loader: DataLoader,
	*,
	target_standardizer: dict[str, Any] | None = None,
) -> tuple:
	model.eval()
	y_true, y_pred = [], []

	with torch.no_grad():
		for data in loader:
			data = data.to(device)
			out = model(data).view(-1)
			true = data.y.view(-1)
			out = invert_standardized_predictions(
				out,
				target_standardizer=target_standardizer,
			)

			y_true.extend(true.detach().cpu().numpy().flatten())
			y_pred.extend(out.detach().cpu().numpy().flatten())

	mse = mean_squared_error(y_true, y_pred)
	rmse = float(np.sqrt(mse))
	r2 = r2_score(y_true, y_pred)
	rho = spearmanr(y_true, y_pred)[0]
	return mse, rmse, r2, rho


def metric_from_name(metric_name: str, mse: float, rmse: float, r2: float, rho: float) -> float:
	values = {
		"mse": mse,
		"rmse": rmse,
		"r2": r2,
		"rho": rho,
	}
	if metric_name not in values:
		raise ValueError(
			f"Unsupported metric '{metric_name}'. Use one of: {list(values.keys())}"
		)
	return values[metric_name]


def metric_description(metric_name: str) -> str:
	descriptions = {
		"mse": "Validation mean squared error; lower is better.",
		"rmse": "Validation root mean squared error; lower is better.",
		"r2": "Validation R2 score; higher is better.",
		"rho": "Validation Spearman rank correlation; higher is better.",
	}
	if metric_name not in descriptions:
		raise ValueError(
			f"Unsupported metric '{metric_name}'. Use one of: {list(descriptions.keys())}"
		)
	return descriptions[metric_name]


def is_improvement(metric_name: str, current: float, best: float, min_delta: float) -> bool:
	# Lower is better for mse/rmse. Higher is better for r2/rho.
	if metric_name in {"mse", "rmse"}:
		return current < (best - min_delta)
	return current > (best + min_delta)


def build_model(model_class, config: dict[str, Any], feature_context: dict[str, Any]):
	return model_class(
		node_in_dim=feature_context["atom_feature_dim"],
		edge_in_dim=feature_context["edge_feature_dim"],
		hidden_dim=config["model"]["hidden_dim"],
		num_conv_layers=config["model"]["num_conv_layers"],
		heads=config["model"]["heads"],
		ffnn_hidden_layers=config["model"]["ffnn_hidden_layers"],
		residual=config["model"].get("residual", True),
		normalization=str(config["model"].get("normalization", "layernorm")).lower(),
		dropout=config["model"]["dropout"],
	).to(device)


def build_summary_dataframe(
	cv_results: list[dict[str, Any]],
	best_fold_idx: int,
	test_mse: float,
	test_rmse: float,
	test_r2: float,
	test_rho: float,
) -> pd.DataFrame:
	summary_data = []
	for fold_result in cv_results:
		summary_data.append(
			{
				"Stage": f"Fold_{fold_result['fold_idx']}_Val",
				"MonitorMetric": fold_result["monitor_metric"],
				"MonitorDescription": fold_result["monitor_metric_description"],
				"BestMonitorValue": fold_result["best_monitor_value"],
				"ValLoss_MSE": fold_result["val_loss_mse"],
				"MSE": fold_result["val_mse"],
				"RMSE": fold_result["val_rmse"],
				"R2": fold_result["val_r2"],
				"Rho": fold_result["val_rho"],
			}
		)

	avg_best_monitor = safe_nanmean([res["best_monitor_value"] for res in cv_results])
	avg_val_mse = safe_nanmean([res["val_mse"] for res in cv_results])
	avg_val_rmse = safe_nanmean([res["val_rmse"] for res in cv_results])
	avg_val_r2 = safe_nanmean([res["val_r2"] for res in cv_results])
	avg_val_rho = safe_nanmean([res["val_rho"] for res in cv_results])

	summary_data.append(
		{
			"Stage": "Average_Val",
			"MonitorMetric": cv_results[0]["monitor_metric"] if cv_results else "",
			"MonitorDescription": (
				cv_results[0]["monitor_metric_description"] if cv_results else ""
			),
			"BestMonitorValue": avg_best_monitor,
			"ValLoss_MSE": avg_val_mse,
			"MSE": avg_val_mse,
			"RMSE": avg_val_rmse,
			"R2": avg_val_r2,
			"Rho": avg_val_rho,
		}
	)
	summary_data.append(
		{
			"Stage": f"Holdout_Test (Model from Fold_{best_fold_idx})",
			"MonitorMetric": "",
			"MonitorDescription": "",
			"BestMonitorValue": "",
			"ValLoss_MSE": "",
			"MSE": test_mse,
			"RMSE": test_rmse,
			"R2": test_r2,
			"Rho": test_rho,
		}
	)

	return pd.DataFrame(summary_data)


def write_losses_file(loss_path: Path, losses: dict[str, dict[str, list[float]]]) -> None:
	with open(loss_path, "w", encoding="utf-8") as handle:
		for fold, loss_dict in losses.items():
			df = pd.DataFrame(loss_dict)
			handle.write(f"{fold}:\n")
			handle.write(df.to_string(index=False))
			handle.write("\n\n")


# ==================== checkpoint helpers ====================
def get_fold_checkpoint_path(target_folder: str, fold_idx: int) -> Path:
	"""Canonical path for a per-fold best-model checkpoint.

	Checkpoints live inside the dataset folder so each dataset mix keeps
	its own set of saved weights and they never overwrite each other.
	"""
	return Path(target_folder) / "checkpoints" / f"best_model_fold_{fold_idx}.pth"


def load_fold_checkpoint(
	checkpoint_path: Path,
	config: dict[str, Any],
	model_class,
	feature_context: dict[str, Any],
) -> tuple:
	"""Load a fold checkpoint and return (model, target_standardizer | None).

	Handles both the current rich format::

		{"state_dict": ..., "fold": int, "target_mean": float,
		 "target_std": float, "val_rmse": float}

	and the legacy bare state_dict format for backward compatibility.
	"""
	ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

	if isinstance(ckpt, dict) and "state_dict" in ckpt:
		state_dict = ckpt["state_dict"]
		target_mean = float(ckpt.get("target_mean", 0.0))
		target_std = float(ckpt.get("target_std", 1.0))
	else:
		# Legacy format: checkpoint is a plain state_dict (OrderedDict).
		state_dict = ckpt
		target_mean = 0.0
		target_std = 1.0

	model = build_model(model_class, config, feature_context)
	model.load_state_dict(state_dict)
	model.eval()

	use_std = _get_target_standardization_config(config)["enabled"]
	target_standardizer = None
	if use_std:
		target_standardizer = build_target_standardizer_from_stats(target_mean, target_std)

	return model, target_standardizer


def run_training_pipeline(config: dict[str, Any], model_class) -> None:
	set_seed(config["experiment"]["seed"])
	feature_context = prepare_feature_config(config)
	print_feature_summary(config, feature_context)

	target_std_cfg = _get_target_standardization_config(config)
	use_target_standardization = target_std_cfg["enabled"]
	if use_target_standardization:
		print(
			"\nTarget standardization: ENABLED (per fold) "
			f"| epsilon={target_std_cfg['epsilon']:.1e}"
		)
	else:
		print("\nTarget standardization: DISABLED")

	target_folder = config["experiment"]["target_folder"]
	actual_test_file = config["experiment"]["actual_test_file"]
	total_folds = config["experiment"]["total_folds"]
	batch_size = config["data"]["batch_size"]
	num_epochs = config["training"]["num_epochs"]
	print_every = config["training"]["print_every"]

	learning_rate = config["optimization"]["learning_rate"]
	weight_decay = config["optimization"]["weight_decay"]
	grad_clip_norm = config["optimization"]["grad_clip_norm"]

	early_stop_cfg = config["training"]["early_stopping"]
	early_stop_enabled = early_stop_cfg["enabled"]
	early_stop_metric = early_stop_cfg["monitor_metric"]
	early_stop_patience = early_stop_cfg["patience"]
	early_stop_min_delta = early_stop_cfg["min_delta"]

	scheduler_cfg = config["scheduler"]
	scheduler_enabled = scheduler_cfg["enabled"]
	scheduler_metric = scheduler_cfg["monitor_metric"]

	print(f"\nLoading holdout test set: {actual_test_file}")
	test_data = load_dataset(actual_test_file, config, feature_context)
	test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

	global_best_val_rmse = float("inf")
	global_best_weights = None
	best_fold_idx = -1
	global_best_target_mean = None
	global_best_target_std = None
	cv_results = []

	losses = {f"fold_{i}": {} for i in range(total_folds)}

	ckpt_dir = Path(target_folder) / "checkpoints"
	ckpt_dir.mkdir(parents=True, exist_ok=True)

	print("\n==================== 5-Fold Cross-Validation ====================")
	for val_idx in range(total_folds):
		print(f"\n-------------------- Fold {val_idx} (validation) --------------------")

		train_paths, val_paths = get_fold_files(
			target_folder,
			val_idx,
			config,
			total_folds=total_folds,
		)
		train_data = load_dataset(train_paths, config, feature_context)
		val_data = load_dataset(val_paths, config, feature_context)

		pretrain_synth_paths = get_synthetic_pretraining_files(
			target_folder,
			val_idx,
			config,
		)
		pretrain_synth_data = []
		if pretrain_synth_paths:
			pretrain_synth_data = load_dataset(pretrain_synth_paths, config, feature_context)


		model = build_model(model_class, config, feature_context)

		run_synthetic_pretraining_stage(
			model,
			pretrain_synth_data,
			config=config,
			fold_idx=val_idx,
			target_std_cfg=target_std_cfg,
		)

		synth_policy = get_synthetic_cv_config(config)
		finetune_train_data = cap_synthetic_train_ratio(
			train_data,
			max_ratio=synth_policy["max_train_synth_to_real_ratio"],
			seed=config["experiment"]["seed"],
			fold_idx=val_idx,
		)

		print_fold_data_summary(val_idx, finetune_train_data, val_data)

		target_standardizer = None
		train_target_mean = 0.0
		train_target_std = 1.0
		if use_target_standardization:
			train_targets = get_targets_from_graphs(finetune_train_data)
			target_standardizer = build_target_standardizer(
				train_targets,
				epsilon=target_std_cfg["epsilon"],
			)
			train_target_mean = target_standardizer["mean"]
			train_target_std = target_standardizer["std"]
			print(
				f"[Fold {val_idx}] target standardizer: "
				f"mean={train_target_mean:.4f} std={train_target_std:.4f}"
			)

		train_loader = DataLoader(
			finetune_train_data,
			batch_size=batch_size,
			shuffle=config["data"]["shuffle_train"],
		)
		val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

		optimizer = torch.optim.Adam(
			model.parameters(), lr=learning_rate, weight_decay=weight_decay
		)
		criterion = torch.nn.MSELoss()

		scheduler = None
		if scheduler_enabled:
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizer,
				mode=scheduler_cfg["mode"],
				factor=scheduler_cfg["factor"],
				patience=scheduler_cfg["patience"],
			)

		print(
			f"[Fold {val_idx}] model params={model_class.count_parameters(model)} "
			f"| lr={learning_rate:.2e} | dropout={config['model']['dropout']:.2f} "
			f"| residual={config['model'].get('residual', True)} "
			f"| norm={str(config['model'].get('normalization', 'layernorm')).lower()}"
		)

		best_val_mse = float("inf")
		best_val_rmse = float("inf")
		best_val_r2 = -float("inf")
		best_val_rho = -float("inf")
		best_epoch = 0
		best_model_weights = None

		if early_stop_metric in {"mse", "rmse"}:
			best_monitor_value = float("inf")
		else:
			best_monitor_value = -float("inf")

		train_losses = []
		val_losses = []
		monitor_values = []
		lr_history = []
		epochs_wo_improv = 0

		for epoch in range(1, num_epochs + 1):
			model.train()
			train_loss = 0.0

			for batch in train_loader:
				batch = batch.to(device)
				optimizer.zero_grad()
				out = model(batch).view(-1)

				y_true = standardize_batch_targets(
					batch.y.view(-1),
					target_standardizer=target_standardizer,
				)
				loss = criterion(out, y_true)
				loss.backward()

				if grad_clip_norm and grad_clip_norm > 0:
					torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

				optimizer.step()
				train_loss += loss.item() * batch.num_graphs

			train_loss /= len(finetune_train_data)
			val_mse, val_rmse, val_r2, val_rho = evaluate(
				model,
				val_loader,
				target_standardizer=target_standardizer,
			)

			# current monitor is the valye we compare aginst best value for early stopping and scheduler decisions
			current_monitor = metric_from_name(
				early_stop_metric, val_mse, val_rmse, val_r2, val_rho
			)

			if is_improvement(
				early_stop_metric, current_monitor, best_monitor_value, early_stop_min_delta
			):
				best_monitor_value = current_monitor
				best_val_mse = val_mse
				best_val_rmse = val_rmse
				best_val_r2 = val_r2
				best_val_rho = val_rho
				best_epoch = epoch
				best_model_weights = copy.deepcopy(model.state_dict())
				epochs_wo_improv = 0
				torch.save(
					{
						"state_dict": best_model_weights,
						"fold": val_idx,
						"target_mean": train_target_mean,
						"target_std": train_target_std,
						"val_rmse": val_rmse,
					},
					get_fold_checkpoint_path(target_folder, val_idx),
				)
			else:
				epochs_wo_improv += 1

			if scheduler is not None:
				scheduler_value = metric_from_name(
					scheduler_metric, val_mse, val_rmse, val_r2, val_rho
				)
				scheduler.step(scheduler_value)

			train_losses.append(train_loss)
			val_losses.append(val_mse)
			monitor_values.append(current_monitor)
			lr_history.append(float(optimizer.param_groups[0]["lr"]))

			if epoch % print_every == 0:
				print(
					f"[Fold {val_idx}][Epoch {epoch:03d}] "
					f"train_mse={'(scaled) ' if use_target_standardization else ''}{train_loss:.4f} "
					f"val_mse={val_mse:.4f} val_rmse={val_rmse:.4f} "
					f"val_r2={val_r2:.4f} val_rho={val_rho:.4f} "
					f"lr={optimizer.param_groups[0]['lr']:.2e} "
					f"best_val_rmse={best_val_rmse:.4f} "
					f"no_improve={epochs_wo_improv}"
				)

			if early_stop_enabled and epochs_wo_improv >= early_stop_patience:
				print(
					f"[Fold {val_idx}] early stopping at epoch {epoch} "
					f"(monitor={early_stop_metric}, patience={early_stop_patience}, "
					f"min_delta={early_stop_min_delta})"
				)
				break

		print(
			f"[Fold {val_idx}] best epoch={best_epoch} | "
			f"val_rmse={best_val_rmse:.4f} val_r2={best_val_r2:.4f} val_rho={best_val_rho:.4f}"
		)

		cv_results.append(
			{
				"fold_idx": val_idx,
				"monitor_metric": early_stop_metric,
				"monitor_metric_description": metric_description(early_stop_metric),
				"best_monitor_value": best_monitor_value,
				"val_loss_mse": best_val_mse,
				"val_mse": best_val_mse,
				"val_rmse": best_val_rmse,
				"val_r2": best_val_r2,
				"val_rho": best_val_rho,
			}
		)

		if best_val_rmse < global_best_val_rmse:
			global_best_val_rmse = best_val_rmse
			global_best_weights = copy.deepcopy(best_model_weights)
			best_fold_idx = val_idx
			global_best_target_mean = train_target_mean
			global_best_target_std = train_target_std
			print(f"[Fold {val_idx}] is current global best.")

		losses[f"fold_{val_idx}"] = {
			"train": train_losses,
			"val": val_losses,
			"monitor": monitor_values,
			"lr": lr_history,
		}

	print("\n-------------------- Holdout Evaluation --------------------")
	print(
		f"Loading global-best model from fold {best_fold_idx} "
		f"(val_rmse={global_best_val_rmse:.4f})"
	)
	if global_best_weights is None:
		raise RuntimeError("No best model weights were saved during cross-validation.")

	best_model = build_model(model_class, config, feature_context)
	best_model.load_state_dict(global_best_weights)
	# Evaluate the best model on the holdout test set

	best_target_standardizer = None
	if use_target_standardization:
		if global_best_target_mean is None or global_best_target_std is None:
			raise RuntimeError(
				"Target standardization is enabled, but best-fold stats were not captured."
			)
		best_target_standardizer = build_target_standardizer_from_stats(
			float(global_best_target_mean),
			float(global_best_target_std),
		)

	test_mse, test_rmse, test_r2, test_rho = evaluate(
		best_model,
		test_loader,
		target_standardizer=best_target_standardizer,
	)
	print(
		f"Holdout -> MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, "
		f"R2: {test_r2:.4f}, Rho: {test_rho:.4f}"
	)

	print("\n-------------------- Final CV Summary --------------------")
	df_results = build_summary_dataframe(
		cv_results,
		best_fold_idx,
		test_mse,
		test_rmse,
		test_r2,
		test_rho,
	)
	print(df_results.to_string(index=False))

	target_path_obj = Path(target_folder)
	folder_name = target_path_obj.name if target_folder != "." else "default_experiment"

	save_path = target_path_obj / f"MPNN_results_{folder_name}.csv"
	df_results.to_csv(save_path, index=False)
	print(f"Saved results to: {save_path.resolve()}")

	loss_path = target_path_obj / f"MPNN_losses_{folder_name}.txt"
	write_losses_file(loss_path, losses)
	print(f"Saved learning curves to: {loss_path.resolve()}")