from __future__ import annotations

from typing import Any

import numpy as np
import torch
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# Only continuous descriptor columns are scaled by sklearn-based feature scaling.
SCALABLE_ATOM_FEATURE_NAMES = {
	"atomic_num",
	"degree",
	"total_num_hs",
	"formal_charge",
	"implicit_valence",
	"mass_scaled",
}
SCALABLE_EDGE_FEATURE_NAMES = {
	"bond_order",
}


# ==================== chemprop featurizer (optional alternative) ====================
_chemprop_featurizer_instance = None


def set_seed(seed: int) -> None:
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def get_feature_scaling_config(config: dict[str, Any]) -> dict[str, Any]:
	scaling_cfg = config.get("features", {}).get("feature_scaling", {})
	if not isinstance(scaling_cfg, dict):
		scaling_cfg = {}

	mode = str(scaling_cfg.get("mode", "none")).lower().strip()
	if mode not in {"none", "standard", "minmax"}:
		raise ValueError(
			"CONFIG['features']['feature_scaling']['mode'] must be one of "
			"{'none', 'standard', 'minmax'}."
		)

	fit_on = str(scaling_cfg.get("fit_on", "real_plus_synthetic")).lower().strip()
	if fit_on not in {"real_plus_synthetic", "real_only"}:
		raise ValueError(
			"CONFIG['features']['feature_scaling']['fit_on'] must be one of "
			"{'real_plus_synthetic', 'real_only'}."
		)
	return {
		"mode": mode,
		"fit_on": fit_on,
		"enabled": mode != "none",
	}


def _build_sklearn_scaler(mode: str):
	if mode == "standard":
		return StandardScaler()
	if mode == "minmax":
		return MinMaxScaler()
	raise ValueError(f"Unsupported feature scaler mode: {mode}")


def _collect_feature_matrix(
	data_list: list[Data],
	*,
	attr_name: str,
	column_indices: list[int],
) -> np.ndarray | None:
	if not column_indices:
		return None

	blocks = []
	for graph in data_list:
		tensor = getattr(graph, attr_name)
		if tensor is None or tensor.numel() == 0 or tensor.shape[0] == 0:
			continue
		arr = tensor.detach().cpu().numpy()
		blocks.append(arr[:, column_indices])

	if not blocks:
		return None
	return np.concatenate(blocks, axis=0)


def _transform_graph_columns(
	graph: Data,
	*,
	attr_name: str,
	column_indices: list[int],
	scaler,
) -> None:
	if scaler is None or not column_indices:
		return
	if not hasattr(graph, attr_name):
		return

	tensor = getattr(graph, attr_name)
	if tensor is None or tensor.numel() == 0 or tensor.shape[0] == 0:
		return

	arr = tensor.detach().cpu().numpy()
	arr[:, column_indices] = scaler.transform(arr[:, column_indices])
	setattr(graph, attr_name, torch.tensor(arr, dtype=torch.float32))


def _resolve_scalable_indices(feature_names: list[str], scalable_names: set[str]) -> list[int]:
	return [idx for idx, name in enumerate(feature_names) if name in scalable_names]


def fit_feature_scalers(
	train_data: list[Data],
	*,
	feature_context: dict[str, Any],
	config: dict[str, Any],
) -> dict[str, Any] | None:
	scaling_cfg = get_feature_scaling_config(config)
	if not scaling_cfg["enabled"]:
		return None
	if feature_context.get("featurizer") != "custom":
		print("[Feature scaling] skipped: sklearn scaling is only supported for featurizer='custom'.")
		return None

	mode = scaling_cfg["mode"]
	fit_on = scaling_cfg["fit_on"]
	scaler_fit_data = train_data
	if fit_on == "real_only":
		real_train_data = [g for g in train_data if not bool(getattr(g, "is_synthetic", False))]
		if real_train_data:
			scaler_fit_data = real_train_data
		else:
			print(
				"[Feature scaling] fit_on='real_only' requested but no real rows were found; "
				"falling back to real_plus_synthetic for scaler fitting."
			)
			scaler_fit_data = train_data

	node_feature_names = feature_context.get("atom_feature_names", [])
	edge_feature_names = feature_context.get("edge_feature_names", [])
	node_indices = _resolve_scalable_indices(node_feature_names, SCALABLE_ATOM_FEATURE_NAMES)
	edge_indices = _resolve_scalable_indices(edge_feature_names, SCALABLE_EDGE_FEATURE_NAMES)

	node_scaler = None
	node_matrix = _collect_feature_matrix(
		scaler_fit_data,
		attr_name="x",
		column_indices=node_indices,
	)
	if node_matrix is not None and node_matrix.shape[0] > 0:
		node_scaler = _build_sklearn_scaler(mode)
		node_scaler.fit(node_matrix)

	edge_scaler = None
	edge_matrix = _collect_feature_matrix(
		scaler_fit_data,
		attr_name="edge_attr",
		column_indices=edge_indices,
	)
	if edge_matrix is not None and edge_matrix.shape[0] > 0:
		edge_scaler = _build_sklearn_scaler(mode)
		edge_scaler.fit(edge_matrix)

	if node_scaler is None and edge_scaler is None:
		print("[Feature scaling] no eligible continuous feature columns found; skipping scaling.")
		return None

	print(
		f"[Feature scaling] mode={mode} fit_on={fit_on} | "
		f"node_cols={len(node_indices)} edge_cols={len(edge_indices)}"
	)
	return {
		"mode": mode,
		"fit_on": fit_on,
		"node_indices": node_indices,
		"edge_indices": edge_indices,
		"node_scaler": node_scaler,
		"edge_scaler": edge_scaler,
	}


def apply_feature_scalers(
	data_list: list[Data],
	*,
	feature_scalers: dict[str, Any] | None,
) -> list[Data]:
	if feature_scalers is None:
		return data_list

	node_indices = feature_scalers.get("node_indices", [])
	edge_indices = feature_scalers.get("edge_indices", [])
	node_scaler = feature_scalers.get("node_scaler")
	edge_scaler = feature_scalers.get("edge_scaler")

	for graph in data_list:
		_transform_graph_columns(
			graph,
			attr_name="x",
			column_indices=node_indices,
			scaler=node_scaler,
		)
		_transform_graph_columns(
			graph,
			attr_name="edge_attr",
			column_indices=edge_indices,
			scaler=edge_scaler,
		)
	return data_list


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
		scaling_cfg = get_feature_scaling_config(config)
		if scaling_cfg["enabled"]:
			print("[Feature scaling] ignored for featurizer='chemprop_simple'; using mode='none'.")
		config["features"]["atom_feature_dim"] = node_dim
		config["features"]["edge_feature_dim"] = edge_dim
		return {
			"featurizer": "chemprop_simple",
			"categorical_encoding_mode": "n/a",
			"feature_scaling_mode": "none",
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
	if encoding_mode == "index_with_unknown":
		encoding_mode = "index"
	if encoding_mode not in {"one_hot", "index"}:
		raise ValueError(
			"CONFIG['features']['categorical_encoding_mode'] must be 'one_hot', "
			"'index', or 'index_with_unknown'."
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

	scaling_cfg = get_feature_scaling_config(config)
	return {
		"featurizer": "custom",
		"categorical_encoding_mode": encoding_mode,
		"feature_scaling_mode": scaling_cfg["mode"],
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
		print(f"  Feature scaling mode: {feature_context.get('feature_scaling_mode', 'none')}")
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
