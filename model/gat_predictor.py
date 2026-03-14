import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from gat_utils import run_training_pipeline


# ==================== migration notes ====================
# Detailed documentation: see 'MPNN_predictor_migration.md'.
#
# Summary of changes vs 'orig_MPNN_predictor.py':
# - Refactor: most non-model logic moved into 'gat_utils.py' (data loading,
#   featurization, training loop, evaluation, exporting).
# - Config: added centralized nested 'CONFIG' dict (paths, data, features, model,
#   optimization, scheduler, training).
# - Model: 'GATModel' is now configurable (num_conv_layers/heads/dropout/FFNN),
#   with optional residual connections and per-conv normalization.
# - Targets: optional per-fold target standardization (fit mean/std on train fold
#   only; invert predictions back to original scale for metrics/reporting).
# - Features: atom/node + bond/edge descriptors are config-driven ('atom_descriptors',
#   'bond_descriptors') with categorical encoding modes (one_hot/index + unknown).
# - Training stability: reproducible seed, gradient clipping, ReduceLROnPlateau,
#   early stopping with min_delta + configurable monitor metric.
# - Reporting: expanded metrics (MSE/RMSE/R2/Spearman rho), standardized CV
#   summary CSV columns, and learning-curve export.


# ==================== changelog ====================
# 2026-03-12
# - Added a centralized nested CONFIG dictionary for all important settings.
# - Switched categorical atom/bond fields to one-hot (+unknown) encoding.
# - Added stronger training stability patches: min-delta early stopping,
#   configurable monitor metric, gradient clipping, and improved scheduler usage.
# - Removed duplicate test-set loading and improved fold-level diagnostics printing.
# - Kept optimization loss as MSE, while early stopping defaults to RMSE.
# - Added per-fold target standardization (fit on train fold only, invert for reported metrics).
# - Added config-driven atom/bond (node/edge) descriptor selection for modular featurization.
# 2026-03-12 PARAM SWEEP - Anti-Overfitting Config
# - DIAGNOSIS: Model overfitting severely (val loss progression 1.5 -> 19.9+)
# wer params)
# Fix: NO DROPOUT IN CONV LAYERS!!!!!
# 2026-03-14
# - Added "featurizer" option to CONFIG["features"]: "custom" (default, RDKit-based,
#   fully configurable via atom_descriptors / bond_descriptors) or "chemprop_simple"
#   (uses chemprop's SimpleMoleculeMolGraphFeaturizer as a drop-in alternative;
#   atom_descriptors / bond_descriptors are ignored in this mode).

# ==================== configuration ====================
CONFIG = {
	"experiment": {
		"target_folder": "./0%",  # Change to ./0% or ./33% or ./67% for other experiments.
		"actual_test_file": "heldout_testset.csv",
		"total_folds": 5,
		"seed": 42,
	},
	"data": {
		"batch_size": 64,
		"shuffle_train": True,
		"real_target_column": "pIC50",
		"synthetic_target_column": "target_pIC50",
		"fallback_target_columns": ["target_pIC50", "pred_pIC50"],
		"synthetic_cv": {
			# Training uses the synthetic file matching the CV iteration index.
			# Even if this is turned off, synthetic data can be used for pre-training
			"include_in_training": True,
			# Options: "matching_fold", "all", "none".
			"train_selection": "matching_fold",
			# Default keeps synthetic data out of validation.
			"include_in_validation": False,
			# Leakage-safe validation strategy:
			# - "all_except_train_and_fold": use all synthetic iterations except
			#   the current fold iteration and any synthetic iterations used in training.
			# - "single_next_non_train": use one deterministic non-overlapping iteration.
			# - "none": do not add synthetic rows to validation.
			"validation_selection": "single_next_non_train",
			# Keep the central fraction of synthetic rows by value (1.0 = keep all).
			# Example: 0.95 keeps the middle 95% and removes 2.5% from each tail.
			"keep_percentile": 0.5,
			# Uniform random row subsample after percentile filtering.
			# Useful for reducing pseudo-label noise volume while keeping diversity.
			# Set to None to disable row subsampling entirely.
			"row_keep_fraction": 1.0,
			# Optional cap for synthetic-to-real ratio in finetuning train folds.
			# Example: 1.0 keeps at most as many synthetic graphs as real graphs.
			# Set to None to disable ratio capping.
			"max_train_synth_to_real_ratio": None,
			# Which synthetic pIC50 column to use both for filtering and training labels.
			# Options: "pred" (pred_pIC50) or "target" (target_pIC50).
			"label_source": "target",
		},
	},
	"features": {
		"scale_atomic_mass_by": 100.0,
		"categorical_encoding_mode": "one_hot", # options: "one_hot", "index", "index_with_unknown"
		# Which molecule-to-graph featurizer to use.
		# Options:
		#   "custom"          - custom RDKit featurizer driven by atom_descriptors and
		#                       bond_descriptors below; fully configurable.
		#   "chemprop_simple" - chemprop's SimpleMoleculeMolGraphFeaturizer (fixed
		#                       ~72-dim atom features, ~14-dim bond features). The
		#                       atom_descriptors and bond_descriptors settings below
		#                       are ignored in this mode.
		"featurizer": "custom",
		"atom_descriptors": [
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
		],
		"bond_descriptors": [
			"bond_type",
			"bond_order",
			"is_conjugated",
			"is_in_ring",
			"bond_stereo",
		],
	},
	"model": {
		"hidden_dim": 256,
		"num_conv_layers": 3,
		"heads": 4,
		"dropout": 0.2,
		"residual": True,
		"normalization": "layernorm", # options: "layernorm", "batchnorm1d", "none"
		"ffnn_hidden_layers": [256], # and was [256,512], [256,128,64] orig was [256], but added an extra layer to increase capacity without widening too much and overfitting.
	},
	"optimization": {
		"learning_rate": 1e-3,
		"weight_decay": 1e-4,
		"grad_clip_norm": 8.0,
	},
	"scheduler": {
		"enabled": True, # Reduce on Plateau. 
		"mode": "min",
		"factor": 0.75,
		"patience": 3,
		"monitor_metric": "rmse",
	},
	"training": {
		"num_epochs": 200,
		"print_every": 1,
		# Stage 1 in two-stage training: synthetic-only pretraining per fold,
		# then Stage 2 finetunes on the fold training set.
		"synthetic_pretraining": {
			"enabled": True,
			"epochs": 25,
			"learning_rate": 8e-4,
			"weight_decay": 1e-4,
			"grad_clip_norm": 8.0,
			"batch_size": 64,
			"shuffle": True, #
			"min_synthetic_graphs": 64, # Ensure at least one batch for pretraining, even if synthetic data is very limited after filtering.
			# Usually False for pseudo labels; turn on only if target scale drift is large.
			"use_target_standardization": False,
		},
		"target_standardization": {
			"enabled": False,
			"epsilon": 1e-8,
		},
		"early_stopping": {
			"enabled": True,
			"monitor_metric": "rmse",
			"patience": 15,
			"min_delta": 0.0,
		},
	},
}

# ==================== model ====================
class GATModel(torch.nn.Module):
	def __init__(
		self,
		node_in_dim: int,
		edge_in_dim: int,
		hidden_dim: int,
		num_conv_layers: int,
		heads: int,
		ffnn_hidden_layers: list[int],
		residual: bool = True,
		normalization: str = "layernorm",
		dropout: float = 0.2,
	):
		super().__init__()

		if num_conv_layers < 1:
			raise ValueError("num_conv_layers must be >= 1")

		normalization = str(normalization).lower()
		if normalization not in {"layernorm", "batchnorm1d", "none"}:
			raise ValueError(
				"normalization must be 'layernorm', 'batchnorm1d', or 'none'."
			)

		self.residual = bool(residual)
		self.normalization = normalization

		self.convs = torch.nn.ModuleList()
		self.norm_layers = torch.nn.ModuleList()
		self.residual_projections = torch.nn.ModuleList()
		conv_in_dim = node_in_dim
		for _ in range(num_conv_layers):
			# NOTE: DO NOT APPLY DROPOUT HERE! GATv2Conv already applies dropout to attention coefficients internally.
			self.convs.append(
				GATv2Conv(
					conv_in_dim,
					hidden_dim,
					heads=heads,
					edge_dim=edge_in_dim,
					concat=False,
				)
			)
			self.norm_layers.append(self._build_norm_layer(hidden_dim, normalization))
			if conv_in_dim == hidden_dim:
				self.residual_projections.append(torch.nn.Identity())
			else:
				self.residual_projections.append(
					torch.nn.Linear(conv_in_dim, hidden_dim, bias=False)
				)
			conv_in_dim = hidden_dim

		if ffnn_hidden_layers is None:
			ffnn_hidden_layers = [hidden_dim]
		if not isinstance(ffnn_hidden_layers, list):
			raise ValueError("ffnn_hidden_layers must be a list, e.g. [128, 256, 512]")
		if any((not isinstance(x, int)) or x <= 0 for x in ffnn_hidden_layers):
			raise ValueError("All values in ffnn_hidden_layers must be positive integers.")

		self.ffnn_layers = torch.nn.ModuleList()
		ffnn_in_dim = hidden_dim * 2
		for layer_dim in ffnn_hidden_layers:
			self.ffnn_layers.append(torch.nn.Linear(ffnn_in_dim, layer_dim))
			ffnn_in_dim = layer_dim

		self.output_layer = torch.nn.Linear(ffnn_in_dim, 1)
		self.dropout_layer = torch.nn.Dropout(p=dropout)

	@staticmethod
	def _build_norm_layer(hidden_dim: int, normalization: str) -> torch.nn.Module:
		if normalization == "layernorm":
			return torch.nn.LayerNorm(hidden_dim)
		if normalization == "batchnorm1d":
			return torch.nn.BatchNorm1d(hidden_dim)
		return torch.nn.Identity()

	def forward(self, data):
		x, edge_index, edge_attr, batch = (
			data.x,
			data.edge_index,
			data.edge_attr,
			data.batch,
		)

		for conv, norm_layer, residual_projection in zip(
			self.convs,
			self.norm_layers,
			self.residual_projections,
		):
			x_in = x
			x = conv(x, edge_index, edge_attr)
			x = F.elu(x)
			x = norm_layer(x)
			if self.residual:
				x = x + residual_projection(x_in)

		x_mean = global_mean_pool(x, batch)
		x_max = global_max_pool(x, batch)
		x = torch.cat([x_mean, x_max], dim=1)

		for layer in self.ffnn_layers:
			x = F.elu(layer(x))
			x = self.dropout_layer(x)
		return self.output_layer(x)

	@staticmethod
	def count_parameters(model):
		return sum(p.numel() for p in model.parameters() if p.requires_grad)

	@staticmethod
	def print_model_info(model):
		total_params = GATModel.count_parameters(model)
		print(f"Total trainable parameters: {total_params}")
		for name, param in model.named_parameters():
			if param.requires_grad:
				print(f"  {name}: {param.numel()} params | shape={tuple(param.shape)}")


# ==================== main ====================
if __name__ == "__main__":
	run_training_pipeline(CONFIG, GATModel)
