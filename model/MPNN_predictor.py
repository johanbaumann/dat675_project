import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from gat_utils import run_training_pipeline


# ==================== migration notes ====================
# Detailed documentation: see `MPNN_predictor_migration.md`.
#
# Summary of changes vs `orig_MPNN_predictor.py`:
# - Refactor: most non-model logic moved into `gat_utils.py` (data loading,
#   featurization, training loop, evaluation, exporting).
# - Config: added centralized nested `CONFIG` dict (paths, data, features, model,
#   optimization, scheduler, training).
# - Model: `GATModel` is now configurable (num_conv_layers/heads/dropout/FFNN),
#   with optional residual connections and per-conv normalization.
# - Targets: optional per-fold target standardization (fit mean/std on train fold
#   only; invert predictions back to original scale for metrics/reporting).
# - Features: atom/node + bond/edge descriptors are config-driven (`atom_descriptors`,
#   `bond_descriptors`) with categorical encoding modes (one_hot/index + unknown).
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

# ==================== configuration ====================
CONFIG = {
	"experiment": {
		"target_folder": "./0%",  # Change to ./33% or ./67% for other experiments.
		"actual_test_file": "heldout_testset.csv",
		"total_folds": 5,
		"seed": 42,
	},
	"data": {
		"batch_size": 64,
		"shuffle_train": True,
		"real_target_column": "pIC50",
		"synthetic_target_column": "pred_pIC50",
		"fallback_target_columns": ["target_pIC50", "pred_pIC50"],
	},
	"features": {
		"scale_atomic_mass_by": 100.0,
		"categorical_encoding_mode": "one_hot",
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
		"ffnn_hidden_layers": [256, 128], # orig was [256], but added an extra layer to increase capacity without widening too much and overfitting.
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
		"patience": 6,
		"monitor_metric": "rmse",
	},
	"training": {
		"num_epochs": 200,
		"print_every": 1,
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
