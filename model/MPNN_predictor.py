import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from gat_utils import run_training_pipeline


# ==================== changelog ====================
# 2026-03-12
# - Added a centralized nested CONFIG dictionary for all important settings.
# - Switched categorical atom/bond fields to one-hot (+unknown) encoding.
# - Added stronger training stability patches: min-delta early stopping,
#   configurable monitor metric, gradient clipping, and improved scheduler usage.
# - Fixed target-column selection logic with explicit real vs synthetic priorities.
# - Removed duplicate test-set loading and improved fold-level diagnostics printing.
# - Kept optimization loss as MSE, while early stopping defaults to RMSE.
# - Added configurable residual connections and per-conv normalization blocks.
# - Standardized the exported CV summary CSV columns to match actual_mpnn.py.
# - Refactored non-model logic into gat_utils.py so this script stays focused on config and GATModel.
# - Added per-fold target standardization (fit on train fold only, invert for reported metrics).
# - Added config-driven atom/bond (node/edge) descriptor selection for modular featurization.


# ==================== configuration ====================
CONFIG = {
	"experiment": {
		"target_folder": "./0%",  # Change to ./33% or ./67% for other experiments.
		"actual_test_file": "heldout_testset.csv",
		"weight_save_path": "best_model_iteration_{fold}.pth",
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
		"scale_atomic_mass_by": 100.0, # to keep atomic mass in a similar range as other features
		"categorical_encoding_mode": "one_hot",  # one_hot or index
		# Atom/node descriptors (enabled in-order). Remove items to ablate features.
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
		# Bond/edge descriptors (enabled in-order).
		"bond_descriptors": [
			"bond_type",
			"bond_order",
			"is_conjugated",
			"is_in_ring",
			"bond_stereo",
		],
	},
	"model": {
		"hidden_dim": 96, # hidden dimension for GAT layers and FFN layers
		"num_conv_layers": 4, # of GAT layers (message-passing steps)
		"heads": 4,
		"dropout": 0.2,
		"residual": True,
		"normalization": "layernorm",  # layernorm, batchnorm1d, or none
		"ffnn_hidden_layers": [128,64],  # Example: [128, 256, 512], or [] for direct output.
	},
	"optimization": {
		"learning_rate": 3e-4,
		"weight_decay": 1e-4,
		"grad_clip_norm": 10.0,
	},
	"scheduler": {
		"enabled": True,
		"mode": "min",
		"factor": 0.7,
		"patience": 4,
		"monitor_metric": "rmse",  # rmse/mse/r2/rho
	},
	"training": {
		"num_epochs": 150,
		"print_every": 1,
		# Standardize targets per fold: y_scaled = (y - mean_train) / std_train
		# Model predicts y_scaled, and we invert predictions for metrics/reporting.
		"target_standardization": {
			"enabled": True,
			"epsilon": 1e-8,
		},
		"early_stopping": {
			"enabled": True,
			"monitor_metric": "rmse",  # rmse/mse/r2/rho
			"patience": 10,
			"min_delta": 1e-4,
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
		ffnn_hidden_layers,
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
			self.convs.append(
				GATv2Conv(
					conv_in_dim,
					hidden_dim,
					heads=heads,
					dropout=dropout,
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
			x = self.dropout_layer(x)

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
