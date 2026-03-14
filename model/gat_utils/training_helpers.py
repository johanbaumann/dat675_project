from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

from .target_scaling import invert_standardized_predictions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
	rho = float(spearmanr(y_true, y_pred)[0])
	pearson = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
	return mse, rmse, r2, rho, pearson


def metric_from_name(
	metric_name: str,
	mse: float,
	rmse: float,
	r2: float,
	rho: float,
	pearson: float,
) -> float:
    """
	MSE = mean squared error (lower is better)
	RMSE = root mean squared error (lower is better)
    R2 = R2 score (higher is better) Coeficent of determination 
    Rho = Spearman rank correlation (higher is better)
    Pearson = Pearson correlation coefficient (higher is better)
	
	
	
	MSE = 1/N * sum((y_true - y_pred)^2)
    RMSE = sqrt(MSE)
    R2 = 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))
    Rho = spearmanr(y_true, y_pred)[0]
	
    Pearson = pearsonr(y_true, y_pred)[0]= p_{y,y_pred} = cov(y_true, y_pred) / (std(y_true) * std(y_pred))
	
	
	
	"""



	values = {
		"mse": mse,
		"rmse": rmse,
		"r2": r2,
		"rho": rho,
		"pearson": pearson,
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
		"pearson": "Validation Pearson correlation coefficient; higher is better.",
	}
	if metric_name not in descriptions:
		raise ValueError(
			f"Unsupported metric '{metric_name}'. Use one of: {list(descriptions.keys())}"
		)
	return descriptions[metric_name]


def is_improvement(metric_name: str, current: float, best: float, min_delta: float) -> bool:
	# Lower is better for mse/rmse. Higher is better for r2/rho/pearson.
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


def safe_nanmean(values) -> float:
	return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def build_summary_dataframe(
	cv_results: list[dict[str, Any]],
	best_fold_idx: int,
	test_mse: float,
	test_rmse: float,
	test_r2: float,
	test_rho: float,
	test_pearson: float,
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
				"Pearson": fold_result["val_pearson"],
			}
		)

	avg_best_monitor = safe_nanmean([res["best_monitor_value"] for res in cv_results])
	avg_val_mse = safe_nanmean([res["val_mse"] for res in cv_results])
	avg_val_rmse = safe_nanmean([res["val_rmse"] for res in cv_results])
	avg_val_r2 = safe_nanmean([res["val_r2"] for res in cv_results])
	avg_val_rho = safe_nanmean([res["val_rho"] for res in cv_results])
	avg_val_pearson = safe_nanmean([res["val_pearson"] for res in cv_results])

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
			"Pearson": avg_val_pearson,
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
			"Pearson": test_pearson,
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
