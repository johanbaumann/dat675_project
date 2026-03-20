

# Random Forest Bioactivity Predictor(RF)

This project includes a machine learning script written in Python (`RF_fixed_predictor.py`) designed to predict the biological activity (**pIC50** value) of a molecule based on its SMILES string.

## Environment
Please ensure that the following dependencies are installed in your Python environment:
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn rdkit
```
## User Guide
### 1.Configuration
Before running the script, open 'RF_fixed_predictor.py' and adjust the following top-level variables to match your dataset name and usage needs:
- `GROUP_FOLDER`：Name of the dataset folder to run (e.g., `"combination_3900_molecules_and_67_%_synthetic"`）。
- `USE_SYNTHETIC`：Boolean value（`True` / `False`）.Whether to incorporate the corresponding synthetic data（`synthetic_data_iteration_x.csv`）during training.
- `HOLDOUT_FILE`：Relative path to the holdout test set file.

### 2.Run the Script
Run the script directly in a terminal or command line:
```bash
python RF_fixed_predictor.py
```

### 3.View Output
After the script finishes running, it will print the cross-validation results and the performance on the external test set to the terminal, and automatically generate the following 4 files in the  `results/RF_model/[GROUP_FOLDER]/` directory:
1. **`RF_metrics_summary.csv`** Including all key metrics
2. **`RF_plot_actual_vs_predicted.png`** Scatter plot of actual pIC50 and predicted pIC50 values in the holdout set
3. **`RF_plot_residuals.png`** Residual plot for the holdout set prediction
4. **`RF_plot_rank_correlation.png`** Scatter plot of actual rank and predicted rank values in the holdout set
   
