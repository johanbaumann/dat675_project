Run the preprocessing for the BACE dataset by running

> python src/original_preprocessing.py

from the project root.



After the synthetic molecules have been generated, the synthetic preprocessing can be done by running

> python src/synthetic_preprocessing.py

from the project root.



The Original Preprocessing processes the molecules and creates a folder of roughly 1300 original molecules. These are then used by the beta CVAE in the vae folder to generate synthetic molecules. The Synthetic Preprocessing then combines the synthetic molecules with the original data and placed in two folders with varying ratios of synthetic data.
