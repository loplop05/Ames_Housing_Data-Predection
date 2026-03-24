# Ames Housing Price Prediction (Elastic Net)

This project builds and tunes an Elastic Net regression model on the engineered Ames Housing dataset to predict home sale prices. The workflow lives in `Linear-Regression-Project.ipynb` and walks through data preparation, model training, hyperparameter search, and evaluation.

## What I built
- Loaded the processed Ames dataset (`AMES_Final_DF.csv`) with 2,925 rows and 274 features.
- Split features/labels (`SalePrice`), then created train/test sets with a 90/10 split and a fixed `random_state=101` for reproducibility.
- Standardized all numeric features with `StandardScaler` to keep the Elastic Net loss well-conditioned.
- Trained an `ElasticNet` regression model and tuned `alpha` and `l1_ratio` using `GridSearchCV` (5-fold CV, 20 candidate settings).
- Selected the best model (`alpha=100`, `l1_ratio=1`, `max_iter=100000`) based on negative MSE.
- Evaluated on the held-out test set, achieving:
  - MAE: **$14,195**
  - RMSE: **$20,559**

## What I learned
- Why scaling is critical for regularized linear models: standardized features prevented coefficient dominance and improved convergence.
- How Elastic Net balances L1/L2 penalties to handle multicollinearity and perform feature shrinkage without over-pruning.
- Using grid search with cross-validation to choose hyperparameters objectively instead of manual guesswork.
- Interpreting regression error metrics: MAE for average absolute error intuition and RMSE to emphasize larger mistakes.

## How to run the notebook
1. Clone the repo and open the notebook:
    ```bash
    git clone https://github.com/loplop05/Ames_Housing_Data-Predection.git
    cd Ames_Housing_Data-Predection
    ```
   *Note: the GitHub repository name retains the original spelling `Ames_Housing_Data-Predection`.*
2. Install requirements (Python 3.10+ recommended):
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```
3. Launch Jupyter and open `Linear-Regression-Project.ipynb`:
   ```bash
   jupyter notebook
   ```
4. Run cells in order to reproduce the training, grid search, and metrics above.

## Project structure
- `Linear-Regression-Project.ipynb` — end-to-end workflow (imports, scaling, Elastic Net tuning, metrics).
- `DATA/AMES_Final_DF.csv` — processed feature matrix used by the notebook (referenced in the notebook; ensure it is present when running).
