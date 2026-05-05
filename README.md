# Abalone

# Abalone — Rings Prediction (Regression) with Scikit-Learn and Keras

This project uses the **Abalone** dataset (UCI Machine Learning Repository) to solve a **regression** task: predicting the number of **Rings** of an abalone from its physical measurements and sex.

## Goal

- Predict `Rings` (numeric target)
- Apply preprocessing (One-Hot Encoding + feature selection + scaling)
- Train and compare multiple models:
  - **Ridge Regression**
  - **Lasso Regression**
  - **Linear Regression** (baseline)
  - **Polynomial Regression**
  - **Neural Network** (Keras/TensorFlow)

## Dataset

- Source: UCI Machine Learning Repository (loaded via `ucimlrepo`)
- Main features:
  - `Sex` (categorical: M/F/I)
  - `Length`, `Diameter`, `Height`
  - `Whole_weight`, `Shucked_weight`, `Viscera_weight`, `Shell_weight` (continuous)
- Target:
  - `Rings`

Dataset is fetched directly via API:

```python
from ucimlrepo import fetch_ucirepo
abalone = fetch_ucirepo(id=1)

X = abalone.data.features
y = abalone.data.targets
```

## Approach
This project mainly consists of a notebook, abalone.ipynb, which follows the following approach: 

### Preprocessing
1) One-Hot Encoding (Sex):
    Sex is encoded with OneHotEncoder, producing columns like:

    sex_F, sex_I, sex_M

2) Correlation heatmap:
    A correlation matrix is computed on the encoded dataframe and visualized with Seaborn. 

3) Outliers removal

4) Feature selection and/or engineering:
    The notebook creates alternative feature sets, for example:

    - dropping: Shucked_weight, Viscera_weight, Shell_weight
    - dropping: Whole_weight
    - feature engineering, like density or weight ratios.


    The resulting dataframe is used as the final X.

5) Scaling
    A StandardScaler is used:
    ```
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ```
### Train/Test Split
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Models

- Ridge Regression (Scikit-Learn) (Linear with alpha=0)

- Lasso Regression (Scikit-Learn)

- Polynomial Regression (Scikit-Learn)

- Neural Network (Keras/TensorFlow)
    A feed-forward MLP for regression:

    - Input: shape=(9,) (after preprocessing/feature selection)
    - Hidden layers (tested variants): Dense + ReLU (e.g. 50x2, 50x3, 32x3)
    - Output: Dense(1)
    - Optimizer: adam
    - Loss: mean_squared_error
    Early stopping:
    - monitor="val_loss"
    - patience=10
    - restore_best_weights=True
    training up to epochs=1000

    CPU-only execution is forced in the notebook.

## Results

Models are evaluated using:

- MSE (Mean Squared Error)
- R² score

The notebook reports example results (Neural Network slightly better than Linear Regression):

- NN: MSE ~ 4.75, R² ~ 0.54 (all features)
- Poly Regression: MSE ~ 5.16, R² ~ 0.56 (removed outliers and all features)