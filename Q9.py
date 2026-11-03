import pandas as pd

from ISLP import load_data
from sklearn.model_selection import train_test_split

# Load College dataset
College = load_data("College")

# Display basic information
print("Dataset shape:", College.shape)
print("\nFirst five rows:")
print(College.head())

# Define predictors and response
# Define predictors and response
X = College.drop(columns=["Apps"])
y = College["Apps"]

# Convert categorical variables into numerical variables
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=123
)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)

# ============================================================
# Question 9(b): Least Squares Regression
# ============================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fit least squares regression
ls_model = LinearRegression()
ls_model.fit(X_train, y_train)

# Predict on test set
y_pred_ls = ls_model.predict(X_test)

# Calculate test MSE
ls_test_mse = mean_squared_error(y_test, y_pred_ls)

print("\n========== Least Squares Results ==========")
print("Test MSE:", ls_test_mse)

# ============================================================
# Question 9(c): Ridge Regression with Cross-Validation
# ============================================================

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Define Ridge regression pipeline
ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

# Define candidate lambda values
lambda_values = np.logspace(-2, 5, 100)

# Cross-validation to select the best lambda
ridge_cv = GridSearchCV(
    ridge_pipeline,
    {"ridge__alpha": lambda_values},
    cv=10,
    scoring="neg_mean_squared_error"
)

ridge_cv.fit(X_train, y_train)

# Best lambda
best_lambda_ridge = ridge_cv.best_params_["ridge__alpha"]

# Predict on test set
y_pred_ridge = ridge_cv.predict(X_test)

# Calculate test MSE
ridge_test_mse = mean_squared_error(y_test, y_pred_ridge)

print("\n========== Ridge Regression Results ==========")
print("Best Lambda:", best_lambda_ridge)
print("Best CV MSE:", -ridge_cv.best_score_)
print("Test MSE:", ridge_test_mse)

# ============================================================
# Question 9(d): Lasso Regression with Cross-Validation
# ============================================================

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Define Lasso regression pipeline
lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(max_iter=100000))
])

# Define candidate lambda values
lambda_values = np.logspace(-2, 5, 100)

# Cross-validation to select the best lambda
lasso_cv = GridSearchCV(
    lasso_pipeline,
    {"lasso__alpha": lambda_values},
    cv=10,
    scoring="neg_mean_squared_error"
)

lasso_cv.fit(X_train, y_train)

# Best lambda
best_lambda_lasso = lasso_cv.best_params_["lasso__alpha"]

# Predict on test set
y_pred_lasso = lasso_cv.predict(X_test)

# Calculate test MSE
lasso_test_mse = mean_squared_error(y_test, y_pred_lasso)

# Count non-zero coefficients
lasso_model = lasso_cv.best_estimator_.named_steps["lasso"]
nonzero_count = np.sum(lasso_model.coef_ != 0)

print("\n========== Lasso Regression Results ==========")
print("Best Lambda:", best_lambda_lasso)
print("Best CV MSE:", -lasso_cv.best_score_)
print("Test MSE:", lasso_test_mse)
print("Number of Non-zero Coefficients:", nonzero_count)

# ============================================================
# Question 9(e): Principal Components Regression (PCR)
# ============================================================

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Define PCR pipeline
pcr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("linear", LinearRegression())
])

# Test different numbers of principal components
M_values = list(range(1, X_train.shape[1] + 1))

# Cross-validation to select the best M
pcr_cv = GridSearchCV(
    pcr_pipeline,
    {"pca__n_components": M_values},
    cv=10,
    scoring="neg_mean_squared_error"
)

pcr_cv.fit(X_train, y_train)

# Best number of principal components
best_M = pcr_cv.best_params_["pca__n_components"]

# Predict on test set
y_pred_pcr = pcr_cv.predict(X_test)

# Calculate test MSE
pcr_test_mse = mean_squared_error(y_test, y_pred_pcr)

print("\n========== PCR Results ==========")
print("Best Number of Components (M):", best_M)
print("Best CV MSE:", -pcr_cv.best_score_)
print("Test MSE:", pcr_test_mse)