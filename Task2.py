

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

import joblib

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Loading California Housing Dataset...\n")

data = fetch_california_housing(as_frame=True)

# Combine features and target
df = pd.concat(
    [data.data, data.target.rename("HousePrice")],
    axis=1
)

# Display first rows
print(df.head())

# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print("\n==============================")
print("DATASET INFORMATION")
print("==============================\n")

print(df.info())

print("\n==============================")
print("MISSING VALUES")
print("==============================\n")

print(df.isnull().sum())

print("\n==============================")
print("STATISTICAL SUMMARY")
print("==============================\n")

print(df.describe())

# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nGenerating Histograms...\n")

df.hist(figsize=(14, 10))

plt.suptitle("Feature Distributions")

plt.tight_layout()

plt.show()

# ============================================================
# 5. CORRELATION MATRIX
# ============================================================

print("\nGenerating Correlation Matrix...\n")

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))

plt.imshow(correlation_matrix, cmap="coolwarm")

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Matrix")

plt.show()

# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("HousePrice", axis=1)

y = df["HousePrice"]

print("\nFeatures Shape:", X.shape)

print("Target Shape:", y.shape)

# ============================================================
# 7. FEATURE SCALING
# ============================================================

print("\nApplying Feature Scaling...\n")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature Scaling Completed")

# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

print("\nSplitting Dataset...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)

print("Testing Data Shape:", X_test.shape)

# ============================================================
# 9. TRAIN MULTIPLE MODELS
# ============================================================

print("\nTraining Multiple Models...\n")

models = {

    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Decision Tree": DecisionTreeRegressor(max_depth=5)

}

# ============================================================
# 10. MODEL EVALUATION
# ============================================================

results = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    # Save results
    results[name] = {

        "RMSE": rmse,

        "R2 Score": r2

    }

    print(f"{name} Completed")

# ============================================================
# 11. RESULTS TABLE
# ============================================================

print("\n==============================")
print("MODEL COMPARISON RESULTS")
print("==============================\n")

results_df = pd.DataFrame(results).T

print(results_df)

# ============================================================
# 12. BEST MODEL SELECTION
# ============================================================

best_model_name = results_df["R2 Score"].idxmax()

print("\nBest Performing Model:")

print(best_model_name)

# ============================================================
# 13. VISUAL PERFORMANCE VALIDATION
# ============================================================

print("\nGenerating Actual vs Predicted Plot...\n")

best_model = models[best_model_name]

y_pred = best_model.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.4
)

plt.xlabel("Actual House Prices")

plt.ylabel("Predicted House Prices")

plt.title(f"Actual vs Predicted ({best_model_name})")

# Reference Line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.show()

# ============================================================
# 14. MODEL COMPARISON BAR CHART
# ============================================================

print("\nGenerating Model Comparison Charts...\n")

# RMSE Chart
plt.figure(figsize=(8, 5))

plt.bar(
    results_df.index,
    results_df["RMSE"]
)

plt.ylabel("RMSE")

plt.title("RMSE Comparison")

plt.xticks(rotation=10)

plt.show()

# R2 Score Chart
plt.figure(figsize=(8, 5))

plt.bar(
    results_df.index,
    results_df["R2 Score"]
)

plt.ylabel("R2 Score")

plt.title("R2 Score Comparison")

plt.xticks(rotation=10)

plt.show()

# ============================================================
# 15. SAVE BEST MODEL
# ============================================================

print("\nSaving Best Model...\n")

joblib.dump(
    best_model,
    "best_house_price_model.pkl"
)

print("Best Model Saved Successfully")

# ============================================================
# 16. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Scaler Saved Successfully")

# ============================================================
# 17. TEST MODEL ON NEW INPUT
# ============================================================

print("\n==============================")
print("TESTING BEST MODEL")
print("==============================\n")

sample_input = np.array([[
    8.3252,
    41.0,
    6.984127,
    1.023810,
    322.0,
    2.555556,
    37.88,
    -122.23
]])

# Scale input
sample_scaled = scaler.transform(sample_input)

# Predict
sample_prediction = best_model.predict(sample_scaled)

print(
    "Predicted House Price:",
    sample_prediction[0]
)

# ============================================================
# 18. CONCLUSION
# ============================================================

print("\n==============================")
print("PROJECT CONCLUSION")
print("==============================\n")

print("""
1. Successfully implemented Feature Scaling.

2. Trained multiple Machine Learning models:
   - Linear Regression
   - Ridge Regression
   - Decision Tree Regressor

3. Compared model performance using:
   - RMSE
   - R2 Score

4. Selected the best-performing model.

5. Saved the trained model and scaler.

6. Built visualization graphs for comparison.

Future Improvements:
- Hyperparameter tuning
- Random Forest Regressor
- XGBoost
- Cross-validation
- Feature engineering
""")