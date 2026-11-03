# ==========================================
# Question 4 (a): Create binary variable mpg01
# ==========================================

from ISLP import load_data

# Load Auto dataset
Auto = load_data("Auto")

# Calculate median of mpg
median_mpg = Auto["mpg"].median()

print("Median MPG:", median_mpg)

# Create binary variable mpg01
Auto["mpg01"] = (Auto["mpg"] > median_mpg).astype(int)

# Display results
print("\nFirst five observations:")
print(Auto[["mpg", "mpg01"]].head())

print("\nClass distribution:")
print(Auto["mpg01"].value_counts())

print("\nClass proportions:")
print(Auto["mpg01"].value_counts(normalize=True))

# Display all feature names
print("\nDataset columns:")
print(Auto.columns.tolist())

# Display dataset information
print("\nDataset information:")
print(Auto.info())

import matplotlib.pyplot as plt
import seaborn as sns

# Select numerical features
features = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "year",
    "origin"
]

# Set figure size
plt.figure(figsize=(14, 10))

# Draw boxplots
for i, feature in enumerate(features, 1):

    plt.subplot(3, 3, i)

    sns.boxplot(
        data=Auto,
        x="mpg01",
        y=feature
    )

    plt.title(feature)
    plt.xlabel("mpg01")
    plt.ylabel(feature)

plt.tight_layout()
plt.savefig("Q4_boxplots.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================
# Question 4 (c): Split the dataset
# ==========================================

from sklearn.model_selection import train_test_split

# Define predictors and target variable
selected_vars = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "year",
    "origin"
]

X = Auto[selected_vars]
y = Auto["mpg01"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=123,
    stratify=y
)

# Display dataset sizes
print("Training set size:", len(X_train))
print("Test set size:", len(X_test))

# Check class distribution
print("\nTraining set distribution:")
print(y_train.value_counts())

print("\nTest set distribution:")
print(y_test.value_counts())

# ==========================================
# Question 4 (d): Linear Discriminant Analysis
# ==========================================

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

# Create LDA model
lda_model = LinearDiscriminantAnalysis()

# Train the model
lda_model.fit(X_train, y_train)

# Predict on test set
y_pred_lda = lda_model.predict(X_test)

# Calculate accuracy and test error
lda_accuracy = accuracy_score(y_test, y_pred_lda)
lda_error = 1 - lda_accuracy

# Display results
print("\n========== LDA Results ==========")

print("Accuracy:", lda_accuracy)
print("Test Error:", lda_error)
print("Test Error (%):", lda_error * 100)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lda))

# ==========================================
# Question 4 (e): Quadratic Discriminant Analysis
# ==========================================

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

# Create QDA model
qda_model = QuadraticDiscriminantAnalysis()

# Train the model
qda_model.fit(X_train, y_train)

# Predict on test set
y_pred_qda = qda_model.predict(X_test)

# Calculate accuracy and test error
qda_accuracy = accuracy_score(y_test, y_pred_qda)
qda_error = 1 - qda_accuracy

# Display results
print("\n========== QDA Results ==========")

print("Accuracy:", qda_accuracy)
print("Test Error:", qda_error)
print("Test Error (%):", qda_error * 100)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_qda))

# ==========================================
# Question 4 (f): Logistic Regression
# ==========================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Fit logistic regression
logistic_model = LogisticRegression(max_iter=10000)

logistic_model.fit(X_train, y_train)

# Predict on test set
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate the model
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
logistic_error = 1 - logistic_accuracy

print("\n========== Logistic Regression Results ==========")
print("Accuracy:", logistic_accuracy)
print("Test Error:", logistic_error)
print("Test Error (%):", logistic_error * 100)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))

# ==========================================
# Question 4 (e):K-Nearest Neighbors
# ==========================================
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Standardize the predictors
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different values of K
k_values = [1, 3, 5, 7, 9, 11, 15, 20, 25, 30]

knn_results = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)

    knn_model.fit(X_train_scaled, y_train)

    y_pred_knn = knn_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred_knn)
    error = 1 - accuracy

    knn_results.append({
        "K": k,
        "Accuracy": accuracy,
        "Test Error (%)": error * 100
    })

# Display results
knn_results_df = pd.DataFrame(knn_results)

print("\n========== KNN Results ==========")
print(knn_results_df.to_string(index=False))

# Find the best K
best_result = knn_results_df.loc[
    knn_results_df["Test Error (%)"].idxmin()
]

print("\nBest K:", best_result["K"])
print("Lowest Test Error (%):", best_result["Test Error (%)"])