

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# ---------------- CONFIG ----------------
RAW_PATH = "C:\\Users\\chand\\OneDrive\\Documents\\DATASET\\Iris.csv"   # <- uploaded dataset path
OUT_DIR = 'knn_outputs'
RANDOM_STATE = 42
TEST_SIZE = 0.2
K_VALUES = list(range(1, 21))     # try k=1..20
CV_FOLDS = 5
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Load dataset ----------------
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Dataset not found at {RAW_PATH}. Place your CSV there or update RAW_PATH.")

# try CSV or Excel
if RAW_PATH.lower().endswith(('.xls', '.xlsx')):
    df = pd.read_excel(RAW_PATH)
else:
    df = pd.read_csv(RAW_PATH)

print("Loaded dataset:", RAW_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ---------------- Detect target column ----------------
# common names for classification targets:
common_targets = ['target','label','class','y','species','diagnosis']
target_col = None
for c in df.columns:
    if c.lower() in common_targets:
        target_col = c
        break

if target_col is None:
    # try columns with two or small unique values
    for c in df.columns[::-1]:  # check from rightmost
        nunique = df[c].nunique(dropna=True)
        if nunique <= 10 and nunique >= 2:
            # prefer columns with <=10 categories and not huge unique
            target_col = c
            break

if target_col is None:
    # fallback to last column
    target_col = df.columns[-1]

print("Detected target column:", target_col)
print(df[target_col].value_counts(dropna=False))

# ---------------- Prepare X, y ----------------
# Drop rows with missing target
df = df.loc[df[target_col].notna()].reset_index(drop=True)
y_raw = df[target_col].copy()

# Encode target to 0..n-1
le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = le.classes_
print("Classes:", classes)

# Build X by dropping the target column
X = df.drop(columns=[target_col])

# Drop any non-informative columns (all unique values -> likely ID)
id_like = [c for c in X.columns if X[c].nunique() == X.shape[0]]
if id_like:
    print("Dropping ID-like columns:", id_like)
    X = X.drop(columns=id_like)

# For categorical features, attempt to convert to numeric (simple)
# We'll keep only numeric columns for KNN (KNN uses distances)
# Convert bools and categorical object columns that are numeric-like
for c in X.columns:
    if X[c].dtype == 'object':
        # try to coerce to numeric; if fails, drop the column (KNN needs numeric)
        try:
            X[c] = pd.to_numeric(X[c])
        except Exception:
            print(f"Dropping non-numeric column for KNN: {c}")
            X = X.drop(columns=[c])

# Final numeric X
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    raise ValueError("No numeric features available for KNN after preprocessing. Provide numeric data or allow encoding.")

X = X[numeric_cols].copy()
print("Final features used:", X.columns.tolist())

# Fill missing numeric values with median
X = X.fillna(X.median())

# ---------------- Train/Test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("Train/Test sizes:", X_train.shape, X_test.shape)

# ---------------- Scale features ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, os.path.join(OUT_DIR, 'scaler.joblib'))

# ---------------- Evaluate K values with cross-validation ----------------
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
mean_scores = []
std_scores = []

for k in K_VALUES:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    mean_scores.append(scores.mean())
    std_scores.append(scores.std())
    print(f"k={k:2d}  CV accuracy: {scores.mean():.4f}  +/- {scores.std():.4f}")

# Save K vs accuracy
k_df = pd.DataFrame({'k': K_VALUES, 'cv_mean_accuracy': mean_scores, 'cv_std': std_scores})
k_df.to_csv(os.path.join(OUT_DIR, 'k_accuracy.csv'), index=False)
print("Saved K vs CV accuracy to", os.path.join(OUT_DIR, 'k_accuracy.csv'))

# ---------------- Plot K vs accuracy ----------------
plt.figure(figsize=(8,5))
plt.errorbar(K_VALUES, mean_scores, yerr=std_scores, fmt='-o')
plt.xlabel('k (n_neighbors)')
plt.ylabel('CV Accuracy')
plt.title('KNN: CV Accuracy vs k')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'k_vs_accuracy.png'))
plt.close()
print("Saved k_vs_accuracy.png")

# ---------------- Select best k (highest mean accuracy, tie -> smaller k) ----------------
best_idx = int(np.argmax(mean_scores))
best_k = K_VALUES[best_idx]
print(f"Best k selected (by CV mean accuracy): {best_k}")

# ---------------- Train final model with best_k ----------------
final_knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
final_knn.fit(X_train_scaled, y_train)
y_pred = final_knn.predict(X_test_scaled)
y_proba = final_knn.predict_proba(X_test_scaled) if hasattr(final_knn, "predict_proba") else None

# ---------------- Metrics ----------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print("\nFinal model metrics on test set:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (weighted): {prec:.4f}")
print(f"Recall (weighted): {rec:.4f}")
print(f"F1 (weighted): {f1:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# Save evaluation metrics
eval_df = pd.DataFrame([{
    'best_k': best_k,
    'accuracy': acc,
    'precision_weighted': prec,
    'recall_weighted': rec,
    'f1_weighted': f1
}])
eval_df.to_csv(os.path.join(OUT_DIR, 'evaluation_metrics.csv'), index=False)
print("Saved evaluation metrics to", os.path.join(OUT_DIR, 'evaluation_metrics.csv'))

# Save confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (Test set)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'))
plt.close()
print("Saved confusion_matrix.png")

# Save model and label encoder
joblib.dump(final_knn, os.path.join(OUT_DIR, f'knn_k{best_k}.joblib'))
joblib.dump(le, os.path.join(OUT_DIR, 'label_encoder.joblib'))
print("Saved trained model and label encoder")

# ---------------- Decision boundary (2D) plot ----------------
# If dataset has >2 features use PCA to reduce to 2D for visualization
n_features = X.shape[1]
if n_features >= 2:
    if n_features > 2:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_vis = pca.fit_transform(scaler.transform(X))
        X_train_vis = pca.transform(scaler.transform(X_train))
        X_test_vis = pca.transform(X_test_scaled)
        x_label = 'PC1'; y_label = 'PC2'
        title_suffix = ' (PCA 2 components)'
    else:
        # exactly 2 features
        X_vis = scaler.transform(X)
        X_train_vis = scaler.transform(X_train)
        X_test_vis = X_test_scaled
        x_label = X.columns[0]; y_label = X.columns[1]
        title_suffix = ''
    # Fit KNN on the 2D transformed train set for plotting decision regions
    viz_knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    viz_knn.fit(X_train_vis, y_train)
    # create mesh
    x_min, x_max = X_vis[:,0].min() - 1.0, X_vis[:,0].max() + 1.0
    y_min, y_max = X_vis[:,1].min() - 1.0, X_vis[:,1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = viz_knn.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='tab10')
    scatter = plt.scatter(X_test_vis[:,0], X_test_vis[:,1], c=y_test, cmap='tab10', edgecolor='k', s=60)
    plt.xlabel(x_label); plt.ylabel(y_label)
    plt.title(f"KNN decision regions (k={best_k}){title_suffix}")
    # legend
    handles = scatter.legend_elements()[0]
    plt.legend(handles, le.classes_, title="Classes")
    plt.tight_layout()
    dec_path = os.path.join(OUT_DIR, f'decision_boundary_k{best_k}.png')
    plt.savefig(dec_path, dpi=150)
    plt.close()
    print("Saved decision boundary plot to", dec_path)
else:
    print("Not enough numeric features to plot 2D decision boundary.")

print("\nAll outputs saved to:", OUT_DIR)
print("Done.")
