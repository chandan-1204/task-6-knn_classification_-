# K-Nearest Neighbors (KNN) Classification – Task 6

This repository contains a complete, ready-to-run K-Nearest Neighbors (KNN) classification project (Task 6).  
The script loads a dataset, preprocesses features, evaluates K values (1–20) using cross-validation, selects the best k, trains the final KNN model, evaluates it, and saves all outputs and visualizations.

------------------------------------------------------------

## Project Structure

knn_classification_task6.py          # Main script
knn_outputs/                         # Auto-generated output folder
│
├── k_accuracy.csv                   # CV mean & std accuracy for k = 1..20
├── k_vs_accuracy.png                # Plot: CV accuracy vs k
├── evaluation_metrics.csv           # Final test metrics + selected k
├── confusion_matrix.png             # Confusion matrix on test set
├── knn_k{best_k}.joblib             # Saved KNN model (best k)
├── scaler.joblib                    # Saved StandardScaler
├── label_encoder.joblib             # Saved LabelEncoder
└── decision_boundary_k{best_k}.png  # 2D decision boundary (PCA if needed)

------------------------------------------------------------

## Dataset

The script expects a dataset file.  
Default dataset path used during development:

/mnt/data/data.csv

If your dataset is stored elsewhere, update the `RAW_PATH` variable inside:

knn_classification_task6.py

------------------------------------------------------------

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

Install all dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn joblib

------------------------------------------------------------

## How the Script Works (High-Level)

1. Loads dataset (supports CSV and Excel).
2. Automatically detects the target column (common target names or class-like columns).
3. Encodes the target values using LabelEncoder (mapped to 0..n-1).
4. Selects numeric features only (drops ID-like and categorical columns).
5. Fills missing numeric values using median imputation.
6. Splits dataset into training/testing (80/20, stratified).
7. Applies StandardScaler to normalize numeric features.
8. Performs Stratified K-Fold cross-validation for k = 1..20.
9. Saves mean and std accuracy for each k to k_accuracy.csv.
10. Selects best k (highest mean CV accuracy; if tie, smallest k).
11. Trains the final KNN model using best k.
12. Evaluates on test set (accuracy, precision, recall, F1).
13. Saves:
    - confusion matrix,
    - metrics CSV,
    - trained KNN model,
    - scaler + label encoder,
    - plot of CV accuracy vs k,
    - decision boundary visualization (PCA used if features > 2).
14. Outputs are stored in the knn_outputs/ directory.

------------------------------------------------------------

## How to Run

1. Place your dataset (CSV or Excel) in the project folder  
   OR update:

RAW_PATH = "/mnt/data/data.csv"

inside knn_classification_task6.py

2. Run the script:

python knn_classification_task6.py

3. All files will be saved inside:

knn_outputs/

------------------------------------------------------------

## Output Files Explained

- k_accuracy.csv  
  Cross-validated mean and std accuracy for k = 1..20.

- k_vs_accuracy.png  
  Plot showing performance trend vs k.

- evaluation_metrics.csv  
  Final test accuracy, precision, recall, F1, and chosen k.

- confusion_matrix.png  
  Confusion matrix for the final model.

- knn_k{best_k}.joblib  
  Final trained KNN model.

- scaler.joblib  
  StandardScaler required for inference.

- label_encoder.joblib  
  Needed to convert model predictions back to original class labels.

- decision_boundary_k{best_k}.png  
  2D PCA decision boundary visualization.

------------------------------------------------------------

## Notes & Tips

- KNN uses distance-based classification, so **all features must be numeric**.  
  The script drops non-numeric columns automatically.

- For imbalanced datasets, consider using class-wise metrics in evaluation.

- PCA in the decision boundary plot is **only for visualization**, not for training.

- For large datasets, KNN may be slow; consider feature selection or approximate nearest neighbors.

------------------------------------------------------------

## Extensions / Next Steps

- Add GridSearchCV to tune more hyperparameters (k, weights, distance metrics).
- Add support for weighted KNN (weights="distance").
- Include SMOTE or class-balancing techniques for imbalanced datasets.
- Convert the script into an interactive Jupyter Notebook.
- Add more visualization options (pair plots, correlation heatmaps).

------------------------------------------------------------
