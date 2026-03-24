import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ==========================================
# CONFIGURATION
# ==========================================
CSV_PATH = "/Users/boting/li_4yp/data/MERGE4PATTERNS/MEGAMEGAFEATURES.csv"  # Path to your CSV generated from extract_features
MODEL_DIR = (
    "/Users/boting/li_4yp/scripts/exp_models_mega/"  # Directory to save trained models
)
TEST_SIZE = 0.2  # 20% for validation
RANDOM_SEED = 42

# Define the grid for "Best Accuracy" search
# XGBoost will try random combinations of these to find the best fit
# PARAM_GRID = {
#     'n_estimators': [100, 200, 300, 500],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'max_depth': [3, 4, 5, 6, 8],
#     'subsample': [0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
#     'gamma': [0, 0.1, 0.2]
# }

PARAM_GRID = {
    # 1. Tree Structure (Complexity)
    # Shallow trees (3-5) are better for avoiding overfitting on small data.
    # Deeper trees (6-10) might capture complex color interactions but risk memorization.
    "max_depth": [3, 4, 5, 6, 8, 10],
    # Controls how much weight is needed in a leaf.
    # Higher values = more conservative model (prevents learning specific noisy samples).
    # For small data, range 1-7 is critical.
    "min_child_weight": [1, 3, 5, 7],
    # 2. Learning Speed vs. Quantity
    # Lower learning rate + More trees is generally more robust but slower.
    "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    "n_estimators": [100, 300, 500, 800, 1000],
    # 3. Stochastic Sampling (Randomness to prevent overfitting)
    # Row sampling: "Don't see every image in every tree"
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    # Column sampling: "Don't see every color feature in every tree"
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # 4. Regularization (CRITICAL for small datasets)
    # Gamma: Minimum loss reduction required to make a split. Acts like a "brake".
    "gamma": [0, 0.1, 0.5, 1, 2, 5],
    # Alpha (L1): Lasso regularization. Good for feature selection (dropping useless color stats).
    "reg_alpha": [0, 0.01, 0.1, 1, 10],
    # Lambda (L2): Ridge regularization. Smooths weights.
    "reg_lambda": [0.5, 1, 2, 5, 10],
}


def load_and_prep_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Identify Feature Columns
    # We select columns that start with our color spaces
    feature_cols = [
        c for c in df.columns if c.startswith(("RGB_", "HSV_", "LAB_", "YUV_"))
    ]

    print(f"Found {len(feature_cols)} features.")

    # 2. Extract X (Features)
    X = df[feature_cols]

    # 3. Extract Y (Targets)
    # We assume your CSV has columns: 'Base', 'Primary', 'Secondary'
    # Fill NaN Secondary with a distinct class (e.g., 10 for 'Empty') if not already handled
    if "Secondary" in df.columns:
        df["Secondary"] = df["Secondary"].fillna(10)

    targets = {
        "Base": df["Base"],
        "Primary": df["Primary"],
        "Secondary": df["Secondary"],
    }

    return X, targets


def train_target(X_train, y_train, X_test, y_test, target_name):
    print(f"\n{'='*40}")
    print(f"TRAINING TARGET: {target_name}")
    print(f"{'='*40}")

    # 1. Encode Labels (e.g., if Base is 6, 7, 8... XGB needs 0, 1, 2)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # 2. Initialize Classifier
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_SEED,
        n_jobs=-1,  # Use all cores
    )

    # 3. Hyperparameter Tuning (Find Best Accuracy)
    print("  > Tuning hyperparameters...")
    search = RandomizedSearchCV(
        xgb_clf,
        param_distributions=PARAM_GRID,
        n_iter=20,  # Try 20 random combinations (increase for better results)
        cv=3,  # 3-Fold Cross Validation
        scoring="accuracy",
        verbose=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    search.fit(X_train, y_train_enc)

    best_model = search.best_estimator_
    print(f"  > Best Params: {search.best_params_}")

    # 4. Evaluate
    y_pred_enc = best_model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)  # Convert back to real labels

    acc = accuracy_score(y_test, y_pred)
    print(f"  > VALIDATION ACCURACY: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return best_model, le, acc


def main():
    # 1. Data Prep
    X, targets = load_and_prep_data(CSV_PATH)

    # 2. Split Data (Stratify based on Base to ensure balanced depth distribution)
    # Note: It's hard to stratify on all 3, so we prioritize Base
    X_train, X_test, indices_train, indices_test = train_test_split(
        X,
        X.index,
        test_size=TEST_SIZE,
        stratify=targets["Base"],
        random_state=RANDOM_SEED,
    )

    results = {}

    # 3. Train Loop for each Target
    for name in ["Base", "Primary", "Secondary"]:
        y_train = targets[name].loc[indices_train]
        y_test = targets[name].loc[indices_test]

        model, encoder, accuracy = train_target(X_train, y_train, X_test, y_test, name)

        # Save results
        results[name] = {"model": model, "encoder": encoder, "accuracy": accuracy}

        # 4. Save to Disk
        # We save the model AND the label encoder (crucial for inference!)
        joblib.dump(model, f"{MODEL_DIR}/xgb_{name.lower()}.pkl")
        joblib.dump(encoder, f"{MODEL_DIR}/le_{name.lower()}.pkl")
        print(f"  > Model saved to {MODEL_DIR}/xgb_{name.lower()}.pkl")

    print(f"\n{'='*40}")
    print("FINAL RESULTS")
    print(f"{'='*40}")
    print(f"Base Accuracy:      {results['Base']['accuracy']:.2%}")
    print(f"Primary Accuracy:   {results['Primary']['accuracy']:.2%}")
    print(f"Secondary Accuracy: {results['Secondary']['accuracy']:.2%}")


if __name__ == "__main__":
    main()
