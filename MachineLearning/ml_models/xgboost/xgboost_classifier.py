import pandas as pd
import numpy as np
import cupy as cp
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from typing import Tuple, Dict, Any, List
import pprint

from MachineLearning import ml_utils
from MachineLearning.config.param_types_ml import (
    TrainingParamsML,
    TrainingHyperparametersContainer,
    XGBoostHyperparams
)


def train_model(
    prepared_df: pd.DataFrame,
    feature_names: List[str],
    target_name: str,
    training_params: TrainingParamsML
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Trains an XGBoostClassifier model using GPU acceleration (CuPy).
    Automatically moves data to GPU and prevents CPU fallbacks.
    """
    print("--- Starting XGBoost (GPU) Training ---")

    # --- 1. Split Data ---
    print("Splitting data into training and testing sets...")
    split_params = training_params.split_params
    train_size = 1.0 - split_params.test_size

    X_train, X_test, y_train, y_test = ml_utils.prepare_and_split_data(
        df=prepared_df,
        feature_cols=feature_names,
        target_col=target_name,
        train_size=train_size
    )

    y_train_flat = y_train.values.ravel()
    print(f"Train shapes: X={X_train.shape}, y={y_train_flat.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    # --- 2. Prepare GPU Data ---
    print("Transferring data to GPU using CuPy...")
    X_train_gpu = cp.asarray(X_train)
    X_test_gpu = cp.asarray(X_test)
    y_train_gpu = cp.asarray(y_train_flat)
    y_test_gpu = cp.asarray(y_test.values.ravel())

    # --- 3. Configure Hyperparameters ---
    if not training_params.hyperparameters.xgboost_classifier:
        raise ValueError("Config missing 'xgboost_classifier' hyperparameter block.")

    hyperparams = training_params.hyperparameters.xgboost_classifier

    xgb_params = {
        "tree_method": "hist",   # XGBoost GPU-compatible method
        "device": "cuda",        # Ensures GPU training + prediction
        "predictor": "gpu_predictor",
        "n_estimators": hyperparams.n_estimators,
        "max_depth": hyperparams.max_depth,
        "min_child_weight": hyperparams.min_samples_leaf,
        "random_state": hyperparams.random_state,
        "n_jobs": hyperparams.n_jobs,
    }

    # --- 4. Compute Class Weights (optional) ---
    sample_weights = None
    if hyperparams.class_weight == "balanced":
        print("Calculating balanced sample weights for XGBoost...")
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_flat)
        sample_weights = cp.asarray(sample_weights)  # move to GPU

    # --- 5. Initialize and Fit Model ---
    print("Initializing XGBClassifier for GPU...")
    model = xgb.XGBClassifier(**xgb_params)

    print("Fitting model...")
    model.fit(X_train_gpu, y_train_gpu, sample_weight=sample_weights)
    print("Model fitting complete.")

    # --- 6. Evaluation ---
    print("\n--- Model Evaluation (on Test Set) ---")
    y_pred_gpu = model.predict(X_test_gpu)
    y_pred = cp.asnumpy(y_pred_gpu)  # convert back to NumPy for sklearn metrics
    y_test_np = cp.asnumpy(y_test_gpu)

    accuracy = accuracy_score(y_test_np, y_pred)
    class_labels = [str(c) for c in model.classes_]
    report = classification_report(
        y_test_np, y_pred,
        labels=model.classes_,
        target_names=class_labels,
        output_dict=True,
        zero_division=0
    )
    conf_matrix = confusion_matrix(y_test_np, y_pred, labels=model.classes_)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classes found by model: {model.classes_}")
    print("Classification Report:")
    pprint.pprint(report)
    print("Confusion Matrix:")
    print(conf_matrix)

    # --- 7. Build Metrics Dict ---
    metrics = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
        "feature_importances": {
            fname: float(imp)
            for fname, imp in zip(feature_names, model.feature_importances_)
        },
    }

    print("--- XGBoost Training Complete ---")
    return model, metrics
