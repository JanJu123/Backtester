import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Dict, Any, List
import pprint

# --- Import your ML config dataclasses ---
from MachineLearning import ml_utils 
from MachineLearning.config.param_types_ml import TrainingParamsML # Import the main container

def train_model(
    prepared_df: pd.DataFrame, 
    feature_names: List[str], 
    target_name: str, 
    training_params: TrainingParamsML # It receives the *entire* training_params block
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Trains a RandomForestClassifier model.
    It dynamically selects its parameters from the training_params object.
    """
    print("\n--- Starting Random Forest Training ---")
    
    # 1. Split Data
    print("Splitting data into training and testing sets...")
    split_params = training_params.split_params
    train_size = 1.0 - split_params.test_size 
    
    X_train, X_test, y_train, y_test = ml_utils.prepare_and_split_data(
        df=prepared_df,
        feature_cols=feature_names,
        target_col=target_name,
        train_size=train_size
    )
    
    # Fix the DataConversionWarning
    y_train_flat = y_train.values.ravel() 

    print(f"Train shapes: X={X_train.shape}, y={y_train_flat.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")

    # 2. Initialize Model using hyperparameters
    print("\nInitializing RandomForestClassifier...")
    
    # --- MODIFICATION: DYNAMIC HYPERPARAMETER SELECTION ---
    
    # Get the *container* object
    hyperparams_container = training_params.hyperparameters
    
    # Get the model_type string (e.g., "random_forest_classifier")
    model_type = training_params.model_type
    
    # Check if the container has an attribute matching the model_type
    if not hasattr(hyperparams_container, model_type):
        raise ValueError(f"Hyperparameter block for '{model_type}' not found in config!")
        
    # Get the *specific* hyperparameter object (e.g., the RandomForestHyperparams object)
    hyperparams_object = getattr(hyperparams_container, model_type) 
    
    if hyperparams_object is None:
         raise ValueError(f"Hyperparameter block for '{model_type}' is null in config!")

    # Now, convert the *specific* object to a dict
    hyperparams_dict = vars(hyperparams_object) 
    # --- END MODIFICATION ---

    print(f"Using hyperparameters: {hyperparams_dict}")
    
    model = RandomForestClassifier(**hyperparams_dict) # Unpack the *correct* dict
    
    # 3. Train Model
    print("Fitting model...")
    model.fit(X_train, y_train_flat) # Use the flattened y_train
    print("Model fitting complete.")

    # 4. Evaluate Model
    print("\n--- Model Evaluation (on Test Set) ---")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    # Get class names from the model itself for the report
    class_labels = [str(c) for c in model.classes_]
    report = classification_report(y_test, y_pred, labels=model.classes_, target_names=class_labels, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classes found by model: {model.classes_}")
    print("Classification Report:")
    pprint.pprint(report)
    print("Confusion Matrix:")
    print(conf_matrix)

    # 5. Return Metrics
    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix, # <-- MODIFICATION: Return NumPy array, not list
        "feature_importances": dict(zip(feature_names, model.feature_importances_)) # <-- MODIFICATION: Added feature importances
    }

    print("--- Random Forest Training Complete ---")
    return model, metrics

