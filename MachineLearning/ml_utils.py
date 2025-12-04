import pandas as pd
import numpy as np
import json
import os
from types import SimpleNamespace
from core.data_preprocessor import DataPreprocessor
from MachineLearning import target_labeling

from MachineLearning.config.param_types_ml import MLConfig, LabelingRequiredFeatures,TrainingParamsML
from dacite import from_dict, DaciteError

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

import pickle
from typing import Tuple, List, Dict, Any, Set
from dataclasses import asdict, is_dataclass, make_dataclass, fields
import time
import importlib

def prepare_and_split_data(df: pd.DataFrame, feature_cols: list, target_col: str, train_size: float = 0.7):
    """
    Separates the DataFrame into features (X) and target (Y) and performs a 
    time-series split based on the specified train size.

    Args:
        df: The DataFrame containing all features and the target column.
        feature_cols: A list of column names for the features (X).
        target_col: The name of the target column (Y).
        train_size: The proportion of data to use for the training set (e.g., 0.7).

    Returns:
        X_train, X_test, Y_train, Y_test (The four split DataFrames/Series).
    """
    # 1. Separate X (Features) and Y (Target)
    X = df[feature_cols]
    Y = df[target_col]

    # 2. Determine the split index
    split_index = int(len(X) * train_size)

    # 3. Perform the Time-Series Split
    # .iloc is used to select rows by integer location, preserving time order
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    Y_train = Y.iloc[:split_index]
    Y_test = Y.iloc[split_index:]

    # print(f"Total Samples: {len(X)}")
    # print(f"Train Samples: {len(X_train)}")
    # print(f"Test Samples: {len(X_test)}")
    
    return X_train, X_test, Y_train, Y_test




def load_params_from_json(config_base_path: str, model_name: str) -> MLConfig: # <-- Changed return type hint
    """
    Loads, VALIDATES (using dataclasses), and returns a specific model's
    configuration as a dataclass object (which supports dot-notation).

    Args:
        config_base_path (str): The base path to the 'configs' folder
                                (e.g., "MachineLearning/configs").
        model_name (str): The full Model ID
                          (e.g., "regime_detection_rf_v1").

    Returns:
        MLConfig: The validated dataclass object for the specified model.
    """

    # --- 1. Parse the model_name ---
    try:
        parts = model_name.rsplit('_', 2)
        folder_name = parts[0]
        file_name = f"{parts[1]}_{parts[2]}"
    except IndexError:
        raise ValueError(f"model_name format is incorrect. Expected 'GROUP_NAME_VERSION', got: {model_name}")

    # --- 2. Construct the full file path ---
    config_file_name = f"{file_name}.json"
    config_path = os.path.join(config_base_path, folder_name, config_file_name)

    # --- 3. Load the RAW JSON data ---
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at path: {config_path}")

    with open(config_path, 'r') as f:
        try:
            # Loads the whole file { "exp_name": { ... } }
            raw_configs_file = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {config_path}: {e}")

    if not isinstance(raw_configs_file, dict):
        raise ValueError(f"Config file {config_path} does not contain a root JSON object.")

    # --- 4. Get the specific experiment's raw dictionary ---
    if model_name not in raw_configs_file:
        raise KeyError(f"Experiment key '{model_name}' not found in {config_path}. Found keys: {list(raw_configs_file.keys())}")

    # Get the inner dict { "model_info": ..., }
    model_config_dict = raw_configs_file[model_name]

    # --- 5. NEW: Validate and parse using dataclasses ---
    try:
        # Use dacite to convert the dictionary to the MLConfig dataclass.
        # This performs the validation automatically.
        validated_config_object = from_dict(data_class=MLConfig, data=model_config_dict)
        print(f"Successfully loaded and validated config for {model_name} using dataclasses.")
        return validated_config_object
    except DaciteError as e: # Catch dacite's specific validation errors
        print(f"--- DATACLASS VALIDATION ERROR for {model_name} in {config_path} ---")
        print("Your JSON config might be missing required fields, have typos, or wrong data types.")
        print(f"Error details: {e}")
        raise ValueError("Failed to validate ML config dictionary against dataclass structure.") from e
    except Exception as e: # Catch any other unexpected errors during parsing
         print(f"An unexpected error occurred during config parsing for {model_name}: {e}")
         raise




def validate_and_map_labeling_columns(df: pd.DataFrame, col_name: list, params):
    """
    Constructs and validates the mapping between generic labeling requirements
    (e.g., 'adx') and the specific, dynamically-named columns calculated by
    the preprocessor (e.g., 'adx_14').

    This ensures the labeling function receives the correct column names
    corresponding to the parameters defined in the configuration's
    `labeling_params.required_features`.
    """

    builded_params = {}

    # Access the required features dataclass
    required_features = params.required_features

    for param in fields(required_features):
        param_name = param.name
        value = getattr(required_features, param_name)  # e.g. AdxParamsML, MaParamsML

        if value is None or not is_dataclass(value):
            continue

        # Skip if feature inactive
        if not getattr(value, "is_active", False):
            continue

        feature_parts = []
        for f in fields(value):
            f_name = f.name
            f_value = getattr(value, f_name)

            if f_name == "is_active" or f_value is None:
                continue

            feature_parts.append(str(f_value))

        if feature_parts:
            base_name = param_name.rsplit("_", 1)[0]  # e.g. "adx" from "adx_params"
            feature_name = "_".join([base_name] + feature_parts)
            builded_params[base_name] = feature_name

    # Validate columns exist
    for key, feature_name in builded_params.items():
        if feature_name not in df.columns:
            print(f"❌ {feature_name} NOT found in df")
        # else:
        #     print(f"✅ {feature_name} exists in df")

    return builded_params





def create_feature_set(df: pd.DataFrame, feature_col: list):
    df_feature = df[feature_col].copy()
    return df_feature, feature_col


def create_labeling_set(df: pd.DataFrame, params: dict, label_map):
    df = df.copy()

    # inverted_label_map = {v: k for k, v in label_map.items()}  # invert mapping, Keys <=> Values
    # df_labels = df.rename(columns=inverted_label_map)

    label_func = target_labeling.get_label_func(params.function_name)
    df_labels, labels_col = label_func(df, params.func_params, label_map)


    return df_labels, labels_col



def preprocess_ml_data(df: pd.DataFrame, config):
    """
    Handles calculation of ALL required features (for model AND labeling)
    and target labeling for the ML pipeline. 
    """

    #TODO:
    # 1. Pridobimo vse parametre, za features in labels/targets
    # 2. Parametre bomo poslali v DataPreproccesor, vsako posebej za features in labels
    # 3. Nazaj dobimo feature in labels col name,
    # 4. Naredimo posebej feature df, ki vsebuje samo features
    # 5. Uporabimo fucnkijo, ki je namenjena za izračunavo labels, in dobimo target df/series
    # 6. Nato naredimo posebej label/target df/series, ki vsebuje samo target
    # 7. Vrnemo, df, features_col_names, target_col_names


    df = df.copy()
    feature_params = config.feature_params
    label_params = config.labeling_params

    df_feature, feature_col = DataPreprocessor(df=df, config=feature_params, mode="ml", extra_arg="features").run()
    df_label, label_col = DataPreprocessor(df=df, config=label_params.required_features, mode="ml", extra_arg="labels").run()


    # feature_col_map = validate_and_map_labeling_columns(df_feature, col_name=feature_col, params=feature_params)
    label_col_map = validate_and_map_labeling_columns(df_label, col_name=label_col, params=label_params)

    df_feature, feature_col = create_feature_set(df_feature, feature_col)
    df_target, target_col = create_labeling_set(df_label, params=label_params, label_map=label_col_map)


    print("\n--- Combining features, label indicators, and targets... ---")

    combined_df = df.copy() 

    combined_df = combined_df.join(df_feature, how='inner') 
    print(f"Shape after joining features: {combined_df.shape}")

    new_label_indicator_cols = [col for col in df_label.columns if col not in combined_df.columns]
    if new_label_indicator_cols:
        combined_df = combined_df.join(df_label[new_label_indicator_cols], how='inner')
        print(f"Shape after joining label indicators: {combined_df.shape}")
        

    combined_df = combined_df.join(df_target, how='inner')
    print(f"Shape after joining targets: {combined_df.shape}")

    print(f"Shape before final dropna: {combined_df.shape}")
    prepared_df = combined_df.dropna().reset_index(drop=True)
    print(f"Shape after final dropna: {prepared_df.shape}")
 
    df_final = prepared_df.loc[:, ~prepared_df.columns.duplicated()]
    print(f"Shape after removing duplicates: {prepared_df.shape}")



    return df_final, feature_col, label_col, target_col


def plot_results(
    metrics: Dict[str, Any],
    labels: List[str],
    filepath: str, # Requires filepath
    show_equity_curve: bool = False,
    show_feature_importance: bool = True
):
    """ Creates a single, combined figure and saves it (overwriting). """
    num_plots = 1 + int(show_equity_curve) + int(show_feature_importance)
    if num_plots == 0: return
    # Basic check for confusion matrix data
    has_cm = 'confusion_matrix' in metrics and isinstance(metrics['confusion_matrix'], np.ndarray) and metrics['confusion_matrix'].sum() > 0
    has_fi = show_feature_importance and 'feature_importances' in metrics and metrics['feature_importances']
    # Check if anything valid to plot exists
    if not has_cm and not show_equity_curve and not has_fi:
        print("No valid data found to plot.")
        return
    # Adjust plot count if CM is missing
    if not has_cm:
        num_plots -=1
        if num_plots == 0: return # Exit if nothing left

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1: axes = [axes]
    ax_index = 0

    # Plot CM if available
    if has_cm:
        ax = axes[ax_index]
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_title("Confusion Matrix", fontsize=12, fontweight='bold')
        ax.set_xlabel("Predicted Label", fontsize=10); ax.set_ylabel("True Label", fontsize=10)
        plt.sca(ax); plt.yticks(rotation=0)
        ax_index += 1

    # Plot FI if available and requested
    if has_fi:
        ax = axes[ax_index]
        importances = metrics['feature_importances']
        feature_importance_df = pd.DataFrame({'feature': list(importances.keys()), 'importance': list(importances.values())}).sort_values(by='importance', ascending=True)
        ax.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='teal')
        ax.set_title("Feature Importance", fontsize=12, fontweight='bold')
        ax.set_xlabel("Feature Importance", fontsize=10)
        ax_index += 1

    # Plot EQ (Placeholder)
    if show_equity_curve:
        ax = axes[ax_index]
        ax.set_title("Equity Curve (Placeholder)", fontsize=12, color='gray')
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax_index += 1

    # Save the combined figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath) # Save to file
    plt.close() # Close figure
    print(f"Saved (Overwritten) Model Summary Plot to: {filepath}")

def _convert_metrics_to_serializable(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively converts numpy types and dataclasses in a nested dict
    to standard Python types for JSON serialization.
    """
    if not isinstance(metrics_dict, dict):
        return metrics_dict

    serializable_dict = {}
    for k, v in metrics_dict.items():
        if isinstance(v, np.ndarray):
            serializable_dict[k] = v.tolist()
        elif isinstance(v, (np.floating)): # Catches np.float32, np.float64, etc.
            serializable_dict[k] = float(v)
        elif isinstance(v, (np.integer)): # Catches np.int32, np.int64, etc.
            serializable_dict[k] = int(v)
        elif isinstance(v, dict):
            # Recurse for nested dictionaries (like classification_report)
            serializable_dict[k] = _convert_metrics_to_serializable(v)
        elif isinstance(v, list):
             # Recurse for items in lists
             serializable_dict[k] = [_convert_metrics_to_serializable(item) if isinstance(item, dict) else item for item in v]
        elif is_dataclass(v):
            serializable_dict[k] = asdict(v)
        elif isinstance(v, (int, float, str, bool, type(None))):
            serializable_dict[k] = v # Already serializable
        else:
            serializable_dict[k] = str(v) # Fallback for other types
    return serializable_dict
# --- End New Helper ---

# --- UPDATED: visualize_and_save_results ---
def visualize_and_save_results(
    trained_model: Any,
    model_metrics: Dict[str, Any],
    config: MLConfig,
    experiment_id: str,
    save_artifact: bool
):
    """ Handles visualization and permanent logging/saving. """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

    # --- Visualization (Save Plot File) ---
    if model_metrics:
        # --- FIX: Remap labels for visualization ---
        # Your XGBoost model predicts [0, 1, 2]. Let's map them to their meaning.
        # Map: Bearish(-1) -> 0, Choppy(0) -> 1, Bullish(1) -> 2
        # So the labels for the plot should be:
        class_labels = ["Bearish (-1)", "Choppy (0)", "Bullish (1)"] 
        print("\n--- Generating Visualizations (Saving to Disk) ---")
        VISUALS_DIR = os.path.join(project_root, "experiment_outputs", "ml_training", "visuals", experiment_id)
        visual_filepath = os.path.join(VISUALS_DIR, "combined_model_summary.png")
        plot_results( 
            metrics=model_metrics, labels=class_labels, filepath=visual_filepath,
            show_equity_curve=False, show_feature_importance=True
        )
    else:
        print("\nSkipping visualization as no metrics were provided.")

    # --- Logging & Optional Artifact Saving ---
    if trained_model and model_metrics:
        print("\n--- Logging Metrics and Optionally Saving Model ---")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        try: group_name = experiment_id.rsplit('_', 2)[0]
        except IndexError: group_name = "unknown_group"; print(f"Warning: Could not parse group name.")
        ARTIFACT_SAVE_DIR = os.path.join(project_root, "trained_artifacts", group_name)
        metrics_filename = f"{experiment_id}_{timestamp}_metrics.json"
        config_filename = f"{experiment_id}_{timestamp}_config.json"
        model_filename = f"{experiment_id}_{timestamp}_model.pkl"
        metrics_filepath = os.path.join(ARTIFACT_SAVE_DIR, metrics_filename)
        config_filepath = os.path.join(ARTIFACT_SAVE_DIR, config_filename)
        model_filepath = os.path.join(ARTIFACT_SAVE_DIR, model_filename)
        os.makedirs(ARTIFACT_SAVE_DIR, exist_ok=True)

        # --- Log Metrics (WITH FIX) ---
        try:
            # Use the new recursive helper function to clean the dict
            serializable_metrics = _convert_metrics_to_serializable(model_metrics)
            
            with open(metrics_filepath, 'w') as f: json.dump(serializable_metrics, f, indent=4)
            print(f"Metrics logged to: {metrics_filepath}")
        except Exception as e: 
            print(f"Error saving metrics: {e}")
            # Print problematic dict for debugging
            # print(f"Problematic metrics dict: {model_metrics}")

        # --- Log Config ---
        try:
            config_dict = asdict(config)
            with open(config_filepath, 'w') as f: json.dump(config_dict, f, indent=4)
            print(f"Config logged to: {config_filepath}")
        except Exception as e: print(f"Error saving config: {e}")

        # --- Save Model Artifact (Optional) - Using Pickle ---
        if save_artifact:
            try:
                with open(model_filepath, 'wb') as f:
                    pickle.dump(trained_model, f) # Use pickle
                print(f"Model artifact saved to: {model_filepath}")
            except Exception as e: print(f"Error saving model artifact with pickle: {e}")
        else:
            print("Skipping model artifact saving (save_artifact=False).")

    elif not trained_model:
        print("\nSkipping logging/saving because model training failed.")





def run_model_training(
    prepared_df: pd.DataFrame,
    feature_names: List[str],
    target_name: str,
    training_params: TrainingParamsML # Use the dataclass for type hint
) -> Tuple[Any, Dict[str, Any]]:
    """
    Dynamically imports and runs the correct model trainer based on config.
    
    Returns:
        Tuple[model_object, metrics_dict]
    """
    print("\n--- Training Model ---")
    model_type = training_params.model_type
    print(f"Attempting to train model type: {model_type}")

    # --- Dynamic Import Logic ---
    try:
        model_group = model_type.split('_')[0]
        trainer_filename = model_type
        module_path = f"MachineLearning.ml_models.{model_group}.{trainer_filename}"
        print(f"Importing trainer module: {module_path}")
        trainer_module = importlib.import_module(module_path)
    except (ModuleNotFoundError, IndexError, ImportError) as e:
        print(f"Error importing trainer module: {e}")
        print(f"Looked for module at: {module_path}")
        raise SystemExit

    if hasattr(trainer_module, 'train_model'):
        train_function = getattr(trainer_module, 'train_model')
        
        # Call the specific model's training function
        trained_model, model_metrics = train_function(
            prepared_df=prepared_df,
            feature_names=feature_names,
            target_name=target_name,
            training_params=training_params # Pass the full params object
        )
        print("\n--- Model Training Summary ---")
        return trained_model, model_metrics
    else:
        print(f"Error: Module {module_path} does not have a 'train_model' function.")
        raise SystemExit