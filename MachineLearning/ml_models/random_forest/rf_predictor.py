import pandas as pd
import joblib 
import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


def load_rf_model(model_version: str):
    """
    Loads the Random Forest model object from the saved .pkl file.

    Args:
        model_version (str): The identifier for the model (e.g., 'first_run').

    Returns:
        The loaded Scikit-learn model object.
    """
    # 1. Define the Save/Load Path (MUST MATCH THE CLASSIFIER SCRIPT)
    # The current directory is '...Razvijanje_Strategij'
    load_dir = os.path.join(os.getcwd(), '..', '..', 'optimization_results', 'ml_models')
    model_filepath = os.path.join(load_dir, f'rf_model_{model_version}.pkl')

    # 2. Load the Model
    print(f"\n--- Loading Model: {model_filepath} ---")
    try:
        model = joblib.load(model_filepath)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_filepath}")
        raise



def predict_and_evaluate(model, X_test: pd.DataFrame, Y_test: pd.Series):
    """
    Uses the trained model to make predictions on test data and evaluates performance.

    Args:
        model: The loaded Random Forest model object.
        X_test (pd.DataFrame): The test features (indicators).
        Y_test (pd.Series): The actual test target (0s and 1s).
    """
    
    # 1. Prediction: Hard Classification (0 or 1)
    Y_pred = model.predict(X_test)
    
    # 2. Prediction: Probability/Confidence Score (CRITICAL for strategy)
    Y_proba = model.predict_proba(X_test)
    
    # --- EVALUATION ---
    print("\n--- Model Evaluation on Test Data ---")
    
    # A. Confusion Matrix (Raw Counts)
    cm = confusion_matrix(Y_test, Y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # B. Classification Report (CRITICAL TRADING METRICS)
    report = classification_report(Y_test, Y_pred, target_names=['HOLD (0)', 'BUY (1)'], output_dict=False)
    print("\nClassification Report:")
    print(report)
    
    # C. Analyze Confidence (Future Strategy Hook)
    # The '1' column contains the probability of a BUY signal
    buy_confidence = Y_proba[:, 1]
    avg_buy_confidence = buy_confidence[Y_pred == 1].mean()
    
    print(f"\nAverage Confidence for BUY Signals: {avg_buy_confidence:.4f}")
    print("-" * 40)



if __name__ == '__main__':
    
    # 🚨 CRITICAL TO CUSTOMIZE: MODEL VERSION 🚨
    # This must match the version saved by rf_classifier.py
    MODEL_VERSION = 'first_run' 
    
    # 1. MOCK X/Y Generation (Identical to Classifier's TEST data split)
    np.random.seed(42)
    N_SAMPLES = 1000
    dates = pd.date_range('2020-01-01', periods=N_SAMPLES, freq='D')
    
    # Mock Features (X)
    mock_X = pd.DataFrame({
        'RSI_14': np.random.uniform(30, 70, N_SAMPLES),
        'ADX_20': np.random.uniform(10, 50, N_SAMPLES),
        'Volume_MA': np.random.uniform(100, 1000, N_SAMPLES)
    }, index=dates)
    
    # Mock Target (Y)
    mock_Y = ((mock_X['RSI_14'] > 60).astype(int) + (np.random.rand(N_SAMPLES) > 0.95)).clip(upper=1)
    mock_Y = pd.Series(mock_Y, index=dates, name='target_buy')
    
    # MOCK TEMPORAL SPLIT
    split_index = int(N_SAMPLES * 0.8) 
    X_test = mock_X.iloc[split_index:]
    Y_test = mock_Y.iloc[split_index:]
    
    print(f"Test Data Loaded. Size: {len(X_test)}")
    # --- END MOCK DATA ---

    # 2. Load the Model
    loaded_model = load_rf_model(MODEL_VERSION)
    
    # 3. Predict and Evaluate
    predict_and_evaluate(loaded_model, X_test, Y_test)