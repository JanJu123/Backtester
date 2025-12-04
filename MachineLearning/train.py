from data import save_load
from MachineLearning import ml_utils
from MachineLearning.ml_models.random_forest import rf_classifier
import pandas as pd
import numpy as np



#TODO Workflow:
# Naloži config za določen model (glede na ime in verzijo modela)
#   Pre izračunaj vse potrebne indicatorje z uporabo data_preprocessor.py glede na config
#   Izračunaj target label, z uporabo določene funkcije iz target_labeling.py
#   Uporabimo funkcijo za treniranje in ji passamo features in label df
#   Nato pa še analiziramo

if __name__ == "__main__":

    DATA_PATH = r"F:\Programiranje\Learning_Trading_Bot\Razvijanje_Strategij\data\files\crypto"
    DATA_FILENAME = "BTCUSDT_15m.csv"
    CONFIG_PATH = r"MachineLearning\config/"
    MODEL_TO_RUN = "regime_detection_rf_v2"
    SAVE_MODEL_ARTIFACT = False


    # --- Workflow ---
    main_df = save_load.load_and_prepare_data(path=DATA_PATH, filename=DATA_FILENAME)
    all_params = ml_utils.load_params_from_json(config_base_path=CONFIG_PATH, model_name=MODEL_TO_RUN)

    print("--- DataFrame Time Range ---")
    print(f"Start Date: {main_df['Datetime'].min().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End Date: {main_df['Datetime'].max().strftime('%Y-%m-%d %H:%M:%S')}")
    print("---------------------------------")

    # # Calculate Features and Targets/Labels
    # df_features, feature_col = ml_utils.create_feature_set(main_df, all_params)
    # print(f"Calculated feature columns: {feature_col}")

    # df_labels, target_col = ml_utils.create_labeling_set(df_features, all_params)
    # print(f"Calculated label columns: {label_col}")

    # labeled_feature_df = df_features.join(df_labels)
    # training_ready_df = labeled_feature_df.dropna()

    training_ready_df, feature_col, label_col, target_col = ml_utils.preprocess_ml_data(
            df=main_df,
            config=all_params
        )


    model, model_metrics = ml_utils.run_model_training(
                                    prepared_df=training_ready_df,
                                    feature_names=feature_col,
                                    target_name=target_col,
                                    training_params=all_params.training_params
                                               )


    ml_utils.visualize_and_save_results(
            trained_model=MODEL_TO_RUN,
            model_metrics=model_metrics,
            config=all_params, 
            experiment_id=MODEL_TO_RUN,
            save_artifact=SAVE_MODEL_ARTIFACT
        )   
    


