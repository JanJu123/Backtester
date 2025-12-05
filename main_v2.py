import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import time

#* .\trading_env\Scripts\activate
# Napiši to ^^^^ v terminal za uproabo 3.10 python verzijo, ker je bolj stabilna


# from core.strategy_engine import StrategyEngine
from data import save_load
from core.stats import PerformanceEvaluator
# from core import backtest
# from strategy_components import stop_loss_strategies
# from strategy_components import take_profit_strategies
# from strategy_components import position_sizing_strategies
# from strategy_components.trading_cost import TradingCost
from core.utils import save_to_sql_database, create_strategy_summary, create_strategy_info, calc_buy_and_hold #,save_to_sql_database_fw
from config.load_params_v1 import load_params_from_json

from core.backtest_runner import initialize_strategy_components, run_single_backtest
from core.data_preprocessor import DataPreprocessor

from config import run_config

if run_config.MODE in ["optimize", "walkforward", "o", "w"]:
    from optimization.optimizer import OptunaOptimizer





if __name__ == "__main__":
    os.system("cls")
    

    # --- Workflow ---
    # Load JSON Params
    params_path = os.path.join(run_config.BASE_DIR, "config", "params_v2.json")
    # params_path = os.path.join(run_config.BASE_DIR, "config", "strategies", f"{run_config.STRATEGY_TO_RUN.lower()}.json")
    
    all_params = load_params_from_json(params_path)

    # Load Data
    main_df = save_load.load_and_prepare_data(path=run_config.DATA_FOLDER, filename=run_config.DATA_FILE_NAME)
    

    print("--- DataFrame Time Range ---")
    print(f"Start Date: {main_df['Datetime'].min().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End Date: {main_df['Datetime'].max().strftime('%Y-%m-%d %H:%M:%S')}")
    print("---------------------------------")

    # --- NEW: Split the data automatically ---
    development_df, final_holdout_df = save_load.split_data(main_df, oos_percentage=run_config.OOS_SPLIT_PERCENTAGE)

    # --- Print the results for confirmation ---
    dev_start = development_df["Datetime"].min().date()
    dev_end = development_df["Datetime"].max().date()
    oos_start = final_holdout_df["Datetime"].min().date()
    oos_end = final_holdout_df["Datetime"].max().date()
    # main_df = development_df

    print(f"---------- Data Split ----------")
    print(f"Development Set: {len(development_df):,} rows from {dev_start} to {dev_end}")
    print(f"Final Holdout Set: {len(final_holdout_df):,} rows from {oos_start} to {oos_end}")
    print("-" * 40)

    strategy_config = all_params.strategy_configs[run_config.STRATEGY_TO_RUN]
    preprocessor = DataPreprocessor(df=development_df, config=strategy_config)
    enriched_df = preprocessor.run()


    # # 1. Define the start and end dates from your image
    # start_date = "2021-01-01"
    # end_date = "2023-08-06"

    # # 2. Ensure your 'Datetime' column is in the correct format
    # main_df["Datetime"] = pd.to_datetime(main_df["Datetime"])

    # # 3. Create a boolean mask to filter the DataFrame
    # mask = (main_df["Datetime"] >= start_date) & (main_df["Datetime"] <= end_date)

    # # 4. Apply the mask to get your final, split DataFrame
    # main_df = main_df[mask]


    # Make sure the files exist to save into
    try:
        os.makedirs(r"data/files/backtest", exist_ok=True)
        os.makedirs(r"data/databases", exist_ok=True) # Mapa za SQLITE datoteke
        os.makedirs(r"data/experiment_outputs", exist_ok=True) # Mapa za Optuna in druge outpute
    except Exception as e:
        print(f"Napaka pri kreiranju map: {e}")
        # Nadaljujemo, ker je napaka morda nepomembna
        pass






    if run_config.MODE in ["single", "s"]:
        # --- MODE 1: Run a single backtest with the default params from the JSON ---
        print("--- Running Single Backtest ---")
        df_final_signals, df_trades, summary, backtest_summary, strategy_info, iteration_log = run_single_backtest(
            df=enriched_df, 
            params=all_params, 
            strategy_name=run_config.STRATEGY_TO_RUN, 
            initial_capital=run_config.INITIAL_CAPITAL,
            trading_cost=True
        )

        if df_trades.empty:
            print(df_trades)
            raise ValueError(f"df_trades is empty, which means there was no trades made!")


        df_buy_and_hold = calc_buy_and_hold(main_df, run_config.INITIAL_CAPITAL)
        # ---- Save Data ----
        print("\n--- Saving Backtest Results ---")
        df_final_signals.to_csv(r"data/files/backtest/backtest_trades.csv", index=False) # Shranimo trejde za analizo
        save_to_sql_database(path=r"data\files\backtest", file_name="backtest.db", df_signals=df_final_signals, df_trades=df_trades, 
                            summary=summary, backtest_summary=backtest_summary, strategy_info=strategy_info, iteration_log=iteration_log,
                            df_buy_and_hold=df_buy_and_hold)






    elif run_config.MODE in ["walkforward", "w"]:
        # --- MODE 2: Run a full Walk-Forward Optimization ---
        print("--- Running Walk-Forward Optimization ---")


        optuna_search_space = [
            {
                "name": "macd_fast",
                "path": f"strategy_configs.{run_config.STRATEGY_TO_RUN}.core_signal_params.macd_params.period_fast",
                "method": "suggest_int",
                "args": [6, 30],
                "kwargs": {"step": 1}
            },
            {
                "name": "macd_slow",
                "path": f"strategy_configs.{run_config.STRATEGY_TO_RUN}.core_signal_params.macd_params.period_slow",
                "method": "suggest_int",
                "args": [20, 70],   
                "kwargs": {"step": 1}
            },
            # {
            #     "name": "adx_threshold",
            #     "path": f"strategy_configs.{STRATEGY_TO_RUN}.filters.adx_filter.threshold",
            #     "method": "suggest_int",
            #     "args": [25, 40],
            #     "kwargs": {"step": 1}
            # }
        ]


        optimizer = OptunaOptimizer(
            df=enriched_df,
            params=all_params,
            strategy_name=run_config.STRATEGY_TO_RUN,
            initial_capital=run_config.INITIAL_CAPITAL,
            search_space=optuna_search_space
        )



        # --- NEW: Split the data automatically ---
        development_df, final_holdout_df = save_load.split_data(main_df, oos_percentage=run_config.OOS_SPLIT_PERCENTAGE)

        # --- Print the results for confirmation ---
        dev_start = development_df["Datetime"].min().date()
        dev_end = development_df["Datetime"].max().date()
        oos_start = final_holdout_df["Datetime"].min().date()
        oos_end = final_holdout_df["Datetime"].max().date()

        print(f"\n--- Data Split ---")
        print(f"Development Set: {len(development_df):,} rows from {dev_start} to {dev_end}")
        print(f"Final Holdout Set: {len(final_holdout_df):,} rows from {oos_start} to {oos_end}")
        print("-" * 20)



        df_final_signals, df_trades, last_study, best_params_dict = optimizer.run_walk_forward(
            start_date="2022-08-01",
            end_date="2024-08-01",
            train_period_months=12,
            test_period_months=3,
            n_trials_per_window=10,
        )

        wf_evaluator = PerformanceEvaluator(trades_df=df_trades, initial_capital=run_config.INITIAL_CAPITAL)
        summary = wf_evaluator.summary()




        print("\n--- FINAL WALK-FORWARD PERFORMANCE SUMMARY ---")
        print(summary)

        print("\n"*2)
        for key, val in best_params_dict.items():
            print(f"Test window: {key}. Best changed params: {val}")

        #TODO: Dodaj, tako da bo shranilo iste rezultate kot pri navadnem testu, ampak brez iteration_log


        df_buy_and_hold = calc_buy_and_hold(main_df, run_config.INITIAL_CAPITAL)



        # ---- Save Data ----
       
        print("\n--- Saving Backtest Results ---")
        df_final_signals.to_csv(r"data/files/backtest/backtest_trades.csv", index=False) # Shranimo trejde za analizo

        output_path = r"data/files/backtest/backtest_trades.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        save_to_sql_database(path=r"data\files\backtest", file_name="backtest.db", df_signals=df_final_signals, df_trades=df_trades, 
                        summary=summary, df_buy_and_hold=df_buy_and_hold)







    elif run_config.MODE in ["optimize", "o"]:
        # --- MODE 3: Run a full  Optimization ---
        print("--- Running Optimization on In-Sample data---")

        optuna_search_space = [
            {
                "name": "bb_std_dev_mult",
                "path": f"strategy_configs.{run_config.STRATEGY_TO_RUN}.core_signal_params.bb_params.std_dev_mult",
                "method": "suggest_float",
                "args": [1,3],
                "kwargs": {"step": 0.5}
            },
            {
                "name": "rsi_smoothing_period",
                "path": f"strategy_configs.{run_config.STRATEGY_TO_RUN}.filters.rsi_filter.smoothing_period",
                "method": "suggest_int",
                "args": [2, 20],   
                "kwargs": {"step": 2}
            },
            {
                "name": "rsi_oversold",
                "path": f"strategy_configs.{run_config.STRATEGY_TO_RUN}.filters.rsi_filter.oversold",
                "method": "suggest_int",
                "args": [10, 40],   
                "kwargs": {"step": 5}
            },
                    {
                "name": "rsi_overbought",
                "path": f"strategy_configs.{run_config.STRATEGY_TO_RUN}.filters.rsi_filter.overbought",
                "method": "suggest_int",
                "args": [60, 90],   
                "kwargs": {"step": 5}
            }
        ]


        optimizer = OptunaOptimizer(
            df=enriched_df,
            params=all_params,
            strategy_name=run_config.STRATEGY_TO_RUN,
            initial_capital=run_config.INITIAL_CAPITAL,
            search_space=optuna_search_space
        )

        optimizer.run_simple_optimization(n_trials=run_config.OPT_N_TRIALS,
                                          storage_location="sqlite:///optimization_results/simple_optimization_result.db")
        # print("\nBest parameters:")
        # optimizer.get_strategy_details()






#TODOS:
# 1. Dodaj tako da bo vse shranjeno v en df, in potem naredimo dict z arrays = Data, # in potem pasamo samo idnekse in ne več slicamo za hitrost bo bolje
# 2. Dodaj main.py config v now file.py ki bo imel shranjene confige  kot so v main_v2.py
#* 3. Dodaj documentation, tako da se bo sam generiral, zato da bom vedel kaksne parametre moram napisati za izbran indicator, sl, tp, strategy ....
# Pozneje: Dodaj MaBasedTraillingStopLoss, ki bo deloval kot traling stop loss z MA, za strategijo: Momentum_zscore_mean_reversion_V2

# 4. Za vsako SL in TP in mogoče tudi Pos sizing class, dodaj funkcijo prepare_data, ki jo bom poklical ob inicializaciji classa in shranil podatke v
        # main df, kjer bo class lahko potem dostopal do teh podatkov, kadar bo hotel. 
        # Primer: Potrebujem ma za ma price crossover: ma_func: ema, ma_period: 20, 
                #  Primer imena columna ma_ema_20, in potem samo zgradim ta col, da imam potem dostop do njega


#TODO: 1. task
# 1. Stop Loss Strategies               Status: Y
# 2. Take profit Strategies             Status: Y
# 3. Position Sizing strategies         Status: Y  (nothing to change)
# 4. Strategies                         Status: Y  (nothing to change)
# 5. Indicators                         Status: Y  (nothing to change)

#TODO: 4. task
# 1. Stop Loss Strategies               Status: X
# 2. Take profit Strategies             Status: X
# 3. Position Sizing strategies         Status: X


