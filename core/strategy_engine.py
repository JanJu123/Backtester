import pandas as pd
import numpy as np
from indicators import function_map
from strategies import strategy_map
from typing import Callable
from dataclasses import asdict





class StrategyEngine():
    def __init__(self, df, params, strategy_to_use: str, test: bool = True):
        """
        Initializes the StrategyEngine object.
        - df (pd.DataFrame): The ENRICHED DataFrame containing OHLCV and all pre-calculated indicators.
        - ...
        """
        self.params = params.strategy_configs[strategy_to_use]
        self.strategy_type = self.params.strategy_type
        self.df = df.copy() # df now contains 'adx', 'rsi', etc.

        # This part remains exactly as you wanted
        description = strategy_to_use.rsplit('_', 1)
        self.core_func = strategy_map.get_strategy_func(name=description[0])
        print(description)
        self.strategy_version = description[1]
        if self.core_func is None:
            raise NotImplementedError(f"Error: {description[0]} not implemented.")
            
        self.filter_params = asdict(self.params.filters)
        self.signals = {
            "Datetime": np.array(df["Datetime"]),
            "core_signal": None,
            "final_signal": None
        }

    def _run_core_func(self):
        """ This function remains exactly as you wanted. """
        result_df = self.core_func(self.df.copy(), self.params.core_signal_params)
        self.df["core_signal"] = np.array(result_df["signal"])
        self.signals["core_signal"] = np.array(result_df["signal"])

        # Only add columns that are not in df already
        for col in result_df.columns:
            if col not in self.df.columns:
                self.df[col] = result_df[col]

        if "context" in result_df.columns:
            self.df["context"] = result_df["context"]






    def apply_filters(self):
        """
        Combines the 'core_signal' with all active, PRE-CALCULATED filter states.
        """
        final_signal = self.df['core_signal'].copy()

        for filter_name, filter_params in self.filter_params.items():
            if not (filter_params and filter_params["is_active"]):
                continue

            # --- NEW LOGIC: Derive state from pre-calculated columns ---
            state = 0 # Default state
            
            if filter_name == "adx_filter":
                state = np.where(self.df["adx"] > filter_params["threshold"], 1, -1)
                # state = pd.Series(state)
                # print(state.value_counts())
                # exit(1)

            
            elif filter_name == "ma_trend_filter":
                state = np.where(self.df["ma_trend"] > self.df["Close"], 1, -1)
            
            elif filter_name == "ma_momentum_filter":
                print(self.df.columns)
                state = np.where(self.df["ma_fast"] > self.df["ma_slow"], 1, -1)
                
            elif filter_name == "rsi_filter":
                state = np.where(self.df["rsi"] >= filter_params["overbought"], -1, 
                        np.where(self.df["rsi"] <= filter_params["oversold"], 1, 0))
            
            elif filter_name == "hurst_exponent_filter":
                state = np.where(self.df["hurst_exponent"] > filter_params["max_hurst_threshold"], 1, -1)

            # ... (Add logic for your other filters here)


            # --- The filtering logic remains the same ---
            final_signal = np.where(
                (final_signal == 1) & (state != filter_params["on_long"]), 
                0, final_signal
            )
            final_signal = np.where(
                (final_signal == -1) & (state != filter_params["on_short"]), 
                0, final_signal
            )


        self.signals['final_signal'] = final_signal

    def run(self):
        """ Main execution method. """

        self._run_core_func()
        self.apply_filters()




        self.df["signal"] = pd.DataFrame(self.signals)["final_signal"].values
        # print((self.df["signal"] != 0).value_counts())
        # exit(1)

        return self.df






        



#TODO: Postopki delovanja FLEXSIBILNEGA StrategyEngine:
    #1. : Core signal funkcijo podamo v StrategyEngine  (macd + bb, ...)
    #2. : Zraven damo še dict, ki vsebuje vse parametre za StrategyEngine (katere filtre uporabiti in njihove parametre...)
    #3. : Na koncu pa dobimo final signal coll, ki vsebuje signale. (zraven tudi vreddosti vseh filtrov.. adx value, ma, ma_slow, ma_fast ....)



#TODO: Postopki za izgradnjo FLEXSIBILNEGA StrategyEngine:
    #1. : 
    #2. :
    #3. :
    #4. :
    #5. :