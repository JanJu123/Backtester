import pandas as pd
import numpy as np
from typing import Tuple, List, Set

from indicators import function_map
from patterns import market_structure, candle_patterns

class DataPreprocessor:
    """
    Intelligently analyzes a strategy's configuration and pre-calculates all necessary
    indicators and patterns, returning an enriched DataFrame.
    """
    def __init__(self, df: pd.DataFrame, config, mode: str = 'backtest', extra_arg="features"):
        self.df = df.copy()
        self.config = config
        self.mode = mode
        self.extra_arg = extra_arg


        self.backtest_calc_map = {
            # Core Signals
            "bb_signal_params": self._calculate_bollinger_bands,
            "macd_signal_params": self._calculate_macd,
            "rsi_signal_params": self._calculate_rsi,

            # Filters
            "adx_filter": self._calculate_adx,
            "ma_trend_filter": self._calculate_ma_trend,
            "rsi_filter": self._calculate_rsi, 
            "hurst_exponent_filter": self._calculate_hurst_exponent,
            "ma_momentum_filter": self._calculate_ma_momentum,

            # Patterns
            "ls_fvg_params": self._calculate_recent_low_high,
            
            # Exit Indicators 
            "atr_exit": self._calculate_atr_for_exits,
            "ma_exit": self._calculate_ma_for_exits,
            "fvg_exit": self._calculate_fvg_for_exits
        }

        self.ml_calc_map = {
            # ML preprocessing
            "ma_params": self._calculate_ma_ML,
            "atr_params": self._calculate_atr_ML,
            "rsi_params": self._calculate_rsi_ML,
            "bb_params": self._calculate_bollinger_bands_ML,
            "adx_params": self._calculate_adx_ML,
            "macd_params": self._calculate_macd_ML,
            "hurst_exponent_params": self._calculate_hurst_exponent_ML,
            "norm_volume_params": self._calculate_norm_volume_ML,
            "pct_close_params": self._calculate_pct_close_ML,
            
        }

        if self.mode == 'ml':
            self.calculation_map = self.ml_calc_map
        else:
            self.calculation_map = self.backtest_calc_map

    def run(self) -> pd.DataFrame | Tuple[pd.DataFrame, List[str]]:
        """
        Main method to discover and run all required pre-calculations.
        """
        initial_columns = set(self.df.columns)
        required_calculations: Set[str] = set()

        if self.mode == 'ml':
            self._discover_ml_requirements(required_calculations)
            print(f"\n--- Pre-calculating ML {self.extra_arg}: {list(required_calculations)} ---")
            for key in required_calculations:
                calculation_function = self.calculation_map[key]
                print(f"Calculating {key}...")
                calculation_function()

            final_columns = set(self.df.columns)
            feature_names = sorted(list(final_columns - initial_columns)) # Sort for consistency
            return self.df, feature_names

        else: # backtest mode
            self._discover_core_signal_requirements(required_calculations)
            self._discover_filter_requirements(required_calculations)
            self._discover_exit_requirements(required_calculations)

            calculatable_features = {f for f in required_calculations if f in self.calculation_map}
            missing_features = required_calculations - calculatable_features

            print(f"--- Pre-calculating required features: {list(calculatable_features)} ---")
            if missing_features:
                 print(f"Warning: No calculation functions found for: {list(missing_features)}")

            for key in calculatable_features:
                calculation_function = self.calculation_map[key]
                calculation_function()

            return self.df
    

    
    # --- Private helper methods for discovery ---
    
    def _discover_core_signal_requirements(self, required_calculations: set):
        """Discovers calculation needs from the core signal configuration."""
        if self.config.core_signal_params:
            for param_key, params in self.config.core_signal_params.__dict__.items():
                if params is not None and param_key in self.calculation_map:
                    required_calculations.add(param_key)

    def _discover_filter_requirements(self, required_calculations: set):
        """Discovers calculation needs from the active filters."""
        if self.config.filters:
            for filter_key, filter_params in self.config.filters.__dict__.items():
                if filter_params is not None and filter_params.is_active and filter_key in self.calculation_map:
                    required_calculations.add(filter_key)

    def _discover_exit_requirements(self, required_calculations: set):
        """Checks the SL/TP strategies and adds their indicators to the calculation list."""
        sl_name = self.config.stop_loss_strategy
        tp_name = self.config.take_profit_strategy

        if sl_name in ["AtrBasedStopLoss", "AtrTrailingStopLoss"] or tp_name in ["AtrBasedTakeProfit"]:
            required_calculations.add("atr_exit") # Add the key

        if sl_name == "MaReversionStopLoss" or tp_name == "MaReversionTakeProfit":
            required_calculations.add("ma_exit") # Add the key
        
        if sl_name == "FvgBasedStopLoss":
            required_calculations.add("fvg_exit")



    def _discover_ml_requirements(self, required_calculations: set):
        """
        Discovers required features and labels for ML mode based on the MLConfig object,
        checking for None and the 'is_active' flag.
        """

        params_dict = vars(self.config)

        for key, params_obj in params_dict.items():
            # Skip if the parameter block is None
            if params_obj is None:
                continue
            # Skip if 'is_active' exists and is False
            if hasattr(params_obj, 'is_active') and not params_obj.is_active:
                continue
            # Only add if a calculation function actually exists in the ML map
            if key in self.ml_calc_map:
                 required_calculations.add(key)

            
    #? --- Private helper methods for each calculation ---
    #* Default way
    def _calculate_bollinger_bands(self):
        bb_params = self.config.core_signal_params.bb_signal_params
        func = function_map.get_indicator_func("bb")
        self.df[["bband_middle", "bband_upper", "bband_lower"]] = func(self.df, bb_params)

    def _calculate_adx(self):
        adx_params = self.config.filters.adx_filter
        func = function_map.get_indicator_func("adx")
        self.df["adx"] = func(self.df, adx_params)

    def _calculate_ma_trend(self):
        ma_params = self.config.filters.ma_trend_filter
        func = function_map.get_indicator_func(ma_params.ma_func)
        self.df["ma_trend"] = func(self.df, ma_params.period)

    def _calculate_ma_momentum(self):
        """Calculates 2 MA, 1 fast and 1 slow"""
        ma_momentum_params = self.config.filters.ma_momentum_filter
        func_fast = function_map.get_indicator_func(ma_momentum_params.ma_func_fast)
        func_slow = function_map.get_indicator_func(ma_momentum_params.ma_func_slow)

        self.df["ma_fast"] = func_fast(self.df, ma_momentum_params.period_fast)
        self.df["ma_slow"] = func_fast(self.df, ma_momentum_params.period_slow)

    def _calculate_rsi(self):
        rsi_params = None
        if self.config.core_signal_params and hasattr(self.config.core_signal_params, "rsi_signal_params") and self.config.core_signal_params.rsi_signal_params:
            rsi_params = self.config.core_signal_params.rsi_params
        elif self.config.filters and self.config.filters.rsi_filter:
            rsi_params = self.config.filters.rsi_filter
        
        func = function_map.get_indicator_func("rsi")
        self.df["rsi"] = func(self.df, rsi_params)
            
    def _calculate_recent_low_high(self):
        lookback = self.config.core_signal_params.ls_fvg_params.range_lookback
        self.df = market_structure.calculate_rolling_high_low(self.df, lookback)

    def _calculate_macd(self):
        # print("Pre-calculating MACD ...")
        macd_params = self.config.core_signal_params.macd_signal_params
        func = function_map.get_indicator_func("macd")
        self.df = func(self.df, macd_params)
    

    def _calculate_hurst_exponent(self):
        """Calculates just the Hurst Exponent value."""
        hurst_params = self.config.filters.hurst_exponent_filter
        func = function_map.get_indicator_func("hurst_exponent")
        
        # This method's only job is to add the raw indicator column
        self.df["hurst_exponent"] = func(self.df, hurst_params)



    #*----- for ML part ------
    def _calculate_adx_ML(self):
        adx_params = self.config.adx_params
        func = function_map.get_indicator_func("adx")
        self.df[f"adx_{adx_params.period}"] = func(self.df, adx_params)

    def _calculate_ma_ML(self):
        ma_params = self.config.ma_params
        func = function_map.get_indicator_func(ma_params.ma_func)
        self.df[f"ma_{ma_params.period}_{ma_params.ma_func}"] = func(self.df, ma_params.period)


    def _calculate_rsi_ML(self):
        rsi_params = self.config.rsi_params

        func = function_map.get_indicator_func("rsi")
        self.df[f"rsi_{rsi_params.period}"] = func(self.df, rsi_params)
          
    def _calculate_macd_ML(self):
        macd_params = self.config.macd_params
        func = function_map.get_indicator_func("macd")
        self.df = func(self.df, macd_params)

        initial_cols = set(self.df.columns)
        new_cols = set(self.df.columns) - initial_cols
        rename_map = {}
        p_fast = macd_params.period_fast
        p_slow = macd_params.period_slow
        p_sig = macd_params.period_signal_line
        for col in new_cols:
             # Basic renaming, adjust based on actual column names from func
             if 'macd' in col.lower() and 'signal' not in col.lower() and 'hist' not in col.lower():
                 rename_map[col] = f"macd_{p_fast}_{p_slow}_{p_sig}"
             elif 'signal' in col.lower():
                 rename_map[col] = f"macdsignal_{p_fast}_{p_slow}_{p_sig}"
             elif 'hist' in col.lower():
                 rename_map[col] = f"macdhist_{p_fast}_{p_slow}_{p_sig}"
        self.df.rename(columns=rename_map, inplace=True)

    def _calculate_bollinger_bands_ML(self):
        bb_params = self.config.bb_params
        func = function_map.get_indicator_func("bb")
        p = bb_params.period
        std = bb_params.std_dev_mult

        # Define dynamic column names
        middle_col = f"bb_{p}_{std}_mid"
        upper_col = f"bb_{p}_{std}_upper"
        lower_col = f"bb_{p}_{std}_lower"
        # Calculate and assign to new columns
        self.df[[middle_col, upper_col, lower_col]] = func(self.df, bb_params)   


    def _calculate_atr_ML(self):
        atr_params = self.config.atr_params

        func = function_map.get_indicator_func("atr")
        self.df[f"atr_{atr_params.atr_period}"] = func(self.df, atr_params)

    def _calculate_hurst_exponent_ML(self):
        hurst_params = self.config.hurst_exponent_params
        func = function_map.get_indicator_func("hurst_exponent")
        p = hurst_params.lookback_period
        self.df[f'hurst_{p}'] = func(self.df, period=p)

    def _calculate_norm_volume_ML(self):
        window = self.config.norm_volume_params.window
        vol = self.df["Volume"]
        self.df[f"norm_volume_{window}"] =  vol / vol.rolling(window).mean()

    def _calculate_pct_close_ML(self):
        method = self.config.pct_close_params.method
        df = self.df.copy()
        if method == "pct":
            self.df[f"pct_close_{method}"] =  df["Close"].pct_change()
        elif method == "log":
            self.df[f"pct_close_{method}"] = np.log(df["Close"] / df["Close"].shift(1))
        else:
            raise ValueError(f"Unknown method for pct_close: {method}")



    #* Calculation methods for exit indicators
    def _calculate_atr_for_exits(self):
            """
            Calculates ATR for exits, finding the correct parameters from either
            the stop-loss or take-profit configuration.
            """
            # print("Pre-calculating ATR for exits...")
            # Try to get params from the SL strategy first
            if "Atr" in self.config.stop_loss_strategy:
                atr_params = self.config.stop_loss_params
            # If not found, try to get them from the TP strategy
            elif self.config.take_profit_strategy and "Atr" in self.config.take_profit_strategy:
                atr_params = self.config.take_profit_params

            func = function_map.get_indicator_func("atr")
            self.df["atr_rm"] = func(self.df, atr_params)


    def _calculate_ma_for_exits(self):
        """
        Calculates a Moving Average for exits, finding the correct parameters
        from either the stop-loss or take-profit configuration.
        """
        # print("Pre-calculating MA for exits...")

        # Try to get params from the SL strategy first
        if "Ma" in self.config.stop_loss_strategy:
            ma_params = self.config.stop_loss_params
        # If not found, try to get them from the TP strategy
        elif self.config.take_profit_strategy and "Ma" in self.config.take_profit_strategy:
            ma_params = self.config.take_profit_params
            
        func = function_map.get_indicator_func(ma_params.ma_func)
        self.df['MA_rm'] = func(self.df, period=ma_params.ma_period)


    def _calculate_fvg_for_exits(self):
        """Calculates Fair Value Gaps for FVG-based exits."""
        self.df = candle_patterns.find_fvg(self.df)
        # self.df.to_csv("debug_after_fvg_calculation.csv", index=False)

