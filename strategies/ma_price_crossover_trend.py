import pandas as pd
from typing import Callable
import numpy as np
from indicators import function_map


    

def ma_crossover(df: pd.DataFrame, params: dict):


    params = params.ma_crossover_signal_params
    period = params.period

    ma_func = function_map.get_indicator_func(name=params.ma_func)

    df["ma"] = ma_func(df=df, period=period) # Funkcija
    df["signal"] = 0

    data = {
    "close": np.array(df["Close"]),
    "ma": np.array(df["ma"]),
    "signal": np.zeros(len(df), dtype=int),
    }

    for n in range(1, len(df)):
        if pd.isna(data["ma"][n]) or pd.isna(data["ma"][n-1]):
            continue
        
        prev_close = data["close"][n-1]
        prev_ma = data["ma"][n-1]
        curr_close = data["close"][n]
        curr_ma = data["ma"][n]

        # Bullish crossover
        if prev_close < prev_ma and curr_close > curr_ma:
            if n + 1 < len(df):
                data["signal"][n] = 1

        # Bearish crossover
        elif prev_close > prev_ma and curr_close < curr_ma:
            if n + 1 < len(df):
                 data["signal"][n] = -1

    df["signal"] = data["signal"]

    return df


def ma_crossover_vectorized(df: pd.DataFrame, ma_func: Callable, params: dict):
    params = params.ma_crossover_signal_params
    period = params.period
    df["ma"] = ma_func(df=df, period=period)
    
    close = df["Close"].to_numpy()
    ma = df["ma"].to_numpy()
    signal = np.zeros(len(df), dtype=int)

    # Create arrays shifted by 1 for previous values
    prev_close = close[:-1]
    prev_ma = ma[:-1]
    curr_close = close[1:]
    curr_ma = ma[1:]

    # Avoid NaNs - create a mask where previous and current MA are not NaN
    valid_mask = (~np.isnan(prev_ma)) & (~np.isnan(curr_ma))

    # Bullish crossover: prev_close < prev_ma & curr_close > curr_ma
    bullish = (prev_close < prev_ma) & (curr_close > curr_ma) & valid_mask

    # Bearish crossover: prev_close > prev_ma & curr_close < curr_ma
    bearish = (prev_close > prev_ma) & (curr_close < curr_ma) & valid_mask

    # Assign signals shifted forward by 1 bar (i.e. at n+1)
    # So signal[2:] because bullish[1] corresponds to signal at index 2, etc.
    # But since bullish and bearish length is len-1, assign starting at index 2
    # However, your original logic assigns signal at n+1, where n starts at 1, so:
    signal_indices = np.arange(1, len(df))
    signal[signal_indices[1:][bullish[1:]]] = 1
    signal[signal_indices[1:][bearish[1:]]] = -1

    df["signal"] = signal
    return df