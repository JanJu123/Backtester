import pandas as pd
from typing import Callable
import numpy as np
from indicators import function_map


    

def ma_crossover(df: pd.DataFrame, params: dict):

    params = params.ma_crossover_signal_params
    period_slow = params.period_slow
    period_fast = params.period_fast


    ma_func_slow = function_map.get_indicator_func(name=params.ma_func_slow)
    ma_func_fast = function_map.get_indicator_func(name=params.ma_func_fast)

    df["ma_slow"] = ma_func_slow(df=df, period=period_slow) # Funkcija
    df["ma_fast"] = ma_func_fast(df=df, period=period_fast) # Funkcija
    df["signal"] = 0
    

    data = {
        "ma_slow": np.array(df["ma_slow"]),
        "ma_fast": np.array(df["ma_fast"]),
        "signal": np.array(df["signal"]),
    }

    for n in range(1, len(df)):
        if (pd.isna(data["ma_slow"][n]) or pd.isna(data["ma_slow"][n-1])) or \
            pd.isna(data["ma_fast"][n]) or pd.isna(data["ma_fast"][n-1]):
            continue
        
        prev_ma_slow = data["ma_slow"][n-1]
        prev_ma_fast = data["ma_fast"][n-1]
        curr_ma_slow = data["ma_slow"][n]
        curr_ma_fast = data["ma_fast"][n]

        # Bullish crossover
        if prev_ma_fast < prev_ma_slow and curr_ma_fast > curr_ma_slow:
            data["signal"][n] = 1

        # Bearish crossover
        elif prev_ma_fast > prev_ma_slow and curr_ma_fast < curr_ma_slow:
            data["signal"][n] = -1

    df["signal"] = data["signal"]

    return df




def ma_crossover_adx(df: pd.DataFrame, params: dict):

    params = params.MA_crossover_adx
    period_slow = params.period_slow
    period_fast = params.period_fast

    adx_threshold = params.adx_threshold
    adx_period = params.adx_period

    ma_func_slow = function_map.get_indicator_func(params.ma_func_slow)
    ma_func_fast = function_map.get_indicator_func(params.ma_func_fast)
    adx_func = function_map.get_indicator_func(params.adx_func)

    df["ma_slow"] = ma_func_slow(df=df, period=period_slow) # Funkcija
    df["ma_fast"] = ma_func_fast(df=df, period=period_fast) # Funkcija
    df["signal"] = 0
    df["adx"] = adx_func(df, period=adx_period)

    data = {
        "ma_slow": np.array(df["ma_slow"]),
        "ma_fast": np.array(df["ma_fast"]),
        "signal": np.array(df["signal"]),
        "adx": np.array(df["adx"]),
    }

    for n in range(1, len(df)):
        if (pd.isna(data["ma_slow"][n]) or pd.isna(data["ma_slow"][n-1])) or \
            pd.isna(data["ma_fast"][n]) or pd.isna(data["ma_fast"][n-1]):
            continue
        
        prev_ma_slow = data["ma_slow"][n-1]
        prev_ma_fast = data["ma_fast"][n-1]
        curr_ma_slow = data["ma_slow"][n]
        curr_ma_fast = data["ma_fast"][n]
        curr_adx = data["adx"][n]

        # Bullish crossover
        if prev_ma_fast < prev_ma_slow and curr_ma_fast > curr_ma_slow and curr_adx >= adx_threshold:
            if n + 1 < len(df):
                data["signal"][n+1] = 1

        # Bearish crossover
        elif prev_ma_fast > prev_ma_slow and curr_ma_fast < curr_ma_slow and curr_adx >= adx_threshold:
            if n + 1 < len(df):
                data["signal"][n+1] = -1

    df["signal"] = data["signal"]

    return df
