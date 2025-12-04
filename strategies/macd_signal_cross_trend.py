import pandas as pd
from typing import Callable
import numpy as np
from indicators import function_map
from indicators import macd_indic


def macd_signal(df: pd.DataFrame, params: dict):
    """
        "macd_params": {
          "period_fast": 12,
          "period_slow": 26,
          "ma_func_slow": "ema",
          "ma_func_fast": "ema",
          "period_signal_line": 9,
          "ma_func_signal_line": "ema"
    """
    # Extract parameters
    params = params.macd_signal_params

    df_macd = macd_indic.calc_macd(df=df, params=params)

    data = {
        "macd_line": np.array(df_macd["macd_line"]),
        "signal_line": np.array(df_macd["signal_line"]),
        "histogram": np.array(df_macd["histogram"]),
        "signal":  np.zeros(len(df), dtype=int),
    }

    # Generate buy/sell signals
    for n in range(1, len(df)):
        if pd.isna(data["macd_line"][n]) or pd.isna(data["macd_line"][n-1]) or \
           pd.isna(data["signal_line"][n]) or pd.isna(data["signal_line"][n-1]):
            continue

        prev_macd = data["macd_line"][n-1]
        prev_signal_line = data["signal_line"][n-1]
        curr_macd = data["macd_line"][n]
        curr_signal_line = data["signal_line"][n]

        # Bullish crossover
        if prev_macd < prev_signal_line and curr_macd > curr_signal_line:
            if n + 1 < len(df):
                data["signal"][n] = 1

        # Bearish crossover
        elif prev_macd > prev_signal_line and curr_macd < curr_signal_line:
            if n + 1 < len(df):
                data["signal"][n] = -1

    df["signal"] = data["signal"]

    df["macd_line"] = data["macd_line"]
    df["signal_line"] = data["signal_line"]
    df["histogram"] = data["histogram"]

    return df





def macd_adx_signal(df: pd.DataFrame, params: dict):
    """
    """
    # Extract parameters
    params = params.MACD_adx

    df_macd = macd_indic.calc_macd(df=df, params=params)

    adx_threshold = params.adx_threshold
    adx_period = params.adx_period
    adx_func = function_map.get_indicator_func(params.adx_func)
    df["adx"] = adx_func(df, adx_period)

    data = {
        "macd_line": np.array(df_macd["macd_line"]),
        "signal_line": np.array(df_macd["signal_line"]),
        "histogram": np.array(df_macd["histogram"]),
        "adx": np.array(df["adx"]),
        "signal":  np.zeros(len(df), dtype=int),
    }

    # Generate buy/sell signals
    for n in range(1, len(df)):
        if pd.isna(data["macd_line"][n]) or pd.isna(data["macd_line"][n-1]) or \
           pd.isna(data["signal_line"][n]) or pd.isna(data["signal_line"][n-1]):
            continue

        prev_macd = data["macd_line"][n-1]
        prev_signal_line = data["signal_line"][n-1]
        curr_macd = data["macd_line"][n]
        curr_signal_line = data["signal_line"][n]
        curr_adx = data["adx"][n]

        # Bullish crossover
        if prev_macd < prev_signal_line and curr_macd > curr_signal_line and curr_adx > adx_threshold:
            if n + 1 < len(df):
                data["signal"][n+1] = 1

        # Bearish crossover
        elif prev_macd > prev_signal_line and curr_macd < curr_signal_line and curr_adx > adx_threshold:
            if n + 1 < len(df):
                data["signal"][n+1] = -1

    df["signal"] = data["signal"]

    df["macd_line"] = data["macd_line"]
    df["signal_line"] = data["signal_line"]
    df["histogram"] = data["histogram"]

    return df




def macd_adx_ma_signal(df: pd.DataFrame, params: dict):
    """
    """
    # Extract parameters
    params = params.MACD_adx_ma

    df_macd = macd_indic.calc_macd(df=df, params=params)

    adx_threshold = params.adx_threshold
    adx_period = params.adx_period
    adx_func = function_map.get_indicator_func(params.adx_func)
    df["adx"] = adx_func(df, adx_period)

    ma_func = function_map.get_indicator_func(params.ma_func)
    ma_period = params.ma_period
    df["ma"] = ma_func(df, period=ma_period)

    data = {
        "close": np.array(df["Close"]),
        "macd_line": np.array(df_macd["macd_line"]),
        "signal_line": np.array(df_macd["signal_line"]),
        "histogram": np.array(df_macd["histogram"]),
        "adx": np.array(df["adx"]),
        "ma": np.array(df["ma"]),
        "signal":  np.zeros(len(df), dtype=int),
    }

    # Generate buy/sell signals
    for n in range(1, len(df)):
        if pd.isna(data["macd_line"][n]) or pd.isna(data["macd_line"][n-1]) or \
           pd.isna(data["signal_line"][n]) or pd.isna(data["signal_line"][n-1]):
            continue

        prev_macd = data["macd_line"][n-1]
        prev_signal_line = data["signal_line"][n-1]
        curr_macd = data["macd_line"][n]
        curr_signal_line = data["signal_line"][n]
        curr_adx = data["adx"][n]
        curr_ma = data["ma"][n]
        curr_close = data["close"][n]

        # Bullish crossover
        if prev_macd < prev_signal_line and curr_macd > curr_signal_line and curr_adx > adx_threshold and curr_close > curr_ma:
            if n + 1 < len(df):
                data["signal"][n+1] = 1

        # Bearish crossover
        elif prev_macd > prev_signal_line and curr_macd < curr_signal_line and curr_adx > adx_threshold and curr_close < curr_ma:
            if n + 1 < len(df):
                data["signal"][n+1] = -1

    df["signal"] = data["signal"]

    df["macd_line"] = data["macd_line"]
    df["signal_line"] = data["signal_line"]
    df["histogram"] = data["histogram"]
    
    return df






def macd_adx_ma_crossover_signal(df: pd.DataFrame, params: dict):
    """
    """
    # Extract parameters
    params = params.MACD_adx_ma_crossover

    df_macd = macd_indic.calc_macd(df=df, params=params)

    adx_threshold = params.adx_threshold
    adx_period = params.adx_period
    adx_func = function_map.get_indicator_func(params.adx_func)
    df["adx"] = adx_func(df, period=adx_period)

    ma_func_slow = function_map.get_indicator_func(params.ma_cross_func_slow)
    ma_func_fast = function_map.get_indicator_func(params.ma_cross_func_fast)
    ma_period_slow = params.ma_period_slow
    ma_period_fast = params.ma_period_fast

    df["ma_slow"] = ma_func_slow(df, period=ma_period_slow)
    df["ma_fast"] = ma_func_fast(df, period=ma_period_fast)



    data = {
        "close": np.array(df["Close"]),
        "macd_line": np.array(df_macd["macd_line"]),
        "signal_line": np.array(df_macd["signal_line"]),
        "histogram": np.array(df_macd["histogram"]),
        "adx": np.array(df["adx"]),
        "ma_slow": np.array(df["ma_slow"]),
        "ma_fast": np.array(df["ma_fast"]),
        "signal":  np.zeros(len(df), dtype=int),
    }

    # Generate buy/sell signals
    for n in range(1, len(df)):
        if pd.isna(data["macd_line"][n]) or pd.isna(data["macd_line"][n-1]) or \
           pd.isna(data["signal_line"][n]) or pd.isna(data["signal_line"][n-1]):
            continue

        prev_macd = data["macd_line"][n-1]
        prev_signal_line = data["signal_line"][n-1]
        curr_macd = data["macd_line"][n]
        curr_signal_line = data["signal_line"][n]
        curr_adx = data["adx"][n]
        curr_ma_slow = data["ma_slow"][n]
        curr_ma_fast = data["ma_fast"][n]
        curr_close = data["close"][n]

        # Bullish crossover
        if prev_macd < prev_signal_line and curr_macd > curr_signal_line and curr_adx > adx_threshold and curr_ma_slow < curr_ma_fast:
            if n + 1 < len(df):
                data["signal"][n+1] = 1

        # Bearish crossover
        elif prev_macd > prev_signal_line and curr_macd < curr_signal_line and curr_adx > adx_threshold and curr_ma_slow > curr_ma_fast:
            if n + 1 < len(df):
                data["signal"][n+1] = -1

    df["signal"] = data["signal"]

    df["macd_line"] = data["macd_line"]
    df["signal_line"] = data["signal_line"]
    df["histogram"] = data["histogram"]
    
    return df
