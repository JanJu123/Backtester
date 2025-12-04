import pandas as pd
from typing import Callable
import numpy as np
from indicators import function_map

def supertrend_trend_folowing(df: pd.DataFrame, params: dict):
    """
    supertrend_params: {
        "atr_period": 10,
        "atr_multiplier": 3

        "ma_func": "ema",
        "ma_period": 200,
    }
    """

    params = params.supertrend_signal_params

    atr_multiplier = params.atr_multiplier
    ma_period = params.ma_period

    ma_func = function_map.get_indicator_func(name=params.ma_func)
    df["ma"] = ma_func(df=df, period=ma_period)

    atr_period = params.atr_period
    atr_func = function_map.get_indicator_func(name="atr")
    df["atr_supertrend"] = atr_func(df=df, params=params)

    data = {
        "close": np.array(df["Close"]),
        "high": np.array(df["High"]),
        "low": np.array(df["Low"]),
        "ma": np.array(df["ma"]),
        "atr": np.array(df["atr_supertrend"]),

        "supertrend": np.zeros(len(df)),
        "direction": np.zeros(len(df), dtype=int),
        "signal": np.zeros(len(df), dtype=int),
    }




    for i in range(1, len(df)):
        # Calculate the basic upper and lower bands
        median_price = (data["high"][i] + data["low"][i]) / 2
        upper_band = median_price + (atr_multiplier * data["atr"][i])
        lower_band = median_price - (atr_multiplier * data["atr"][i])

        # If the previous trend was up (direction was 1)
        if data["direction"][i-1] == 1:
            data["supertrend"][i] = max(lower_band, data["supertrend"][i-1])
        # If the previous trend was down (direction was -1)
        else:
            data["supertrend"][i] = min(upper_band, data["supertrend"][i-1])


        # Check for a trend flip
        if data["close"][i] > data["supertrend"][i]:
            data["direction"][i] = 1
        else:
            data["direction"][i] = -1
            if data["direction"][i-1] == 1:
                data["supertrend"][i] = upper_band






    df["supertrend_direction"] = data["direction"]

    is_bullish_regime = df["Close"] > df["ma"]
    is_bearish_regime = df["Close"] < df["ma"]

    long_signal = (df["supertrend_direction"] == 1) & (df["supertrend_direction"].shift(1) == -1)
    short_signal = (df["supertrend_direction"] == -1) & (df["supertrend_direction"].shift(1) == 1)

    conditions = [
        is_bullish_regime & long_signal,
        is_bearish_regime & short_signal
    ]
    choices = [1, -1]
    df["signal"] = np.select(conditions, choices, default=0)
    
    return df
