from . import function_map
import numpy as np
import pandas as pd

def calc_macd(df: pd.DataFrame, params):
    """
        "macd_params": {
          "period_fast": 12,
          "period_slow": 26,
          "ma_func_slow": "vwema",
          "ma_func_fast": "vwema",
          "period_signal_line": 9,
          "ma_func_signal_line": "ema"
        }
    """

    
    period_slow = params.period_slow
    period_fast = params.period_fast
    ma_func_slow = function_map.get_indicator_func(name=params.ma_func_slow)
    ma_func_fast = function_map.get_indicator_func(name=params.ma_func_fast)
    period_signal_line = params.period_signal_line
    ma_func_signal_line = function_map.get_indicator_func(name=params.ma_func_signal_line)


    data = {
        "close": np.array(df["Close"]),
        "ma_slow": np.array(ma_func_slow(df=df, period=period_slow)),
        "ma_fast": np.array(ma_func_fast(df=df, period=period_fast)),
        "ma_signal_line": np.zeros(len(df), dtype=float),
        "macd_line": np.zeros(len(df), dtype=float),
        "histogram": np.zeros(len(df), dtype=float),
    }

    data["macd_line"] = data["ma_fast"] - data["ma_slow"]
    data["ma_signal_line"] = ma_func_signal_line(
                df=pd.DataFrame({"Close": data["macd_line"]}, index=df.index), 
                period=period_signal_line)
    
    data["histogram"] = data["macd_line"] - data["ma_signal_line"]


    df["macd_line"] = data["macd_line"]
    df["signal_line"] = data["ma_signal_line"]
    df["histogram"] = data["histogram"]


    return df

