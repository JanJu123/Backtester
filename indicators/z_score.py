import pandas as pd
import numpy as np
from . import function_map

def zscore(df: pd.DataFrame, params):
    """
    Calculates the z-score of a pandas Series.
    If window is None → uses full-series mean & std.
    If window is provided → uses rolling z-score.

    params{
        window: 20,
        ma_func: "ma"
    }
    """
    col_name = getattr(params, "column", "Close")


    ts = df[col_name]
    window = params.window
    ma_func = function_map.get_indicator_func(params.ma_func)

    mean = ma_func(df, window, col_name)
    std = ts.rolling(window).std(ddof=0)
    return (ts - mean) / std