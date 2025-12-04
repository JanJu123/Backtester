import pandas as pd
from ta.momentum import RSIIndicator


def rsi(df: pd.DataFrame, params):
    """
        "rsi_params": {
          "period": 14,
    """
    df = df.copy()
    period = params.period
    
    col_name = getattr(params, "column", "Close")
    ts=df[col_name]
    df["rsi"] = RSIIndicator(close=ts, window=period).rsi()


    if "smoothing_period" in vars(params):
        smoothing_period = params.smoothing_period
        if smoothing_period is not None:
            df["rsi"] = RSIIndicator(close=ts, window=period).rsi()
            df["rsi"] = df["rsi"].rolling(window=smoothing_period).mean()
        

    return df["rsi"]
