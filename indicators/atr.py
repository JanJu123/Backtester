import pandas as pd
from ta.volatility import AverageTrueRange


def atr(df: pd.DataFrame, params):
    """
        "atr_params": {
            "atr_period": 10,
            "atr_multiplier": 3,   # Tukaj ni potreben, potreben je samo v strategy function
    """

    period = params.atr_period
    High = df["High"]
    Low = df["Low"]
    Close =df["Close"]

    atr_series = AverageTrueRange(high=High, low=Low, close=Close, window=period).average_true_range()
    
    # Enforce NaN
    atr_series.iloc[:period] = float('nan')


    return atr_series