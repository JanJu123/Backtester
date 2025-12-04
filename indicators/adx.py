import pandas as pd
from ta.trend import ADXIndicator


def adx(df: pd.DataFrame, params):
    """
        "adx_params": {
          "period": 14,
          "threshold": 35,
    """
    # if not type(params) == dict:
    #     params = {"period": params}
    period = params.period
    High = df["High"]
    Low = df["Low"]
    Close =df["Close"]
    adx_series = ADXIndicator(high=High, low=Low, close=Close, window=period).adx()

    # Enforce NaN

    first_real_idx = adx_series.ne(0).idxmax()
    adx_series.iloc[:first_real_idx] = float("nan")

  
    return adx_series