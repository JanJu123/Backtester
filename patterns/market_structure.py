import numpy as np
import pandas as pd


"""Contains functions that identify key market structure points and events, such as swing highs, swing lows, and breaks of structure (BOS)."""

def find_all_swing_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies all swing high and swing low points without lookahead bias.
    A swing point at index 'i-1' is only confirmed at the close of index 'i'.
    Adds two boolean columns: 'is_swing_high' and 'is_swing_low'.
    """
    df_copy = df.copy()

    # A swing high is confirmed at the current bar 'i' if the high of the PREVIOUS bar 'i-1'
    # was higher than its two neighbors ('i-2' and 'i').
    df_copy['is_swing_high'] = (
        (df_copy['High'].shift(1) > df_copy['High'].shift(2)) & 
        (df_copy['High'].shift(1) > df_copy['High'])
    )

    # A swing low is confirmed at the current bar 'i' if the low of the PREVIOUS bar 'i-1'
    # was lower than its two neighbors ('i-2' and 'i').
    df_copy['is_swing_low'] = (
        (df_copy['Low'].shift(1) < df_copy['Low'].shift(2)) & 
        (df_copy['Low'].shift(1) < df_copy['Low'])
    )

    return df_copy

def calculate_rolling_high_low(data: pd.DataFrame, lookback_period: int) -> pd.DataFrame:
    """
    Calculates the rolling minimum of the 'Low' column and the rolling maximum 
    of the 'High' column over a given lookback period and adds them to the DataFrame.
    """
    
    df = data.copy()
    df["recent_low"] = df["Low"].rolling(window=lookback_period).min()
    df["recent_high"] = df["High"].rolling(window=lookback_period).max()
    return df