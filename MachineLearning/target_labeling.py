import pandas as pd
from data import save_load
from config.load_params_v1 import load_params_from_json
import numpy as np

def get_label_func(name: str):
    mapping = {
        "label_simple_return_regime": label_simple_return_regime,
        "label_adx_ma_regime": label_adx_ma_regime,
    }
    return mapping.get(name.lower())


def label_simple_return_regime(df: pd.DataFrame, params) -> pd.Series:
    """
    Calculates a simple 3-state market regime (Bullish, Bearish, Choppy)
    based on future price movement. This provides a basic structure for
    a regime detection model.

    - Regime 1 (Bullish): Price increases by >= threshold within the lookahead_period.
    - Regime -1 (Bearish): Price decreases by >= threshold within the lookahead_period.
    - Regime 0 (Choppy): Price moves less than the threshold (in either direction).

    Args:
        df (pd.DataFrame): DataFrame containing the price data.
        price_column (str): The name of the column to use for price (e.g., 'Close').
        lookahead_period (int): The number of future candles to check.
        threshold (float): The minimum percentage change to define a regime (e.g., 0.01 for 1%).

    Returns:
        pd.Series: A Series of 0s, 1s, and 2s, representing the target regime.
    """
    price_column = params.price_column
    lookahead_period = params.lookahead_period
    threshold = params.threshold


    # 1. Calculate future price and return
    future_price = df[price_column].shift(-lookahead_period)
    future_return = (future_price / df[price_column]) - 1

    # 2. Define conditions and choices for np.select
    conditions = [
        future_return >= threshold,  # Bullish
        future_return <= -threshold  # Bearish
    ]
    choices = [ 1, 2 ] # Bullish=1, Bearish=-1

    # 3. Calculate target array
    target_values = np.select(conditions, choices, default=0) # Choppy=0

    # 4. Create DataFrame with the named target column
    target_name = 'target_regime' # Define the name
    # Fill NaN values that occur due to shift and np.select, then cast
    target_df = pd.DataFrame({target_name: target_values}, index=df.index)
    target_df[target_name] = target_df[target_name].fillna(0).astype(int)

    # If you add more targets later, calculate them and add as columns here:
    # target_df['another_target'] = calculate_another_target(...)

    # 5. Get the list of target column names
    target_names = list(target_df.columns)

    # 6. Return the DataFrame AND the list of its column names
    return target_df, target_names


def label_adx_ma_regime(df: pd.DataFrame, params, label_map):
    """
    Calculates a simple 3-state market regime (Bullish, Bearish, Choppy)
    based on ma and adx. This provides a basic structure for
    a regime detection model.
    """

    adx_treshold = params.adx_threshold

    adx_series = df[label_map["adx"]]
    ma_series = df[label_map["ma"]]
    close_series = df["Close"]

    conditions = [
        (adx_series >=adx_treshold) & (close_series >= ma_series),
        (adx_series >=adx_treshold) & (close_series <= ma_series)
    ]
    choices = [1, 2]

    target_values = np.select(conditions, choices, default=0)
    target_name = "target_regime"

    target_df = pd.DataFrame({target_name: target_values}, index=df.index)
    target_df[target_name] = target_df[target_name].fillna(0).astype(int)

    target_names = list(target_df.columns)

    return target_df, target_names
