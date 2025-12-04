import numpy as np
import pandas as pd


"""Contains functions that identify specific candlestick patterns based on their shape and a small number of surrounding bars (e.g., FVG, Engulfing, Doji)."""

def find_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies Fair Value Gaps (FVG) and returns their state and boundaries.
    This version is hardened to prevent the creation of "zero-height" FVGs.
    """
    df_copy = df.copy()

    # --- Conditions for FVGs (Bias-Free) ---
    is_bullish_fvg = df_copy['Low'] > df_copy['High'].shift(2)
    is_bearish_fvg = df_copy['High'] < df_copy['Low'].shift(2)

    # --- Initialize columns with default values ---
    df_copy['fvg_state'] = 0
    df_copy['fvg_top'] = np.nan
    df_copy['fvg_bottom'] = np.nan

    # --- Assign State and Boundaries for BULLISH FVGs ---
    df_copy.loc[is_bullish_fvg, 'fvg_state'] = 1
    df_copy.loc[is_bullish_fvg, 'fvg_bottom'] = df_copy['High'].shift(2)
    df_copy.loc[is_bullish_fvg, 'fvg_top'] = df_copy['Low']

    # --- Assign State and Boundaries for BEARISH FVGs ---
    df_copy.loc[is_bearish_fvg, 'fvg_state'] = -1
    df_copy.loc[is_bearish_fvg, 'fvg_top'] = df_copy['Low'].shift(2)
    df_copy.loc[is_bearish_fvg, 'fvg_bottom'] = df_copy['High']

    # print("--- Saving fvg_01_before_fix.csv ---")
    # df_copy.to_csv("fvg_01_before_fix.csv", index=False)


    # --- THE FIX: Invalidate any zero-height FVGs ---
    zero_height_condition = df_copy['fvg_top'] == df_copy['fvg_bottom']
    df_copy.loc[zero_height_condition, ['fvg_state', 'fvg_top', 'fvg_bottom']] = [0, np.nan, np.nan]


    # --- DEBUG STEP 2: Save the data AFTER the fix is applied ---
    # This file will prove whether the fix is working as intended.
    # print("--- Saving fvg_02_after_fix.csv ---")
    # df_copy.to_csv("fvg_02_after_fix.csv", index=False)

    return df_copy