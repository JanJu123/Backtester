import numpy as np
import pandas as pd
from typing import Union
from config.param_schemas.filter_schemas import HurstExponentParams

# Set the minimum number of data points required for a meaningful calculation
MIN_SAMPLES_FOR_REGRESSION = 5

def _compute_hurst_rs_optimized(ts: np.ndarray) -> float:
    """
    Optimized core R/S Analysis calculation for a single time series window.
    It computes the Hurst Exponent (H) for the given array (ts).
    
    This version simplifies segment definition for better performance in a 
    rolling().apply() context, maximizing NumPy array operations.
    """
    n_samples = len(ts)
    
    # 1. Calculate increments (returns)
    increments = ts[1:] - ts[:-1]
    if len(increments) < MIN_SAMPLES_FOR_REGRESSION:
        return np.nan
    
    # Define a set of segment sizes (n) to cover the window
    # We use a limited, fixed set of segment sizes for speed.
    segment_sizes = np.unique(
        (2 ** np.arange(1, np.log2(n_samples // 2))).astype(int)
    )
    
    if len(segment_sizes) < 2:
        return np.nan

    log_n = []
    log_rs = []

    for n in segment_sizes:
        if n < MIN_SAMPLES_FOR_REGRESSION:
            continue

        # Split the increments into segments of size n
        num_segments = len(increments) // n
        if num_segments == 0:
            continue

        segments = increments[:n * num_segments].reshape(num_segments, n)

        # Vectorized R/S calculation across all segments:
        
        # A. Centering (Mean of each segment subtracted)
        segment_means = np.mean(segments, axis=1, keepdims=True)
        centered_segments = segments - segment_means
        
        # B. Cumulative Sum (across each segment row)
        cumulative_sum = np.cumsum(centered_segments, axis=1)
        
        # C. Range (Max - Min for each cumulative sum row)
        max_sum = np.max(cumulative_sum, axis=1)
        min_sum = np.min(cumulative_sum, axis=1)
        range_val = max_sum - min_sum
        
        # D. Standard Deviation (for the original increments in each segment)
        std_dev = np.std(segments, axis=1, ddof=1)
        
        # E. Ratio (R/S) - filter out divisions by zero
        # Use np.divide and a mask for safety and speed
        with np.errstate(divide='ignore', invalid='ignore'):
            rs_ratios = np.divide(range_val, std_dev, out=np.full_like(range_val, np.nan), where=std_dev!=0)
        
        valid_rs = rs_ratios[~np.isnan(rs_ratios)]

        if valid_rs.size > 0:
            avg_rs = np.mean(valid_rs)
            if avg_rs > 0:
                log_n.append(np.log(n))
                log_rs.append(np.log(avg_rs))

    if len(log_n) < 2:
        return np.nan

    # 3. Linear regression (H is the slope)
    try:
        hurst_exponent = np.polyfit(log_n, log_rs, 1)[0]
        return hurst_exponent
    except Exception:
        return np.nan 


def hurst_exponent(df: pd.DataFrame, params: HurstExponentParams) -> pd.Series:
    """
    Calculates the rolling Hurst Exponent (H) using the optimized Rescaled Range (R/S) analysis.
    
    H < 0.5: Mean-Reverting (Anti-persistent)
    H > 0.5: Trending (Persistent)

    Args:
        df: DataFrame with 'Close' prices.
        period: The lookback window for the R/S analysis (e.g., 50).
    
    Returns:
        pd.Series: Series containing the rolling Hurst Exponent values.
    """
    df = df.copy()
    if 'Close' not in df.columns:
        return pd.Series(np.nan, index=df.index)

    period = params.lookback_period if hasattr(params, 'lookback_period') else params

    # Use the natural log of the prices (standard practice for R/S)
    log_prices = df['Close'].apply(np.log) # Use apply for guaranteed speed if df is large
    
    # Calculate the rolling Hurst Exponent using the optimized R/S helper function
    # raw=True ensures x is passed as a numpy.ndarray
    hurst_values = log_prices.rolling(window=period).apply(
        lambda x: _compute_hurst_rs_optimized(x), # <--- FIX: Removed .values
        raw=True, 
    )
    
    return hurst_values
