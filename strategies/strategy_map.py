from . import RSI_2p_mr, bollinger_bands_mr, liquidity_sweep_fvg_struct, ma_crossover_trend, ma_price_crossover_trend, macd_signal_cross_trend
from . import supertrend_trend, momentum_zscore_mr



def get_strategy_func(name: str):
    """
    Vrne funkcijo  glede na ime.

    Parametri:
        name (str): ime MA metode (npr. "sma", "ema", "wma", "vwma", "vwema")

    Returns:
        Callable: ustrezna funkcija za izračun ali None, če ime ni definirano.
    """
    name = name.lower()
    mapping = {
        "ma_crossover": ma_crossover_trend.ma_crossover,
        "ma_price_crossover": ma_price_crossover_trend.ma_crossover,
        "macd": macd_signal_cross_trend.macd_signal,
        "rsi_2p": RSI_2p_mr.strategy_RSI_2p,
        "bollinger_bands_mean_reversion": bollinger_bands_mr.bollinger_bands_mean_reversion,
        "bollinger_bands_breakout": bollinger_bands_mr.bollinger_bands_breakout,
        "supertrend_trend_following": supertrend_trend.supertrend_trend_folowing,
        "liquiditysweep_fvg_entry": liquidity_sweep_fvg_struct.liquidity_sweep_fvg,
        "momentum_zscore_mean_reversion": momentum_zscore_mr.momentum_zscore,
    }
    return mapping.get(name.lower())