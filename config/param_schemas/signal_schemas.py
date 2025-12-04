from dataclasses import dataclass
from typing import Optional
# Import the ingredients from your indicator library
from .indicator_schemas import *

# ------------------------------------------------------------------
# STRATEGY SIGNAL CONTRACTS (The Recipes)
# ------------------------------------------------------------------

@dataclass
class MACDSignalParams:
    period_fast: int
    period_slow: int
    period_signal_line: int
    ma_func_fast: str
    ma_func_slow: str
    ma_func_signal_line: str

@dataclass
class MaCrossoverSignalParams:
    period_slow: int
    period_fast: int
    ma_func_slow: str
    ma_func_fast: str

@dataclass
class MaPriceCrossoverSignalParams:
    period: int
    ma_func: str

@dataclass
class Rsi2pSignalParams:
    oversold: float
    overbought: float
    cumulative_window: int

    period: int
    column: str = "Close"
    smoothing_period: Optional[int] = None





@dataclass
class BollingerBandsSignalParams:
    # Exit Params
    exit_ma_func: str
    exit_ma_period: int

    period: int
    std_dev_mult: float
    column: str = "Close"

@dataclass
class SuperTrendSignalParams:
    atr_period: int
    atr_multiplier: float
    ma_period: int
    ma_func: str

@dataclass
class LsFvgSignalParams:
    range_lookback: int
    confirmation_window: int




# --- Composite Z-Score Params (Decoupled Logic) ---
@dataclass
class MomentumZscoreSignalParams:
    # Price Z-Score Params
    price_z_period: int
    price_z_ma_func: str
    price_z_thresh_buy: float
    price_z_thresh_sell: float
    
    # Momentum Z-Score Params
    mom_z_period: int
    mom_z_ma_func: str
    mom_z_thresh_buy: float
    mom_z_thresh_sell: float
    
    # Exit Params
    exit_ma_func: str
    exit_ma_period: int