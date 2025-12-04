from dataclasses import dataclass
from typing import Optional

# ------------------------------------------------------------------
# FILTER SCHEMAS
# ------------------------------------------------------------------

@dataclass
class ADXFilterParams:
    is_active: bool
    period: int
    threshold: float
    on_long: int
    on_short: int

@dataclass
class MATrendFilterParams:
    is_active: bool
    period: int
    ma_func: str
    on_long: int
    on_short: int

@dataclass
class MAMomentumFilterParams:
    is_active: bool
    period_fast: int
    period_slow: int
    on_long: int
    on_short: int
    ma_func_fast: str = "ema"
    ma_func_slow: str = "ema"

@dataclass
class BollingerBandFilterParams:
    is_active: bool
    period: int
    std_dev_mult: float
    on_long: int
    on_short: int

@dataclass
class RsiFilterParams:
    is_active: bool
    period: int
    oversold: float
    overbought: float
    on_long: int
    on_short: int
    smoothing_period: Optional[int] = None

@dataclass
class HurstExponentParams:
    is_active: bool
    lookback_period: int
    max_hurst_threshold: float
    on_long: int
    on_short: int