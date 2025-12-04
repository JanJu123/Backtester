from dataclasses import dataclass
from typing import Optional

# ------------------------------------------------------------------
# RAW INDICATOR INPUTS (The Ingredients)
# These schemas match the arguments required by your indicator functions.
# ------------------------------------------------------------------

@dataclass
class ADXInputs:
    """Inputs for adx.py"""
    period: int

@dataclass
class ATRInputs:
    """Inputs for atr.py"""
    atr_period: int

@dataclass
class BollingerBandsInputs:
    """Inputs for bollinger_bands.py"""
    period: int
    std_dev_mult: float
    column: str = "Close"

@dataclass
class HurstExponentInputs:
    """Inputs for hurst_exponent.py"""
    period: int # Your code calls this 'period' (though JSON calls it 'lookback_period')

@dataclass
class MAInputs:
    """Generic inputs for ma.py functions (sma, ema, etc.)"""
    period: int
    column: str = "Close"

@dataclass
class MACDInputs:
    """Inputs for macd_indic.py"""
    period_fast: int
    period_slow: int
    period_signal_line: int
    ma_func_fast: str
    ma_func_slow: str
    ma_func_signal_line: str

@dataclass
class RSIInputs:
    """Inputs for rsi.py"""
    period: int
    smoothing_period: Optional[int] = None
    column: str = "Close"

@dataclass
class ZScoreInputs:
    """Inputs for z_score.py"""
    window: int
    ma_func: str
    column: str = "Close"