from dataclasses import dataclass
from typing import Optional

# ------------------------------------------------------------------
# STOP LOSS SCHEMAS (SL)
# ------------------------------------------------------------------
@dataclass
class FixedPctBasedSLParams:
    stop_loss_pct: float
    is_dynamic: bool = True

@dataclass
class AtrBasedSLParams:
    atr_period: int
    atr_multiplier: float
    is_dynamic: bool = False

@dataclass
class AtrTrailingBasedSLParams:
    atr_period: int
    atr_multiplier: float
    is_dynamic: bool = True

@dataclass
class FvgBasedSLParams:
    is_dynamic: bool = False

@dataclass
class PctTrailingBasedSLParams:
    trail_pct: float
    is_dynamic: bool = True

@dataclass
class RRRBasedSLParams:
    rrr: float
    is_dynamic: bool = False

@dataclass
class RRRBasedTrailingSLParams:
    rrr: float
    target_col: str = "Close"
    is_dynamic:bool = True

@dataclass
class FixedRRRBasedTrailingSLParams:
    rrr: float
    target_col: str = "Close"
    is_dynamic: bool = True

    




# ------------------------------------------------------------------
# TAKE PROFIT SCHEMAS (TP)
# ------------------------------------------------------------------

@dataclass
class AtrBasedTPParams:
    atr_period: int
    atr_multiplier: float
    is_dynamic: bool = False

@dataclass
class RiskRewardTPParams:
    risk_reward_ratio: float
    is_dynamic: bool = False




# ------------------------------------------------------------------
# POSITION SIZING SCHEMAS
# ------------------------------------------------------------------


@dataclass
class FixedPctSizingParams:
    percentage_to_risk: float