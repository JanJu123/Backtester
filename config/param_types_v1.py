from dataclasses import dataclass
from typing import Optional, Dict, Union

# ------------------------------------------------------------------
# 1. ALIASED IMPORTS (The "Source of Truth")
# ------------------------------------------------------------------
# We use aliases (rs, ins, fs, ss) to know exactly where a type comes from.
from .param_schemas import risk_schemas as rs
from .param_schemas import indicator_schemas as ins
from .param_schemas import filter_schemas as fs
from .param_schemas import signal_schemas as ss


# ------------------------------------------------------------------
# 2. TYPE UNIONS (The Flexible Links)
# ------------------------------------------------------------------

# Risk Unions
SL_PARAM_TYPE = (rs.AtrBasedSLParams | rs.RRRBasedSLParams | rs.AtrTrailingBasedSLParams | rs.FvgBasedSLParams | 
        rs.PctTrailingBasedSLParams | rs.RRRBasedTrailingSLParams | rs.FixedRRRBasedTrailingSLParams)

TP_PARAM_TYPE = rs.AtrBasedTPParams | rs.RiskRewardTPParams
SIZING_PARAM_TYPE = rs.FixedPctSizingParams

# Core Signal Union (Universal Strategy Contract)
# 🚨 RULE: Only use 'ss' (Strategy Signal) types here. Never 'ins' (Indicator) types.
CORE_SIGNAL_PARAM_TYPE = (
    ss.MACDSignalParams |
    ss.MaCrossoverSignalParams |
    ss.MaPriceCrossoverSignalParams |
    ss.Rsi2pSignalParams |
    ss.BollingerBandsSignalParams |
    ss.SuperTrendSignalParams |
    ss.LsFvgSignalParams |
    ss.MomentumZscoreSignalParams
)


# ------------------------------------------------------------------
# 3. CONTAINER DATACLASSES (The Switchboards)
# ------------------------------------------------------------------

@dataclass
class CoreSignalParams:
    """
    The Switchboard for Strategy Signals.
    Only ONE field should be populated in the JSON, matching the 'core_signal_func'.
    
    IMPORTANT: The field names here MUST match the keys in your params_v2.json.
    """
    # Trend Strategies
    macd_signal_params: Optional[ss.MACDSignalParams] = None
    ma_crossover_signal_params: Optional[ss.MaCrossoverSignalParams] = None
    ma_price_crossover_signal_params: Optional[ss.MaPriceCrossoverSignalParams] = None
    supertrend_signal_params: Optional[ss.SuperTrendSignalParams] = None
    
    # Mean Reversion / Breakout Strategies
    rsi_2p_signal_params: Optional[ss.Rsi2pSignalParams] = None
    bb_signal_params: Optional[ss.BollingerBandsSignalParams] = None  # Used for both Breakout & MR
    momentum_zscore_signal_params: Optional[ss.MomentumZscoreSignalParams] = None
    
    # Price Action Strategies
    ls_fvg_signal_params: Optional[ss.LsFvgSignalParams] = None


@dataclass
class Filters:
    """
    A container for all possible filter parameters.
    These use the 'fs' (filter_schemas) definitions.
    """
    adx_filter: Optional[fs.ADXFilterParams] = None
    ma_trend_filter: Optional[fs.MATrendFilterParams] = None
    ma_momentum_filter: Optional[fs.MAMomentumFilterParams] = None
    bb_filter: Optional[fs.BollingerBandFilterParams] = None
    rsi_filter: Optional[fs.RsiFilterParams] = None
    hurst_exponent_filter: Optional[fs.HurstExponentParams] = None


# ------------------------------------------------------------------
# 4. FINAL CONTRACT: StrategyConfig (The Top-Level Structure)
# ------------------------------------------------------------------

@dataclass
class StrategyConfig:
    # Embedded Risk & Sizing Profiles
    stop_loss_strategy: Optional[str]
    stop_loss_params: Optional[SL_PARAM_TYPE] 
    take_profit_strategy: Optional[str]
    take_profit_params: Optional[TP_PARAM_TYPE] 
    sizing_strategy: str
    sizing_params: SIZING_PARAM_TYPE

    # Core Strategy Logic
    strategy_type: str
    core_signal_func: str
    core_signal_params: CoreSignalParams # <--- The Container defined above
    filters: Optional[Filters] = None
    exit_on_neutral_signal: bool = False


# ------------------------------------------------------------------
# 5. ROOT CONTAINERS (For the JSON root)
# ------------------------------------------------------------------

@dataclass
class TradingCostParams:
    fees: float
    slippage: float

@dataclass
class Params:
    strategy_configs: Dict[str, StrategyConfig]
    trading_cost_params: TradingCostParams