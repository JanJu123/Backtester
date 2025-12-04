import json
import pprint
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Type, TypeVar

# --- Only Import Trading Cost if needed ---
from config.params_schema.param_types_v2 import TradingCostParams

@dataclass
class MACDParamsML:
    is_active: bool = True 
    period_slow: int = 26
    period_fast: int = 12
    ma_func_slow: str = 'sma'
    ma_func_fast: str = 'sma'
    period_signal_line: int = 9
    ma_func_signal_line: str = 'sma'

@dataclass
class RsiParamsML:
    is_active: bool = True
    period: int = 14
    smoothing_period: Optional[int] = None

@dataclass
class BollingerBandsParamsML:
    is_active: bool = True
    period: int = 20
    std_dev_mult: float = 2.0

@dataclass
class AdxParamsML:
    is_active: bool = True
    period: int = 14
    threshold: Optional[int] = None # Threshold might be useful as a feature itself

@dataclass
class AtrParamsML:
    is_active: bool = True
    atr_period: int = 10

@dataclass
class MaParamsML:
    # Non-default first
    period: int
    is_active: bool = True
    ma_func: str = 'sma'

@dataclass
class MsParamsML: # For Market Structure
    is_active: bool = True
    lookback: int = 20

@dataclass
class FvgParamsML:
    is_active: bool = True
    # Add specific params if your find_fvg function takes any

@dataclass
class HurstExponentParamsML:
     # Non-default first
     lookback_period: int
     is_active: bool = True

@dataclass
class NormVolumeParamsML:
    is_active: bool = True
    window: int = 20  # rolling period for normalization (e.g., 20 bars)

@dataclass
class PctCloseParamsML:
    is_active: bool = True
    method: str = "log"  # "pct" or "log"





# --- ML Feature Container ---
# Now uses the locally defined ML param types with is_active
@dataclass
class FeatureParamsML:
    """A container for all possible feature parameters for ML."""
    # Features used by the MODEL
    adx_params: Optional[AdxParamsML] = None
    atr_params: Optional[AtrParamsML] = None
    rsi_params: Optional[RsiParamsML] = None
    bb_params: Optional[BollingerBandsParamsML] = None
    ma_params: Optional[MaParamsML] = None
    macd_params: Optional[MACDParamsML] = None
    ms_params: Optional[MsParamsML] = None
    fvg_params: Optional[FvgParamsML] = None
    hurst_exponent_params: Optional[HurstExponentParamsML] = None
    norm_volume_params: Optional[NormVolumeParamsML] = None
    pct_close_params: Optional[PctCloseParamsML] = None
    # Add other local param types here if needed

# --- Labeling Requirements Container ---
@dataclass
class LabelingRequiredFeatures:
    """Specifies indicators needed ONLY by the labeling function."""
    # Use the SAME ML dataclass types for consistency
    adx_params: Optional[AdxParamsML] = None
    ma_params: Optional[MaParamsML] = None
    # Add others if needed by future labeling functions (RSI, BBands etc)

@dataclass
class LabelingFuncParams:
    """Parameters specific TO the labeling function's logic (e.g., thresholds)."""
    price_column: str = 'Close'
    lookahead_period: Optional[int] = None # Needed by simple return labeler
    threshold: Optional[float] = None      # Needed by simple return labeler
    adx_threshold: Optional[int] = None    # Needed by ADX/MA labeler
    # Add other potential logic params here

@dataclass
class LabelingParamsML:
    """Defines the labeling strategy and its requirements."""
    function_name: str
    required_features: LabelingRequiredFeatures = field(default_factory=LabelingRequiredFeatures)
    func_params: LabelingFuncParams = field(default_factory=LabelingFuncParams)

# --- Other ML Specific Config Blocks ---
@dataclass
class ModelInfoML:
    model_name: str
    version: str
    description: str

@dataclass
class SplitParamsML:
    test_size: float = 0.2
    shuffle: bool = False # Usually False for time series


@dataclass
class RandomForestHyperparams:
    """Hyperparameters specifically for RandomForestClassifier."""
    n_estimators: int = 100
    max_depth: Optional[int] = 10
    min_samples_leaf: int = 5
    n_jobs: int = -1
    random_state: int = 42
    class_weight: Optional[str] = None
    criterion: Optional[str] = 'gini'

@dataclass
class XGBoostHyperparams:
    """Hyperparameters specifically for XGBClassifier."""
    n_estimators: int = 100
    max_depth: Optional[int] = 10
    min_samples_leaf: int = 5 # This will be mapped to 'min_child_weight' in the trainer
    n_jobs: int = -1
    random_state: int = 42
    class_weight: Optional[str] = None # This will be mapped to 'sample_weight'
    # XGBoost-specific params you can add later:
    # learning_rate: float = 0.1
    # subsample: float = 1.0
    # colsample_bytree: float = 1.0

@dataclass
class TrainingHyperparametersContainer:
    """A container to hold different hyperparameter sets.
    The key (e.g., 'random_forest_classifier') MUST match the model_type string.
    """
    random_forest_classifier: Optional[RandomForestHyperparams] = None
    xgboost_classifier: Optional[XGBoostHyperparams] = None
    # Add new models here, e.g.:
    # linear_regression_classifier: Optional[LinearRegressionHyperparams] = None

@dataclass
class TrainingParamsML:
    model_type: str # e.g., "random_forest_classifier"
    split_params: SplitParamsML = field(default_factory=SplitParamsML)
    hyperparameters: TrainingHyperparametersContainer = field(default_factory=TrainingHyperparametersContainer)

# --- The Top-Level Config Dataclass for ONE ML Experiment ---
@dataclass
class MLConfig:
    model_info: ModelInfoML
    feature_params: FeatureParamsML
    labeling_params: LabelingParamsML
    training_params: TrainingParamsML
    trading_cost_params: Optional[TradingCostParams] = None # Still optional



# --- The File-Level Container (Similar to your Params dataclass) ---
@dataclass
class MLParamsFile:
    experiments: Dict[str, MLConfig] = field(default_factory=dict)

