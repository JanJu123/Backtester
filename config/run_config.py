import os

# ==============================================================================
# 1. FILE SYSTEM ARCHITECTURE
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, "data", "files", "crypto")
DB_FOLDER = os.path.join(BASE_DIR, "data", "databases")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "experiment_outputs")

# ==============================================================================
# 2. EXECUTION MODE
# ==============================================================================
# Options: "single", "optimize", "walkforward"
MODE = "single"

# ==============================================================================
# 3. STRATEGY SELECTION
# ==============================================================================
STRATEGY_TO_RUN = "Momentum_zscore_mean_reversion_V2" 

# ==============================================================================
# 4. DATA CONFIGURATION
# ==============================================================================
TIMEFRAME = "1h"
DATA_FILE_NAME = f"BTCUSDT_{TIMEFRAME}.csv"
DATA_PATH = os.path.join(DATA_FOLDER, DATA_FILE_NAME)


# ==============================================================================
# 5. BACKTEST & RISK SETTINGS
# ==============================================================================
INITIAL_CAPITAL = 10000
WARM_UP_PERIOD = 200 
USE_TRADING_COSTS = True

# ==============================================================================
# 6. OPTIMIZATION & SPLIT SETTINGS
# ==============================================================================
# Out-of-Sample Split (0.25 = 25% of data reserved for final validation)
OOS_SPLIT_PERCENTAGE = 0.25 

# Walk-Forward Optimization Defaults
WF_TRAIN_MONTHS = 12
WF_TEST_MONTHS = 3
WF_N_TRIALS = 10

# Simple Optimization Defaults
OPT_N_TRIALS = 200