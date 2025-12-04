import pandas as pd
from typing import Callable
import numpy as np
from indicators import function_map
from config.param_schemas.signal_schemas import BollingerBandsSignalParams

def bollinger_bands_mean_reversion(df: pd.DataFrame, params: dict):
    """
    bb_params: {
        "period": 20,
        "std_dev_mult": 2
    }
    """

    params = params.bb_signal_params

    func = function_map.get_indicator_func("bb")
    df[["bband_middle", "bband_upper", "bband_lower"]] = func(df, params)

    
    ma_func = function_map.get_indicator_func(params.exit_ma_func)
    df["mr_target"] = ma_func(df, period=params.exit_ma_period, column="Close")


    df["signal"] = 0

    data = {
    "close": np.array(df["Close"]),
    "bband_middle": np.array(df["bband_middle"]),
    "bband_upper": np.array(df["bband_upper"]),
    "bband_lower": np.array(df["bband_lower"]),
    "mr_target": np.array(df["mr_target"]),
    "signal": np.zeros(len(df), dtype=int),
    }

    state = 0
    for i in range(1, len(data['close'])):
    
        
        # LONG ENTRY
        if state==0 and data["close"][i] < data["bband_lower"][i] and data["close"][i-1] > data["bband_lower"][i-1]:
            data["signal"][i] = 1
            state = 1
                
        # SHORT ENTRY
        elif state==0 and data["close"][i] > data["bband_upper"][i] and data["close"][i-1] < data["bband_upper"][i-1]:
            continue
            data["signal"][i] = -1
            state = -1
        
        # LONG Exit
        elif state == 1:
            if data["close"][i] >= data["mr_target"][i]:
                data["signal"][i] = 0
                state = 0
            else:
                data["signal"][i] = 1
            continue
        
        #SHORT Exit
        elif state == -1:
            if data["close"][i] <= data["mr_target"][i]:
                data["signal"][i] = 0
                state = 0
            else:
                data["signal"][i] = -1
            continue


    df["signal"] = data["signal"]
    
    return df




def bollinger_bands_breakout(df: pd.DataFrame, params: BollingerBandsSignalParams):
    """
    bb_params: {
        "period": 20,
        "std_dev_mult": 2
    }
    """
    params = params.bb_signal_params

    func = function_map.get_indicator_func("bb")
    df[["bband_middle", "bband_upper", "bband_lower"]] = func(df, params)


    ma_func = function_map.get_indicator_func(params.exit_ma_func)
    df["mr_target"] = ma_func(df, period=params.exit_ma_period, column="Close")


    df["signal"] = 0


    data = {
    "close": np.array(df["Close"]),
    "bband_middle": np.array(df["bband_middle"]),
    "bband_upper": np.array(df["bband_upper"]),
    "bband_lower": np.array(df["bband_lower"]),
    "mr_target": np.array(df["mr_target"]),
    "signal": np.zeros(len(df), dtype=int),
    }

    state = 0
    for i in range(1, len(data['close'])):

        # LONG ENTRY
        if state==0 and data["close"][i] < data["bband_lower"][i] and data["close"][i-1] > data["bband_lower"][i-1]:
            data["signal"][i] = -1
            state = -1
                
        # SHORT ENTRY
        elif state==0 and data["close"][i] > data["bband_upper"][i] and data["close"][i-1] < data["bband_upper"][i-1]:
            data["signal"][i] = 1
            state = 1
        
        # LONG Exit
        elif state == -1:
            if data["close"][i] >= data["mr_target"][i]:
                data["signal"][i] = 0
                state = 0
            else:
                data["signal"][i] = -1
            continue
        
        #SHORT Exit
        elif state == 1:
            if data["close"][i] <= data["mr_target"][i]:
                data["signal"][i] = 0
                state = 0
            else:
                data["signal"][i] = 1
            continue

            

    df["signal"] = data["signal"]



    
    return df