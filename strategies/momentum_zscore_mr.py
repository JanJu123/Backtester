import pandas as pd
from typing import Callable
import numpy as np
from indicators import function_map

from config.param_schemas.signal_schemas import MomentumZscoreSignalParams
from config.param_schemas.indicator_schemas import ZScoreInputs


def momentum_zscore(df: pd.DataFrame, params: MomentumZscoreSignalParams):
    """
        "momentum_zscore_params":{
            "price_z_period": 20,
            "price_z_ma_func": "sma",
            "price_z_thresh_buy": -2.0,  
            "price_z_thresh_sell": 2.0,  

            "mom_z_period": 40,
            "mom_z_ma_func": "sma",
            "mom_z_thresh_buy": 1.0,     
            "mom_z_thresh_sell": -1.0,
        }
    """
    params = params.momentum_zscore_signal_params


    z_score_func = function_map.get_indicator_func("z_score")
    df["pct_change"] = df["Close"].pct_change()


    price_inputs = ZScoreInputs(
        window=params.price_z_period,
        ma_func=params.price_z_ma_func,
        column="Close"
    )
    
    mom_inputs = ZScoreInputs(
        window=params.mom_z_period,
        ma_func=params.mom_z_ma_func,
        column="pct_change"
    )


    # exit_params = {
    #     "window": params.exit_ma_period,
    #     "column": "Close",
    #     "ma_func": params.exit_ma_func
    # }
    
    # Izračunamo Z Score, in momentum Z Score
    df["z_score"] = z_score_func(df, price_inputs)
    df["momentum_z_score"] = z_score_func(df, mom_inputs)

    # Dobimo ma funkcijo in izračunamo MA za exit
    ma_func = function_map.get_indicator_func(params.exit_ma_func)
    df["mr_target"] = ma_func(df, period=params.exit_ma_period, column="Close")

    data = {
        "Close":  np.array(df["Close"]),
        "z_score":  np.array(df["z_score"]),
        "momentum_z_score":  np.array(df["momentum_z_score"]),
        "mr_target": np.array(df["mr_target"]),
        "signal":  np.zeros(len(df), dtype=int)
    }

    p_sell_thresh = params.price_z_thresh_sell
    p_buy_thresh  = params.price_z_thresh_buy
    
    m_buy_thresh  = params.mom_z_thresh_buy
    m_sell_thresh = params.mom_z_thresh_sell


    current_state = 0

    for n in range(len(df)):
        if np.isnan(data["z_score"][n]) or np.isnan(data["momentum_z_score"][n]) or np.isnan(data["mr_target"][n]):
            continue
            

        curr_z = data["z_score"][n]
        curr_mom_z = data["momentum_z_score"][n]
        curr_close = data["Close"][n]
        curr_mr_target = data["mr_target"][n]



        # A. IF FLAT (0) -> Look for Entry
        if current_state == 0:
            # LONG: Cheap Price + Momentum Up
            if curr_z < p_buy_thresh and curr_mom_z > m_buy_thresh:
                current_state = 1
            
            # SHORT: Expensive Price + Momentum Down
            elif curr_z > p_sell_thresh and curr_mom_z < m_sell_thresh:
                current_state = -1


        # B. IF LONG (1) -> Look for Exit
        elif current_state == 1:

            if curr_close > curr_mr_target:
                current_state = 0 
        
        # C. IF SHORT (-1) -> Look for Exit
        elif current_state == -1:
            # EXIT: Price reverts to Mean (Crosses below MA)
            if curr_close < curr_mr_target:
                current_state = 0

        data["signal"][n] = current_state


    df["signal"] = data["signal"]

    return df