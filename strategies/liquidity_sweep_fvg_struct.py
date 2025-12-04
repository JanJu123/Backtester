import pandas as pd
import numpy as np
from patterns import candle_patterns, market_structure
from core.context import SignalContext



def liquidity_sweep_fvg(df: pd.DataFrame, params: dict):
    """
    The final, robust, dual-direction version of the Liquidity Sweep & FVG Entry model.
    It includes the critical validation check for both long and short trades to prevent
    illogical entries and ensure stability.
    """
    # --- Parameters ---
    params = params.ls_fvg_signal_params
    confirmation_window = params.confirmation_window
    warm_up_period = params.range_lookback

    df = df.copy()

    # --- NumPy Conversion ---
    highs = df['High'].to_numpy()
    lows = df['Low'].to_numpy()
    closes = df['Close'].to_numpy()
    recent_highs = df['recent_high'].to_numpy()
    recent_lows = df['recent_low'].to_numpy()
    fvg_states = df['fvg_state'].to_numpy()
    fvg_tops = df['fvg_top'].to_numpy()
    fvg_bottoms = df['fvg_bottom'].to_numpy()

    # --- Output Arrays ---
    signals = np.zeros(len(df), dtype=int)
    contexts = np.full(len(df), None, dtype=object)
    
    # --- State Machine Variables ---
    state = 0
    sweep_index = -1
    fvg_to_watch = None
    bos_target_high = None
    bos_target_low = None
    trade_direction = 0  # 0: Neutral, 1: Long, -1: Short

    def reset_state():
        """A centralized reset function to clean all state variables."""
        nonlocal state, sweep_index, fvg_to_watch, bos_target_high, bos_target_low, trade_direction
        state = 0
        sweep_index = -1
        fvg_to_watch = None
        bos_target_high = None
        bos_target_low = None
        trade_direction = 0

    # --- Main Loop ---
    for i in range(warm_up_period, len(df)):

        is_bullish_sweep = lows[i] < recent_lows[i-1] and closes[i] > recent_lows[i-1]
        is_bearish_sweep = highs[i] > recent_highs[i-1] and closes[i] < recent_highs[i-1]

        if is_bullish_sweep:
            reset_state()
            state = 1
            trade_direction = 1 # Set for a LONG pattern
            sweep_index = i
            bos_target_high = recent_highs[i-1]
            continue
        
        elif is_bearish_sweep:
            reset_state()
            state = 1
            trade_direction = -1 # Set for a SHORT pattern
            sweep_index = i
            bos_target_low = recent_lows[i-1]
            continue
        
        if state == 0:
            continue

        # --- State 1: Awaiting Break of Structure (BOS) ---
        if state == 1:
            if i > sweep_index + confirmation_window: reset_state(); continue

            # --- LOGIC FOR LONG TRADE (BULLISH) ---
            if trade_direction == 1:
                if fvg_states[i] == 1 and fvg_to_watch is None: fvg_to_watch = (fvg_tops[i], fvg_bottoms[i])
                if fvg_to_watch and lows[i] < fvg_to_watch[1]: reset_state(); continue
                if highs[i] > bos_target_high:
                    if fvg_to_watch: state = 2
                    else: reset_state()
                        
            # --- LOGIC FOR SHORT TRADE (BEARISH) ---
            elif trade_direction == -1:
                if fvg_states[i] == -1 and fvg_to_watch is None: fvg_to_watch = (fvg_tops[i], fvg_bottoms[i])
                if fvg_to_watch and highs[i] > fvg_to_watch[0]: reset_state(); continue
                if lows[i] < bos_target_low:
                    if fvg_to_watch: state = 2
                    else: reset_state()

        # --- State 2: Awaiting Pullback to the FVG ---
        elif state == 2:
            if i > sweep_index + confirmation_window: reset_state(); continue
            
            # --- PULLBACK LOGIC FOR LONG TRADE ---
            if trade_direction == 1:
                if fvg_to_watch and lows[i] <= fvg_to_watch[0]:
                    # FINAL VALIDATION: Signal candle must close ABOVE the stop loss level.
                    if closes[i] > fvg_to_watch[1]:
                        signals[i] = 1
                        contexts[i] = SignalContext(fvg_level=fvg_to_watch[1]) # SL is FVG bottom
                        reset_state()
                    else:
                        reset_state() # Failed validation, reset

            # --- PULLBACK LOGIC FOR SHORT TRADE ---
            elif trade_direction == -1:
                if fvg_to_watch and highs[i] >= fvg_to_watch[1]:
                    # FINAL VALIDATION: Signal candle must close BELOW the stop loss level.
                    if closes[i] < fvg_to_watch[0]:
                        signals[i] = -1
                        contexts[i] = SignalContext(fvg_level=fvg_to_watch[0]) # SL is FVG top
                        reset_state()
                    else:
                        reset_state() # Failed validation, reset

    # --- Return the results ---
    df["signal"] = signals
    df["context"] = contexts

    return df