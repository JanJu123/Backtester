import pandas as pd
import numpy as np
import ta
import sqlite3
import os
import ta
from typing import Callable, Optional, Dict, Any
from datetime import datetime
import json
import functools
import time
from dataclasses import is_dataclass, asdict
import json


def calculate_ATR(df, period: int=5):
    """
    Izračuna Average True Range (ATR) za podan DataFrame.

    :param df: ["High", "Low", "Close"]

    :param period: Koliko candlov naj bo uproabljenih  za MA ATR

    Returns:
        pd.Series: ATR values.

    """
    df_ATR = pd.DataFrame()
    # df_ATR['High-Low'] = df['High'] - df['Low']
    # df_ATR['High-Close'] = abs(df['High'] - df['Close'].shift(1))
    # df_ATR['Low-Close'] = abs(df['Low'] - df['Close'].shift(1))

    # df_ATR['TR'] = df_ATR[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    # df_ATR['ATR'] = df_ATR['TR'].rolling(window=period, min_periods=1).mean()  # SMA ATR

    df_ATR['ATR'] = ta.atr(df["High"], df["Low"], df["Close"], length=period)

    # df = df_ATR["ATR"].copy()

    return df_ATR["ATR"].copy()



def calculate_RSI(df, period: int=14, price_col: str = "Close"):
    """
    Izračuna  Relative Strength Index (RSI) za podan DataFrame.

    :param df (pd.DataFrame): DataFrame containing at least a 'Close' price column.
    :param period (int): Lookback period for RSI. Default is 14.
    :param price_col (str): Name of the column to use for price. Default is "Close".

    Returns:
        pd.Series: RSI values.
    """
    delta = df[price_col].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_cumulative_RSI(df, rsi_period: int = 14, cumulative_window: int = 2, price_col: str = "Close"):
    """
    Izračuna cumulative Relative Strength Index (RSI) kot vsoto preteklih N RSI vrednosti.

    :param df (pd.DataFrame): DataFrame z vsaj ceno ('Close' oz. določeno v price_col).
    :param rsi_period (int): Koliko candlov se upošteva pri izračunu RSI. Privzeto 14.
    :param cumulative_window (int): Koliko zadnjih RSI vrednosti se sešteje. Privzeto 2.
    :param price_col (str): Katera cena naj se uporablja za izračun RSI. Privzeto "Close".

    Returns:
        pd.Series: Cumulative RSI (vsota preteklih RSI vrednosti).
    """
    
    rsi = calculate_RSI(df, period=rsi_period, price_col=price_col)
    cumulative_rsi = rsi.rolling(window=cumulative_window).sum()

    return cumulative_rsi








def slice_data_dict(data_dict: Dict[str, np.ndarray], index: int) -> Dict[str, np.ndarray]:
    """Slices all NumPy arrays in a dictionary up to a specific index."""
    sliced_dict = {}
    for key, arr in data_dict.items():
        # Slices each NumPy array from the start up to the 'index'
        sliced_dict[key] = arr[:index]
    return sliced_dict



def calc_trading_cost(trade_size:float, trading_cost_params=None):
    """
        Izračuna trading cost glede na velikost trejda.
        Trading cost: fees + slippage + spread
        
        Parameters:
            trade_size (float): Veilkost trejda (in USD, EUR, etc.).

            
        Returns:
            total_cost (float): Total cost in currency units.
    """

    if trading_cost_params is None:
        trading_cost_params = {}


    fees = trading_cost_params.fees
    slippage = trading_cost_params.slippage

    fee_cost = trade_size * fees
    slippage_cost = trade_size * slippage
    total_cost = fee_cost + slippage_cost
    return total_cost

def exit_trade(data_dict, entry_price, exit_price, entry_time, exit_time, stop_loss_price, take_profit_price, direction, index, signal, reason,
               position_size=1, use_trading_cost=True, trading_cost_class: Callable = None):
    """
    #TODO: Posodobi desctiption, da se ne uporablja vec Df
    Exita trade in odstrani vse enake signale, ki se ponavljajo in vrne nazaj df, ki vsebuje vse pomembne podatke:
    {
        "Datetime": exit_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "position_size": position_size,
        "cumulative_rsi": df["cumulative_rsi"].iloc[index],
        "signal": direction,
        "pnl": profit,
        "reason": reason,
    }

    :param df (pd.DataFrame): DataFrame z vsaj ceno ('Close' oz. določeno v price_col).
    :param entry_price (float): Cena, ko smo entrali v trade
    :param exit_price (float): Cena, ko smo exitali trade
    :param entry_time: Date/Datetime, ko smo entrali trade
    :param exit_time: Date/Datetime, ko smo exitali trade
    :param stop_loss_price (float): Cena/Pozicija stop_lossa
    :param take_profit_price (float): Cena/Pozicija take_profita
    :param position_size (float): Velikost pozicije (Če trejdamo na primer CDF, mora biti pozicija celo število oz. int)
    :param direction (int): Ali smo šli Long (1) ali Short (-1)
    :param index (int): Trenutni index, za katerega želimo exitati
    :param signal (int): 1 (buy) , -1(sell)
    :param reason (str): Razlog, zakaj smo exitali (stop_loss_hit ...) za lažjo analizo in debuganje

    Returns:
        {"df": df, "trade_info": df_trade, "profit": profit}

    df: Df ki vsebuje spremembe,
    trade_info: Vsebuje vse pomembne podatke za kasnejšo analizo
    profit: Profit tega trejda
    """

    # Izračunamo position value glede na entry price
    position_value = position_size * entry_price

    profit_per_share = (exit_price - entry_price) * direction
    profit = profit_per_share * position_size # Pomnožimo, da dobimo dejanski profit

    if use_trading_cost:
        # Izračunamo trading_cost
        trading_cost = trading_cost_class.calculate_fees_and_slippage(position_value)

        # Upoštevamo trading_cost
        profit -= trading_cost




    # Remove upcoming duplicate signals
    # m = index
    # current_signal = signal
    # while m < len(df):
    #     future_signal = df.at[m, "signal"]
    #     if future_signal == current_signal:
    #         df.at[m, "signal"] = 0
    #     elif future_signal == 0:
    #         break
    #     else:
    #         break
    #     m += 1

    df_trade = {
        "Datetime": exit_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "position_size": position_size,
        "position_value": position_value,
        "signal": direction,
        "pnl": profit,
        "reason": reason,
    }
    return {"df": data_dict, "trade_info": df_trade, "profit": profit}


def exit_trade_v2( entry_price, exit_price, entry_time, exit_time, stop_loss_price, take_profit_price, direction, index, signal, reason, data_dict: dict,
               position_size=1, use_trading_cost=True, trading_cost_class: Callable = None):
    """
    #TODO: Posodobi desctiption, da se ne uporablja vec Df
    Exita trade in odstrani vse enake signale, ki se ponavljajo in vrne nazaj df, ki vsebuje vse pomembne podatke:
    {
        "Datetime": exit_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "position_size": position_size,
        "cumulative_rsi": df["cumulative_rsi"].iloc[index],
        "signal": direction,
        "pnl": profit,
        "reason": reason,
    }

    :param df (pd.DataFrame): DataFrame z vsaj ceno ('Close' oz. določeno v price_col).
    :param entry_price (float): Cena, ko smo entrali v trade
    :param exit_price (float): Cena, ko smo exitali trade
    :param entry_time: Date/Datetime, ko smo entrali trade
    :param exit_time: Date/Datetime, ko smo exitali trade
    :param stop_loss_price (float): Cena/Pozicija stop_lossa
    :param take_profit_price (float): Cena/Pozicija take_profita
    :param position_size (float): Velikost pozicije (Če trejdamo na primer CDF, mora biti pozicija celo število oz. int)
    :param direction (int): Ali smo šli Long (1) ali Short (-1)
    :param index (int): Trenutni index, za katerega želimo exitati
    :param signal (int): 1 (buy) , -1(sell)
    :param reason (str): Razlog, zakaj smo exitali (stop_loss_hit ...) za lažjo analizo in debuganje

    Returns:
        {"df": df, "trade_info": df_trade, "profit": profit}

    df: Df ki vsebuje spremembe,
    trade_info: Vsebuje vse pomembne podatke za kasnejšo analizo
    profit: Profit tega trejda
    """

    # Izračunamo position value glede na entry price
    position_value = position_size * entry_price

    profit_per_share = (exit_price - entry_price) * direction
    profit = profit_per_share * position_size # Pomnožimo, da dobimo dejanski profit

    if use_trading_cost:
        # Izračunamo trading_cost
        trading_cost = trading_cost_class.calculate_fees_and_slippage(position_value)

        # Upoštevamo trading_cost
        profit -= trading_cost




    # Remove upcoming duplicate signals
    # m = index
    # current_signal = signal
    # while m < len(df):
    #     future_signal = df.at[m, "signal"]
    #     if future_signal == current_signal:
    #         df.at[m, "signal"] = 0
    #     elif future_signal == 0:
    #         break
    #     else:
    #         break
    #     m += 1

    df_trade = {
        "Datetime": exit_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_loss_price": stop_loss_price,
        "take_profit_price": take_profit_price,
        "position_size": position_size,
        "position_value": position_value,
        "signal": direction,
        "pnl": profit,
        "reason": reason,
    }
    return {"data_dict": data_dict, "trade_info": df_trade, "profit": profit}


def create_strategy_summary(sl_class, tp_class, pos_class, trading_cost_class, summary):
    """
    Combine strategy details and summary metrics into a single dictionary.

    Parameters:
    - sl_class: Stop loss strategy instance.
    - tp_class: Take profit strategy instance.
    - pos_class: Position sizing strategy instance.
    - trading_cost_class: Trading cost strategy instance.
    - summary: Dictionary containing summary statistics of the backtest.

    Returns:
    - dict: Merged dictionary of flattened strategy info and summary metrics.
    """
    def to_serializable(obj):
        try:
            # If object has __dict__, return that (attributes)
            return vars(obj)
        except TypeError:
            # Otherwise fallback to string representation
            return str(obj)
        
    def to_json_serializable(obj):
        # Convert to dict or string, then dump to JSON string
        serializable_obj = to_serializable(obj)
        return json.dumps(serializable_obj)
    
    strategy_info = {
        "backtest_timestamp": datetime.now().isoformat(),
        "stop_loss_strategy": sl_class.__class__.__name__ if sl_class is not None else None,
        "stop_loss_params": to_json_serializable(sl_class.params) if sl_class is not None else None,
        "take_profit_strategy": tp_class.__class__.__name__ if tp_class is not None else None,
        "take_profit_params": to_json_serializable(tp_class.params) if tp_class is not None else None,
        "position_sizing_strategy": pos_class.__class__.__name__,
        "position_sizing_params": to_json_serializable(pos_class.params),
        "trading_cost_strategy": trading_cost_class.__class__.__name__,
        "trading_cost_params": to_json_serializable(trading_cost_class.trading_cost_params),
    }

    combined = {**strategy_info, **summary}  # Merge both dictionaries
    return combined


def create_strategy_info(sl_class, tp_class, pos_class, trading_cost_class):
    """
    Create a dictionary containing the class names of the key strategy components.

    Parameters:
    - sl_class: Instance of the stop loss strategy.
    - tp_class: Instance of the take profit strategy.
    - pos_class: Instance of the position sizing strategy.
    - trading_cost_class: Instance of the trading cost strategy.

    Returns:
    - dict: Dictionary with the strategy class names as values in lists.
    """
        
    strategy_info = {
        "stop_loss_strategy": sl_class.__class__.__name__,
        "take_profit_strategy": tp_class.__class__.__name__,
        "position_sizing_strategy": pos_class.__class__.__name__,
        "trading_cost_strategy": trading_cost_class.__class__.__name__,
    }
    return strategy_info

def ensure_utc(series):
    """
        Pregleda če je datetime v utc pasovnem času in če ni ga premeni v UTC.
    """
    series = pd.to_datetime(series)
    if series.dt.tz is None:
        # tz-naive → localize to UTC
        return series.dt.tz_localize('UTC')
    else:
        # tz-aware → convert to UTC
        return series.dt.tz_convert('UTC')
    


def set_nested_attr(obj, path, value):
    """
    Sets a nested attribute on an object given a dot-separated path.
    For example, set_nested_attr(my_obj, 'a.b.c', 10) is equivalent to my_obj.a.b.c = 10
    """
    parts = path.split('.')
    parent_obj = functools.reduce(getattr, parts[:-1], obj)
    setattr(parent_obj, parts[-1], value)


def set_nested_value(obj, path, value):
    """
    Sets a value in a nested structure that can contain both objects and dictionaries.
    """
    keys = path.split('.')
    current_level = obj

    # Navigate down to the parent element, handling both objects and dicts
    for key in keys[:-1]:
        try:
            # First, try to access it like a dictionary key
            current_level = current_level[key]
        except (TypeError, KeyError):
            # If it fails, access it like an object attribute
            current_level = getattr(current_level, key)

    # Set the value on the final element
    final_key = keys[-1]
    try:
        # First, try to set it like a dictionary item
        current_level[final_key] = value
    except TypeError:
        # If it fails, set it like an object attribute
        setattr(current_level, final_key, value)


def calc_buy_and_hold(df: pd.DataFrame, initial_capital):
    starting_price = df["Close"].iloc[0]
    num_of_shares = initial_capital / starting_price
    df["buy_hold"] = df["Close"] * num_of_shares
    return df[["Datetime","Close","buy_hold"]]




def _create_fast_visualization_df(df_signals, df_trades, iteration_log, df_buy_and_hold):
    """
    Creates the visualization DataFrame using fast, vectorized pandas operations.
    This version is corrected to only use the 'signal' column from df_signals.
    """
    df_vis = df_signals.copy()

    # 1. Prepare trade entry "event" table WITHOUT the 'signal' column
    # This is the key change to prevent duplicate signal columns.
    entry_events = df_trades[['entry_time', 'entry_price']].rename(
        columns={'entry_time': 'Datetime'}
    ).sort_values('Datetime')
    
    # Prepare exit events as before
    exit_events = df_trades[['exit_time', 'exit_price', 'pnl', 'cumulative_capital']].rename(
        columns={'exit_time': 'Datetime'}
    )

    # 2. Use merge_asof to efficiently map entry data forward in time
    # This will now only add 'entry_price', not a new signal column.
    df_vis = pd.merge_asof(
        df_vis.sort_values('Datetime'),
        entry_events,
        on='Datetime',
        direction='backward'
    )

    # 3. Use a standard merge to place exit data only on the exact exit timestamp
    df_vis = df_vis.merge(exit_events, on='Datetime', how='left')

    # 4. Clean up: Invalidate entry data and signals for periods when not in a trade.
    # The original 'signal' column is correctly modified here.
    is_not_in_trade = df_vis['cumulative_capital'].bfill().isna()
    df_vis.loc[is_not_in_trade, ['entry_price', 'signal']] = [np.nan, 0]

    # 5. Merge dynamic stop-loss/take-profit from iteration_log if it exists
    if iteration_log is not None and not iteration_log.empty:
        iteration_log_copy = iteration_log.copy()
        iteration_log_copy["Datetime"] = ensure_utc(iteration_log_copy["Datetime"])
        df_vis = df_vis.merge(
            iteration_log_copy[["Datetime", "take_profit_price", "stop_loss_price"]],
            on="Datetime", how="left"
        )

    # 6. Add Buy & Hold and handle starting capital
    if df_buy_and_hold is not None:
        df_buy_and_hold_copy = df_buy_and_hold.copy()
        df_buy_and_hold_copy['Datetime'] = ensure_utc(df_buy_and_hold_copy['Datetime'])
        df_vis = pd.merge(df_vis, df_buy_and_hold_copy[['Datetime', 'buy_hold']], on='Datetime', how='left')
        df_vis.rename(columns={'buy_hold': 'buy_and_hold'}, inplace=True)

    if not df_trades.empty:
        starting_capital = df_trades["cumulative_capital"].iloc[0] - df_trades["pnl"].iloc[0]
        first_trade_index = df_vis['cumulative_capital'].first_valid_index()
        if first_trade_index is not None and first_trade_index > 0:
            df_vis.loc[df_vis.index < first_trade_index, 'cumulative_capital'] = starting_capital
        df_vis["cumulative_capital"] = df_vis["cumulative_capital"].ffill()

    return df_vis


def save_to_sql_database(
    path: str,
    file_name: str,
    df_trades: pd.DataFrame,
    summary: dict,
    df_signals: Optional[pd.DataFrame] = None,
    backtest_summary: Optional[dict] = None,
    strategy_info: Optional[dict] = None,
    iteration_log: Optional[pd.DataFrame] = None,
    df_buy_and_hold: Optional[pd.DataFrame] = None
):
    """
    Saves all backtest results to a SQLite database using fast, vectorized operations.
    Handles both single runs and walk-forward results by checking for optional data.
    """
    if df_trades is None or df_trades.empty:
        print("No trades to save. Skipping database write.")
        return

    os.makedirs(path, exist_ok=True)
    db_path = os.path.join(path, file_name)
    conn = sqlite3.connect(db_path)

    # --- Helper function to safely serialize context objects ---
    def serialize_context(df_to_serialize):
        if 'context' in df_to_serialize.columns:
            # Important: Work on a copy to avoid modifying the original DataFrame
            df_copy = df_to_serialize.copy()
            df_copy['context'] = df_copy['context'].apply(
                lambda x: json.dumps(asdict(x)) if is_dataclass(x) else None
            )
            return df_copy
        return df_to_serialize

    # --- Prepare Core DataFrames ---
    df_trades_copy = df_trades.copy()
    for col in ["entry_time", "exit_time"]:
        df_trades_copy[col] = ensure_utc(df_trades_copy[col])
    df_trades_copy["entry_time_epoch"] = df_trades_copy["entry_time"].astype("int64") // 10**9
    df_trades_copy["exit_time_epoch"] = df_trades_copy["exit_time"].astype("int64") // 10**9

    # --- Time the Visualization DataFrame creation ---
    start_time_vis = time.time()
    df_visualization = None
    df_signals_to_save = None

    if df_signals is not None and not df_signals.empty:
        df_signals_copy = df_signals.copy()
        df_signals_copy["Datetime"] = ensure_utc(df_signals_copy["Datetime"])
        df_signals_copy["epoch_time"] = df_signals_copy["Datetime"].astype("int64") // 10**9
        
        # Create the visualization DF *before* serializing context for the database
        df_visualization = _create_fast_visualization_df(df_signals_copy, df_trades_copy, iteration_log, df_buy_and_hold)
        
        # Now, prepare the final versions for saving by serializing the context
        df_signals_to_save = serialize_context(df_signals_copy)
        df_visualization_to_save = serialize_context(df_visualization)

    end_time_vis = time.time()
    print(f"Time to create visualization DataFrame: {end_time_vis - start_time_vis:.4f} seconds")

    # --- Time the Database Writing ---
    start_time_db = time.time()
    chunk_size = 1000 # Optimal chunk size for SQLite bulk inserts

    # Save all data, checking if optional DataFrames exist
    df_trades_copy.to_sql("trades", conn, if_exists="replace", index=False, method='multi', chunksize=chunk_size)
    pd.DataFrame([summary]).to_sql("summary", conn, if_exists="replace", index=False)

    if df_signals_to_save is not None and df_visualization_to_save is not None:
         df_signals_to_save.to_sql("signals", conn, if_exists="replace", index=False, method='multi', chunksize=chunk_size)
         df_visualization_to_save.to_sql("all_data", conn, if_exists="replace", index=False, method='multi', chunksize=chunk_size)

    if strategy_info is not None:
        pd.DataFrame([strategy_info]).to_sql("strategy_info", conn, if_exists="replace", index=False)
    
    if backtest_summary is not None:
        pd.DataFrame([backtest_summary]).to_sql("backtest_runs", conn, if_exists="append", index=False)

    end_time_db = time.time()
    print(f"Time to write to database: {end_time_db - start_time_db:.4f} seconds")

    conn.close()
    print(f"Saved to {db_path}")







def clean_dict_for_printing(d):
    """
    Recursively converts NumPy types in a dictionary to standard
    Python types for cleaner pprint output.
    """
    if not isinstance(d, dict):
        return d
    
    clean = {}
    for k, v in d.items():
        if isinstance(v, dict):
            clean[k] = clean_dict_for_printing(v) # Recurse
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            clean[k] = round(float(v), 4) # Convert to float and round
        elif isinstance(v, (np.integer, np.int64, np.int32)):
            clean[k] = int(v)
        elif isinstance(v, str) and v.startswith('{') and v.endswith('}'):
            # Try to parse and pretty-print nested JSON strings
            try:
                import json
                clean[k] = json.loads(v)
            except json.JSONDecodeError:
                clean[k] = v # Keep as-is if not valid JSON
        else:
            clean[k] = v # Keep as-is
    return clean