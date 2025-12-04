import pandas as pd
import pandas_ta as ta
import numpy as np
from indicators import function_map

def run_backtest_pessimistic(data_path: str, initial_capital: float = 10000.0):
    """
    Runs a commission-free backtest with a PESSIMISTIC exit logic.
    If both SL and TP are hit on the same candle, it counts as a stop-loss.
    """
    
    # 1. Load and Prepare Data
    try:
        df = pd.read_csv(data_path, parse_dates=['Datetime'])
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        return None, None

    # Calculate SMAs using the custom function_map
    func = function_map.get_indicator_func("sma")
    df["SMA_50"] = func(df, 50)
    df["SMA_200"] = func(df, 200)

    df['ATR'] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    
    # Note: Assuming 'ATR' column is already in the CSV file with the name 'Atr'
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)




    # 2. Initialize Backtest Variables
    capital = initial_capital
    equity_curve = [initial_capital]
    trades = []
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_time = None
    position_size = 0   # Position size in units/shares (calculated for 1% risk)
    position_value = 0  # NEW: Notional value of the position (Entry Price * Position Size)
    risk_per_share = 0  # NEW: ATR value, which represents the risk in price units
    atr_at_entry = 0 
    
    # Variables to store SMA values at the signal point
    sma_50_at_entry = 0
    sma_200_at_entry = 0

    # 3. Main Backtesting Loop
    # We loop up to len(df) - 2 to ensure we can look one candle ahead for entry/exit
    for i in range(1, len(df) - 1):
        # --- Check for Entry Conditions ---
        if position == 0:
            # Set SMA values at the signal candle (index i)
            sma_50_at_entry = df['SMA_50'][i]
            sma_200_at_entry = df['SMA_200'][i]

            # Long Entry Signal: SMA_50 crosses above SMA_200
            if df['SMA_50'][i-1] < df['SMA_200'][i-1] and df['SMA_50'][i] > df['SMA_200'][i]:
                position = 1
                entry_price = df['Open'][i+1]
                entry_time = df['Datetime'][i+1]
                
                atr_value = df['ATR'][i]
                atr_at_entry = atr_value
                stop_loss = entry_price - (1 * atr_value)
                take_profit = entry_price + (1 * atr_value)
                
                risk_per_trade_usd = capital * 0.01
                
                # Calculation for Position Sizing (Long)
                risk_per_share = entry_price - stop_loss
                position_size = risk_per_trade_usd / risk_per_share if risk_per_share > 0 else 0
                position_value = entry_price * position_size

            # Short Entry Signal: SMA_50 crosses below SMA_200
            elif df['SMA_50'][i-1] > df['SMA_200'][i-1] and df['SMA_50'][i] < df['SMA_200'][i]:
                position = -1
                entry_price = df['Open'][i+1]
                entry_time = df['Datetime'][i+1]

                atr_value = df['ATR'][i]
                atr_at_entry = atr_value
                stop_loss = entry_price + (1 * atr_value)
                take_profit = entry_price - (1 * atr_value)
                
                risk_per_trade_usd = capital * 0.01
                
                # Calculation for Position Sizing (Short)
                risk_per_share = stop_loss - entry_price
                position_size = risk_per_trade_usd / risk_per_share if risk_per_share > 0 else 0
                position_value = entry_price * position_size


        # --- Check for Exit Conditions on the SAME candle as entry ---
        if position != 0: 
            exit_price = None
            reason = None

            if position == 1: # --- LOGIC FOR LONG POSITION ---
                # Pessimistic check: Stop-Loss is checked first on the entry candle
                if df['Low'][i+1] <= stop_loss:
                    exit_price, reason = stop_loss, "Stop-Loss Hit"
                    
                elif df['High'][i+1] >= take_profit:
                    exit_price, reason = take_profit, "Take-Profit Hit"

                if exit_price:
                    pnl = (exit_price - entry_price) * position_size
                    
            elif position == -1: # --- LOGIC FOR SHORT POSITION ---
                # Pessimistic check: Stop-Loss is checked first on the entry candle
                if df['High'][i+1] >= stop_loss:
                    exit_price, reason = stop_loss, "Stop-Loss Hit"
                elif df['Low'][i+1] <= take_profit:
                    exit_price, reason = take_profit, "Take-Profit Hit"

                if exit_price:
                    pnl = (entry_price - exit_price) * position_size

            # If an exit occurred
            if exit_price:
                capital += pnl
                trades.append({
                    "Entry Time": entry_time, "Exit Time": df['Datetime'][i+1],
                    "Direction": "Long" if position == 1 else "Short", "Entry Price": entry_price,
                    "Exit Price": exit_price, "PnL": pnl, "Reason": reason,
                    "Stop-Loss": stop_loss, "Take-Profit": take_profit,
                    "ATR at Entry": atr_at_entry,
                    # NEW: Include all position details
                    "Risk Per Share": risk_per_share,
                    "Position Size (units)": position_size,
                    "Position Value (notional)": position_value,
                    "SMA_50": sma_50_at_entry,
                    "SMA_200": sma_200_at_entry,
                    "Cumulative Capital": capital
                })
                position = 0
        
        equity_curve.append(capital)
                
    # 4. Calculate and Return Performance Metrics
    if not trades:
        print("No trades were executed.")
        return pd.DataFrame(), pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades_df)
    wins = trades_df[trades_df['PnL'] > 0]
    losses = trades_df[trades_df['PnL'] <= 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    
    equity_series = pd.Series(equity_curve)
    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    performance = {
        "Ending Capital": [f"${capital:,.2f}"],
        "Total Return": [f"{(capital - initial_capital) / initial_capital:.2%}"],
        "Total Trades": [total_trades],
        "Win Rate (%)": [f"{win_rate:.2f}%"],
        "Average Win": [f"${wins['PnL'].mean():,.2f}" if not wins.empty else "$0.00"],
        "Average Loss": [f"${losses['PnL'].mean():,.2f}" if not losses.empty else "$0.00"],
        "Max Drawdown (%)": [f"{max_drawdown:.2f}%"]
    }
    
    return trades_df, pd.DataFrame(performance)

# --- Main execution block ---
if __name__ == "__main__":
    path_to_data = r'data\files\crypto\BTCUSDT_1h.csv'
    
    trade_log, performance_summary = run_backtest_pessimistic(data_path=path_to_data, initial_capital=10000.0)

    if trade_log is not None:
        print("\n--- Performance Summary (Pessimistic ATR Backtest) ---")
        print(performance_summary.to_string(index=False))
        
        print("\n--- Trade Log ---")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        trade_log_formatted = trade_log.copy()
        
        # MODIFIED: Added new columns for comprehensive position logging
        display_cols = ['Entry Price', 'Exit Price', 'PnL', 'Stop-Loss', 'Take-Profit', 
                        'ATR at Entry', 'Risk Per Share', 'Position Size (units)', 
                        'Position Value (notional)', 'SMA_50', 'SMA_200', 'Cumulative Capital']
        
        # Format the numeric columns
        for col in display_cols:
            if col in trade_log_formatted.columns:
                trade_log_formatted[col] = trade_log_formatted[col].apply(lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x)
            else:
                pass 
                
        # Reorder and print
        print(trade_log_formatted[['Entry Time', 'Exit Time', 'Direction', 'Reason'] + display_cols])