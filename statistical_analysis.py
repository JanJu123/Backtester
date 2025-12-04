import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional
from data.save_load import read_table_from_sqlite
import matplotlib.pyplot as plt





def plot_monte_carlo_results(mc_results: pd.DataFrame, final_profits: pd.Series, median_profit: float, worst_case_profit: float, starting_capital: float):
    """
    Generates and displays the two Monte Carlo visualization plots within a single figure 
    with a dark theme and custom-defined large spacing.
    """
    
    # Apply dark mode style for high contrast
    plt.style.use('dark_background')
    
    # Increased figure size (24, 7) for overall better visuals and to accommodate the large gap
    fig, axes = plt.subplots(1, 2, figsize=(24, 7)) 

    fig.tight_layout() 
    plt.subplots_adjust(wspace=0.15) 

    # ====================================================================
    # Subplot 1 (Left): Monte Carlo Simulation: Equity Curve Distribution
    # ====================================================================
    ax1 = axes[0]
    
    # Plot all simulations in a very faint gray color without labels for the legend
    for col in mc_results.columns:
        ax1.plot(mc_results[col], color='#7f8c8d', alpha=0.05, linewidth=1)
    
    # Plot the key statistical lines
    ax1.plot(mc_results.median(axis=1), color='#00FFFF', linewidth=3, label='Median Equity Curve')
    ax1.plot(mc_results.quantile(0.05, axis=1), color='#e74c3c', linestyle='--', linewidth=1.5, label='5th Percentile (Worst-Case)')
    ax1.plot(mc_results.quantile(0.95, axis=1), color='#2ecc71', linestyle='--', linewidth=1.5, label='95th Percentile (Best-Case)')

    ax1.set_title('Monte Carlo Simulation: Equity Curve Distribution', fontsize=16)
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel(f'Equity Value (Starting Capital: ${starting_capital:,.0f})')
    ax1.grid(True, linestyle=':', alpha=0.4, color='darkgray')
    ax1.legend(loc='lower left', frameon=False)
    
    # ====================================================================
    # Subplot 2 (Right): Monte Carlo: Distribution of Final Profits
    # ====================================================================
    ax2 = axes[1]
    
    # Plot histogram
    ax2.hist(final_profits, bins=50, edgecolor='white', color='#3498db', alpha=0.8)
    
    # Add vertical lines for key statistics
    ax2.axvline(median_profit, color='#00FFFF', linestyle='-', linewidth=2, label=f'Median Profit: ${median_profit:,.0f}')
    ax2.axvline(worst_case_profit, color='#e74c3c', linestyle='--', linewidth=2, label=f'5th Percentile: ${worst_case_profit:,.0f}')
    # Use a high-contrast color for the break-even line
    ax2.axvline(0, color='yellow', linestyle='-', linewidth=1, label='Break-Even') 

    ax2.set_title('Monte Carlo: Distribution of Final Profits (After 1,000 Runs)', fontsize=16)
    ax2.set_xlabel('Final Profit ($)')
    ax2.set_ylabel('Frequency (Number of Runs)')
    ax2.legend(loc='upper right', frameon=False)
    ax2.grid(axis='y', linestyle=':', alpha=0.4, color='darkgray')
    
    plt.show()
    # Reset to default style to avoid affecting future non-plot operations if run interactively
    plt.style.use('default')


def plot_pnl_distribution(pnl_series: np.ndarray, t_stat: float, p_value: float):
    """
    Plots the distribution of PnL values along with the mean and the null hypothesis (mean=0),
    with a dark mode aesthetic.
    """
    # Apply dark mode style
    plt.style.use('dark_background')
    
    plt.figure(figsize=(10, 6))
    
    mean_pnl = np.mean(pnl_series)
    
    # Plot PnL distribution
    plt.hist(pnl_series, bins=30, edgecolor='white', alpha=0.8, color='#3498db', density=True)
    
    # Calculate Standard Error of the Mean (SEM) for CI
    std_err = stats.sem(pnl_series)
    alpha = 0.05
    df = len(pnl_series) - 1
    t_critical = stats.t.ppf(1 - alpha/2, df) 
    margin_of_error = t_critical * std_err 
    ci_lower = mean_pnl - margin_of_error
    ci_upper = mean_pnl + margin_of_error

    # Add lines for mean and null hypothesis
    plt.axvline(mean_pnl, color='#00FFFF', linestyle='-', linewidth=2, label=f'Mean PnL: ${mean_pnl:.2f}') # Bright Cyan for Mean
    plt.axvline(0, color='#e74c3c', linestyle='--', linewidth=2, label='Null Hypothesis ($\mu=0$)') # Bright Red for Null
    
    # Add Confidence Interval shading
    plt.axvspan(ci_lower, ci_upper, color='#00FFFF', alpha=0.1, label='95% Confidence Interval for Mean')

    # Add titles and labels
    plt.title(f'Trade PnL Distribution (T-Stat: {t_stat:.2f}, P-Value: {p_value:.5f})', fontsize=16)
    plt.xlabel('Individual Trade PnL ($)')
    plt.ylabel('Density / Frequency')
    plt.legend(loc='upper right', frameon=False)
    plt.grid(axis='y', linestyle=':', alpha=0.4, color='darkgray')
    
    plt.show()
    # Reset to default style
    plt.style.use('default')


# --- 2. Monte Carlo Simulation for Robustness (Updated) ---

def monte_carlo_equity_simulation(
    df_trades: pd.DataFrame, 
    num_simulations: int = 1000, 
    num_trades_per_sim: Optional[int] = None, # Changed default to None
    starting_capital: float = 10000.0,
    plot_simulation: bool = False # New variable added here
) -> pd.DataFrame:
    """
    Performs Monte Carlo simulation by randomly reordering trades (bootstrapping)
    to analyze the range of potential equity curve outcomes and sequence risk.

    If num_trades_per_sim is None, it defaults to the total number of historical trades.

    Args:
        df_trades: DataFrame of closed trades, must include 'pnl' column.
        num_simulations: The number of random equity curves to generate.
        num_trades_per_sim: The number of trades to include in each simulation (resampling size).
        starting_capital: The initial capital used for the base equity curve.
        plot_simulation: If True, generates and displays plots for the results.
    
    Returns:
        pd.DataFrame: A DataFrame where each column is one simulated equity curve.
    """
    if 'pnl' not in df_trades.columns:
        raise ValueError("df_trades must contain a 'pnl' column.")
        
    pnl_series = df_trades['pnl'].dropna().values
    
    n_historical_trades = len(pnl_series)
    
    if n_historical_trades < 10: 
        print("Warning: Insufficient trade count for meaningful simulation.")
        return pd.DataFrame()

    # DECISION POINT: Set trades per simulation
    if num_trades_per_sim is None:
        # Default behavior: use the size of the historical sample (Bootstrapping)
        num_trades_per_sim = n_historical_trades
    elif num_trades_per_sim > n_historical_trades:
        # Safety check: Cannot simulate more unique trades than exist
        print(f"Warning: Requested {num_trades_per_sim} trades, but only {n_historical_trades} exist. Using {n_historical_trades}.")
        num_trades_per_sim = n_historical_trades
    
    print(f"Running Monte Carlo: {num_simulations} simulations, each with {num_trades_per_sim} trades.")
    
    simulations = {}
    for i in range(num_simulations):
        # Sample with replacement from historical PnL values
        simulated_pnl = np.random.choice(pnl_series, size=num_trades_per_sim, replace=True)
        
        # Calculate the cumulative PnL for the simulation
        equity_curve = np.cumsum(simulated_pnl) + starting_capital
        simulations[f'Sim_{i+1}'] = equity_curve

    mc_results = pd.DataFrame(simulations)

    # --- Conditional Plotting and Statistics Generation (NEW LOGIC) ---
    if not mc_results.empty:
        final_profits = mc_results.iloc[-1] - starting_capital
        median_profit = final_profits.median()
        worst_case_profit = final_profits.quantile(0.05)
        
        print(f"Monte Carlo Runs: {len(mc_results.columns)}")
        print(f"Trades per Run: {len(mc_results)}")
        print(f" - Median Expected Profit: ${median_profit:,.2f}")
        print(f" - 5th Percentile (Worst-Case Risk): ${worst_case_profit:,.2f} 📉")
        
        if worst_case_profit > 0:
            print("--> CONCLUSION: HIGHLY ROBUST - 95% of runs ended positive. 👍")
        elif worst_case_profit < 0 and median_profit > 0:
            print("--> CONCLUSION: ROBUSTNESS WARNING - There is a 5% chance of severe loss.")

        if plot_simulation:
            print("\nVisualizing Monte Carlo Results...")
            plot_monte_carlo_results(mc_results, final_profits, median_profit, worst_case_profit, starting_capital)

    return mc_results

# --- 3. P-Value Calculation for Statistical Edge (Updated) ---

def calculate_p_value(df_trades: pd.DataFrame, plot_distribution: bool = False) -> Tuple[float, float]:
    """
    Performs a one-sample T-test on the trade PnL to determine the t-statistic and p-value
    against the null hypothesis that the mean PnL is zero.
    
    The T-statistic measures the size of the difference relative to the variation 
    in your sample data. A larger magnitude (further from 0) indicates stronger evidence
    against the null hypothesis.

    Args:
        df_trades: DataFrame of closed trades, must include 'pnl' column.
        plot_distribution: If True, generates and displays a plot of the PnL distribution.
        
    Returns:
        Tuple[float, float]: (T-statistic, P-value). Returns (np.nan, np.nan) if data is insufficient.
    """
    pnl = df_trades['pnl'].dropna().values
    
    # Needs a reasonable number of samples for the t-test to be reliable
    if len(pnl) < 30: 
        print("Warning: Insufficient trade count for reliable P-Value calculation (N < 30).")
        # Return NaNs for both t_stat and p_value if data is insufficient
        return np.nan, np.nan
        
    # T-test comparing the mean PnL to 0
    t_stat, p_value = stats.ttest_1samp(pnl, 0)
    
    if plot_distribution:
        print("\nVisualizing PnL Distribution...")
        plot_pnl_distribution(pnl, t_stat, p_value)
    
    return t_stat, p_value # Now returns both

# --- 4. Seasonal/Calendar Analysis ---

def analyze_monthly_performance(df_trades: pd.DataFrame) -> pd.DataFrame:
    """
    Groups trade performance by the calendar month the trade was CLOSED to find 
    monthly biases or weaknesses.
    """
    required_cols = ['pnl', 'exit_time']
    if not all(col in df_trades.columns for col in required_cols):
        raise ValueError(f"df_trades must contain columns: {required_cols}")

    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    df_trades = df_trades.set_index('exit_time')

    # Calculate PnL by month
    monthly_pnl_series = df_trades['pnl'].resample('M').sum()
    
    # Group by the month number to get aggregate statistics
    monthly_group = monthly_pnl_series.groupby(monthly_pnl_series.index.month)
    
    # Calculate statistics
    monthly_stats = pd.DataFrame({
        'Month Name': monthly_group.first().index.map(lambda x: pd.Timestamp(year=2000, month=x, day=1).strftime('%B')),
        'Avg Monthly PnL': monthly_group.mean(),
        'Median Monthly PnL': monthly_group.median(),
        'Monthly Win %': monthly_group.apply(lambda x: (x > 0).mean() * 100),
        'Num Trading Months': monthly_group.size 
    })
    
    monthly_stats = monthly_stats.reset_index(drop=True).set_index('Month Name')
    
    return monthly_stats.sort_values('Avg Monthly PnL', ascending=False)


if __name__ == "__main__":
    db_path = r"data\files\backtest\backtest.db"
    table_name = "trades"
    df_trades = read_table_from_sqlite(db_path=db_path, table_name=table_name)

    if not df_trades.empty:
        # --- 1. Run Monte Carlo Simulation ---
        print("--- Running Monte Carlo Analysis ---")
        # Corrected typo and stored the result
        df_simulations = monte_carlo_equity_simulation(
            df_trades=df_trades,
            num_simulations=1000,
            starting_capital=1000,
            plot_simulation=True  # This function will print its own results and plot
        )
        
        # --- 2. Calculate Statistical Significance (P-Value) ---
        print("\n--- Running Statistical Significance Test ---")
        t_stat, p_value = calculate_p_value(df_trades=df_trades, plot_distribution=True)
        if not np.isnan(p_value):
            print(f"T-statistic: {t_stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            if p_value < 0.05:
                print("--> CONCLUSION: The result is statistically significant (p < 0.05). 👍")
            else:
                print("--> CONCLUSION: The result is not statistically significant (p >= 0.05).")

        # # --- 3. Analyze Monthly Performance ---
        # print("\n--- Analyzing Monthly Performance ---")
        # monthly_stats = analyze_monthly_performance(df_trades=df_trades)
        # print(monthly_stats)

