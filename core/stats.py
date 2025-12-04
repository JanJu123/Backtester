
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PerformanceEvaluator:
    def __init__(self, trades_df, initial_capital: float=1000):
        """
        trades_df = pd.DataFrame with columns:
        ['Datetime', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 'position_size', 'signal', "reason"]
        """
        self.trades = trades_df.copy()
        self.trades['Datetime'] = pd.to_datetime(self.trades['Datetime'])
        self.trades["entry_time"] = pd.to_datetime(self.trades["entry_time"])
        self.trades["exit_time"] = pd.to_datetime(self.trades["exit_time"])
        self.initial_capital = initial_capital

        
        self.initial_capital = initial_capital
        self.trades['holding_time'] = (
            self.trades['exit_time'] - self.trades['entry_time']
        ).dt.total_seconds() / 3600  # in hours

    def total_return(self):
        total_profit = self.trades['pnl'].sum()
        return (total_profit / self.initial_capital) * 100
    
    def total_profit(self):
        return self.trades['pnl'].sum()

    def average_return(self):
        return self.trades['pnl'].mean()

    def sharpe_ratio(self):
        returns = self.trades['pnl']
        return np.mean(returns) / (np.std(returns) + 1e-10)

    def max_drawdown(self):
        equity = self.trades['pnl'].cumsum() + self.initial_capital
        peak = equity.cummax()
        drawdown = equity - peak
        return drawdown.min()

    def max_drawdown_pct(self):
        equity = self.trades['pnl'].cumsum() + self.initial_capital
        peak = equity.cummax()
        drawdown = equity - peak

        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown_pct = drawdown / peak
            drawdown_pct = drawdown_pct.replace([np.inf, -np.inf], 0).fillna(0)
        return drawdown_pct.min() * 100

    def win_rate(self):
        wins = self.trades[self.trades['pnl'] > 0]
        return len(wins) / len(self.trades) if len(self.trades) > 0 else np.nan


    def planned_risk_reward_ratio(self):
        risk = abs(self.trades['entry_price'] - self.trades['stop_loss_price'])
        reward = abs(self.trades['take_profit_price'] - self.trades['entry_price'])

        valid = (risk > 0) & (reward > 0)
        risk = risk[valid]
        reward = reward[valid]

        if len(risk) == 0:
            return np.nan


        planned_rr = reward / risk
        return planned_rr.mean()

    def realized_risk_reward_ratio(self):
        wins = self.trades[self.trades['pnl'] > 0]['pnl']
        losses = self.trades[self.trades['pnl'] < 0]['pnl']
        if len(wins) == 0 or len(losses) == 0:
            return np.nan
        return wins.mean() / abs(losses.mean())

    def max_win(self):
        return self.trades['pnl'].max()

    def max_loss(self):
        return self.trades['pnl'].min()

    def avg_holding_time(self):
        return self.trades['holding_time'].mean()

    def min_holding_time(self):
        return self.trades['holding_time'].min()

    def max_holding_time(self):
        return self.trades['holding_time'].max()

    def num_trades(self):
        return len(self.trades)

    def num_buy_trades(self):
        return len(self.trades[self.trades['signal'] == 1])

    def num_sell_trades(self):
        return len(self.trades[self.trades['signal'] == -1])
    

    def expectancy(self):
        # Filter winning and losing trades
        wins = self.trades[self.trades['pnl'] > 0]
        losses = self.trades[self.trades['pnl'] < 0]

        if len(self.trades) == 0:
            return 0

        # Calculate initial risk per trade for all trades (Series)
        initial_risks = (abs(self.trades["entry_price"] - self.trades["stop_loss_price"]) * self.trades["position_size"])

        # Avoid division by zero or zero risk trades by masking zeros with NaN
        initial_risks = initial_risks.replace(0, np.nan)

        # Calculate R-multiples for wins and losses
        wins_r = wins['pnl'] / initial_risks.loc[wins.index]
        losses_r = losses['pnl'] / initial_risks.loc[losses.index]

        win_rate = len(wins) / len(self.trades)
        loss_rate = 1 - win_rate

        avg_win_r = wins_r.mean() if not wins_r.empty else 0
        avg_loss_r = abs(losses_r.mean()) if not losses_r.empty else 0

        return (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

    def average_r_multiple(self):
        # Calculate initial risk per trade
        self.trades['initial_risk'] = (
            abs(self.trades["entry_price"] - self.trades["stop_loss_price"]) * self.trades["position_size"]
        )

        self.trades['initial_risk'] = self.trades['initial_risk'].replace(0, np.nan)

        # Calculate R-multiple per trade
        self.trades['r_multiple'] = self.trades['pnl'] / self.trades['initial_risk']


        return self.trades['r_multiple'].mean()
    
    def count_take_profit_hits(self):
        return (self.trades["reason"] == "take_profit_hit").sum()
    
    def count_stop_loss_hits(self):
        return (self.trades["reason"] == "stop_loss_hit").sum()

    def summary(self):
        return {
            "total_profit": self.total_profit(),
            "total_return": self.total_return(),
            "avg_return": self.average_return(),
            "sharpe": self.sharpe_ratio(),
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_pct": self.max_drawdown_pct(),
            "win_rate": self.win_rate(),
            "risk_reward_ratio": self.planned_risk_reward_ratio(),
            "realized_risk_reward_ratio": self.realized_risk_reward_ratio(),
            "max_win": self.max_win(),
            "max_loss": self.max_loss(),
            "num_trades": self.num_trades(),
            "num_buy_trades": self.num_buy_trades(),
            "num_sell_trades": self.num_sell_trades(),
            "avg_holding_time_hours": self.avg_holding_time(),
            "min_holding_time_hours": self.min_holding_time(),
            "max_holding_time_hours": self.max_holding_time(),
            "expectancy": self.expectancy(),
            "avg_r_multiple": self.average_r_multiple(),
            "take_profit_hits": self.count_take_profit_hits(),
            "stop_loss_hits": self.count_stop_loss_hits()
        }
        
    def plot_performance(self):
        if not self.trades.empty:
            cumulative_pnl = self.trades['pnl'].cumsum()
            peak = cumulative_pnl.cummax()
            drawdown = cumulative_pnl - peak
            
            fig, axs = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1, 1]})  # slightly shorter height
            fig.suptitle('Trading Performance Overview', fontsize=16)
            
            # 1. Cumulative PnL with ideal straight line
            axs[0].plot(self.trades['Datetime'], cumulative_pnl, label='Cumulative PnL', color='green')
            
            # Ideal straight line from start to end
            x_vals = np.array([self.trades['Datetime'].iloc[0].toordinal(), self.trades['Datetime'].iloc[-1].toordinal()])
            y_vals = np.array([cumulative_pnl.iloc[0], cumulative_pnl.iloc[-1]])
            
            # Plot the line with low opacity and dashed style
            axs[0].plot(self.trades['Datetime'], 
                        np.interp(self.trades['Datetime'].map(lambda d: d.toordinal()), x_vals, y_vals),
                        label='Ideal Linear Growth', color='gray', linestyle='--', alpha=0.5)
            
            axs[0].fill_between(self.trades['Datetime'], drawdown, 0, color='red', alpha=0.3, label='Drawdown')
            axs[0].set_ylabel('Cumulative PnL')
            axs[0].legend()
            axs[0].grid(True)
            
            # 2. Drawdown over time
            axs[1].plot(self.trades['Datetime'], drawdown, color='red')
            axs[1].set_ylabel('Drawdown')
            axs[1].grid(True)
            
            # 3. Win/Loss Distribution Histogram
            axs[2].hist(self.trades['pnl'], bins=30, color='blue', alpha=0.7)
            axs[2].set_xlabel('PnL per Trade')
            axs[2].set_ylabel('Frequency')
            axs[2].grid(True)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        
        else:
            print(self.trades)
            print("Warning: self.trades is empty. Cannot compute a graph...")
    

    #TODO Opiši vizualize na grafana
    def visualize_on_grafana():
        """
        
        """




