import pandas as pd
import numpy as np
import os
import pprint


from core.strategy_engine import StrategyEngine
from core.stats import PerformanceEvaluator
from core import backtest
from strategy_components import stop_loss_strategies
from strategy_components import take_profit_strategies
from strategy_components import position_sizing_strategies
from strategy_components.trading_cost import TradingCost
from core.utils import create_strategy_summary, create_strategy_info


def initialize_strategy_components(strategy_config, global_params):
    """Dynamically initializes all plug-and-play components based on the config."""
    
    sl_strategy_name = strategy_config.stop_loss_strategy
    tp_strategy_name = strategy_config.take_profit_strategy
    pos_strategy_name = strategy_config.sizing_strategy

    # --- Component Maps ---
    if sl_strategy_name is not None:
        stop_loss_map = {
            "AtrBasedStopLoss": stop_loss_strategies.AtrBasedStopLoss, 
            "AtrTrailingBasedStopLoss": stop_loss_strategies.AtrTrailingBasedStopLoss,
            "FixedPctStopLoss": stop_loss_strategies.FixedPctStopLoss,
            "RRRBasedStopLoss": stop_loss_strategies.RRRBasedStopLoss,
            "FvgBasedStopLoss": stop_loss_strategies.FvgBasedStopLoss,
            "PctTrailingBasedStopLoss": stop_loss_strategies.PctTrailingBasedStopLoss,
            "RRRBasedTrailingStopLoss": stop_loss_strategies.RRRBasedTrailingStopLoss,
            "FixedRRRBasedTrailingStopLoss": stop_loss_strategies.FixedRRRBasedTrailingStopLoss
        }

        # --- Stop loss Initialization ---
        sl_class = stop_loss_map[sl_strategy_name](
            stop_loss_params=strategy_config.stop_loss_params
        )if sl_strategy_name else None

    else: sl_class = None

    if tp_strategy_name is not None:
        take_profit_map = {
            "AtrBasedTakeProfit": take_profit_strategies.AtrBasedTakeProfit,
            "FixedPctTakeProfit": take_profit_strategies.FixedPctTakeProfit,
            "MaReversionTakeProfit": take_profit_strategies.MaReversionTakeProfit,
            "RiskRewardTakeProfit": take_profit_strategies.RiskRewardTakeProfit
        }
         # --- Take profit Initialization ---
        tp_class = take_profit_map[tp_strategy_name](
            take_profit_params=strategy_config.take_profit_params
        ) if tp_strategy_name else None

    else: tp_class = None

    if pos_strategy_name is not None:
        position_sizing_map = {
            "FixedPctPositionSizing": position_sizing_strategies.FixedPctPositionSizing, 
            "FixedSharePositionSizing": position_sizing_strategies.FixedSharePositionSizing
        }
         # --- Take profit Initialization ---
        pos_class = position_sizing_map[pos_strategy_name](
            pos_params=strategy_config.sizing_params
        )

    else: pos_class = None

    if global_params.trading_cost_params:
         trading_cost_class = TradingCost(global_params.trading_cost_params)
    else:
         trading_cost_class = None


    return sl_class, tp_class, pos_class, trading_cost_class



def run_single_backtest(df, params, strategy_name, initial_capital, print_summary = True,
                        test: bool = False, trading_cost: bool = True):
    """Runs one complete backtest and returns all necessary artifacts."""
    
    # --- Look up the correct profiles for this strategy ---
    strategy_config = params.strategy_configs[strategy_name]

    # 1. Generate Signals
    strategy_engine = StrategyEngine(strategy_to_use=strategy_name, df=df, params=params,
                                     test=test)
    df_signals = strategy_engine.run()
    


    # 2. Initialize Components
    sl, tp, pos, cost = initialize_strategy_components(strategy_config, params)
    # print(f"sl: {sl}")
    # print(f"tp: {tp}")
    # print(f"pos: {pos}")
    # print(f"cost: {cost}")
    # exit(1)
    # 3. Run Backtest
    df_trades, iteration_log = backtest.backtest_third_stage_v2(
        df=df_signals, 
        capital=initial_capital,
        stop_loss_class=sl,
        take_profit_class=tp,
        position_sizing_class=pos,
        trading_cost_class=cost,
        use_stop_loss=True if sl else False,
        use_take_profit=True if tp else False,
        use_position_sizing=True,
        use_integer_position=False,
        use_trading_cost=trading_cost,
        warm_up_period=20,
        exit_on_neutral_signal=strategy_config.exit_on_neutral_signal
    )
    


    # 4. Evaluate and create summaries
    evaluator = PerformanceEvaluator(trades_df=df_trades, initial_capital=initial_capital)
    summary = evaluator.summary()
    backtest_summary = create_strategy_summary(sl, tp, pos, cost, summary)
    strategy_info = create_strategy_info(sl, tp, pos, cost)

    if print_summary:
        print(f"\n--- Backtest Summary for {strategy_name} ---")
        # cleared_summary = {}
        # for key, value in summary.items():
        #     cleared_summary[key] = round((float(value)), 5)
        # print(cleared_summary)

        print(summary)
    
    return df_signals, df_trades, summary, backtest_summary, strategy_info, iteration_log
