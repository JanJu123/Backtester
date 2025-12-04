# Vsebuje: 
#   backtest_first_stage()
#   backtest_second_stage()
#   backtest_third_stage()

import pandas as pd
from .utils import exit_trade, calc_trading_cost, calculate_ATR, slice_data_dict
from indicators import function_map
from typing import Callable
import numpy as np
import ta
from core.data_preprocessor import DataPreprocessor



def backtest_third_stage_v2(df: pd.DataFrame, capital:float=1000, stop_loss_class: Callable = None, take_profit_class: Callable = None,
                         use_stop_loss=True, use_take_profit=True,
                         position_sizing_class: Callable = None, use_position_sizing=False, exit_on_neutral_signal=False,
                         use_integer_position=False, use_trading_cost=True, trading_cost_class=None,
                         warm_up_period: int = 0, test: bool = False):
    """
    Isto kot "backtest_third_stage" funkcija, ampak veliko hitrejša ker uporabljamo dict, ki vsebuje arraye. Zgrajen pa je isto kot Dataframe.
    """


    # Naredimo dict, ki vsebuje arraye
    # data = {
    #     "Datetime": np.array(df["Datetime"]),
    #     "Close": np.array(df["Close"]),
    #     "Open": np.array(df["Open"]),
    #     "High": np.array(df["High"]),
    #     "Low": np.array(df["Low"]),
    #     "Volume": np.array(df["Volume"]),
    #     "signal": np.array(df["signal"])
    # }


    # * Spremenimo df v dict z np.array
    data = {col: np.array(df[col]) for col in df.columns}

    # Ali uporabljamo signal, ki ga generiramo tudi za exit ali ne
    EXIT_ON_NEUTRAL_SIGNAL = exit_on_neutral_signal

    entry_price = 0
    entry_index = 0
    exit_price = 0
    profit = 0
    trade = False
    stop_loss_price = None
    take_profit_price = None
    direction = 0
    trade_info = None

    initial_capital = capital

    # Shranjuje, vse kar se dodaja vsak itteration, če je trade aktiven shranjuje st, tp, pos size.... (Za kasnejšo uproabo v Grafani)
    iteration_log_dict = {
        "Datetime": np.array(df["Datetime"], ),
        "Close": np.array(df["Close"]),
        "Open": np.array(df["Open"]),
        "High": np.array(df["High"]),
        "Low": np.array(df["Low"]),
        "Volume": np.array(df["Volume"]),
        "signal": np.array(df["signal"]),
        "stop_loss_price": np.full(len(df), np.nan, dtype=float),
        "take_profit_price": np.full(len(df), np.nan, dtype=float),
        "position_size": np.zeros(len(df), dtype=float),
        "position_value": np.zeros(len(df), dtype=float),
    }


    
    # Če je stop_loss_func None, potem raisamo error.
    if use_stop_loss==True and stop_loss_class is None :
        raise ValueError("Stop-loss class is missing. Please provide a stop_loss_class to proceed.")
    elif use_stop_loss==True:
        stop_loss_func = stop_loss_class.sl_function
        sl_is_dynamic= stop_loss_class.is_dynamic


    # Če je take_profit_class None, potem raisamo error.
    if use_take_profit==True and take_profit_class is None:
        raise ValueError("Take-profit class is missing. Please provide a take_profit_class to proceed.")
    elif use_take_profit==True:
        take_profit_func = take_profit_class.tp_function
        tp_is_dynamic = take_profit_class.is_dynamic
    

    # Če je position_sizing_class None, potem raisamo error.
    if use_position_sizing==True and position_sizing_class is None:
        raise ValueError("position-sizing class is missing. Please provide a position_sizing_class to proceed.")
    elif use_position_sizing==True:
        position_sizing_func = position_sizing_class.pos_function
    
    # Če je trading_cost_class None, potem raisamo error.
    if use_trading_cost==True and trading_cost_class is None:
        raise ValueError("trading-cost class is missing. Please provide a trading_cost_class to proceed.")


    trades_dict = {
        "Datetime": np.array(df["Datetime"]),
        "entry_time": np.empty(len(df), dtype=object),
        "exit_time": np.empty(len(df), dtype=object),
        "entry_price": np.zeros(len(df), dtype=float),
        "exit_price": np.zeros(len(df), dtype=float),
        "stop_loss_price": np.full(len(df), np.nan, dtype=float),
        "take_profit_price": np.full(len(df), np.nan, dtype=float),
        "position_size": np.zeros(len(df), dtype=float),
        "position_value": np.zeros(len(df), dtype=float),
        "cumulative_rsi": np.zeros(len(df), dtype=float),
        "signal": np.zeros(len(df), dtype=float),
        "pnl": np.zeros(len(df), dtype=float),
        "cumulative_capital": np.zeros(len(df), dtype=float),
        "reason": np.empty(len(df), dtype=object),
    }


    for index in range(1, len(df)):
        if index < warm_up_period:
            continue
        
        signal = data["signal"][index-1]

        # Get the previous signal (handle index=1 case safely)
        prev_signal = data["signal"][index-2] if index >= 2 else 0


        if signal in [1, -1] and not trade and (signal != prev_signal):

            entry_price = data["Open"][index]
            entry_time = data["Datetime"][index]
            direction = signal
            entry_index = index

            # 1. Check if a 'context' column even exists in the data.
            if 'context' in df.columns:
                # If it exists, get the context object for this signal.
                context = data["context"][index-1]
            else:
                # If it doesn't exist, just set context to None.
                context = None

            # 2. This part of your code now works perfectly for both cases.
            context_args = vars(context) if context is not None else {}

            if use_stop_loss:
                # stop_loss_price = stop_loss_func(data=slice_data_dict(data_dict=data, index=index), entry_price=entry_price, direction=direction, entry_time=entry_time,
                #                                  previus_stop_loss_price=stop_loss_price)    # Funkcija
                # stop_loss_price = stop_loss_func(data=df.iloc[:index], entry_price=entry_price, direction=direction, entry_time=entry_time,
                #                     previus_stop_loss_price=stop_loss_price)    # Funkcija
                # stop_loss_price = entry_price-(atr * 1 * direction)

                stop_loss_price = stop_loss_func(
                    data=data,
                    index=index-1,  # Avoid Look-Ahead bias,  close price not avaible for current day, (only open)
                    entry_index = entry_index,
                    entry_price=entry_price, 
                    direction=direction, 
                    entry_time=entry_time,
                    **context_args
                    )


            #! Naredi funkcijo za position sizing tako kot pri stop_loss in take_profit in jo napiši v helper.utils zato da bo manj kode in lažje berljivo
            if use_position_sizing:
                if use_stop_loss:
                    risk_per_unit = abs(entry_price - stop_loss_price)
                else:
                    risk_per_unit = entry_price
                # Izračunamo Position size
                position_size = position_sizing_func(capital=capital, risk_per_unit=risk_per_unit)
                
                # if use_integer_position:
                #     # 1. Round down to integer units if trading full shares (CDF, ...)
                #     position_size = int(position_size) # Position_size spremenimo v celo število

                #     # 2. Se prepričamo da, positon_size ni večji od našega capitala
                #     max_affordable_size = int(capital / entry_price) # Spremenimo v celo število
                # else:
                #     max_affordable_size = capital / entry_price # Spremenimo v celo število
                
                #     if position_size > max_affordable_size:
                #         # print(f"{str(entry_time).split(" ")[0]} : Desired position size ({position_size}: {position_size*entry_price}) exceeds affordable size based on available capital. Adjusting position size to {max_affordable_size} : {max_affordable_size*entry_price}. Capital: {capital}")
                #         position_size = min(position_size, max_affordable_size)

                # if position_size == 0:
                #     print(f"Insufficient capital to open a position. Available: ${capital:.2f}, Required: ${entry_price:.2f}, max affordable:{max_affordable_size}, Problem: Risk is higher than capital_to_risk")
                
                #Izračunamo position value
                position_value= position_size * entry_price

                
            else:
                position_size = 1
            

            # Iračunamo trading cost, in ga odštejemo
            if use_trading_cost:
                total_cost = trading_cost_class.calculate_fees_and_slippage(position_value)
                position_value -= total_cost
                position_size = position_value / entry_price

                # total_cost = trading_cost_class.calculate_fees_and_slippage(position_value)
                # capital -= total_cost

                
            if use_take_profit:
                # take_profit_price = take_profit_func(data=slice_data_dict(data_dict=data, index=index), 
                #                                      entry_price=entry_price, direction=direction, sl_price=stop_loss_price)    # Funkcija
                # take_profit_price = take_profit_func(data=df.iloc[:index], entry_price=entry_price, direction=direction)    # Funkcija
                # take_profit_price = entry_price+(atr * 1 * direction)

                take_profit_price = take_profit_func(
                    data=data,
                    index=index-1, # Avoid Look-Ahead bias,  close price not avaible for current day, (only open)
                    entry_price=entry_price, 
                    direction=direction,
                    sl_price=stop_loss_price,
                    **context_args # <-- Also pass context here for future flexibility
                    )

            
            if stop_loss_price == take_profit_price and (use_stop_loss and use_take_profit):
                print(f"SL price and TP price are the same!!!: TP: {take_profit_price}    ==     SL: {stop_loss_price}  , EntryPrice: {entry_price}  , Direction: {direction}")


            trade = True

                

        if trade:
            #* Če je stop-loss dinamičen (pomeni, da se spreminja stop-loss vsak candle) potem bomo spreminjali ceno vsak nov candle
            if use_stop_loss==True and sl_is_dynamic:
                old_price = stop_loss_price
                stop_loss_price = stop_loss_func(data=data, index=index-1, entry_index=entry_index, entry_price=entry_price, direction=direction, entry_time=entry_time,
                                                  previus_stop_loss_price=stop_loss_price)    # Funkcija



            #* Če je take-profit dinamičen (pomeni, da se spreminja take-profit vsak candle) potem bomo spreminjali ceno vsak nov candle
            if use_take_profit==True and tp_is_dynamic:
                take_profit_price = take_profit_func(data=data, index=index-1, entry_price=entry_price, direction=direction)    # Funkcija
            

        
            #----------------------------------------------------#
            #* Stop-loss logika

            #* Ko smo Long, potem gledamo če je Low nižji kot naš stop_loss
            if use_stop_loss==True and stop_loss_price >= data["Low"][index] and direction == 1:
                exit_price = stop_loss_price
                exit_time = data["Datetime"][index]
                trade = False
                info = exit_trade(data_dict=data, entry_price=entry_price, exit_price=exit_price, exit_time=exit_time, stop_loss_price=stop_loss_price,
                                  take_profit_price=take_profit_price, direction=direction, index=index, signal=direction,entry_time=entry_time, 
                                  reason="stop_loss_hit",
                                  position_size=position_size if use_position_sizing else 1, # Nastavmi position size, če pa ga ne pa nastavimo kot 1
                                  use_trading_cost=use_trading_cost, trading_cost_class=trading_cost_class) # Iračunamo trading cost, in ga odštejemo
                #! Posodobi da bo posredovalo data: dict, in ne df

                # stop_loss_price = None
                # take_profit_price = None
                

                # data = info["df"]
                trade_info = info["trade_info"]
                profit = info["profit"]

                # Profit dodamo k capitalu
                capital += profit

                direction= 0
                # Shranimo v trades_dict
                for key, value in trade_info.items():
                    trades_dict[key][index] = value

            #* Ko smo Short, potem gledamo če je High višji kot naš stop_loss
            elif use_stop_loss==True and stop_loss_price <= data["High"][index] and direction == -1:
                exit_price = stop_loss_price
                exit_time = data["Datetime"][index]
                trade = False
                info = exit_trade(data_dict=data, entry_price=entry_price, exit_price=exit_price, exit_time=exit_time, stop_loss_price=stop_loss_price,
                                  take_profit_price=take_profit_price, direction=direction, index=index, signal=direction,entry_time=entry_time, 
                                  reason="stop_loss_hit",
                                  position_size=position_size if use_position_sizing else 1, # Nastavmi position size, če pa ga ne pa nastavimo kot 1
                                  use_trading_cost=use_trading_cost, trading_cost_class=trading_cost_class) # Iračunamo trading cost, in ga odštejemo
            
                # stop_loss_price = None
                # take_profit_price = None

                # df = info["df"]
                trade_info = info["trade_info"]
                profit = info["profit"]

                # Profit dodamo k capitalu
                capital += profit

                direction= 0

                # Shranimo v trades_dict
                for key, value in trade_info.items():
                    trades_dict[key][index] = value
            

            #----------------------------------------------------#
            #* Take-profit logika

            #* Ko smo Long, potem gledamo če je High višjo kot naš take_profit
            elif use_take_profit==True and take_profit_price <= data["High"][index] and direction == 1:
                exit_price = take_profit_price
                exit_time = data["Datetime"][index]
                trade = False
                info = exit_trade(data_dict=data, entry_price=entry_price, exit_price=exit_price, exit_time=exit_time, stop_loss_price=stop_loss_price,
                                  take_profit_price=take_profit_price, direction=direction, index=index, signal=direction,entry_time=entry_time, 
                                  reason="take_profit_hit",
                                  position_size=position_size if use_position_sizing else 1, # Nastavmi position size, če pa ga ne pa nastavimo kot 1
                                  use_trading_cost=use_trading_cost, trading_cost_class=trading_cost_class) # Iračunamo trading cost, in ga odštejemo
                



                # stop_loss_price = None
                # take_profit_price = None

                # df = info["df"]
                trade_info = info["trade_info"]
                profit = info["profit"]

                # Profit dodamo k capitalu
                capital += profit

                direction= 0
                for key, value in trade_info.items():
                    trades_dict[key][index] = value


            #* Ko smo Short, potem gledamo če je High višjo kot naš take_profit
            elif use_take_profit==True and take_profit_price >= data["Low"][index] and direction == -1:
                exit_price = take_profit_price
                exit_time = data["Datetime"][index]
                trade = False
                info = exit_trade(data_dict=data, entry_price=entry_price, exit_price=exit_price, exit_time=exit_time, stop_loss_price=stop_loss_price,
                                  take_profit_price=take_profit_price, direction=direction, index=index, signal=direction,entry_time=entry_time, 
                                  reason="take_profit_hit", 
                                  position_size=position_size if use_position_sizing else 1, # Nastavmi position size, če pa ga ne pa nastavimo kot 1
                                  use_trading_cost=use_trading_cost, trading_cost_class=trading_cost_class) # Iračunamo trading cost, in ga odštejemo
                
                # stop_loss_price = None
                # take_profit_price = None

                # df = info["df"]
                trade_info = info["trade_info"]
                profit = info["profit"]

                # Profit dodamo k capitalu
                capital += profit

                direction= 0

                for key, value in trade_info.items():
                    trades_dict[key][index] = value
            
            #* Če EXIT_ON_NEUTRAL_SIGNAL je True, potem bomo exitali če je signal 0
            if trade == True and EXIT_ON_NEUTRAL_SIGNAL:
                if signal == 0:
                    exit_price = data["Open"][index]
                    exit_time = data["Datetime"][index]
                    trade = False
                    info = exit_trade(data_dict=data, entry_price=entry_price, exit_price=exit_price, exit_time=exit_time, stop_loss_price=stop_loss_price,
                                    take_profit_price=take_profit_price, direction=direction, index=index, signal=direction,entry_time=entry_time, reason="signal_0_exit",
                                  position_size=position_size if use_position_sizing else 1, # Nastavmi position size, če pa ga ne pa nastavimo kot 1
                                  use_trading_cost=use_trading_cost, trading_cost_class=trading_cost_class) # Iračunamo trading cost, in ga odštejemo

                    trade_info = info["trade_info"]
                    profit = info["profit"]

                    # Profit dodamo k capitalu
                    capital += profit
                    direction= 0

                    for key, value in trade_info.items():
                        trades_dict[key][index] = value



            # Shranimo info o trenutnem iteration-u
            iteration_log_dict["take_profit_price"][index] = take_profit_price
            iteration_log_dict["stop_loss_price"][index] = stop_loss_price
            iteration_log_dict["position_size"][index] = position_size
            iteration_log_dict["position_value"][index] = position_value

            if trade == False:
                stop_loss_price = None
                take_profit_price = None


            
    # Dodamo cumulative_capital inplace, ker smo že prej alocirali prostor za cumulative_capital
    trades_dict["cumulative_capital"][:] = initial_capital + np.cumsum(trades_dict["pnl"])

    

    # #Ostranimo rows kjer trejdov ni bilo
    mask = ~((trades_dict["position_size"] == 0) & (trades_dict["entry_price"] == 0))
    for key in trades_dict:
        trades_dict[key] = trades_dict[key][mask]


    return pd.DataFrame(trades_dict), pd.DataFrame(iteration_log_dict)
