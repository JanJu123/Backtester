import pandas as pd
import numpy as np
import os




from data import save_load
from strategies.MovingAverage import RSI_2p
from core.utils import modify_columns
from indicators import ma
from config.load_params import load_params_from_json
import time

from strategies.MovingAverage import ma_crossover, ma_price_crossover, macd_signal_cross

full_df = save_load.load_data(path=r"F:\Programiranje\Learning_Trading_Bot\Razvijanje_Strategij\data\files\crypto", 
                         filename="BTCUSDT_1h.csv", 
                         type_of_file="csv")
df = full_df


df = modify_columns(df)  # Spremenimo imena columnov in jih sortiramo


json_path = r"config/params.json"
all_params = load_params_from_json(file_path=json_path)


# df_signals = RSI_2p.strategy_RSI_2p_first_stage(df=df, params=params["params_strategy"])
# df_signals = RSI_2p.strategy_RSI_2p_second_stage(df=df, params=all_params.strategy_params.strategy_RSI_2p)




df = pd.concat([df] * 10, ignore_index=True)
print(df.shape)

runs = 2
times = []

for _ in range(runs):
    start = time.perf_counter()
    macd_signal_cross.macd_signal(df=df, params=all_params.strategy_params)
    end = time.perf_counter()
    times.append(end - start)

print(f"Average time over {runs} runs: {sum(times)/runs:.6f} seconds")  