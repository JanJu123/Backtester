import pandas as pd
import matplotlib.pyplot as plt

df_mine = pd.read_csv(r"F:\Programiranje\Learning_Trading_Bot\Razvijanje_Strategij\data\files\backtest\backtest_trades.csv")
df_mine['Datetime'] = pd.to_datetime(df_mine['Datetime'])
df_mine = df_mine.sort_values('Datetime')
df_mine.set_index('Datetime', inplace=True)

df_backtest = pd.read_csv(r"test_backtest_trades.csv")
df_backtest['Entry Time'] = pd.to_datetime(df_backtest['Entry Time'])
df_backtest = df_backtest.sort_values('Entry Time')
df_backtest.set_index('Entry Time', inplace=True)

print(df_mine)
print(df_backtest)
