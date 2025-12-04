import pandas as pd
import requests
from datetime import datetime

def download_binance_klines(symbol, interval, start_time, end_time=None):
    """
    Prenese zgodovinske podatke (candles) za izbran simbol in interval iz Binance API-ja.

    Funkcija postopoma pridobiva podatke po 1000 candles (omejitev Binance API-ja) od `start_time` do `end_time` 
    (če je podan). Med prenosom izpiše število prenesenih candles, na koncu pa vrne urejen pandas DataFrame 
    s stolpci: Datetime, Close, Open, High, Low, Volume.

    Parametri:
    ----------
    symbol : str
        Tržni simbol, npr. "BTCUSDT".
    interval : str
        Časovni interval candles, npr. "1h", "1d", "5m".
    start_time : str ali datetime
        Začetni čas za prenos podatkov (date/time). Funkcija ga pretvori v milisekunde.
    end_time : str ali datetime, opcijsko
        Končni čas za prenos podatkov. Če ni podan, se prenese do zadnjih razpoložljivih podatkov.

    Vrne:
    -------
    pandas.DataFrame
        DataFrame z zgodovinskimi candles, ki vsebuje stolpce:
        ['Datetime', 'Close', 'Open', 'High', 'Low', 'Volume'].

    Dodatno:
    --------
    - Postopno prenese podatke v blokih po 1000 candles zaradi omejitev Binance API.
    - Prikazuje napredek prenosa za lažje spremljanje.
    """
        
    url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    all_klines = []
    
    start_ts = int(pd.to_datetime(start_time).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_time).timestamp() * 1000) if end_time else None

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_ts
        }
        if end_ts:
            params["endTime"] = end_ts

        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            print("No more data to download.")
            break
        
        all_klines.extend(data)

        last_open_time = data[-1][0]
        start_ts = last_open_time + 1  # move past last candle
        
        print(f"Downloaded {len(all_klines)} candles so far...")
        
        if len(data) == limit:
            print(f"Received max limit ({limit}) candles, fetching next batch...")
        elif len(data) < limit:
            print("Received fewer than limit candles, reached end of available data.")
            break

    df = pd.DataFrame(all_klines)
    df.columns = [
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]

    df['Datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    cols_to_num = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[cols_to_num] = df[cols_to_num].apply(pd.to_numeric)
    df = df.sort_values('Datetime')
    result = df[['Datetime', 'Close', 'Open', 'High', 'Low', 'Volume']].reset_index(drop=True)
    
    print(f"Total downloaded candles: {len(result)}")
    return result


BINANCE_START_DATE = "2017-08-17"

symbol="BTCUSDT"
intervals = ["1d", "1h", "15m", "5m"]  # 1d, 1h, 5m
selected_interval = input(f"Select interval {intervals}: ")

if selected_interval in intervals:
    if selected_interval == "1d":
        df = download_binance_klines(symbol, "1d", BINANCE_START_DATE, "2025-08-01")
    elif selected_interval == "1h":
        df = download_binance_klines(symbol, "1h", BINANCE_START_DATE, "2025-08-01")
    elif selected_interval == "5m":
        df = download_binance_klines(symbol, "5m", BINANCE_START_DATE, "2025-08-01")
    elif selected_interval == "15m":
        df = download_binance_klines(symbol, "15m", BINANCE_START_DATE, "2025-08-01")
else:
    print(f"Selected interval is not avaible: {selected_interval}")

print(df)
df.to_csv(fr"F:\Programiranje\Learning_Trading_Bot\Razvijanje_Strategij\data\files\crypto\{symbol}_{selected_interval}.csv", index=False)
