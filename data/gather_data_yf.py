import pandas as pd
import yfinance as yf
import pickle
import time



def clean_data(df):
    df.reset_index(inplace=True)
    df.columns = [col[0] for col in df.columns]
    if "Date" in df.columns:
        df.rename(columns={"Date": "Datetime"}, inplace=True)
    return df


def get_data(ticker, period="2y", interval="1h", save=False):

    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    # time.sleep(1)

    df = pd.DataFrame(data)
    
    df = clean_data(df)

    df.reset_index(drop=True, inplace=True)

    
    with open(f"data/files/data_{interval}.pkl", "wb") as f:
        pickle.dump(df, f)

    if save:
        df.to_csv(f"data/files/{ticker}_{interval}.csv", index=False)


    return df


def get_time_synced_data(df_1h, df_1d):
    # Calculate the 200-day moving average for the 1d dataset
    df_1d['200_MA'] = df_1d['Close'].rolling(window=200).mean()
    df_1d.to_csv("SPY_1d.csv", index=False)



    # Create a new column for 200_MA in the hourly data (df_1h)
    df_1h['200_MA'] = pd.NaT  # Initializing with NaN values

    # Ensure '200_MA' is a numeric type before applying fillna()
    df_1h['200_MA'] = pd.to_numeric(df_1h['200_MA'], errors='coerce')  # Convert to numeric values
    df_1h['200_MA'] = df_1h['200_MA'].fillna(0)  # Fill NaN with 0

    # Iterate through the 1h dataset to fill the 200_MA column
    for i in range(len(df_1h)):
        # Get the current row's date (this is assumed to be in datetime format)
        current_date = df_1h.loc[i, 'Datetime'].date()  # Extract the date part (without time)
        
        # Find the corresponding 200d MA value from the 1d dataset
        daily_ma_value = df_1d[df_1d['Datetime'].dt.date == current_date]['200_MA'].values
        df_1h['200_MA'] = df_1h['200_MA'].astype('float64')
        
        if len(daily_ma_value) > 0:
            # Apply the 200d MA value for the entire day (same value for all 7 hourly candles)
            df_1h.loc[i, '200_MA'] = daily_ma_value[0]

    return df_1h


def prepare_data(ticker="", period_1h="2y", period_1d="40y", df_1h=None, df_1d=None, save_1h=False, save_1d=False):

    # Če že imamo podatke in jih moramo samo uskladiti
    if df_1d is not None and df_1d is not None:
        return get_time_synced_data(df_1h, df_1d)


    df_1h = get_data(ticker=ticker, period=period_1h, interval="1h", save=save_1h)
    df_1d = get_data(ticker=ticker, period=period_1d, interval="1d", save=save_1d)
    return get_time_synced_data(df_1h, df_1d)


def get_data_tickers(tickers):

    data = yf.download(tickers, period="3y", interval="1d", auto_adjust=True, group_by="ticker") 

    ticker_data_1d_list = []


    for ticker in tickers:
        # Extract the data for the specific ticker
        data_for_ticker = data[ticker][['Close', 'High', 'Low', 'Open', 'Volume']]
        
        # Rename columns for clarity
        data_for_ticker.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        
        # Add the Ticker column for identification (optional)
        data_for_ticker['Ticker'] = ticker
        
        # Reset the index to have a proper Datetime index
        data_for_ticker.reset_index(inplace=True)
        
        # Append this DataFrame to the list
        ticker_data_1d_list.append(data_for_ticker)



    #--------------------------------------------------------------------------#
    # get 1h data, 2y

    data = yf.download(tickers, period="2y", interval="1h", auto_adjust=True, group_by="ticker") 

    ticker_data_1h_list = []


    for ticker in tickers:
        # Extract the data for the specific ticker
        data_for_ticker = data[ticker][['Close', 'High', 'Low', 'Open', 'Volume']]
        
        # Rename columns for clarity
        data_for_ticker.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        
        # Add the Ticker column for identification (optional)
        data_for_ticker['Ticker'] = ticker
        
        # Reset the index to have a proper Datetime index
        data_for_ticker.reset_index(inplace=True)
        
        # Append this DataFrame to the list
        ticker_data_1h_list.append(data_for_ticker)


    #--------------------------------------------------------------------------#
    # Podatke združimo

    df_list = []

    print("Last step")
    for (ticker, df_1h, df_1d) in zip(tickers, ticker_data_1h_list, ticker_data_1d_list):
        df = prepare_data(ticker=ticker, df_1h=df_1h, df_1d=df_1d)
        df_list.append(df)


    return df_list


if __name__ == "__main__":

    # ticker = "^GSPC"  # S&P 500
    # ticker="SPY"
    tickers = ["AVGO", "KO", "HES"]
    ticker="SPY"

    tickers = ["MCD"]




    # for ticker in tickers:
    #     df_1h = prepare_data(ticker)

    #     # with open("files/data.pkl", "wb") as f:
    #     #     pickle.dump(df_1h, f)
    #     # with open(f"stock_data/{ticker}.pkl", "wb") as f:
    #     #     pickle.dump(df_1h, f)
    #     # df_1h.to_csv(f"stock_data/{ticker}.csv", index=False)

    #     print(df_1h)
    #     print("Sucesfully gathered and prepared data...")
    #     print("Ticker: "+ticker)

    df_1h = prepare_data(ticker=ticker, save_1d=False, save_1h=False)
    df_1h.to_csv("SPY_1h.csv", index=False)

    # import os

    # current_dir = os.getcwd()
    # print(current_dir)



