from core import utils
import pandas as pd
import numpy as np



def calculate_cumulative_RSI(df: pd.DataFrame, rsi_period: int = 14, cumulative_window: int = 2, price_col: str = "Close"):
    """
    Izračuna cumulative Relative Strength Index (RSI) kot vsoto preteklih N RSI vrednosti.

    :param df (pd.DataFrame): DataFrame z vsaj ceno ('Close' oz. določeno v price_col).
    :param rsi_period (int): Koliko candlov se upošteva pri izračunu RSI. Privzeto 14.
    :param cumulative_window (int): Koliko zadnjih RSI vrednosti se sešteje. Privzeto 2.
    :param price_col (str): Katera cena naj se uporablja za izračun RSI. Privzeto "Close".

    Returns:
        pd.Series: Cumulative RSI (vsota preteklih RSI vrednosti).
    """
    
    rsi = utils.calculate_RSI(df, period=rsi_period, price_col=price_col)
    cumulative_rsi = rsi.rolling(window=cumulative_window).sum()

    return cumulative_rsi


def strategy_RSI_2p(df: pd.DataFrame, params: dict):
    """
    Prva faza strategije RSI 2P. Generira signale za nakup/prodajo glede na kumulativni RSI brez dodatnih filtrov.

    Potrebni vhodni stolpci v `df`:
        - "Close": cena zaprtja sveče.

    Parametri (`params` dict):
        - "oversold" (float): spodnja meja RSI za nakupni signal.
        - "overbought" (float): zgornja meja RSI za prodajni signal.
        - "rsi_period" (int): Koliko nazaj naj gleda nazaj, za izračunavo RSI (deafult: 14)
        - "comulative_window":(int) Koliko RSI vrendosti naj sešteje (deafult: 2)

    "2pRSI_params": {
        "oversold": 50,
        "overbought": 150,
        "rsi_period": 14
        "comulative_window": 2
    """
    params = params.rsi_2p_signal_params
    df["cumulative_rsi"] = calculate_cumulative_RSI(df=df, rsi_period=params.period, cumulative_window=params.cumulative_window)

    df['signal'] = 0
    df.loc[df["cumulative_rsi"] < params.oversold, "signal"] = 1   # Buy signal
    df.loc[df["cumulative_rsi"] > params.overbought, "signal"] = -1 # Sell signal

    df['signal'] = df['signal'].shift(1)
    df['signal'] = df['signal'].fillna(0).astype(int)

    return df

def strategy_RSI_2p_second_stage(df, params: dict = {}):
    """
    Druga faza strategije RSI 2P. Doda filter z 200p MA za dodatno potrjevanje signalov.

    Nakupni signal (1): cumulative RSI < oversold in cena > 200_MA  
    Prodajni signal (-1): cumulative RSI > overbought in cena < 200_MA

    Potrebni vhodni stolpci v `df`:
        - "Close"
        - "200_MA"

    Parametri (`params` dict):
        - "oversold" (float): spodnja meja RSI za nakupni signal.
        - "overbought" (float): zgornja meja RSI za prodajni signal.
        - "rsi_period" (int): Koliko nazaj naj gleda nazaj, za izračunavo RSI (deafult: 14)
        - "comulative_window":(int) Koliko RSI vrendosti naj sešteje (deafult: 2)

    Doda oz. vrne naslednje nove stolpce v DataFrame:
        - "cumulative_rsi": kumulativna vsota RSI vrednosti.
        - "signal": trgovalni signal:
            - 1 = buy signal
            - -1 = sell signal
            - 0 = no signal

    Returns:
        pd.DataFrame: Originalni DataFrame z dodanimi stolpci 'cumulative_rsi' in 'signal'.
    """
    params = params.rsi_2p_signal_params
    df["cumulative_rsi"] = calculate_cumulative_RSI(df=df, rsi_period=params.rsi_period, cumulative_window=params.cumulative_window)
    df = df.copy()

    if "200_MA" not in df.columns:
        df["200_MA"] = df["Close"].rolling(window=200, min_periods=1).mean()


    df['signal'] = 0
    oversold_treshhold = params.oversold
    overbought_treshhold = params.overbought

    df.loc[(df["cumulative_rsi"] < oversold_treshhold)  & (df["Close"] > df["200_MA"]), "signal"] = 1   # Buy signal
    df.loc[(df["cumulative_rsi"] > overbought_treshhold) & (df["Close"] < df["200_MA"]), "signal"] = -1 # Sell signal


    return df




params = {
    "oversold": 50,
    "overbought": 150
}



if __name__ == "__main__":
    pass
