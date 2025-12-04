import pandas as pd


def sma(df: pd.DataFrame, period:int, column:str = "Close"):
    """
    
    """
    sma = df[column].rolling(window=period).mean()
    return sma

def ema(df: pd.DataFrame, period:int, column:str = "Close"):
    """
    
    """
    try:
        ema = df[column].ewm(span=period, adjust=False).mean()
    except AttributeError as e:
        print(f"Error: {e}")
        print(type(df))
        print(df)
        print(type())
        exit(1)
    except KeyError:
        print(df)

    return ema

def wma(df: pd.DataFrame, period:int, column:str = "Close"):
    """
    """
    weights = pd.Series(range(1, period + 1))
    wma = df[column].rolling(window=period).apply(lambda prices: (prices * weights).sum() / weights.sum(), raw=True)
    return wma

def vwma(df: pd.DataFrame, period:int, column:str = "Close"):
    """
    
    """
    vwma = (df["Volume"]*df[column]).rolling(period).sum()/df["Volume"].rolling(period).sum()
    # vwma = (vwma).rolling(period).mean()
    return vwma

def vwema(df: pd.DataFrame, period:int, column:str = "Close"):
    """
    
    """

    vw_price = df[column] * df['Volume']
    ema_vw_price = vw_price.ewm(span=period, adjust=False).mean()
    ema_volume = df['Volume'].ewm(span=period, adjust=False).mean()
    vwema = ema_vw_price / ema_volume
    
    return vwema