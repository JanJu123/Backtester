from . import ma
from . import adx
from . import macd_indic
from . import bollinger_bands
from . import rsi
from . import atr
from . import hurst_exponent
from . import z_score


def get_indicator_func(name: str):
    """
    Vrne funkcijo  glede na ime.

    Parametri:
        name (str): ime MA metode (npr. "sma", "ema", "wma", "vwma", "vwema")

    Returns:
        Callable: ustrezna funkcija za izračun ali None, če ime ni definirano.
    """

    mapping = {
        "sma": ma.sma,
        "ema": ma.ema,
        "wma": ma.wma,
        "vwma": ma.vwma,
        "vwema": ma.vwema,
        "adx": adx.adx,
        "bb": bollinger_bands.bollinger_bands,
        "rsi": rsi.rsi,
        "atr": atr.atr,
        "hurst_exponent": hurst_exponent.hurst_exponent,
        "z_score": z_score.zscore
    }
    return mapping.get(name.lower())

#TODO!: Naredi tako da bo v vsak indicator bill passan samo dictionary, ki vsebuje parametre, 
#!      ki so potrebni in nič več, tako kot pri momentum_zscore strategiji, Passam samo parametre za indicator!!!!!!!!!