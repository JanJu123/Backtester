import pandas as pd
import numpy as np
from core.utils import calculate_ATR


class FixedPctTakeProfit():
    def __init__(self, take_profit_params):
        self.is_dynamic = False
        self.params = take_profit_params

    # fixed_pct_take_profit
    def tp_function(self, data, index, entry_price, direction, previus_take_profit_price=None, sl_price=None):
        """
        Izračuna fiksen take-profit glede na entry price in določen odstotek.

        Parametri:
        ----------
        df : pd.DataFrame ali None
            Ni potreben za ta izračun, lahko ostane None.
        entry_price : float
            Cena, po kateri je bila odprta pozicija.
        direction : int
            Smer pozicije: 1 za long, -1 za short.
        take_profit_params : dict ali None, privzeto None
            Dodatni parametri, kjer je pomemben 'take_profit_pct' (float), 
            ki določa fiksni odstotek stop lossa (npr. 0.01 pomeni 1%).
        index: int ali None, privzeto None
            Uporaben za slicing df do trenutega podatka (pazimo na look-ahead bias)

        previus_take_profit_price : float
            Prejšna cena take-profita. Ni podatek ni potreben

        Vrne:
        -------
        float
            Izračunana cena stop lossa glede na vstopno ceno in smer pozicije.
        """

    

        take_profit__pct = self.params.take_profit_pct
        take_profit_price = entry_price+(entry_price * take_profit__pct * direction)
        return take_profit_price


class AtrBasedTakeProfit():
    def __init__(self, take_profit_params):
        self.is_dynamic = False
        self.params = take_profit_params

    # atr_based_take_profit
    def tp_function(self, data, index, entry_price, direction, previus_take_profit_price=None, sl_price=None):
        """
        Izračuna dinamično take-profit ceno na podlagi ATR indikatorja.

        Parametri:
        ----------
        df : pd.DataFrame
            DataFrame s podatki o cenah, ki vsebuje potrebne stolpce za izračun ATR.
            Pomembno: df mora vsebovati le podatke do trenutka, za katerega se izračunava stop loss,
            brez vključevanja podatkov, ki so kasnejši od tega datuma (t.i. brez 'gledanja v prihodnost').
        entry_price : float
            Cena, po kateri je bila odprta pozicija.
        direction : int
            Smer pozicije: 1 za long, -1 za short.
        take_profit_params : dict ali None, privzeto None
            Dodatni parametri, kjer so pomembni:
            - 'atr_period' (int): število obdobij za izračun ATR (privzeto 5),
            - 'atr_multiplier' (float): faktor za razdaljo take-profita glede na ATR (privzeto 1).
        previus_take_profit_price : float
            Prejšna cena take-profita. Ni podatek ni potreben
        index: int ali None, privzeto None
            Uporaben za slicing df do trenutega podatka (pazimo na look-ahead bias)
        Vrne:
        -------
        float
            Izračunana cena take-profita, prilagojena volatilnosti in smeri pozicije.
        """




        atr_period = self.params.atr_period
        atr_multiplier = self.params.atr_multiplier

        # atr_df = calculate_ATR(df=df, period=atr_period)
        # take_profit_price = entry_price+(atr_df.iloc[-1] * atr_multiplier * direction)

        take_profit_price = entry_price+(data["atr_rm"][index] * atr_multiplier * direction)


        return take_profit_price



class MaCrossTakeProfit():

    def __init__(self, take_profit_params):
        self.is_dynamic = False
        self.params = take_profit_params

    # ma_cross_take_profit
    def tp_function(self, data, index, entry_price=None, direction=None, previus_take_profit_price=None, sl_price=None):
        """
        Izračuna dinamično take-profit ceno na podlagi drsečega povprečja (MA).

        Parametri:
        ----------
        df : pd.DataFrame
            DataFrame s podatki o cenah, uporabljen za izračun drsečega povprečja.
            Pomembno: df naj vsebuje samo podatke do trenutka preverjanja, brez kasnejših vrednosti.
        direction : int, opcijsko
            Smer pozicije: 1 za long, -1 za short.
        entry_price : float, opcijsko
            Vstopna cena pozicije (ni nujno potrebna za MA logiko, vendar omogoča fleksibilnost).
        take_profit_params : dict ali None, privzeto None
            Parametri za izračun, kjer je pomemben:
            - 'ma_period' (int): obdobje drsečega povprečja (privzeto 5).
            
        previus_take_profit_price : float
            Prejšna cena take-profita. Ni podatek ni potreben

        index: int ali None, privzeto None
            Uporaben za slicing df do trenutega podatka (pazimo na look-ahead bias)

        Vrne:
        -------
        float
            Cena drsečega povprečja, ki se uporablja kot dinamičen nivo za take-profit,
            ne glede na smer (za long in short pozicije enak izračun – cena MA).
        """

        #! check main TODO task 1

        ma_period = self.params.ma_period
        data["ma"] = np.mean(ma_period)
        take_profit_price = data["ma"].iloc[index]

        return take_profit_price



class MaReversionTakeProfit():
    """
    ma_func: sma  \n
    ma_period: 20
    """

    def __init__(self, take_profit_params):
        self.is_dynamic = False
        self.params = take_profit_params
    
    def tp_function(self, data, index, entry_price, direction, previus_take_profit_price=None, sl_price=None):
        """
        Calculates the dynamic Take Profit target price, set at the value of 
        the Moving Average (the 'mean').
        """

        if "MA_rm" not in data:
            raise ValueError("MA for Risk Managment (tp, sl calculation) data not found. Ensure 'data' dictionary contains 'MA_rm' key.")
        
        return data["MA_rm"][index]
    

class RiskRewardTakeProfit():
    def __init__(self, take_profit_params):
        self.is_dynamic = False
        self.params = take_profit_params

    def tp_function(self, data, index, entry_price, direction, previus_take_profit_price=None, sl_price=None, **kwargs):
        """
        Calculates Take Profit based on sl distance from entry price and RR we want.
        """
        if sl_price is None:
            raise ValueError("RiskRewardTakeProfit requires a valid sl_price to be calculated.")

        rrr = self.params.risk_reward_ratio
        sl_distance = abs(entry_price-sl_price)
        take_profit_distance = sl_distance * rrr

        take_profit_price = entry_price + (direction*take_profit_distance)

        return take_profit_price

