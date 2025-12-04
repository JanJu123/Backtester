import pandas as pd
from core.utils import calculate_ATR
import ta
import numpy as np

class FixedPctStopLoss():
    from config.param_schemas.risk_schemas import FixedPctBasedSLParams
    def __init__(self, stop_loss_params: FixedPctBasedSLParams):
        self.params = stop_loss_params
        self.is_dynamic = self.params.is_dynamic

    # fixed_pct_stop_loss
    def sl_function(self, entry_price, entry_index, direction, index=None, entry_time=None, data=None, previus_stop_loss_price=None):
        """
        Izračuna fiksen stop-loss glede na entry price in določen odstotek.

        Parametri:
        ----------
        df : pd.DataFrame ali None
            Ni potreben za ta izračun, lahko ostane None.
        entry_price : float
            Cena, po kateri je bila odprta pozicija.
        direction : int
            Smer pozicije: 1 za long, -1 za short.
        stop_loss_params : dict ali None, privzeto None
            Dodatni parametri, kjer je pomemben 'stop_loss_pct' (float), 
            ki določa fiksni odstotek stop lossa (npr. 0.01 pomeni 1%).
        previus_stop_loss_price : float ali None, privzeto None
            Ni potreben! Potreben je samo kadar primerjamo prejšen stop_loss z trenutnim

        Vrne:
        -------
        float
            Izračunana cena stop lossa glede na vstopno ceno in smer pozicije.
        """

        
        stop_loss_pct = self.params.stop_loss_pct
        stop_loss_price = entry_price-(entry_price * stop_loss_pct * direction)
        return stop_loss_price


class AtrBasedStopLoss():
    from config.param_schemas.risk_schemas import AtrBasedSLParams
    def __init__(self, stop_loss_params: AtrBasedSLParams):
        self.params = stop_loss_params
        self.is_dynamic = self.params.is_dynamic
        # self.stop_loss_params = stop_loss_params.atr_based_stop_loss

    # atr_based_stop_loss
    def sl_function(self, data, index, entry_index, entry_price, direction, entry_time=None, previus_stop_loss_price=None, atr=None):
        """


        Izračuna dinamično stop-loss ceno na podlagi ATR indikatorja.

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
        stop_loss_params : dict ali None, privzeto None
            Dodatni parametri, kjer so pomembni:
            - 'atr_period' (int): število obdobij za izračun ATR (privzeto 5),
            - 'atr_multiplier' (float): faktor za razdaljo stop lossa glede na ATR (privzeto 1).
        previus_stop_loss_price : float ali None, privzeto None
            Ni potreben! Potreben je samo kadar primerjamo prejšen stop_loss z trenutnim

        Vrne:
        -------
        float
            Izračunana cena stop lossa, prilagojena volatilnosti in smeri pozicije.
        """

        atr_period = self.params.atr_period
        atr_multiplier = self.params.atr_multiplier


        stop_loss_price = entry_price-(data["atr_rm"][index] * atr_multiplier * direction)


        return stop_loss_price


class PctTrailingBasedStopLoss():
    from config.param_schemas.risk_schemas import PctTrailingBasedSLParams
    def __init__(self, stop_loss_params: PctTrailingBasedSLParams):
        self.params = stop_loss_params
        self.is_dynamic = self.params.is_dynamic
    
    # trailing_based_stop_loss
    def sl_function(self, data, index, entry_index, entry_price, direction, entry_time=None, previus_stop_loss_price=None):
        """
        Izračuna trailing stop-loss ceno na podlagi fiksnega  x% premika od najvišje dosežene cene (za long pozicije) oziroma najnižje (za short pozicije).

        Parametri:
        ----------
        df : pd.DataFrame
            DataFrame s podatki o cenah, ki vsebuje stolpce z zgodovino cen za izračun trailing stop lossa.
            Pomembno: df mora vsebovati podatke do trenutka, za katerega se izračunava stop loss,
            brez vključevanja podatkov, ki so kasnejši od tega datuma (brez 'gledanja v prihodnost').
        entry_price : float
            Cena, po kateri je bila odprta pozicija.
        direction : int
            Smer pozicije: 1 za long, -1 za short.
        trail_pct : float
            Fiksni odstotek (npr. 0.01 za 1 %), ki določa, kako daleč od vršne cene naj bo trailing stop loss.
        previus_stop_loss_price : float ali None, privzeto None
            Ni potreben! Potreben je samo kadar primerjamo prejšen stop_loss z trenutnim


        Vrne:
        -------
        float
            Izračunana cena trailing stop lossa, ki je vedno 1 % pod najvišjo ceno (za long) ali 1 % nad najnižjo ceno (za short), glede na gibanje trga in smer pozicije.
        """

        trail_pct = self.params.trail_pct

        df = df.set_index("Datetime")
        if previus_stop_loss_price is None:
            stop_loss_price = entry_price - ( entry_price * trail_pct * direction)

            return stop_loss_price

        else:
            if direction == 1:
                peak_price = df["High"].max()
                stop_loss_price = peak_price * (1 - trail_pct)
                if previus_stop_loss_price in [None, 0]:
                    return stop_loss_price
                return max(previus_stop_loss_price, stop_loss_price)

            elif direction == -1:
                bottom_price = df["Low"].min()
                stop_loss_price = bottom_price * (1 + trail_pct)
                if previus_stop_loss_price in [None, 0]:
                    return stop_loss_price
                return min(previus_stop_loss_price, stop_loss_price)

            return previus_stop_loss_price




class AtrTrailingBasedStopLoss():
    from config.param_schemas.risk_schemas import AtrTrailingBasedSLParams
    def __init__(self, stop_loss_params: AtrTrailingBasedSLParams):
        self.params = stop_loss_params
        self.is_dynamic = self.params.is_dynamic
    
    # trailing_based_stop_loss
    def sl_function(self, data, index, entry_index, entry_price, direction, entry_time=None, previus_stop_loss_price=None):
        """
        Izračuna trailing stop-loss ceno na podlagi ATR premika od najvišje dosežene cene (za long pozicije) oziroma najnižje (za short pozicije).

        Parametri:
        ----------
        df : pd.DataFrame
            DataFrame s podatki o cenah, ki vsebuje stolpce z zgodovino cen za izračun trailing stop lossa.
            Pomembno: df mora vsebovati podatke do trenutka, za katerega se izračunava stop loss,
            brez vključevanja podatkov, ki so kasnejši od tega datuma (brez 'gledanja v prihodnost').
        entry_price : float
            Cena, po kateri je bila odprta pozicija.
        direction : int
            Smer pozicije: 1 za long, -1 za short.
        previus_stop_loss_price : float ali None, privzeto None
            Ni potreben! Potreben je samo kadar primerjamo prejšen stop_loss z trenutnim


        Vrne:
        -------
        float
            Izračunana cena trailing stop lossa, ki je vedno 1 % pod najvišjo ceno (za long) ali 1 % nad najnižjo ceno (za short), glede na gibanje trga in smer pozicije.
        """


        atr_period = self.params.atr_period
        atr_multiplier = self.params.atr_multiplier

        # atr  = ta.atr(df["High"], df["Low"], df["Close"], length=atr_period).iloc[-1]

        # --- Initial Stop-Loss Calculation (when a trade starts) ---
        if previus_stop_loss_price is None:
            if direction == 1: # Long trade
                stop_loss_price = entry_price - (data["atr_rm"][index] * atr_multiplier)
            elif direction == -1: # Short trade
                stop_loss_price = entry_price + (data["atr_rm"][index] * atr_multiplier)
            return stop_loss_price

        # --- Trailing Stop-Loss Update (for an active trade) ---
        else:
            if direction == 1: # Long trade
                # Find the highest price since the trade started
                peak_price = data["High"].max()
                # Calculate the new potential stop-loss
                new_stop_loss = peak_price - (data["atr_rm"][index] * atr_multiplier)
                # The stop can only move up, never down. So we take the higher of the two values.
                return max(previus_stop_loss_price, new_stop_loss)

            elif direction == -1: # Short trade
                # Find the lowest price since the trade started
                bottom_price = data["Low"].min()
                # Calculate the new potential stop-loss
                new_stop_loss = bottom_price + (data["atr_rm"][index] * atr_multiplier)
                # The stop can only move down, never up. So we take the lower of the two values.
                return min(previus_stop_loss_price, new_stop_loss)

        return previus_stop_loss_price
    


class RRRBasedStopLoss():
    from config.param_schemas.risk_schemas import RRRBasedSLParams
    """
    RiskReward: 2
    """

    def __init__(self, stop_loss_params: RRRBasedSLParams):
        self.params = stop_loss_params
        self.is_dynamic = self.params.is_dynamic
    
    def sl_function(self, data, index, entry_index, entry_price, direction, entry_time=None, previus_stop_loss_price=None):
        """
        Calculates the dynamic Stop Loss target price, based on value of 
        the Moving Average (the 'mean').
        """

        if "mr_target" not in data:
            raise ValueError("MA for Risk Managment (tp, sl calculation) data not found. Ensure 'data' dictionary contains 'mr_target' key.")
        

        ma_price = data["mr_target"][index]
        price = entry_price
        dist_to_mean = abs(entry_price - ma_price)

        muiltiplier = self.params.rrr

        sl_price = price - direction * (dist_to_mean/muiltiplier)
        return sl_price
    

class FvgBasedStopLoss():
    from config.param_schemas.risk_schemas import FvgBasedSLParams
    """
    A stop-loss strategy that places the stop at the bottom of a recently
    formed Fair Value Gap (FVG).
    """
    def __init__(self, stop_loss_params:FvgBasedSLParams):
        self.params = stop_loss_params
        self.is_dynamic = self.params.is_dynamic

    def sl_function(self, data: dict, index, entry_index, entry_price: float, direction: int, entry_time=None, previus_stop_loss_price=None, fvg_level=None, **kwargs):
        """
        Calculates the stop-loss price based on the FVG boundary corresponding
        to the trade direction.

        This function expects 'fvg_bottom' and 'fvg_top' arrays in the 'data'
        dictionary, pre-calculated by the DataPreprocessor.
        """
        # stop_loss_price = np.nan

        # For a LONG trade, the fvg_level should be the bottom of a bullish FVG.
        if direction == 1 and fvg_level is not None:
            return fvg_level 
        
        # For a SHORT trade, the fvg_level should be the top of a bearish FVG.
        elif direction == -1 and fvg_level is not None:
            return fvg_level

        print(f"Warning: FvgBasedStopLoss did not receive a valid fvg_level for direction {direction} at entry time {entry_time}.")
        return np.nan
    

class RRRBasedTrailingStopLoss():
    from config.param_schemas.risk_schemas import RRRBasedTrailingSLParams
    def __init__(self, stop_loss_params: RRRBasedTrailingSLParams):
        self.params = stop_loss_params
        self.is_dynamic = self.params.is_dynamic

        self.rrr = self.params.rrr
        self.target_col = getattr(self.params, "target_col", "mr_target")

    def sl_function(self, data: dict, index, entry_index, entry_price: float, direction: int, entry_time=None, previus_stop_loss_price=None, fvg_level=None, **kwargs):

        if self.target_col not in data:
            raise ValueError(f"RRRBasedTrailingStopLoss: Target_col: {self.target_col} NOT found in data.")


        target_price = data[self.target_col][index]
        high = data["High"][index]
        low = data["Low"][index]


        if previus_stop_loss_price is None:
            reward = abs(target_price - entry_price)
            risk = reward / self.rrr

            if direction == 1:  # long
                return entry_price - risk
            else:  # short
                return entry_price + risk



        if direction == 1:  # LONG
            reward_now = abs(target_price - high)
            risk_now = reward_now / self.rrr
            new_sl = high - risk_now

            # SL cannot move downward (for long)
            return max(new_sl, previus_stop_loss_price)

        else:  # SHORT
            reward_now = abs(target_price - low)
            risk_now = reward_now / self.rrr
            new_sl = low + risk_now

            # SL cannot move upward (for short)
            return min(new_sl, previus_stop_loss_price)
    


class FixedRRRBasedTrailingStopLoss():
    from config.param_schemas.risk_schemas import FixedRRRBasedTrailingSLParams
    def __init__(self, stop_loss_params: FixedRRRBasedTrailingSLParams):
        self.params = stop_loss_params
        self.is_dynamic = self.params.is_dynamic

        self.target_col = self.params.target_col
        self.rrr = self.params.rrr

        self.fixed_risk_width = {}

    def sl_function(self, data: dict, index, entry_index, entry_price: float, direction: int, entry_time=None, previus_stop_loss_price=None, fvg_level=None, **kwargs):

        if self.target_col not in data:
            raise KeyError(f"Target column '{self.target_col}' missing in data keys: {list(data.keys())}")


        trade_key = (entry_price, entry_time)
        current_price = data["Close"][index]
        
        # --- 1. DETERMINE THE FIXED WIDTH (Calculated ONCE per trade) ---
        if trade_key not in self.fixed_risk_width:
            
            if entry_index is None:
                return np.nan # Cannot calculate fixed width without entry index
                
            # Target price at the moment the trade opened
            target_at_entry = data[self.target_col][entry_index]
            
            # Initial distance calculated at entry time
            dist_at_entry = abs(entry_price - target_at_entry)
            
            if dist_at_entry <= 0:
                return np.nan 

            # Calculate Fixed Risk Width (This value never changes)
            fixed_width = dist_at_entry / self.rrr
            self.fixed_risk_width[trade_key] = fixed_width

        fixed_width = self.fixed_risk_width[trade_key]
        
        # --- 2. TRAIL THE CURRENT PRICE BY THAT FIXED WIDTH ---
        
        if direction == 1: # Long
            new_sl = current_price - fixed_width
        else: # Short
            new_sl = current_price + fixed_width

        # --- 3. Trailing logic — SL only tightens (Uses previous SL) ---
        previus_stop_loss_price = kwargs.get("previus_stop_loss_price")
        
        if previus_stop_loss_price is None:
            return new_sl
        
        if direction == 1:
            # Long: SL moves up (Max of previous and new)
            return max(previus_stop_loss_price, new_sl)
        else:
            # Short: SL moves down (Min of previous and new)
            return min(previus_stop_loss_price, new_sl)

