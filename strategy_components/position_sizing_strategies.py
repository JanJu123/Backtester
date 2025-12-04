

class KellyPositionSizing():
    def __init__(self, pos_params):
        self.pos_params = pos_params

    # kelly_position_size
    def pos_function(self, capital: float = None):
        """
        Izračuna velikost pozicije na podlagi Kellyjevega kriterija z možnostjo prilagoditve agresivnosti.

        Parametri:
        ----------
        params : dict
            - 'win_rate': float
                Verjetnost dobičkonosne pozicije (npr. 0.55 pomeni 55 %).
            - 'avg_win': float
                Povprečni dobiček na dobičkonosni poziciji.
            - 'avg_loss': float
                Povprečna absolutna izguba na izgubljeni poziciji.
            - 'kelly_fraction_use': float, neobvezno
                Delež Kellyjevega kriterija, ki ga želimo uporabiti (med 0 in 1), privzeto 1 (polni Kelly).

        capital : float
            Skupni razpoložljivi kapital, potreben za izračun velikosti pozicije.

        Vrne:
        -------
        float
            Velikost pozicije (znesek kapitala, ki ga tvegamo pri eni poziciji).
        """
        
        win_rate = self.pos_params.win_rate
        avg_win = self.pos_params.avg_win
        avg_loss = self.pos_params.avg_loss
        kelly_fraction_use = self.pos_params.kelly_fraction_use

        # Preverimo za mankajoče podatke
        if capital is None:
            raise ValueError("Capital was not given, Capital is needed to calculate position size")
        if avg_loss == None or win_rate is None or avg_win is None or avg_loss is None:
            raise ValueError(f"Missing params: \n win_rate: {win_rate}\n avg_win: {avg_win}\n avg_loss: {avg_loss}\n kelly_fraction_use: {kelly_fraction_use}")


        reward_risk = avg_win / abs(avg_loss)
        raw_kelly_fraction = win_rate - (1 - win_rate) / reward_risk
        raw_kelly_fraction = max(0, raw_kelly_fraction)  # prevent negative sizing

        # Apply user-defined fraction of Kelly
        final_kelly_fraction = raw_kelly_fraction * kelly_fraction_use
        position_size = capital * final_kelly_fraction

        return position_size

class FixedPctPositionSizing():
    def __init__(self, pos_params):
        self.params = pos_params

    # fixed_pct_position_sizing
    def pos_function(self,  risk_per_unit: float, capital: float = None):
        """
        Izračuna velikost pozicije glede na določen odstotek tveganja kapitala.

        Parametri:
        ----------
        params : dict
            - 'percentage_to_risk': float
                Delež kapitala, ki ga želimo tvegati (npr. 0.01 pomeni 1 % kapitala).

        capital : float
            Skupni razpoložljivi kapital, potreben za izračun velikosti pozicije.
        
        risk_per_unit: float
            Kako daleč je stop loss od trenutne cene

        Vrne:
        -------
        float
            Velikost pozicije na osnovi določenega odstotka tveganja kapitala.
        """

        if capital is None:
            raise ValueError("Capital was not given, Capital is needed to calculate position size")
        
        
        percentage_to_risk = self.params.percentage_to_risk
        capital_to_risk = capital * percentage_to_risk
        position_size = capital_to_risk / risk_per_unit

        if capital_to_risk == capital:
            print("aaaaaa, errorrrr")
            exit(1)


        return position_size


#TODO: Dodaj opis v documentation.md
class FixedSharePositionSizing():
    def __init__(self, pos_params):
        self.params = pos_params

    # fixed_share_position_sizing
    def pos_function(self, capital: float = None, risk_per_unit: float = None):
        """
        Izračuna velikost pozicije glede na določeno velikost uzgube v najslabšem primeru, če cena zadene stop_loss.
        """

        if capital is None:
            raise ValueError("Capital was not given, Capital is needed to calculate position size")
        
        num_shares = self.params.number_of_shares
        return num_shares


#TODO: Dodaj še kakšne druge position sizing strategije