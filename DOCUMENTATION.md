# Backtesting Framework
## About the project
Projekt je nastal za lažje izdelovanje, testiranje in razvijanje strategij. Program deluje kot Backtester, ki že obstajajo, ampak je narejen po osebnih preferencah.

## Installation
Najprej je potrebno namestiti vse knjižnice, ki se nahajajo v `requirements.txt`, med drugim vključuje:
- `pandas`
- `datetime`
- `matplotlib`
- `ta`
- `optuna`
- ...


Namestitev poteka z ukazom:

```bash
pip install -r requirements.txt
```

## Usage
Glavni del projekta se nahaja v main.py, ki skrbi za povezovanje vseh funkcij v enoten backtesting sistem.

Za delovanje, main.py potrebuje:
- Vhodne podatke (OHLC — Open, High, Low, Close)
- Parametre za strategijo (npr. RSI, MA itd.)
- Parametre za stop loss
- Parametre za take profit
- Position sizing parametre
- Import funkcij iz ostalih .py datotek (modularna struktura)


## Modular Files
Vsaka komponenta sistema je razdeljena v svojo datoteko:

- <b>stop_loss_strategies.py: </b>
Funkcije za izračun različnih strategij stop loss (fixed %, ATR, MA...).

- <b>take_profit_strategies.py:</b>
Funkcije za določitev take-profit ciljev (fixed %, ATR, MA...).

- <b>position_sizing.py:</b>
Funkcije za izračun velikosti pozicije (kelly, ...).

- <b>optimization.py :</b>
Uporablja Optuna za iskanje najboljših parametrov glede na Sharpe razmerje, Win rate, ipd.

- <b>utils.py :</b>
Funkcije, ki so uporabljene v drugih datotekah, zato da so lažje dostopna in se ne ponavljajo. (npr. calculate_ATR, calculate_RSI, exit_trade, ...)

- <b>stats.py :</b>
Class, ki izračuna vse pomembne matrice/statistike o strageji. (win rate, RR, ...)

- <b>backtest.py :</b>
Vsebuje funkcije, ki so uporabljene za backtesting. Združuje stop_loss, take_profit in position sizing v backtest funkcijo.

<br><br>
# File descriptions
V tem delu opisujemo, vse datoteke/mape, ki jih vsebuje projekt.

## Strategies
To je mapa, ki vsebuje datoteke z vsemi strategijami za generairanje signalov, za vsako strategijo obstaja svoja datoteka, ki nam omogoča razvijanje in izolirano testiranje delovanja.  

### RSI_2p.py
Vsebuje funkcije, ki so uporabljene za generaranje signalov z, 2 periodim RSI indikatorjem. Potrebuje pa:  
- Dataframe, ki vsebuje OHLC podatke o instrumentu
- Parametre: ( dict )
    - "oversold" (float): spodnja meja RSI za sell signal.
    - "overbought" (float): zgornja meja RSI za buy signal.
    - "rsi_period" (int): Koliko nazaj naj gleda nazaj, za izračunavo RSI (deafult: 14)
    - "comulative_window":(int) Koliko RSI vrendosti naj sešteje (deafult: 2)

Datoteka, vsebuje 3 funkcije:
```python
def calculate_cumulative_RSI(df, rsi_period: int = 14, cumulative_window: int = 2, price_col: str = "Close"):
# Uporabljena je za izračunavo comulative RSI
# DataFrame potrebuje vsaj Close vrednosti/cene
```
```python
def strategy_RSI_2p_first_stage(df, params: dict = {}):
# Uporabljena je za generiranje signalov, buy (1) ko je cena pod oversold oz. sell (-1) ko je cena nad overbought tresholdom
# V tej funkciji potrebujemo samo Date in Close
```
```python
def strategy_RSI_2p_second_stage(df, params: dict = {}):
# Uporabljena je za generiranje signalov, ko je cena pod oversold oz. ko je cena nad overbough tresholdom. V tej funkciji, pa se generirajo signali samo če:
# Buy (1) cumulative RSI < oversold in cena > 200_MA  
# Sell (-1): cumulative RSI > overbought in cena < 200_MA
# V tej funkciji potrebujemo tudi 200_MA, poleg Close
```
Primer parametrov, za podani funkciji:
```python
params = {
    "oversold": 50,   # "oversold": RSI_2p.params["oversold"],
    "overbought": 180,  # RSI_2p.params["overbought"]
    "rsi_period": 6,
    "cumulative_window": 2,
}
```
### Opis drugih funkcij: prihodnje...

<br>

## stop_loss_strategies.py
Vsebuje funkcije, ki so uporabljene za določevanje/nastavljanje stop-lossa, ki je uporabljen za omejevanje izgub, če gre instrument v nasprotno smer kot bi si to želeli.  
Vsebuje naslednje funkcije:

```python
def fixed_pct_stop_loss(entry_price, direction, entry_time, df=None, stop_loss_params=None, previus_stop_loss_price=None):
# Uporabljen za izračunavo fiksnega stop lossa s %
# Stop_loss_params morajo vsebovati:
stop_loss_params = {
    "stop_loss_pct": 0.01, # 1%
    "is_dynamic": True, # Ali se mora stop loss spreminjati vsak nov candle ali ne
}
```

```python
def atr_based_stop_loss(df, entry_price, direction, entry_time=None, stop_loss_params=None, previus_stop_loss_price=None):
# Uporabljen za izračunavo fiksnega stop lossa s %
# Stop_loss_params morajo vsebovati:
stop_loss_params = {
    "stop_loss_pct": 0.01, # 1%
    "is_dynamic": True, # Ali se mora stop loss spreminjati vsak nov candle ali ne
}
```

```python
def trailing_based_stop_loss(df, entry_price, direction, entry_time=None, stop_loss_params=None, previus_stop_loss_price=None):
# Uporabljen za izračunavo stop lossa z uporabo trailing stopa, stop loss se premika glede na High oz. Low od časa ko smo kupili intrument.
# Stop_loss_params morajo vsebovati:
stop_loss_params = {
    "trail_pct": 0.0005, # Kako odaljen je stop loss od High-a oz. Low-a
    "is_dynamic": True, # Ali se mora stop loss spreminjati vsak nov candle ali ne
}
```
Vsaka stop-loss funkcija potrebuje naslednje parametre:
- <b>df</b>: Vsebuje Dataframe, z Datetime/Date in OHLC podatke
- <b>entry_price</b>: (float) Cena, za katero izračunavamo stop_loss
- <b>direction</b>: (int) Ali so Long ali pa Short. Long(1) in Short(-1). Pomembno za pozicijo stop-lossa
- <b>entry_time</b>: Datetime/Date, za katerega izračunavamo stop_loss. Potreben je samo pri določenih funkcijah (trailing_based_stop_loss, ...)
- <b>stop_loss_params</b>: (dict) Vsebuje vse potrebne za izbrano stop-loss strategijo. Kaj mora vsebovati je bilo razloženo malo prej. Če parametri niso podani, se uporabijo default nastavitve, za določeno strategijo/funkcijo
- <b>previus_stop_loss_price</b>: (float) Cena oz. postavitev prejšnega stop-lossa, če smo ga sploh imeli. Potreben je samo pri določenih funkcijah (trailing_based_stop_loss, ...)

Primer 1h timeframa za DataFrame, ki potrebuje ga df:  
(To so podatki, ki jih dobi funkcija v df, ampak potrebni so samo OHLC potaki in Datetime, za izračun pozicije stop-lossa)
```excel
                  Datetime    Open    High     Low   Close  tick_volume  spread  real_volume     200_MA  cumulative_rsi  signal
0      2015-01-02 00:00:00  2063.0  2073.0  2044.7  2053.0         9258       0            0        NaN             NaN       0
1      2015-01-05 00:00:00  2053.1  2054.0  2016.0  2024.5        11893       0            0        NaN             NaN       0
2      2015-01-06 00:00:00  2024.7  2030.2  1991.5  2001.7        14371       0            0        NaN             NaN       0
3      2015-01-07 00:00:00  2001.1  2030.0  2000.9  2027.0        11994       0            0        NaN             NaN       0
4      2015-01-08 00:00:00  2027.2  2064.8  2026.7  2059.8         9008       0            0        NaN             NaN       0
...                    ...     ...     ...     ...     ...          ...     ...          ...        ...             ...     ...
35887  2022-01-19 16:00:00  4601.4  4611.1  4592.8  4609.9         4809      20            0  4435.7628      200.000000       0
35888  2022-01-19 17:00:00  4609.9  4610.9  4571.0  4578.9         5860      20            0  4435.7628      141.176471       0
35889  2022-01-19 18:00:00  4578.8  4584.5  4563.5  4583.3         5612      20            0  4435.7628       84.399914       0
35890  2022-01-19 19:00:00  4583.3  4598.4  4583.0  4596.9         4281      20            0  4435.7628       96.954787       0
35891  2022-01-19 20:00:00  4597.0  4597.5  4577.0  4580.2         4349      20            0  4435.7628       96.399613       0

[35892 rows x 11 columns]
```

<br>

## take_profit_strategies.py
Vsebuje funkcije, ki so uporabljene za določevanje/nastavljanje take-profita, ki je uporabljen za exitanje trejdov, ko so bili naši cilji doseženi.
Vsebuje naslednje funkcije:


```python
def fixed_pct_take_profit(entry_price, direction, df=None, take_profit_params=None, previus_take_profit_price=None):
# Uporabljen za izračunavo fiksnega take-profita s %
# take_profit_params morajo vsebovati:
stop_loss_params = {
    "take_profit_pct": 0.01, # 1%
    "is_dynamic": False, # Ali se mora stop loss spreminjati vsak nov candle ali ne
}
```

```python
def atr_based_take_profit(df, entry_price, direction, take_profit_params=None, 
previus_take_profit_price=None):
# Uporabljen za izračunavo take-profita z uporablo ATR (Average True Range) indikatorja, ki nam pove koliko se cena premika v denarju
# take_profit_params morajo vsebovati:
stop_loss_params = {
    "atr_period": 10,, # (int) število obdobij za izračun ATR (privzeto 5),
    "atr_multiplier": 1 # (float) faktor za razdaljo take-profita glede na ATR (privzeto 1).
    "is_dynamic": False, # Ali se mora stop loss spreminjati vsak nov candle ali ne
}
```

```python
def ma_cross_take_profit(df, entry_price=None, direction=None, take_profit_params=None,ž
previus_take_profit_price=None):
# Uporabljen za določitev take-profita, ko cena prečka MA (Moving Average)
# take_profit_params morajo vsebovati:
stop_loss_params = {
    "ma_period": 10, # (int) obdobje za MA (Moving Average),  (privzeto 5).
    "is_dynamic": False, # Ali se mora stop loss spreminjati vsak nov candle ali ne
}
```
### Opis drugih funkcij/datotek: prihodnje...

Vsaka take-profit funkcija potrebuje naslednje parametre:
- <b>df</b>: Vsebuje Dataframe, z Datetime/Date in OHLC podatke
- <b>entry_price</b>: (float) Cena, za katero izračunavamo take_profit
- <b>direction</b>: (int) Ali so Long ali pa Short. Long(1) in Short(-1). Pomembno za pozicijo take_profita
- <b>entry_time</b>: Datetime/Date, za katerega izračunavamo take_profit. Potreben je samo pri določenih funkcijah (...)
- <b>take_profit_params</b>: (dict) Vsebuje vse potrebne za izbrano take_profit strategijo. Kaj mora vsebovati je bilo razloženo malo prej. Če parametri niso podani, se uporabijo default nastavitve, za določeno strategijo/funkcijo
- <b>previus_take_profit_price</b>: (float) Cena oz. postavitev prejšnega take-profita, če smo ga sploh imeli. Potreben je samo pri določenih funkcijah (...)

Primer 1h timeframa za DataFrame, ki potrebuje ga df:  
(To so podatki, ki jih dobi funkcija v df, ampak potrebni so samo OHLC potaki in Datetime, za izračun pozicije take_profita)
```excel
                  Datetime    Open    High     Low   Close  tick_volume  spread  real_volume     200_MA  cumulative_rsi  signal
0      2015-01-02 00:00:00  2063.0  2073.0  2044.7  2053.0         9258       0            0        NaN             NaN       0
1      2015-01-05 00:00:00  2053.1  2054.0  2016.0  2024.5        11893       0            0        NaN             NaN       0
2      2015-01-06 00:00:00  2024.7  2030.2  1991.5  2001.7        14371       0            0        NaN             NaN       0
3      2015-01-07 00:00:00  2001.1  2030.0  2000.9  2027.0        11994       0            0        NaN             NaN       0
4      2015-01-08 00:00:00  2027.2  2064.8  2026.7  2059.8         9008       0            0        NaN             NaN       0
...                    ...     ...     ...     ...     ...          ...     ...          ...        ...             ...     ...
35887  2022-01-19 16:00:00  4601.4  4611.1  4592.8  4609.9         4809      20            0  4435.7628      200.000000       0
35888  2022-01-19 17:00:00  4609.9  4610.9  4571.0  4578.9         5860      20            0  4435.7628      141.176471       0
35889  2022-01-19 18:00:00  4578.8  4584.5  4563.5  4583.3         5612      20            0  4435.7628       84.399914       0
35890  2022-01-19 19:00:00  4583.3  4598.4  4583.0  4596.9         4281      20            0  4435.7628       96.954787       0
35891  2022-01-19 20:00:00  4597.0  4597.5  4577.0  4580.2         4349      20            0  4435.7628       96.399613       0

[35892 rows x 11 columns]
```

<br>

## position_sizing.py
Vsebuje funkcije za izračun velikosti pozicije pri odpiranju trejdov, glede na tveganje in izbrano strategijo pozicioniranja (npr. fiksni znesek, Kelly formula, odstotek tveganja itd.). Pravilno določanje velikosti pozicije pomaga uravnavati tveganje in optimizirati donosnost strategije.  
Vsebuje naslednje funkcije:

### Prihodnje bodo dodane funkcije in opisi



<br>

## utils.py
Vsebuje pogosto uporabljene funkcije, ki se uporabljajo v več delih projekta (npr. calculate_ATR, calculate_RSI, exit_trade ...). Na ta način se izognemo podvajanju kode, omogočimo lažje vzdrževanje ter hitrejše spreminjanje logike na enem mestu. Vsebuje naslednje funkcije:
- <b>calculate_ATR</b>: Izračuna Average True Range (ATR) za podan DataFrame.
- <b>calculate_RSI</b>: Izračuna  Relative Strength Index (RSI) za podan DataFrame.
- <b>calculate_cumulative_RSI</b>: Izračuna cumulative Relative Strength Index (RSI) kot vsoto preteklih N RSI vrednosti.
- <b>exit_trade</b>: Exita trade in odstrani vse enake signale v nadaljevanju, ki se ponavljajo in vrne nazaj df, ki vsebuje vse pomembne podatke.
Opis funkcij:
```python
def calculate_ATR(df, period: int=5):
    """
    Izračuna Average True Range (ATR) za podan DataFrame.

    :param df: ["High", "Low", "Close"]

    :param period: Koliko candlov naj bo uproabljenih  za MA ATR

    Returns:
        pd.Series: ATR values.

    """
```

```python
def calculate_RSI(df, period: int=14, price_col: str = "Close"):
    """
    Izračuna  Relative Strength Index (RSI) za podan DataFrame.

    :param df (pd.DataFrame): DataFrame containing at least a 'Close' price column.
    :param period (int): Lookback period for RSI. Default is 14.
    :param price_col (str): Name of the column to use for price. Default is "Close".

    Returns:
        pd.Series: RSI values.
    """
```

```python
def calculate_cumulative_RSI(df, rsi_period: int = 14, cumulative_window: int = 2, price_col: str = "Close"):
    """
    Izračuna cumulative Relative Strength Index (RSI) kot vsoto preteklih N RSI vrednosti.

    :param df (pd.DataFrame): DataFrame z vsaj ceno ('Close' oz. določeno v price_col).
    :param rsi_period (int): Koliko candlov se upošteva pri izračunu RSI. Privzeto 14.
    :param cumulative_window (int): Koliko zadnjih RSI vrednosti se sešteje. Privzeto 2.
    :param price_col (str): Katera cena naj se uporablja za izračun RSI. Privzeto "Close".

    Returns:
        pd.Series: Cumulative RSI (vsota preteklih RSI vrednosti).
    """
```

```python
def exit_trade(df, entry_price, exit_price, entry_time, exit_time, stop_loss_price, direction, index, signal, reason):
    """
    Exita trade in odstrani vse enake signale, ki se ponavljajo in vrne nazaj df, ki vsebuje vse pomembne podatke:
    {
        "Datetime": exit_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_loss_price": stop_loss_price,
        "cumulative_rsi": df["cumulative_rsi"].iloc[index],
        "signal": direction,
        "pnl": profit,
        "reason": reason,
    }

    :param df (pd.DataFrame): DataFrame z vsaj ceno ('Close' oz. določeno v price_col).
    :param entry_price (float): Cena, ko smo entrali v trade
    :param exit_price (float): Cena, ko smo exitali trade
    :param entry_time: Date/Datetime, ko smo entrali trade
    :param exit_time: Date/Datetime, ko smo exitali trade
    :param stop_loss_price (float): Cena/Pozicija stop_lossa
    :param direction (int): Ali smo šli Long (1) ali Short (-1)
    :param index (int): Trenutni index, za katerega želimo exitati
    :param signal (int): 1 (buy) , -1(sell)
    :param reason (str): Razlog, zakaj smo exitali (stop_loss_hit ...) za lažjo analizo in debuganje

    Returns:
        {"df": df, "trade_info": df_trade, "profit": profit}

    df: Df ki vsebuje spremembe,
    trade_info: Vsebuje vse pomembne podatke za kasnejšo analizo
    profit: Profit tega trejda
    """
```

### Prihodnje bo dodanih več funkcij in opisov


<br>

## stats.py
Vsebuje class, ki omogoča analizo trejdov. Izračuna razne matrice/rezultate, kot so (win rate, RR, total profit, expectancy, holding time, ...). Class se imenuje `PerformanceEvaluator`. Omogoča analizo in vizualizacijo rezultata in drawdowna.
Vsebuje 2 pomembi funkciji:
- <b>summary</b>: Izračuna vse pomembne podatke in nam jih vrne v dict. Uporablja naslednje funkcije:
    - <b>total_profit</b>: Vrne profit v $ oz. €
    - <b>total_return</b>: Vrne profit v % glede na začetni kapital
    - <b>average_return</b>: Vrne povprečni gain na trade
    - <b>sharpe_ratio</b>: Vrne Sharpe ratio, ki pomeni kako volatilen je naš "profit", equity curve
    - <b>max_drawdown</b>: Vrne velikost drawdowna v $ oz. €
    - <b>max_drawdown_pct</b>: Vrne velikost drawdowna v %
    - <b>win_rate</b>: Vrne win rate
    - <b>max_loss</b>: Vrne največjo izgubo v enem trejdu v $ oz. €
    - <b>num_trades</b>: Vrne število trejdov
    - <b>num_buy_trades</b>: Vrne število Long/Buy trejdov
    - <b>num_sell_trades</b>: Vrne število Short/Sell trejdov
    - <b>avg_holding_time</b>: Vrne povprečni čas, ki ga je porabil trejd
    - <b>min_holding_time</b>: Vrne najkrajši čas, ki ga je porabil trejd
    - <b>max_holding_time</b>: Vrne najdaljši čas, ki ga je porabil trejd
    - <b>expectancy</b>: Vrne expectancy, ki pomeni povprečni gain na trade
    - <b>average_r_multiplier</b>: Vrne povprečen R multiplier, ki pomeni koliko dobimo glede na riskirano vsoto
    - <b>reasons_for_exit</b>: Vrne reason zakaj smo exitali trejd (stop_loss_hit, ...)

- <b>plot_performance</b>: Omogoča vizualizacijo podatkov (Graf prikazuje equity curve, drawdonws, ...).


Summary returna naslednje podatke (primer):  
```python
total_profit: 1650.7859500000081
total_return: 165.0785950000008
avg_return: 1.2659401457055277
sharpe: 0.24351734617446558
max_drawdown: -28.673699999996643
max_drawdown_pct: -2.063302266436262
win_rate: 0.3895705521472393
risk_reward_ratio: 3.74987301643093
max_win: 44.899999999999636
max_loss: -2.393949999999677
num_trades: 1304
num_buy_trades: 1304
num_sell_trades: 0
avg_holding_time_hours: 1.7453987730061349
min_holding_time_hours: 1.0
max_holding_time_hours: 54.0
expectancy: 1.2557486262579873
avg_r_multiple: 0.8006982253174545
exit_reason: reason
stop_loss_hit    762
signal_0_exit    542
Name: count, dtype: int64
```

<br>

## backtest.py
Vsebuje 3 funkcije, ki so razdeljne po stopnjah. To so:
- <b>`backtest_first_stage`</b>: Izvede osnovno backtestiranje strategije na podlagi signalov buy/sell. Exita trade, ko je signal 0
- <b>`backtest_second_stage`</b>: Izvede backtest strategije z možnostjo različnih tipov stop lossa. Exita trade, ko je signal 0
- <b>`backtest_third_stage`</b>: Izvede backtest strategije z možnostjo različnih tipov stop-lossa in take-profita, z dodatno podporo dinamičnemu prilagajanju stop-loss in take-profit cen na vsakem candlu.   
Exita trade, ko je dosežen stop-loss ali take-profit, pri čemer se lahko ti nivoji na vsakem candlu dinamično preračunajo z ustrezno funkcijo.



```python
def backtest_first_stage(df, capital=1000):
    """
    Izvede osnovno backtestiranje trgovinske strategije na podlagi signalov nakupa/prodaje.
    Exita trade, ko je signal 0

    Parametri:
    -----------
    df : pd.DataFrame   vsaj: ["Datetime", "Open", "signal", "cumulative_rsi"]
        DataFrame, ki vsebuje zgodovinske podatke o cenah in signalih. 
        Mora vsebovati naslednje stolpce:
        - 'Datetime' : časovni žig vsakega zapisa (tip datetime ali pretvorljiv v datetime)
        - 'Open' : začetna cena (odpiranje) za posamezen časovni interval
        - 'signal' : trgovalni signal (1 = nakup, -1 = prodaja, 0 = brez pozicije)
        - 'cumulative_rsi' : dodatni indikator (RSI), ki se lahko uporablja za analizo (ni nujno za osnovno logiko)
    
    capital : float, privzeto 1000
        Začetni kapital, s katerim začnemo trgovati.

    Kaj funkcija počne:
    -------------------
    - Prebere podatke iz `df` in na podlagi signalov odpre ali zapre pozicije.
    - Ko se signal spremeni iz 0 v 1 ali -1, se odpre nova pozicija (long ali short).
    - Ko se signal vrne na 0, se pozicija zapre in izračuna se dobiček ali izguba (PnL).
    - Beleži čas entry in exita, cene entry in exita, ter druge informacije.
    - Sledi kumulativnemu kapitalu po vsaki zaprti trgovini.
    - Privzame pozicijsko velikost 1 za vse posle (to se lahko kasneje nadgradi).

    Zakaj se uporablja:
    -------------------
    Ta funkcija služi kot osnovni "prvi korak" backtestiranja, kjer želimo preveriti osnovno logiko trgovanja na podlagi signalov, brez kompleksnih nastavitev pozicijske velikosti, stop-lossov ipd.
    Omogoča hitro preverjanje, kako bi strategija delovala na zgodovinskih podatkih.

    Vrnitev:
    --------
    pd.DataFrame
        DataFrame z zabeleženimi posameznimi trgovinami, ki vsebuje stolpce:
        - 'Datetime': čas zaprtja trgovine,
        - 'entry_time': čas odprtja pozicije,
        - 'exit_time': čas zaprtja pozicije,
        - 'entry_price': cena ob odprtju pozicije,
        - 'exit_price': cena ob zaprtju pozicije,
        - 'cumulative_rsi': vrednost indikatorja RSI ob zaprtju,
        - 'signal': smer pozicije (1 ali -1),
        - 'pnl': dobiček/izguba posamezne trgovine,
        - 'cumulative_capital': kapital po zaključenih trgovinah,
        - 'positions_size': velikost pozicije (trenutno fiksna 1).
    """
```

```python
def backtest_second_stage(df, capital=1000, stop_loss_func=None, stop_loss_params=None):
"""
    Izvede backtest  strategije z možnostjo različnih tipov stop lossa in prilagodljivo velikostjo pozicije (position sizing).
    Exita trade, ko je signal 0 ali ko je dosežen stop loss.

    :param df: pd.DataFrame
        DataFrame s podatki o trgovanju, ki mora vsebovati stolpce kot so 'Datetime', 'Open', 'High', 'Low', 'signal', 'cumulative_rsi'.
    capital : float, privzeto 1000
        Začetni kapital za backtest.
    stop_loss_func : callable ali None, privzeto None
        Funkcija, ki določa logiko stop lossa. Lahko je funkcija, ki sprejme vhodne parametre, kot so entry_price, direction itd.
        Če ni podana, se stop loss ne upošteva.
    stop_loss_params : dict ali None, privzeto None
        Dodatni parametri, ki se posredujejo funkciji stop_loss_func.
    position_sizing_func : callable ali None, privzeto None
        Funkcija, ki določa logiko velikosti pozicije glede na kapital in tveganje na enoto. Sprejme parametre kapitala in dodatne parametre.
    position_sizing_params : dict ali None, privzeto None
        Dodatni parametri, ki se posredujejo funkciji position_sizing_func.
    use_position_sizing : bool, privzeto False
        Določa, ali se uporablja dinamični izračun velikosti pozicije (True) ali fiksna velikost 1 (False).

    Opis:
    ------
    Funkcija omogoča dinamično uporabo različnih strategij stop lossa in prilagodljivo velikost pozicije (position sizing).
    Če je podana funkcija stop_loss_func, se le-ta uporabi za izračun stop loss cene.
    Če je use_position_sizing=True, se velikost pozicije izračuna glede na definirano funkcijo position_sizing_func in parametre.
    Če ni podana funkcija stop_loss_func ali position_sizing_func, se sproži napaka.
    To omogoča enostavno testiranje različnih pristopov in kasnejšo nadgradnjo.

    :return:
    
    pd.DataFrame
        DataFrame z zabeleženimi posameznimi trgovinami, ki vsebuje stolpce:
        - 'Datetime': čas zaprtja trgovine,
        - 'entry_time': čas odprtja pozicije,
        - 'exit_time': čas zaprtja pozicije,
        - 'entry_price': cena ob odprtju pozicije,
        - 'exit_price': cena ob zaprtju pozicije,
        - 'stop_loss_price': stop loss cena,
        - 'position_size': velikost pozicije (dinamično izračunana ali 1),
        - 'position_value': vrednost pozicije ob odprtju,
        - 'cumulative_rsi': vrednost indikatorja RSI ob zaprtju,
        - 'signal': smer pozicije (1 ali -1),
        - 'pnl': dobiček/izguba posamezne trgovine,
        - 'cumulative_capital': kapital po zaključenih trgovinah,
        - 'reason': razlog za zaprtje pozicije (stop_loss_hit, signal_0_exit itd.).
    """
```

```python
def backtest_third_stage(df, capital:float=1000, stop_loss_func=None, stop_loss_params=None, take_profit_func=None, take_profit_params=None,
                         use_stop_loss=True, use_take_profit=True):
    """
    Izvede backtest strategije z možnostjo različnih tipov stop-lossa, take-profita ter prilagodljivo velikostjo pozicije (position sizing),
    z dodatno podporo dinamičnemu prilagajanju stop-loss in take-profit cen na vsakem candlu.

    Exita trade, ko je dosežen stop-loss, take-profit ali signal za izhod (signal=0), pri čemer se lahko ti nivoji na vsakem
    candlu dinamično preračunajo z ustrezno funkcijo.



    :param df: pd.DataFrame
        DataFrame s podatki o trgovanju, ki mora vsebovati stolpce kot so 'Datetime', 'Open', 'High', 'Low', 'signal', 'cumulative_rsi'.
    capital : float, privzeto 1000
        Začetni kapital za backtest.
    stop_loss_func : callable ali None, privzeto None
        Funkcija, ki določa logiko stop lossa. Sprejme parametre kot so df (zgodovina do trenutnega svečnika), entry_price, direction in stop_loss_params.
        Če ni podana in je use_stop_loss=True, se sproži napaka.
        Funkcija lahko implementira statične ali dinamične stop-loss strategije (npr. trailing stop).
    stop_loss_params : dict ali None, privzeto None
        Dodatni parametri, ki se posredujejo funkciji stop_loss_func. Lahko vsebuje tudi 'is_dynamic' (bool), ki določa ali se stop-loss preračunava dinamično vsak svečnik.
    take_profit_func : callable ali None, privzeto None
        Funkcija, ki določa logiko take-profita. Sprejme parametre kot so df, entry_price, direction in take_profit_params.
        Če ni podana in je use_take_profit=True, se sproži napaka.
        Funkcija lahko implementira statične ali dinamične take-profit strategije.
    take_profit_params : dict ali None, privzeto None
        Dodatni parametri, ki se posredujejo funkciji take_profit_func. Lahko vsebuje tudi 'is_dynamic' (bool) za dinamično preračunavanje.
    use_stop_loss : bool, privzeto True
        Določa, ali se uporablja stop-loss mehanizem.
    use_take_profit : bool, privzeto True
        Določa, ali se uporablja take-profit mehanizem.
    position_sizing_func : callable ali None, privzeto None
        Funkcija, ki določa logiko velikosti pozicije glede na kapital in tveganje na enoto. Sprejme parametre kapitala in dodatne parametre.
    position_sizing_params : dict ali None, privzeto None
        Dodatni parametri, ki se posredujejo funkciji position_sizing_func.
    use_position_sizing : bool, privzeto False
        Določa, ali se uporablja dinamični izračun velikosti pozicije (True) ali fiksna velikost 1 (False).

    Opis:
    ------
    Funkcija izvaja backtest strategije s fleksibilno podporo različnim stilom stop-loss in take-profit mehanizmov ter prilagodljivo velikostjo pozicije.
    - Omogoča uporabo statičnih ali dinamičnih (preračunanih na vsakem candlu) stop-loss in take-profit strategij.
    - Stop-loss in take-profit cene se lahko vsak svečnik preračunajo s podano funkcijo, če je 'is_dynamic' nastavljeno na True v ustreznih parametrih.
    - Velikost pozicije se izračuna glede na funkcijo position_sizing_func in parametre, če je use_position_sizing=True.
    - Uporabnik lahko neodvisno vključi ali izključi uporabo stop-loss, take-profit in position sizing z ustreznimi boolean parametri.
    - Trade se zaključi, ko je dosežen stop-loss ali take-profit nivo, oziroma, če sta oba izklopljena, ob signalu za zaprtje pozicije (signal=0).

    :return:
    --------
    pd.DataFrame
        DataFrame z zabeleženimi posameznimi trgovinami, ki vsebuje stolpce:
        - 'Datetime': čas zaprtja trgovine,
        - 'entry_time': čas odprtja pozicije,
        - 'exit_time': čas zaprtja pozicije,
        - 'entry_price': cena ob odprtju pozicije,
        - 'exit_price': cena ob zaprtju pozicije,
        - 'stop_loss_price': stop loss cena (če uporabljeno),
        - 'position_size': velikost pozicije (dinamično izračunana ali 1),
        - 'position_value': vrednost pozicije ob odprtju,
        - 'cumulative_rsi': vrednost indikatorja RSI ob zaprtju,
        - 'signal': smer pozicije (1 ali -1),
        - 'pnl': dobiček/izguba posamezne trgovine,
        - 'cumulative_capital': kapital po zaključenih trgovinah,
        - 'reason': razlog za zaprtje pozicije (stop_loss_hit, take_profit_hit, signal_0_exit itd.).
    """
```

<br>

## main.py
V main.py se nahajajo vse pomembi podatki/parametri, kjer se povezujejo vse funkcije v en delujoč program.   
<b>Postopek delovanja</b>: Generiranje signalov => uporaba signalov oz. trejdanje singalov => analiza trejdov (vizualizacija rezultatov in drugih pomembnih podatkov)

### Za delovanje, <b>main.py</b> potrebuje:
- Vhodne podatke (OHLC — Open, High, Low, Close)
- Parametre za strategijo (npr. RSI, MA itd.)
- Parametre za stop loss
- Parametre za take profit
- Position sizing parametre
- Import funkcij iz ostalih .py datotek (modularna struktura)

### Postopek delovanja <b>main.py</b>:
- <b>Prebere/Naloži podatke</b> (OHLC podatki)
- <b>Generira signale</b> na podlagi izbrane strategije
- <b>Izvede backtest</b> (upošteva generirane signale, stop-loss in take-profit strategije)
- <b>Izračuna statistiko</b> (performance metrics: win-rate, profit, sharpe ratio, drawdown itd.)
- <b>Nariše grafe</b> (equity curve, drawdown  ...)

<br>
Primer parametrov:  

```python
params_strategy = {
    "oversold": 50,
    "overbought": 180,
    "rsi_period": 6,
    "cumulative_window": 2,
}

stop_loss_params = {
    # "stop_loss_pct": 0.01,

    # "atr_period": 17,
    # "atr_multiplier": 0.1,
    # "is_dynamic": False,

    "trail_pct": 0.0005,
    "is_dynamic": True,
}
take_profit_params = {
    # "take_profit_pct": 0.01,
    # "is_dynamic": False,

    # "atr_period": 10,
    # "atr_multiplier": 1,
    # "is_dynamic": False,

    "ma_period": 2,
    "is_dynamic": False,
}
```

<br>

## optimization.py
Omogoča optimizacijo parametrov glede na strategijo in metrike uspešnosti. Za optimizacijo se uporablja knjižnica <b>Optuna</b>, ki omogoča avtomatsko iskanje optimalnih kombinacij parametrov.  
Optuna je orodje za optimizacijo hiperparametrov, primarno razvito za področje strojnega učenja, a je tukaj uporabljeno za optimizacijo strategij v backtestingu.

### Glavne značilnosti:
- <b>Avtomatska izbira parametrov:</b> sami določimo, kateri parametri so pomembni (indikatorji, stop-loss, take-profit, itd.)

- <b>Ocena strategije:</b> na podlagi različnih metrik, kot so <b>Sharpe ratio</b>, <b>win rate</b>, <b>total return</b>, ipd.

- <b>Trajno shranjevanje rezultatov:</b> Optuna rezultate shranjuje v lokalno SQLite bazo, kar omogoča nadaljevanje optimizacije kasneje brez izgube podatkov.

- <b>Prikaz rezultatov preko spletnega vmesnika:</b> Optuna omogoča enostavno vizualizacijo in analizo iskanja.

### Kako deluje:
1. V <b>objective()</b> funkciji določimo, katere parametre želimo optimizirati in znotraj kakšnih mej:

```python

params = {
    "oversold": trial.suggest_int("oversold", 20, 60),
    "overbought": trial.suggest_int("overbought", 130, 180),
    "rsi_period": trial.suggest_int("rsi_period", 5, 14),
    ...
}
```
2. Izračunamo rezultat strategije preko funkcije <b>run_main()</b> iz <b>main.py</b>, ki vrne vse potrebne metrike.

3. Določimo <b>score funkcijo</b>, kjer sami določimo uteži, katera metrika ima večjo pomembnost:

```python
score = summary["sharpe"] * 2 + summary["win_rate"] * 1 + (summary["total_return"] / 1000) * 0.2
```
4. Optuna išče najboljšo kombinacijo parametrov, ki maksimizirajo izračunan <b>score</b>.

### Shranjevanje rezultatov:
Rezultati optimizacije se shranjujejo v SQLite bazo (lokalno):

```python
storage_name = "sqlite:///optuna_study.db"
study_name = "optuna_study"
```
To omogoča:
- nadaljevanje optimizacije ob prekinitvi,
- primerjavo rezultatov,
- enostaven pregled preko spletnega vmesnika Optune:

```bash
optuna-dashboard sqlite:///optuna_study.db
```
Prednost takšne zasnove:
- <b>Generično:</b> enostavno razširljivo za nove strategije

- <b>Prilagodljivo:</b> sami lahko prilagajamo metrike, uteži, in parametre

- <b>Robustno:</b> optimizacijo lahko zaženemo večkrat, jo ustavimo in nadaljujemo kadarkoli