import pandas as pd
from ta.volatility import BollingerBands



def bollinger_bands(df: pd.DataFrame, params):
    """
    bb_params: {
        "period": 20,
        "std_dev_mult": 2
    }
    """
    period = params.period
    std_dev_mult = params.std_dev_mult

    col_name = getattr(params, "column", "Close")

    # Initialize the BollingerBands indicator object from the ta library
    indicator_bb = BollingerBands(
        close=df[col_name], 
        window=period, 
        window_dev=std_dev_mult
    )

    middle_band = indicator_bb.bollinger_mavg()
    upper_band = indicator_bb.bollinger_hband()
    lower_band = indicator_bb.bollinger_lband()


    bands_df = pd.DataFrame({
        'middle': middle_band, 
        'upper': upper_band, 
        'lower': lower_band
    }, index=df.index) # Ensure the index matches the original DataFrame
    
    return bands_df