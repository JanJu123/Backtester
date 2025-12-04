import dukascopy_python as duka
import pandas as pd
from datetime import datetime, timedelta

# SETTINGS
symbol = "USA500IDX" # Note: No dot for some versions, try "USA500.IDX" if this fails
start = datetime(2023, 1, 1)
end = datetime(2023, 12, 31)
timeframe = duka.INTERVAL_MIN_15

print(f"Starting download for {symbol}...")

all_data = []

# Generate a list of VALID days only (skips Nov 31, Feb 30, etc.)
date_range = pd.date_range(start=start, end=end, freq='D')

for day in date_range:
    # Convert pandas Timestamp to python datetime
    current_day = day.to_pydatetime()
    next_day = current_day + timedelta(days=1)
    
    print(f"Downloading {current_day.strftime('%Y-%m-%d')}...", end=" ")
    
    try:
        # Fetch ONE day at a time
        df = duka.fetch(
            instrument=symbol,
            interval=timeframe,
            start=current_day,
            end=next_day, # Fetch until the start of the next day
            offer_side=duka.OFFER_SIDE_BID
        )
        
        if len(df) > 0:
            all_data.append(df)
            print(f"✅ ({len(df)} bars)")
        else:
            print("⚠️ (Empty)")
            
    except Exception as e:
        print(f"❌ Error: {e}")

# Combine all days into one CSV
if all_data:
    full_df = pd.concat(all_data)
    filename = "SP500_15min_2023_FIXED.csv"
    full_df.to_csv(filename)
    print(f"\n🎉 DONE! Saved {len(full_df)} rows to {filename}")
else:
    print("\n❌ No data downloaded.")