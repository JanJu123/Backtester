import os
import pandas as pd
from datetime import datetime
import calendar

def combine_contracts(folder_name="CME_ES"):
    # Month codes map
    month_codes = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4,
        'K': 5, 'M': 6, 'N': 7, 'Q': 8,
        'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }

    def third_friday(year, month):
        """Return the date of the third Friday of given month and year."""
        c = calendar.Calendar(firstweekday=calendar.MONDAY)
        monthcal = c.monthdatescalendar(year, month)
        # Fridays are index 4 (Monday=0)
        fridays = [day for week in monthcal for day in [week[4]] if day.month == month]
        return fridays[2]  # third Friday

    folder_name="CME_NG"
    csv_folder = f"data/files/archive/{folder_name}"

    df_list = []

    for filename in os.listdir(csv_folder):
        if not filename.endswith(".csv"):
            continue
        
        full_path = os.path.join(csv_folder, filename)
        print(f"Loading {filename} ...")
        df = pd.read_csv(full_path)
        
        # Extract contract code from filename, e.g. 'CME_NQH2000.csv' -> 'NQH2000'
        contract = filename.replace("CME_", "").replace(".csv", "")
        
        # Parse contract expiration:
        # Format assumed: 'NQH2000' where H=month code, 2000=year
        # Extract month code and year from contract string
        try:
            month_code = contract[2]  # e.g. 'H'
            year_str = contract[3:]   # e.g. '2000'
            year = int(year_str)
            month = month_codes[month_code]
            expiration_date = third_friday(year, month)
        except Exception as e:
            print(f"Warning: Failed to parse expiration for contract {contract}: {e}")
            expiration_date = pd.NaT
        
        # Add columns
        df['Contract'] = contract
        df['Expiration'] = expiration_date
        
        # Convert 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'].str.strip())
        
        df_list.append(df)

    # Concatenate all contracts data
    full_df = pd.concat(df_list, ignore_index=True)

    # Sort by Date, then Expiration (so front month comes first for each date)
    full_df = full_df.sort_values(by=['Date', 'Expiration']).reset_index(drop=True)

    full_df.drop_duplicates(subset=['Date'], keep='first', inplace=True)


    print("✅ Combined dataset preview:")
    print(full_df.head())
    print(f"✅ Total rows: {len(full_df)}")


    # Save combined CSV
    full_df.to_csv(f"data/files/{folder_name}_combined_2000_2022.csv", index=False)
