import pandas as pd
import pickle
import os
import sqlite3

def load_data(path: str, filename: str, type_of_file: str):
    """
    Load data from a specified file type.

    Args:
        path (str): Directory path where file is located.
        filename (str): Name of the file.
        type_of_file (str): Type of file ('csv', 'pkl', 'json').

    Returns:
        Data loaded from file (usually a pandas DataFrame or Python object).
    """
    full_path = os.path.join(path, filename)

    if type_of_file == 'csv':
        return pd.read_csv(full_path)
    elif type_of_file == 'pkl':
        with open(full_path, 'rb') as f:
            return pickle.load(f)
    elif type_of_file == 'json':
        return pd.read_json(full_path)
    else:
        raise ValueError(f"Unsupported file type: {type_of_file}")
    


def save_data(data, path: str, filename: str, type_of_file: str):
    """
    Save data to a specified file type.

    Args:
        data: Data to save (DataFrame, dict, etc.).
        path (str): Directory path where file will be saved.
        filename (str): Name of the file.
        type_of_file (str): Type of file ('csv', 'pkl', 'json').

    Returns:
        None
    """
    full_path = os.path.join(path, filename)

    # # Create directory if it does not exist
    # os.makedirs(path, exist_ok=True)

    if type_of_file == 'csv':
        data.to_csv(full_path, index=False)
    elif type_of_file == 'pkl':
        with open(full_path, 'wb') as f:
            pickle.dump(data, f)
    elif type_of_file == 'json':
        data.to_json(full_path)
    else:
        raise ValueError(f"Unsupported file type: {type_of_file}")
    

def read_table_from_sqlite(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Reads a specific table from a SQLite database into a Pandas DataFrame.

    Args:
        db_path (str): Path to the SQLite database file (.db).
        table_name (str): Name of the table to read.

    Returns:
        pd.DataFrame: The contents of the table as a DataFrame.
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)

        # Read the table into a DataFrame
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

        return df

    except Exception as e:
        print(f"Error reading table '{table_name}': {e}")
        return pd.DataFrame()  # return empty DataFrame if failed

    finally:
        # Always close the connection
        if 'conn' in locals():
            conn.close()




def modify_columns(df: pd.DataFrame):
    """
    Uredi dataframe tako da so vedno vsa imena columnov enaka z veliko začetnico in, da je vrstni red columnov enak ter spremeni Datetime column v datetime series
    Returns:
        df: Vrne posodobljen DataFrame
    """

    df = df.drop(columns=['Unnamed: 0'], errors='ignore')


    # Convert all column names to uppercase if not already
    df.columns = [col if "_MA" in col else col.capitalize() for col in df.columns]
    # Identify columns that might be datetime by checking their names (e.g., 'TIMESTAMP')
    for col in df.columns:
        if 'Time' in col or 'Date' in col:  # basic check for datetime column
            df[col] = pd.to_datetime(df[col], errors='coerce')  # convert to datetime, invalid parse becomes NaT
            df.rename(columns={col: 'Datetime'}, inplace=True)  # Spremenimo ime columna v Datetime, zato da je vse povsod enako

    # Spremenimo vrsti red collumnov, zato, da bodo vedno enaki
    front_cols = ['Datetime', 'Close', 'Open', 'High', 'Low', 'Volume']
    front_cols_existing = [col for col in front_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in front_cols_existing]
    df = df[front_cols_existing + other_cols]

    return df


def load_and_prepare_data(path, filename):
    """Handles loading and initial preparation of the data."""
    full_df = load_data(path=path, filename=filename, type_of_file="csv")
    df = modify_columns(full_df)
    print(f"Data loaded and prepared. Shape: {df.shape}")
    return df

def split_data(df, oos_percentage=0.25):
    """
    Splits a DataFrame into in-sample and out-of-sample sets based on a percentage.
    
    Args:
        df (pd.DataFrame): The input DataFrame, sorted by date.
        oos_percentage (float): The percentage of data to use for the out-of-sample set (e.g., 0.25 for 25%).
        
    Returns:
        tuple: A tuple containing the in-sample DataFrame and the out-of-sample DataFrame.
    """
    split_index = int(len(df) * (1 - oos_percentage))
    
    in_sample_df = df.iloc[:split_index].copy()
    out_of_sample_df = df.iloc[split_index:].copy()
    
    return in_sample_df, out_of_sample_df


if __name__ == "__main__":
    pass