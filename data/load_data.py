import pandas as pd
import os

def load_data(data_source='csv', data_path='dataset/5_vn30_vnsi_symbols_data.xlsx', vnindex_path='dataset/vnindex.xlsx'):
    """
    Load stock data from CSV or API.
    For API, assumes data is fetched and saved to file.
    """
    if data_source == 'api':
        # Assume data is fetched using utils/fetch_vnstock_raw.py and saved
        # For now, load from file
        pass
    
    if os.path.exists(data_path):
        df = pd.read_excel(data_path)
    else:
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    vnindex_df = None
    if os.path.exists(vnindex_path):
        vnindex_df = pd.read_excel(vnindex_path)
    
    return df, vnindex_df


def load_processed_data(processed_path='dataset/processed/processed.csv'):
    """
    Load the processed dataset saved by preprocess_data.py.
    """
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data file not found: {processed_path}")
    return pd.read_csv(processed_path, parse_dates=['date'])