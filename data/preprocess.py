import pandas as pd
import numpy as np
from stockstats import StockDataFrame as Sdf
from finrl.meta.preprocessor.preprocessors import data_split

TECHNICAL_INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma', 'change'
]

def preprocess_data(df, vnindex_df=None):
    """
    Preprocess the stock data: add technical indicators, VNINDEX, covariance.
    """
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Strip column names
    df.columns = df.columns.str.strip()
    
    # Sort by date and tic
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Add VNINDEX if available
    if vnindex_df is not None:
        df = add_vnindex(df, vnindex_df)
    
    # Add covariance matrix
    df = add_covariance_matrix(df)
    
    return df

def add_technical_indicators(df):
    """
    Add technical indicators using stockstats.
    """
    final_df = pd.DataFrame()
    unique_tickers = df.tic.unique()
    
    for ticker in unique_tickers:
        temp_df = df[df.tic == ticker].copy()
        temp_df = temp_df.sort_values('date').reset_index(drop=True)
        
        original_dates = temp_df['date'].values
        
        stock = Sdf.retype(temp_df)
        
        for indicator in TECHNICAL_INDICATORS:
            _ = stock[indicator]
        
        stock = pd.DataFrame(stock)
        stock = stock.reset_index()
        
        if 'date' not in stock.columns:
            stock['date'] = original_dates
        
        stock['tic'] = ticker
        
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume'] + TECHNICAL_INDICATORS
        available_cols = [c for c in required_cols if c in stock.columns]
        stock = stock[available_cols]
        
        stock = stock.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        
        final_df = pd.concat([final_df, stock], ignore_index=True)
    
    final_df = final_df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    return final_df

def add_vnindex(df, vnindex_df):
    """
    Add VNINDEX return to the dataframe.
    """
    vnindex_df['date'] = pd.to_datetime(vnindex_df['date'])
    vnindex_df['vni_return'] = vnindex_df['close'].pct_change() * 100
    vnindex_context = vnindex_df[['date', 'vni_return']].fillna(0)
    
    df = df.merge(vnindex_context, on='date', how='left')
    return df

def add_covariance_matrix(df, lookback=252):
    """
    Add rolling covariance matrix.
    """
    df = df.sort_values(['date', 'tic'], ignore_index=True)
    df.index = df.date.factorize()[0]
    
    cov_list = []
    return_list = []
    
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i - lookback:i, :]
        price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)
        
        covs = return_lookback.cov().values
        cov_list.append(covs)
    
    df_cov = pd.DataFrame({'date': df.date.unique()[lookback:], 'cov_list': cov_list, 'return_list': return_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    return df

def split_data(df, train_start='2015-01-01', train_end='2023-12-31', test_start='2024-01-01', test_end='2024-12-31'):
    """
    Split data into train and test.
    """
    train = data_split(df, train_start, train_end)
    test = data_split(df, test_start, test_end)
    return train, test