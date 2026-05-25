import pandas as pd
import numpy as np
from stockstats import StockDataFrame as Sdf

TECHNICAL_INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma', 'change'
]


def preprocess_data(df, vnindex_df=None):
    """
    Preprocess the stock data: add technical indicators, VNINDEX, covariance.
    This function is deterministic and does not perform any train/validation/test splitting.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.columns = df.columns.str.strip()
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    df = add_technical_indicators(df)

    if vnindex_df is not None:
        df = add_vnindex(df, vnindex_df)

    df = add_covariance_matrix(df)
    return df


def add_technical_indicators(df):
    """
    Add technical indicators using stockstats.
    """
    final_df = pd.DataFrame()
    unique_tickers = df['tic'].unique()

    for ticker in unique_tickers:
        temp_df = df[df['tic'] == ticker].copy()
        temp_df = temp_df.sort_values('date').reset_index(drop=True)
        original_dates = temp_df['date'].values

        stock = Sdf.retype(temp_df)
        for indicator in TECHNICAL_INDICATORS:
            _ = stock[indicator]

        stock = pd.DataFrame(stock).reset_index()
        if 'date' not in stock.columns:
            stock['date'] = original_dates
        stock['tic'] = ticker

        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume'] + TECHNICAL_INDICATORS
        available_cols = [c for c in required_cols if c in stock.columns]
        stock = stock[available_cols]
        stock = stock.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

        final_df = pd.concat([final_df, stock], ignore_index=True)

    final_df = final_df.sort_values(['date', 'tic']).reset_index(drop=True)
    return final_df


def add_vnindex(df, vnindex_df):
    """
    Add VNINDEX return to the dataframe.
    """
    vnindex_df = vnindex_df.copy()
    vnindex_df['date'] = pd.to_datetime(vnindex_df['date'])
    vnindex_df['vn30_return'] = vnindex_df['close'].pct_change() * 100
    vnindex_context = vnindex_df[['date', 'vn30_return']].fillna(0)
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
        cov_list.append(covs.tolist())
    
    df_cov = pd.DataFrame({'date': df.date.unique()[lookback:], 'cov_list': cov_list, 'return_list': return_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    return df


def split_data(df,
               train_start='2015-01-01', train_end='2023-12-31',
               val_start='2024-01-01', val_end='2024-12-31',
               test_start='2025-01-01', test_end='2025-12-31',
               config=None):
    """
    Split a preprocessed dataframe into train, validation, and test sets based on explicit date boundaries.

    The split preserves chronological order, avoids shuffling, and prevents leakage.
    """
    if config is not None:
        train_start = getattr(config, 'train_start', train_start)
        train_end = getattr(config, 'train_end', train_end)
        val_start = getattr(config, 'val_start', val_start)
        val_end = getattr(config, 'val_end', val_end)
        test_start = getattr(config, 'test_start', test_start)
        test_end = getattr(config, 'test_end', test_end)

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    train_start = pd.to_datetime(train_start)
    train_end = pd.to_datetime(train_end) if train_end is not None else None
    val_start = pd.to_datetime(val_start)
    val_end = pd.to_datetime(val_end) if val_end is not None else None
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end) if test_end is not None else None

    if not (train_start < val_start < test_start):
        raise ValueError('train_start < val_start < test_start must hold for explicit split boundaries.')

    if train_end is not None and train_end >= val_start:
        raise ValueError('train_end must be before val_start.')
    if val_end is not None and val_end >= test_start:
        raise ValueError('val_end must be before test_start.')
    if train_end is not None and train_end < train_start:
        raise ValueError('train_end must be on or after train_start.')
    if val_end is not None and val_end < val_start:
        raise ValueError('val_end must be on or after val_start.')
    if test_end is not None and test_end < test_start:
        raise ValueError('test_end must be on or after test_start.')

    train_conditions = (df['date'] >= train_start)
    if train_end is not None:
        train_conditions &= (df['date'] <= train_end)
    else:
        train_conditions &= (df['date'] < val_start)

    val_conditions = (df['date'] >= val_start)
    if val_end is not None:
        val_conditions &= (df['date'] <= val_end)
    else:
        val_conditions &= (df['date'] < test_start)

    test_conditions = (df['date'] >= test_start)
    if test_end is not None:
        test_conditions &= (df['date'] <= test_end)

    train_df = df[train_conditions].copy()
    val_df = df[val_conditions].copy()
    test_df = df[test_conditions].copy()

    return train_df, val_df, test_df