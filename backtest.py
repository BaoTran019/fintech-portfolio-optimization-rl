import os
import pandas as pd
import matplotlib.pyplot as plt
from config.parser import get_parser
from validate import load_model, model_predict, compute_metrics
from data.load_data import load_data
from data.preprocess import preprocess_data, split_data
from env.trading_env import StockPortfolioEnv

TECHNICAL_INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma', 'change'
]

def backtest(algo, seed, data_source='csv', data_path='dataset/5_vn30_vnsi_symbols_data.xlsx', vnindex_path='dataset/vnindex.xlsx'):
    # Load data
    df, vnindex_df = load_data(data_source, data_path, vnindex_path)
    df = preprocess_data(df, vnindex_df)
    
    # Split
    _, test = split_data(df)
    
    # Clean test data
    unique_tickers = test.tic.unique()
    test = test[test.tic.isin(unique_tickers)]
    test = test.sort_values(['date', 'tic'])
    test = test.drop_duplicates(subset=['date', 'tic'], keep='last')
    test = test.dropna(subset=['cov_list', 'return_list'])
    test = test.sort_values(['date', 'tic']).reset_index(drop=True)
    test.index = test.date.factorize()[0]
    
    # Environment
    stock_dimension = len(test.tic.unique())
    state_space = stock_dimension
    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 10000000,
        "transaction_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": TECHNICAL_INDICATORS, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4
    }
    
    env = StockPortfolioEnv(df=test, **env_kwargs)
    
    # Load model
    model = load_model(algo, seed)
    
    # Run backtest
    df_daily_return, df_actions = model_predict(model, env)
    
    # Compute metrics
    metrics = compute_metrics(df_daily_return)
    
    # Plot cumulative return
    plt.figure(figsize=(10, 6))
    plt.plot(df_daily_return['date'], (df_daily_return['daily_return'] + 1).cumprod() - 1)
    plt.title(f'Cumulative Return - {algo} Seed {seed}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.savefig(f'results/{algo}_seed_{seed}_backtest.png')
    plt.close()
    
    # Save results
    os.makedirs('results/', exist_ok=True)
    df_daily_return.to_csv(f'results/{algo}_seed_{seed}_returns.csv')
    df_actions.to_csv(f'results/{algo}_seed_{seed}_actions.csv')
    
    print(f"Backtest completed for {algo} seed {seed}")
    print(metrics)

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    backtest(args.algo, args.seed, args.data_source, args.data_path, args.vnindex_path)

if __name__ == "__main__":
    main()