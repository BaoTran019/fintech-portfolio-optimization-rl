import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config.parser import get_parser
from validate import load_model, model_predict, compute_metrics
from data.load_data import load_processed_data
from data.preprocess import split_data
from env.trading_env import StockPortfolioEnv

TECHNICAL_INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma', 'change'
]

def compute_min_variance_weights(cov_matrix):
    cov_matrix = np.array(cov_matrix, dtype=float)
    inv_cov = np.linalg.pinv(cov_matrix)
    weights = inv_cov.dot(np.ones(inv_cov.shape[0]))
    weights = np.maximum(weights, 0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()
    return weights


def build_min_variance_benchmark(test, initial_amount=10000000):
    price_matrix = test.pivot_table(index='date', columns='tic', values='close')
    returns = price_matrix.pct_change().shift(-1)
    dates = sorted(test.date.unique())
    benchmark_values = [initial_amount]
    benchmark_returns = []
    weight_records = []

    for date in dates[:-1]:
        row = test[test.date == date].iloc[0]
        cov = row['cov_list']
        weights = compute_min_variance_weights(cov)
        next_returns = returns.loc[date].values
        benchmark_returns.append(np.dot(weights, next_returns))
        benchmark_values.append(benchmark_values[-1] * (1 + benchmark_returns[-1]))
        weight_records.append(weights)

    benchmark_df = pd.DataFrame({
        'date': dates[1:],
        'benchmark_return': benchmark_returns,
        'benchmark_value': benchmark_values[1:]
    })
    weight_df = pd.DataFrame(weight_records, columns=price_matrix.columns, index=dates[:-1])
    weight_df.index.name = 'date'
    return benchmark_df, weight_df


def plot_comparison(df_daily_return, benchmark_df, algo, seed, save_path='results/'):
    os.makedirs(save_path, exist_ok=True)
    account_values = (1 + df_daily_return['daily_return']).cumprod() * 10000000
    benchmark_values = benchmark_df['benchmark_value']

    plt.figure(figsize=(10, 6))
    plt.plot(df_daily_return['date'], account_values, label=f'{algo} Portfolio')
    plt.plot(benchmark_df['date'], benchmark_values, label='Min-Variance Benchmark')
    plt.title(f'Account Value Comparison - {algo} Seed {seed}')
    plt.xlabel('Date')
    plt.ylabel('Account Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'{algo.lower()}_seed_{seed}_account_value_comparison.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df_daily_return['date'], (1 + df_daily_return['daily_return']).cumprod() - 1, label=f'{algo} Portfolio')
    plt.plot(benchmark_df['date'], (benchmark_df['benchmark_value'] / benchmark_df['benchmark_value'].iloc[0]) - 1, label='Min-Variance Benchmark')
    plt.title(f'Cumulative Return Comparison - {algo} Seed {seed}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'{algo.lower()}_seed_{seed}_cumulative_return_comparison.png'))
    plt.close()


def plot_stock_distribution(df_actions, algo, seed, save_path='results/'):
    os.makedirs(save_path, exist_ok=True)
    latest_weights = df_actions.iloc[-1].values
    tickers = df_actions.columns.tolist()

    plt.figure(figsize=(10, 10))
    plt.pie(latest_weights, labels=tickers, autopct='%1.1f%%', startangle=140)
    plt.title(f'Stock Distribution - {algo} Seed {seed}')
    plt.savefig(os.path.join(save_path, f'{algo.lower()}_seed_{seed}_distribution.png'))
    plt.close()


def backtest(algo, seed, processed_data_path='dataset/processed/processed.csv', config=None):
    # Load processed data
    df = load_processed_data(processed_data_path)
    
    # Split
    _, _, test = split_data(df, config)
    
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
    
    # Save results
    os.makedirs('results/', exist_ok=True)
    os.makedirs('results/metrics/', exist_ok=True)
    df_daily_return.to_csv(f'results/{algo}_seed_{seed}_returns.csv', index=False)
    df_actions.to_csv(f'results/{algo}_seed_{seed}_actions.csv')
    metrics.to_csv(f'results/metrics/{algo.lower()}_seed_{seed}_metrics.csv')

    # Generate benchmark and plots
    benchmark_df, benchmark_weights = build_min_variance_benchmark(test)
    plot_comparison(df_daily_return, benchmark_df, algo, seed, save_path='results/')
    plot_stock_distribution(df_actions, algo, seed, save_path='results/')
    benchmark_weights.to_csv(f'results/{algo.lower()}_seed_{seed}_benchmark_weights.csv')
    
    print(f"Backtest completed for {algo} seed {seed}")
    print(metrics)

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    backtest(args.algo, args.seed, args.processed_data_path, args)

if __name__ == "__main__":
    main()