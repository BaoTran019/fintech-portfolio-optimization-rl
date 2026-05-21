import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config.parser import get_parser
from validate import model_predict, compute_metrics
from data.load_data import load_processed_data
from data.preprocess import split_data
from env.trading_env import StockPortfolioEnv
from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3

TECHNICAL_INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma', 'change'
]

def load_model(algo, seed, save_path='results/models/'):
    model_path = os.path.join(save_path, f'{algo.lower()}_seed_{seed}_best.zip')
    if algo == 'A2C':
        model = A2C.load(model_path)
    elif algo == 'PPO':
        model = PPO.load(model_path)
    elif algo == 'DDPG':
        model = DDPG.load(model_path)
    elif algo == 'SAC':
        model = SAC.load(model_path)
    elif algo == 'TD3':
        model = TD3.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    return model

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

def build_vn30_benchmark(
    vn30_path,
    df_daily_return,
    initial_amount=10000000
):

    vn30 = pd.read_excel(vn30_path)

    vn30['date'] = pd.to_datetime(vn30['date'])

    vn30 = vn30.sort_values('date')

    test_start = df_daily_return['date'].min()
    test_end = df_daily_return['date'].max()

    vn30 = vn30[(vn30["date"] >= test_start) & (vn30["date"] <= test_end)].copy()

    # Daily return
    vn30['benchmark_return'] = (
        vn30['close'].pct_change()
    )

    vn30 = vn30.dropna().reset_index(drop=True)

    # Portfolio value
    vn30['benchmark_value'] = (
        (1 + vn30['benchmark_return']).cumprod()
        * initial_amount
    )

    benchmark_df = vn30[[
        'date',
        'benchmark_return',
        'benchmark_value'
    ]]

    return benchmark_df

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


def plot_min_variance_comparison(df_daily_return, benchmark_df, algo, seed, save_path='results/'):
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

def plot_vn30_comparison(
    df_daily_return,
    vn30_df,
    algo,
    seed,
    save_path='results/'
):

    os.makedirs(save_path, exist_ok=True)

    # RL account value
    rl_account_values = (
        (1 + df_daily_return['daily_return']).cumprod()
        * 10000000
    )

    # VN30 benchmark value
    vn30_values = vn30_df['benchmark_value']

    # ==============================
    # ACCOUNT VALUE PLOT
    # ==============================

    plt.figure(figsize=(10, 6))

    plt.plot(
        df_daily_return['date'],
        rl_account_values,
        label=f'{algo} Portfolio'
    )

    plt.plot(
        vn30_df['date'],
        vn30_values,
        label='VN30 Benchmark'
    )

    plt.title(
        f'VN30 Benchmark Comparison - {algo} Seed {seed}'
    )

    plt.xlabel('Date')

    plt.ylabel('Account Value')

    plt.legend()

    plt.grid(True)

    plt.savefig(
        os.path.join(
            save_path,
            f'{algo.lower()}_seed_{seed}_vn30_account_value_comparison.png'
        )
    )

    plt.close()

    # ==============================
    # CUMULATIVE RETURN PLOT
    # ==============================

    plt.figure(figsize=(10, 6))

    plt.plot(
        df_daily_return['date'],
        (1 + df_daily_return['daily_return']).cumprod() - 1,
        label=f'{algo} Portfolio'
    )

    plt.plot(
        vn30_df['date'],
        (vn30_df['benchmark_value']
         / vn30_df['benchmark_value'].iloc[0]) - 1,
        label='VN30 Benchmark'
    )

    plt.title(
        f'VN30 Cumulative Return Comparison - {algo} Seed {seed}'
    )

    plt.xlabel('Date')

    plt.ylabel('Cumulative Return')

    plt.legend()

    plt.grid(True)

    plt.savefig(
        os.path.join(
            save_path,
            f'{algo.lower()}_seed_{seed}_vn30_cumulative_return_comparison.png'
        )
    )

    plt.close()

def plot_stock_distribution(
    df_actions,
    algo,
    seed,
    save_path='results/'
):

    os.makedirs(save_path, exist_ok=True)

    # ==========================================
    # Ensure datetime index if possible
    # ==========================================

    df_plot = df_actions.copy()

    try:
        df_plot.index = pd.to_datetime(df_plot.index)
    except:
        pass

    # ==========================================
    # STACKED AREA PLOT
    # ==========================================

    plt.figure(figsize=(16, 8))

    plt.stackplot(
        df_plot.index,
        df_plot.T.values,
        labels=df_plot.columns
    )

    plt.title(
        f'Portfolio Allocation Over Time - {algo} Seed {seed}',
        fontsize=14
    )

    plt.xlabel('Date', fontsize=12)

    plt.ylabel('Portfolio Weight', fontsize=12)

    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        fontsize=9
    )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_path,
            f'{algo.lower()}_seed_{seed}_allocation_over_time.png'
        )
    )

    plt.close()

    # ==========================================
    # OPTIONAL:
    # AVERAGE PORTFOLIO WEIGHT
    # ==========================================

    avg_weights = df_plot.mean()

    plt.figure(figsize=(10, 6))

    avg_weights.sort_values(ascending=False).plot(
        kind='bar'
    )

    plt.title(
        f'Average Portfolio Allocation - {algo} Seed {seed}',
        fontsize=14
    )

    plt.xlabel('Stock')

    plt.ylabel('Average Weight')

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_path,
            f'{algo.lower()}_seed_{seed}_average_allocation.png'
        )
    )

    plt.close()


def backtest(algo, seed, processed_data_path='dataset/processed/processed.csv', vn30_path='dataset/vn30.xlsx', config=None):
    # Load processed data
    df = load_processed_data(processed_data_path)
    
    # Split
    _, _, test = split_data(df, config = config)
    
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

    # Compare with min-variance
    benchmark_df, benchmark_weights = build_min_variance_benchmark(test)
    plot_min_variance_comparison(df_daily_return, benchmark_df, algo, seed, save_path='results/')

    # Compare with vn30
    vn30_df = build_vn30_benchmark(vn30_path, df_daily_return)
    plot_vn30_comparison(df_daily_return, vn30_df, algo, seed, save_path='results/')

    plot_stock_distribution(df_actions, algo, seed, save_path='results/')
    benchmark_weights.to_csv(f'results/{algo.lower()}_seed_{seed}_benchmark_weights.csv')
    
    print(f"Backtest completed for {algo} seed {seed}")
    print(metrics)

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    backtest(args.algo, args.seed, args.processed_data_path, args.vn30_path, args)

if __name__ == "__main__":
    main()