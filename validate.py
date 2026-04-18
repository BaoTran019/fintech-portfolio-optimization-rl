import os
import pandas as pd
from config.parser import get_parser
from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
from data.load_data import load_data
from data.preprocess import preprocess_data, split_data
from env.trading_env import StockPortfolioEnv

TECHNICAL_INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma', 'change'
]

def load_model(algo, seed, save_path='results/models/'):
    model_path = os.path.join(save_path, f'{algo.lower()}_seed_{seed}.zip')
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

def validate(algo, seed, data_source='csv', data_path='dataset/5_vn30_vnsi_symbols_data.xlsx', vnindex_path='dataset/vnindex.xlsx'):
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
    
    # Run prediction
    df_daily_return, df_actions = model_predict(model, env)
    
    # Compute metrics
    metrics = compute_metrics(df_daily_return)
    
    # Save metrics
    os.makedirs('results/metrics/', exist_ok=True)
    metrics.to_csv(f'results/metrics/{algo}_seed_{seed}_metrics.csv')
    
    print(metrics)

def model_predict(model, env):
    """
    Run prediction on environment.
    """
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
    
    df_daily_return = env.save_asset_memory()
    df_actions = env.save_action_memory()
    return df_daily_return, df_actions

def compute_metrics(df_daily_return, initial_amount=10000000):
    """
    Compute performance metrics for a portfolio.
    """
    returns = df_daily_return['daily_return']
    cumulative_return = (1 + returns).prod() - 1
    annual_return = ((1 + cumulative_return) ** (252 / len(returns))) - 1 if len(returns) > 0 else 0
    sharpe = (252 ** 0.5) * returns.mean() / returns.std() if returns.std() != 0 else 0
    account_value = initial_amount * (1 + returns).cumprod()
    peak_value = account_value.cummax()
    drawdown = (account_value - peak_value) / peak_value
    max_drawdown = drawdown.min()

    metrics = pd.Series({
        'Annual Return': annual_return,
        'Cumulative Return': cumulative_return,
        'Sharpe': sharpe,
        'Mean Return': returns.mean(),
        'Std Return': returns.std(),
        'Max Drawdown': max_drawdown
    })
    return metrics

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    validate(args.algo, args.seed, args.data_source, args.data_path, args.vnindex_path)

if __name__ == "__main__":
    main()