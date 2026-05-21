import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
from config.parser import get_parser
from utils.seed import set_global_seed
from data.load_data import load_processed_data
from data.preprocess import split_data
from env.trading_env import StockPortfolioEnv
from agents.build_agent import build_agent
from validate import (
    model_predict,
    compute_metrics
)

TECHNICAL_INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma', 'change'
]


def setup_logging(log_path='results/logs/'):
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_path, 'training.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def train_single_seed(algo, timesteps, seed, processed_data_path, save_path, eval_freq, config):
    # Set seed
    set_global_seed(seed)

    # Load preprocessed dataset
    df = load_processed_data(processed_data_path)

    # Split data by time into train/validation/test
    train, val, test = split_data(df, config = config)

    # Clean train data
    train = train.dropna(subset=['cov_list', 'return_list']).copy()
    train = train.drop_duplicates(subset=['date', 'tic'], keep='last')
    train = train.sort_values(['date', 'tic']).reset_index(drop=True)
    train.index = train.date.factorize()[0]

    # Clean validation data
    val = val.dropna(subset=['cov_list', 'return_list']).copy()
    val = val.drop_duplicates(subset=['date', 'tic'], keep='last')
    val = val.sort_values(['date', 'tic']).reset_index(drop=True)
    val.index = val.date.factorize()[0]
    
    # Environment setup
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension
    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 10000000,
        "transaction_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": TECHNICAL_INDICATORS, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4,
        "seed": seed
    }
    
    env = StockPortfolioEnv(df=train, **env_kwargs)
    env_sb, _ = env.get_sb_env()
    
    # Build agent
    agent, model = build_agent(algo, env_sb, seed)
    
    # Train with FinRL wrapper
    best_sharpe = -float('inf')
    best_timestep = 0

    validation_history = []

    for current_step in range(0, timesteps, eval_freq):

        # ======================================
        # Train one chunk using FinRL wrapper
        # ======================================

        model = agent.train_model(
            model=model,
            tb_log_name=algo.lower(),
            total_timesteps=eval_freq
        )

        # ======================================
        # Validation
        # ======================================

        val_env = StockPortfolioEnv(df=val, **env_kwargs)

        df_daily_return, _ = model_predict(
            model,
            val_env
        )

        metrics = compute_metrics(df_daily_return)

        sharpe = metrics['Sharpe']

        current_total_step = current_step + eval_freq

        # ======================================
        # Save validation history
        # ======================================

        validation_history.append({
            'algo': algo,
            'seed': seed,
            'timesteps': current_total_step,
            'sharpe': sharpe,
            'cumulative_return': metrics['Cumulative Return'],
            'max_drawdown': metrics['Max Drawdown'],
            'volatility': metrics['Volatility']
        })

        # ======================================
        # Save best model
        # ======================================

        if sharpe > best_sharpe:

            best_sharpe = sharpe

            best_timestep = current_total_step

            os.makedirs(save_path, exist_ok=True)

            best_model_path = os.path.join(
                save_path,
                f'{algo.lower()}_seed_{seed}_best.zip'
            )

            model.save(best_model_path)

            logging.info(
                f'NEW BEST MODEL | '
                f'Algo={algo} | '
                f'Seed={seed} | '
                f'Timestep={best_timestep} | '
                f'Sharpe={best_sharpe:.4f}'
            )

        print(
            f'[{algo}] '
            f'Timestep={current_total_step} | '
            f'Validation Sharpe={sharpe:.4f}'
        )

    history_df = pd.DataFrame(validation_history)

    os.makedirs('results/metrics/', exist_ok=True)

    history_df.to_csv(
        f'results/metrics/{algo.lower()}_seed_{seed}_validation.csv',
        index=False
    )
    
    # Save last model
    os.makedirs(save_path, exist_ok=True)
    model_path_last = os.path.join(save_path, f'{algo.lower()}_seed_{seed}_last.zip')
    model.save(model_path_last)
    
    # Log
    logging.info(
    f'TRAINING FINISHED | '
    f'Algo={algo} | '
    f'Seed={seed} | '
    f'Best Timestep={best_timestep} | '
    f'Best Sharpe={best_sharpe:.4f}'
)

    plt.figure(figsize=(10, 6))

    plt.plot(
        history_df['timesteps'],
        history_df['sharpe']
    )

    plt.xlabel('Timesteps')

    plt.ylabel('Validation Sharpe')

    plt.title(
        f'Validation Sharpe vs Timesteps - '
        f'{algo} Seed {seed}'
    )

    plt.grid(True)

    plt.savefig(
        os.path.join(
            'results',
            f'{algo.lower()}_seed_{seed}_validation_sharpe.png'
        )
    )

    plt.close()
    
    return model

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.timesteps is None:
        parser.error("--timesteps is required for training")
    
    setup_logging()
    
    if args.n_seeds > 1:
        models = []
        for s in range(args.seed, args.seed + args.n_seeds):
            model = train_single_seed(args.algo, args.timesteps, s, args.processed_data_path, args.save_path, args.eval_freq, args)
            models.append(model)
        logging.info(f'Trained {args.n_seeds} seeds for {args.algo}')
    else:
        train_single_seed(args.algo, args.timesteps, args.seed, args.processed_data_path, args.save_path, args.eval_freq, args)

if __name__ == "__main__":
    main()