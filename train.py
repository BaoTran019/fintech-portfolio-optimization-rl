import logging
import os
import matplotlib.pyplot as plt
from config.parser import get_parser
from utils.seed import set_global_seed
from data.load_data import load_processed_data
from data.preprocess import split_data
from env.trading_env import StockPortfolioEnv
from agents.build_agent import build_agent

TECHNICAL_INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma', 'change'
]

def setup_logging(log_path='results/logs/'):
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_path, 'training.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def train_single_seed(algo, timesteps, seed, processed_data_path, save_path, config):
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
    trained_model = agent.train_model(
        model=model,
        total_timesteps=timesteps
    )
    model = trained_model
    
    # Save model
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f'{algo.lower()}_seed_{seed}.zip')
    model.save(model_path)
    
    # Log
    final_reward = env.portfolio_value  # Approximate
    logging.info(f'Algo={algo} | Seed={seed} | Timesteps={timesteps} | Reward={final_reward}')
    
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
            model = train_single_seed(args.algo, args.timesteps, s, args.processed_data_path, args.save_path, args)
            models.append(model)
        logging.info(f'Trained {args.n_seeds} seeds for {args.algo}')
    else:
        train_single_seed(args.algo, args.timesteps, args.seed, args.processed_data_path, args.save_path, args)

if __name__ == "__main__":
    main()