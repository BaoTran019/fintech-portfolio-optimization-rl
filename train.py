import logging
import os
import matplotlib.pyplot as plt
from config.parser import get_parser
from utils.seed import set_global_seed
from data.load_data import load_processed_data
from data.preprocess import split_data
from env.trading_env import StockPortfolioEnv
from agents.build_agent import build_agent
from stable_baselines3.common.callbacks import BaseCallback

class LossCallback(BaseCallback):
    """
    Callback for capturing training losses
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.losses = []
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Capture loss from the model's logger
        if hasattr(self.model, 'logger'):
            logger_values = getattr(self.model.logger, 'name_to_value', {}) or {}
            loss = None
            for key in ['train/loss', 'loss', 'train/policy_loss', 'policy_loss', 'train/value_loss', 'value_loss']:
                if key in logger_values:
                    loss = logger_values[key]
                    break

            if loss is None and hasattr(self.model, 'policy') and hasattr(self.model.policy, 'loss'):
                try:
                    loss = self.model.policy.loss.item()
                except Exception:
                    loss = None

            if loss is not None:
                self.losses.append(loss)
        return True

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
    model = build_agent(algo, env_sb, seed)
    
    # Create loss callback
    loss_callback = LossCallback()
    
    # Train with callback
    model.learn(total_timesteps=timesteps, callback=loss_callback)
    
    # Plot and save loss diagram
    if loss_callback.losses:
        save_dir = os.path.normpath(save_path)
        if os.path.basename(save_dir) == 'models':
            plots_dir = os.path.join(os.path.dirname(save_dir), 'plots')
        else:
            plots_dir = os.path.join(save_dir, 'plots')
        plots_dir = os.path.normpath(plots_dir)
        os.makedirs(plots_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(loss_callback.losses)
        plt.title(f'Training Loss - {algo} (Seed {seed})')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        loss_plot_path = os.path.join(plots_dir, f'{algo.lower()}_loss_seed_{seed}.png')
        plt.savefig(loss_plot_path)
        plt.close()
        logging.info(f'Training loss plot saved to {loss_plot_path}')
    else:
        logging.warning('No training loss values captured; loss plot was not saved.')
    
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