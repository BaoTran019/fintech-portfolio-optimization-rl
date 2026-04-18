import os
import argparse
import numpy as np
from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
import matplotlib.pyplot as plt
from data.load_data import load_data
from data.preprocess import preprocess_data, split_data
from env.trading_env import StockPortfolioEnv

TECHNICAL_INDICATORS = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
    'close_30_sma', 'close_60_sma', 'change'
]

def load_trained_model(algo, seed, model_path='results/models/'):
    """
    Load a trained model from file.
    """
    model_file = os.path.join(model_path, f'{algo.lower()}_seed_{seed}.zip')

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    if algo == 'A2C':
        model = A2C.load(model_file)
    elif algo == 'PPO':
        model = PPO.load(model_file)
    elif algo == 'DDPG':
        model = DDPG.load(model_file)
    elif algo == 'SAC':
        model = SAC.load(model_file)
    elif algo == 'TD3':
        model = TD3.load(model_file)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    return model

def create_test_environment(data_source, data_path, vnindex_path, seed):
    """
    Create the test environment for SHAP analysis.
    """
    # Load and preprocess data
    df, vnindex_df = load_data(data_source, data_path, vnindex_path)
    df = preprocess_data(df, vnindex_df)

    # Split data (use test set)
    _, test = split_data(df)

    # Clean test data
    unique_tickers = test.tic.unique()
    test = test[test.tic.isin(unique_tickers)]
    test = test.sort_values(['date', 'tic'])
    test = test.drop_duplicates(subset=['date', 'tic'], keep='last')
    test = test.dropna(subset=['cov_list', 'return_list'])
    test = test.sort_values(['date', 'tic']).reset_index(drop=True)
    test.index = test.date.factorize()[0]

    # Environment setup
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
        "reward_scaling": 1e-4,
        "seed": seed
    }

    env = StockPortfolioEnv(df=test, **env_kwargs)
    return env

def compute_shap_explanation(model, env, n_samples=100, max_evals=200):
    """
    Compute SHAP values for the trained model.
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Install with: pip install shap")
        return None

    print(f"Computing SHAP values with {n_samples} samples...")

    # Sample states from environment
    states = []
    env.reset()
    for _ in range(n_samples):
        action, _ = model.predict(env.state)  # Use current state
        state, _, done, _ = env.step(action)
        states.append(state)
        if done:
            env.reset()

    if not states:
        print("No states collected for SHAP analysis")
        return None

    states = np.array(states)
    state_shape = states.shape[1:]

    # Convert observations to flat feature vectors for SHAP
    if states.ndim == 3:
        states_flat = states.reshape(states.shape[0], -1)
    else:
        states_flat = states

    def reshape_for_model(flat_states):
        flat_states = np.array(flat_states)
        if flat_states.ndim == 1:
            flat_states = flat_states.reshape(1, -1)
        if states.ndim == 3:
            return flat_states.reshape((-1,) + state_shape)
        return flat_states

    # Create SHAP explainer
    def predict_fn(flat_states):
        obs = reshape_for_model(flat_states)
        actions, _ = model.predict(obs, deterministic=True)
        return actions

    # Use a subset for explanation to avoid memory issues
    background_states = states_flat[:min(50, len(states_flat))]
    test_states = states_flat[:min(20, len(states_flat))]

    try:
        explainer = shap.KernelExplainer(predict_fn, background_states)
        shap_values = explainer.shap_values(test_states, max_evals=max_evals)

        print(f"SHAP analysis completed. Shape: {np.array(shap_values).shape}")
        return shap_values, test_states

    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        return None

def plot_shap_summary(shap_values, feature_names, algo, seed, save_path='results/plots/'):
    """
    Create and save SHAP summary plot.
    """
    try:
        import shap

        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary Plot - {algo} (Seed {seed})')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{algo.lower()}_shap_summary_seed_{seed}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"SHAP summary plot saved to {save_path}")

    except Exception as e:
        print(f"Error creating SHAP plot: {e}")

def explain_model(algo, seed, data_source='csv', data_path='dataset/5_vn30_vnsi_symbols_data.xlsx',
                 vnindex_path='dataset/vnindex.xlsx', model_path='results/models/', n_samples=100):
    """
    Main function to explain a trained model using SHAP.
    """
    print(f"Explaining {algo} model (seed {seed})...")

    # Load trained model
    model = load_trained_model(algo, seed, model_path)

    # Create test environment
    env = create_test_environment(data_source, data_path, vnindex_path, seed)

    # Compute SHAP values
    result = compute_shap_explanation(model, env, n_samples)

    if result is not None:
        shap_values, test_states = result

        # Create feature names for plotting
        stock_dim = env.state.shape[1]
        state_rows = env.state.shape[0]
        feature_names = []

        # Generate names for flattened covariance and technical indicator rows
        for row in range(state_rows):
            if row < stock_dim:
                for col in range(stock_dim):
                    feature_names.append(f'cov_{row}_{col}')
            else:
                indicator_idx = row - stock_dim
                indicator_name = TECHNICAL_INDICATORS[indicator_idx]
                for col in range(stock_dim):
                    feature_names.append(f'{indicator_name}_{col}')

        # Plot SHAP summary
        plot_shap_summary(shap_values, feature_names, algo, seed)

        print(f"SHAP explanation completed for {algo} (seed {seed})")
    else:
        print("SHAP analysis failed")

def main():
    parser = argparse.ArgumentParser(description="Explain trained RL models using SHAP")
    parser.add_argument('--algo', type=str, required=True, choices=['A2C', 'PPO', 'DDPG', 'SAC', 'TD3'],
                       help='Algorithm to explain')
    parser.add_argument('--seed', type=int, required=True, help='Seed of the trained model')
    parser.add_argument('--data_source', type=str, default='csv', choices=['api', 'csv'],
                       help='Data source (default: csv)')
    parser.add_argument('--data_path', type=str, default='dataset/5_vn30_vnsi_symbols_data.xlsx',
                       help='Path to main data file')
    parser.add_argument('--vnindex_path', type=str, default='dataset/vnindex.xlsx',
                       help='Path to VNINDEX data file')
    parser.add_argument('--model_path', type=str, default='results/models/',
                       help='Path to trained models')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples for SHAP analysis')

    args = parser.parse_args()

    explain_model(
        algo=args.algo,
        seed=args.seed,
        data_source=args.data_source,
        data_path=args.data_path,
        vnindex_path=args.vnindex_path,
        model_path=args.model_path,
        n_samples=args.n_samples
    )

if __name__ == "__main__":
    main()