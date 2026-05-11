import os
import shap
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from data.load_data import load_processed_data
from data.preprocess import split_data
from env.trading_env import StockPortfolioEnv


TECHNICAL_INDICATORS = [
    'macd',
    'boll_ub',
    'boll_lb',
    'rsi_30',
    'cci_30',
    'dx_30',
    'close_30_sma',
    'close_60_sma',
    'change'
]

STOCKS = [
    'FPT', 'GAS', 'HPG', 'MSN', 'MWG',
    'SSI', 'STB', 'VCB', 'VIC', 'VNM'
]


# =========================================================
# CREATE ENV
# =========================================================

def create_environment(processed_data_path, seed=42):

    df = load_processed_data(processed_data_path)

    _, _, test = split_data(df)

    test = test.sort_values(['date', 'tic']).reset_index(drop=True)

    test.index = test.date.factorize()[0]

    stock_dimension = len(test.tic.unique())

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 10000000,
        "transaction_cost_pct": 0.001,
        "state_space": stock_dimension,
        "stock_dim": stock_dimension,
        "tech_indicator_list": TECHNICAL_INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "seed": seed
    }

    env = StockPortfolioEnv(df=test, **env_kwargs)

    return env


# =========================================================
# COLLECT STATES
# =========================================================

def collect_states(model, env, n_samples=200):

    states = []

    state = env.reset()

    for _ in range(n_samples):

        action, _ = model.predict(state, deterministic=True)

        next_state, reward, done, info = env.step(action)

        states.append(next_state)

        state = next_state

        if done:
            state = env.reset()

    return np.array(states)


# =========================================================
# EXTRACT TECHNICAL FEATURES ONLY
# =========================================================

def extract_stock_features(states, stock_idx):

    """
    states shape:
    (n_samples, 19, 10)

    first 10 rows:
        covariance matrix

    last 9 rows:
        technical indicators
    """

    technical_rows = states[:, 10:, :]

    # extract one stock only
    X = technical_rows[:, :, stock_idx]

    return X


# =========================================================
# SHAP EXPLAINER
# =========================================================

def create_predict_function(model, stock_idx):

    def predict_fn(X):

        X = np.array(X)

        batch_size = X.shape[0]

        predictions = []

        for i in range(batch_size):

            technical_part = X[i]

            full_state = np.zeros((19, 10))

            # keep covariance = 0
            # only inject technical indicators
            full_state[10:, stock_idx] = technical_part

            action, _ = model.predict(full_state, deterministic=True)

            predictions.append(action[stock_idx])

        return np.array(predictions)

    return predict_fn


# =========================================================
# MAIN SHAP FUNCTION
# =========================================================

def explain_stock(
    model_path,
    processed_data_path,
    stock_name='MWG',
    n_samples=200,
    background_size=50,
    seed=42
):

    os.makedirs("results/shap", exist_ok=True)

    stock_idx = STOCKS.index(stock_name)

    print(f"\nExplaining stock: {stock_name}")
    print(f"Stock index: {stock_idx}")

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model = PPO.load(model_path)

    # -----------------------------------------------------
    # ENV
    # -----------------------------------------------------

    env = create_environment(processed_data_path, seed)

    # -----------------------------------------------------
    # COLLECT STATES
    # -----------------------------------------------------

    print("\nCollecting states...")

    states = collect_states(model, env, n_samples)

    print("States shape:", states.shape)

    # -----------------------------------------------------
    # EXTRACT FEATURES
    # -----------------------------------------------------

    X = extract_stock_features(states, stock_idx)

    print("Feature matrix shape:", X.shape)

    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    predict_fn = create_predict_function(model, stock_idx)

    background = X[:background_size]

    test_samples = X[background_size:background_size + 50]

    print("\nCreating SHAP explainer...")

    explainer = shap.KernelExplainer(
        predict_fn,
        background
    )

    print("\nComputing SHAP values...")

    shap_values = explainer.shap_values(
        test_samples,
        nsamples=100
    )

    shap_values = np.array(shap_values)

    print("SHAP values shape:", shap_values.shape)

    # =====================================================
    # SUMMARY PLOT
    # =====================================================

    print("\nCreating summary plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        test_samples,
        feature_names=TECHNICAL_INDICATORS,
        show=False
    )

    plt.title(f"SHAP Summary Plot - {stock_name}")

    plt.tight_layout()

    plt.savefig(
        f"results/shap/shap_summary_{stock_name}.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    # =====================================================
    # WATERFALL PLOT
    # =====================================================

    print("\nCreating waterfall plot...")

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=test_samples[0],
        feature_names=TECHNICAL_INDICATORS
    )

    plt.figure(figsize=(10, 6))

    shap.plots.waterfall(
        explanation,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f"results/shap/shap_waterfall_{stock_name}.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print("\nDONE")
    print(f"Saved summary plot:")
    print(f"results/shap/shap_summary_{stock_name}.png")

    print(f"\nSaved waterfall plot:")
    print(f"results/shap/shap_waterfall_{stock_name}.png")


# =========================================================
# RUN
# =========================================================

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", type=str, default="PPO")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--processed_data_path",
        type=str,
        required=True
    )

    args = parser.parse_args()

    model_path = f"results/models/{args.algo.lower()}_seed_{args.seed}.zip"

    explain_stock(
        model_path=model_path,
        processed_data_path=args.processed_data_path,
        stock_name="MWG",
        n_samples=200,
        background_size=50,
        seed=args.seed
    )