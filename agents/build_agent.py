from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3

def build_agent(algo, env, seed):
    """
    Build and return the RL agent based on algorithm.
    """
    if algo == 'A2C':
        # Params from notebook
        model_kwargs = {"n_steps": 10, "ent_coef": 0.005, "learning_rate": 0.0001}
        model = A2C("MlpPolicy", env, seed=seed, **model_kwargs)
    elif algo == 'PPO':
        model_kwargs = {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 0.0001, "batch_size": 128}
        model = PPO("MlpPolicy", env, seed=seed, **model_kwargs)
    elif algo == 'DDPG':
        model = DDPG("MlpPolicy", env, seed=seed)
    elif algo == 'SAC':
        model = SAC("MlpPolicy", env, seed=seed)
    elif algo == 'TD3':
        model = TD3("MlpPolicy", env, seed=seed)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    
    return model