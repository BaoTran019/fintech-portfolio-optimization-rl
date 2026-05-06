from finrl.agents.stablebaselines3.models import DRLAgent

def build_agent(algo, env, seed):
    """
    Build and return the RL agent and model based on algorithm.
    """
    algo_key = algo.lower()
    agent = DRLAgent(env=env)

    if algo_key == 'a2c':
        model_kwargs = {"n_steps": 10, "ent_coef": 0.005, "learning_rate": 0.0001, "seed": seed}
    elif algo_key == 'ppo':
        model_kwargs = {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 0.0001, "batch_size": 128, "seed": seed}
    elif algo_key in ('ddpg', 'sac', 'td3'):
        model_kwargs = {"seed": seed}
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    model = agent.get_model(algo_key, model_kwargs=model_kwargs)
    return agent, model