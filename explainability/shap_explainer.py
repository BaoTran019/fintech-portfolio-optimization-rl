import shap
import numpy as np

def compute_shap_values(model, env, n_samples=100):
    """
    Compute SHAP values for the trained model using state samples.
    """
    # Sample states from environment
    states = []
    env.reset()
    for _ in range(n_samples):
        action = env.action_space.sample()  # Random action
        state, _, done, _ = env.step(action)
        states.append(state)
        if done:
            env.reset()
    
    states = np.array(states)
    
    # Use SHAP to explain model predictions
    # Assuming model has predict method
    def predict_fn(states):
        actions, _ = model.predict(states)
        return actions
    
    explainer = shap.Explainer(predict_fn, states)
    shap_values = explainer(states)
    
    return shap_values