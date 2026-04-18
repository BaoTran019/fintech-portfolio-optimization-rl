import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="RL Portfolio Optimization Training and Evaluation")
    
    # Required arguments
    parser.add_argument('--algo', type=str, required=True, choices=['A2C', 'PPO', 'DDPG', 'SAC', 'TD3'], help='Algorithm to use')
    parser.add_argument('--timesteps', type=int, required=False, default=None, help='Number of training timesteps (required only for train.py and run.py)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    
    # Optional arguments
    parser.add_argument('--n_seeds', type=int, default=1, help='Number of seeds to run (default: 1)')
    parser.add_argument('--data_source', type=str, default='csv', choices=['api', 'csv'], help='Data source (default: csv)')
    parser.add_argument('--data_path', type=str, default='dataset/5_vn30_vnsi_symbols_data.xlsx', help='Path to main data file')
    parser.add_argument('--vnindex_path', type=str, default='dataset/vnindex.xlsx', help='Path to VNINDEX data file')
    parser.add_argument('--save_path', type=str, default='results/models/', help='Path to save models')
    
    return parser