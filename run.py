import subprocess
import sys
from config.parser import get_parser

def run_pipeline(algo, timesteps, seed, n_seeds=1, data_source='csv', data_path='dataset/5_vn30_vnsi_symbols_data.xlsx', vnindex_path='dataset/vnindex.xlsx', processed_data_path='dataset/processed/processed.csv'):
    # Train
    cmd_train = [sys.executable, 'train.py', '--algo', algo, '--timesteps', str(timesteps), '--seed', str(seed), '--n_seeds', str(n_seeds), '--processed_data_path', processed_data_path]
    
    # Validate
    for s in range(seed, seed + n_seeds):
        cmd_validate = [sys.executable, 'python', 'validate.py', '--algo', algo, '--seed', str(s), '--processed_data_path', processed_data_path]
        subprocess.run(cmd_validate)
        
        # Backtest
        cmd_backtest = [sys.executable, 'python', 'backtest.py', '--algo', algo, '--seed', str(s), '--processed_data_path', processed_data_path]
        subprocess.run(cmd_backtest)

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    run_pipeline(
        args.algo,
        args.timesteps,
        args.seed,
        args.n_seeds,
        args.data_source,
        args.data_path,
        args.vnindex_path,
        args.processed_data_path
    )

if __name__ == "__main__":
    main()