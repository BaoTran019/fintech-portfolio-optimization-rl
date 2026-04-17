import subprocess
import sys
from config.parser import get_parser

def run_pipeline(algo, timesteps, seed, n_seeds=1, data_source='csv', data_path='dataset/5_vn30_vnsi_symbols_data.xlsx', vnindex_path='dataset/vnindex.xlsx'):
    # Train
    cmd_train = [sys.executable, 'train.py', '--algo', algo, '--timesteps', str(timesteps), '--seed', str(seed), '--n_seeds', str(n_seeds), '--data_source', data_source, '--data_path', data_path, '--vnindex_path', vnindex_path]
    
    # Validate
    for s in range(seed, seed + n_seeds):
        cmd_validate = [sys.executable, 'validate.py', '--algo', algo, '--seed', str(s), '--data_source', data_source, '--data_path', data_path, '--vnindex_path', vnindex_path]
        subprocess.run(cmd_validate)
        
        # Backtest
        cmd_backtest = [sys.executable, 'backtest.py', '--algo', algo, '--seed', str(s), '--data_source', data_source, '--data_path', data_path, '--vnindex_path', vnindex_path]
        subprocess.run(cmd_backtest)

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    run_pipeline(args.algo, args.timesteps, args.seed, args.n_seeds, args.data_source, args.data_path, args.vnindex_path)

if __name__ == "__main__":
    main()