import argparse
import os
from data.load_data import load_data
from data.preprocess import preprocess_data


def main():
    parser = argparse.ArgumentParser(description='Preprocess raw stock data into a reproducible dataset CSV file.')
    parser.add_argument('--data_source', type=str, default='csv', choices=['api', 'csv'], help='Data source for raw input')
    parser.add_argument('--data_path', type=str, default='dataset/5_vn30_vnsi_symbols_data.xlsx', help='Path to raw main data file')
    parser.add_argument('--vnindex_path', type=str, default='dataset/vnindex.xlsx', help='Path to raw VNINDEX data file')
    parser.add_argument('--output_path', type=str, default='dataset/processed/processed.csv', help='Path to save processed CSV')
    parser.add_argument('--force', action='store_true', help='Overwrite existing processed data file')

    args = parser.parse_args()

    if os.path.exists(args.output_path) and not args.force:
        print(f'Processed dataset already exists at {args.output_path}. Use --force to overwrite.')
        return

    df, vnindex_df = load_data(args.data_source, args.data_path, args.vnindex_path)
    processed_df = preprocess_data(df, vnindex_df)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    processed_df.to_csv(args.output_path, index=False)
    print(f'Processed data saved to {args.output_path}')


if __name__ == '__main__':
    main()
