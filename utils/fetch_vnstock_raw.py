def process_data(symbol, source, start, end):
    try:
        quote = Quote(symbol=symbol, source=source)
        quote_df = quote.history(start=start, end=end, interval='1D')
        quote_df['tic'] = symbol
        quote_df['day'] = quote_df['time'].dt.dayofweek
        return quote_df
    except Exception as e:
        print(f"Error when processing {symbol}: {e}")
        return None
    
    
def init_data_set():

    # Tạo thư mục chứa dữ liệu nếu chưa có
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    
    file_path = os.path.join(args.folder, args.output)
    all_df = []

    print(f"--- Begin to get data from {args.start} to {args.end} ---")

    for symbol in args.symbols:
        print(f"Processing {symbol}...")
        df = process_data(symbol, args.source, args.start, args.end)
        if df is not None and not df.empty:
            # Ensure 'date' column exists, if not, try to rename 'time' to 'date'
            # If vnstock returns 'time', we rename it to 'date'
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'date'})
            
            # If for some reason the 'date' column is still missing (only index exists)
            if 'date' not in df.columns:
                df = df.reset_index().rename(columns={'index': 'date', 'level_0': 'date'})

            # Buộc cột 'tic' phải có (VNINDEX thường bị thiếu cột này)
            df['tic'] = symbol
            
            all_df.append(df)

    if all_df:
        # Merge all dataframes into one final dataframe
        final_df = pd.concat(all_df, ignore_index=True)
        
        # Arrange records by date and ticker
        final_df = final_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        
        # saving Excel file
        final_df.to_excel(file_path, index=False)
        print(f"\nGetting data successful: Saved {len(final_df)} lines into {file_path}")
    else:
        print("\nFail to get data.")


if __name__ == "__main__":
    from vnstock import Quote

    import argparse
    import pandas as pd
    import os
    
    parser = argparse.ArgumentParser(description="Getting stock data from vnstock and saving to Excel")
    
    symbols_list_1 = ['FPT', 'VIC', 'GAS', 'SSI', 'HPG', 'VNM', 'VCB', 'STB', 'MSN', 'MWG']
    symbols_list_2 = ['MWG', 'BID', 'VCB', 'VIC', 'MBB']

    source = 'VCI'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, '..', 'dataset')

    START_DATE = '2015-01-01'
    END_DATE = '2025-12-31'

    # Config default parameters
    parser.add_argument('--symbols', nargs='+', default=symbols_list_2)
    parser.add_argument('--start', type=str, default=START_DATE)
    parser.add_argument('--end', type=str, default=END_DATE)
    parser.add_argument('--source', type=str, default=source)
    parser.add_argument('--output', type=str, default='symbols_data.xlsx')
    parser.add_argument('--folder', type=str, default='../dataset')

    args = parser.parse_args()
    
    init_data_set()