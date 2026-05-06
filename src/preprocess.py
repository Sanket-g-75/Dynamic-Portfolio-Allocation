import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------
# Combine all processed feature files
# -------------------------------
def combine_data(features):
    df = pd.DataFrame()

    for i in features:
        file_path = os.path.join(BASE_DIR, 'data', 'processed', f'p-{i}.csv')

        if df.empty:
            df = pd.read_csv(file_path, index_col=0)
            df.set_index('Date', inplace=True)
        else:
            right = pd.read_csv(file_path, index_col=0)
            right.set_index('Date', inplace=True)
            df = pd.concat([df, right], join='inner', axis=1)

    # Save final dataset
    output_path = os.path.join(BASE_DIR, 'data', 'final', 'Dataset.csv')
    df.to_csv(output_path)

    print(f"Dataset saved at: {output_path}")

def combine_stocks(stocks):
    df = pd.DataFrame()

    for i in stocks:
        file_path = os.path.join(BASE_DIR, 'data', 'processed', f'p-{i}.csv')

        if df.empty:
            df = pd.read_csv(file_path, index_col=0)
            df.set_index('Date', inplace=True)
        else:
            right = pd.read_csv(file_path, index_col=0)
            right.set_index('Date', inplace=True)
            df = pd.concat([df, right], join='inner', axis=1)

    # Save final dataset
    output_path = os.path.join(BASE_DIR, 'data', 'final', 'Stocks.csv')
    df.to_csv(output_path)

    print(f"Dataset saved at: {output_path}")


# -------------------------------
# Scaling function
# -------------------------------
def scaling(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled


# -------------------------------
# Run as script
# -------------------------------
if __name__ == '__main__':
    import data.metadata as d

    print("Starting data preprocessing...")
    combine_data(d.features)
    print("Data preprocessing complete.")
    
    print("Starting for stocks")
    combine_stocks(d.stocks)
    print("Stocks preprocessing complete.")