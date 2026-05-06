import os
import pandas as pd
import numpy as np

from src.preprocess import scaling
from src.pfutils import generate_weekly_allocations

LOOKBACK = 60


def create_latest_sequence(df):
    df_scaled = scaling(df.values)
    df_scaled = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)

    last_seq = df_scaled.iloc[-LOOKBACK:].values
    return np.expand_dims(last_seq, axis=0)


def run_inference(model):
    dataset_path = os.path.join('data', 'final', 'Dataset.csv')
    stocks_path = os.path.join('data', 'final', 'Stocks.csv')

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    # Load Stocks.csv to get the correct tickers (target variables)
    Y_df = pd.read_csv(stocks_path, index_col=0, parse_dates=True)
    tickers = [c for c in Y_df.columns if c.endswith('-Close') and 'VIX' not in c]

    X_input = create_latest_sequence(df)

    alloc = generate_weekly_allocations(
        model,
        X_input,
        [df.index[-1]],
        tickers=tickers,
        rebalance_weeks=1,
        temperature=0.1
    )

    latest_date = str(alloc.index[-1].date())

    weights = alloc.iloc[-1].to_dict()

    cleaned_weights = {
        k.replace('-Close', ''): round(float(v), 4)
        for k, v in weights.items()
    }

    return {
        "date": latest_date,
        "portfolio_weights": cleaned_weights
    }