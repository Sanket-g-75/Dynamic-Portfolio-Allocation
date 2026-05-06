import pandas as pd
import numpy as np
import os
import yfinance as yf
import datetime
import data.metadata as d

path = os.getcwd()

def download_new_ticker(ticker):
    import yfinance as yf
    import os

    print(f"Downloading {ticker}...")

    data = yf.download(
        ticker,
        period="10y",
        interval="1d",
        progress=False
    )

    # Check if data is empty
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")

    data.reset_index(inplace=True)

    # Rename columns
    for col in data.columns:
        if col != "Date":
            data.rename(columns={col: f"{ticker}-{col}"}, inplace=True)

    # Save file
    file_path = os.path.join(path, 'data', 'raw', f'{ticker}.csv')
    data.to_csv(file_path, index=False)

def update_ticker(ticker,path):
    # data = yf.Tickers(ticker).history(period='10y')
    label = f'{ticker}.csv'
    data = pd.read_csv(os.path.join(path,'data','raw',label))
    data = data.iloc[2:,:]
    data.rename(columns={'Price':'Date'},inplace=True)
    for i in data.columns:
        if i != 'Date':
            data.rename(columns={f'{i}':f'{ticker}-{i}'},inplace=True)
    data.to_csv(os.path.join(path,'data','processed',f'p-{ticker}.csv'))
    # d.features.append(ticker)
   #  d.lastdate = data.iloc[[-1]]['Date']




def update_data():
    print("Starting data ingestion process...")

    for ticker in d.features:
        try:
            print(f"Downloading data for {ticker}...")
            download_new_ticker(ticker)

            print(f"Processing data for {ticker}...")
            update_ticker(ticker, path)

        except Exception as e:
            print(f"Failed for {ticker}: {e}")

    print("Data ingestion complete.")

if __name__ == "__main__":
    update_data()
