import os
import time
import requests
import datetime

# Save directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(SAVE_DIR, exist_ok=True)

# Yahoo Finance uses UNIX timestamps
def get_unix_time(year, month, day):
    return int(datetime.datetime(year, month, day).timestamp())

# Date range (10+ years)
start = get_unix_time(2010, 1, 1)
end = get_unix_time(2025, 1, 1)

# Your tickers (use simple ones first)
tickers = ["NVDA", "TSLA", "GOOG", "JPM", "XOM"]

def download_csv(ticker):
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start}&period2={end}&interval=1d&events=history"

    print(f"Downloading {ticker}...")

    try:
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed for {ticker} (status {response.status_code})")
            return

        file_path = os.path.join(SAVE_DIR, f"{ticker}.csv")

        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"Saved: {file_path}")

        time.sleep(1)  # avoid rate limit

    except Exception as e:
        print(f"Error for {ticker}: {e}")


if __name__ == "__main__":
    for t in tickers:
        download_csv(t)

    print("Download complete.")