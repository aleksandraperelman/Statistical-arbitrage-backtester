import os
import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

def fetch_prices(
        tickers: list[str],
        start: str = '2016-01-01',
        end: str = '2026-01-01',
        use_cache: bool = True,
) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = "_".join(sorted(tickers)) + f"_{start}_{end}.csv"
    cache_path = os.path.join(CACHE_DIR, cache_key)

    if use_cache and os.path.exists(cache_path):
        print(f'[fetcher] Loading from cache: {cache_path}')
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"[fetcher] Downloading {tickers} from {start} to {end}...")
    raw = yf.download(tickers, start = start, end = end, auto_adjust = True, progress = False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.ffill().dropna()

    if use_cache:
        prices.to_csv(cache_path)
        print(f"[fetcher] Cached to {cache_path}")

    return prices

def to_log_prices(prices = pd.DataFrame) -> pd.DataFrame:
    return np.log(prices)

def get_returns(prices = pd.DataFrame) -> pd.DataFrame:
    return np.log(prices/prices.shift(1)).dropona()
