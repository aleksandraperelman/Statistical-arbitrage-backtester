import numpy as np
import pandas as pd
import dataclasses as dataclass

@dataclass
class SignalConfig:
    lookback: int = 60
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5

def compute_rolling_spread(
    log_prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    lookback: int,
) -> tuple[pd.Series, pd.Series]:

    y = log_prices[ticker_a]
    x = log_prices[ticker_b]
    n = len(y)

    spreads = np.full(n, np.nan)
    hedge_ratios = np.full(n, np.nan)

    for i in range(lookback, n):
        y_win = y.iloc[i - lookback:i].values
        x_win = x.iloc[i - lookback:i].values

        X = np.column_stack([np.ones(lookback), x_win])
        coefs, *_ = np.linalg.lstsq(X, y_win, rcond= None)
        alpha, beta = coefs

        hedge_ratios[i] = beta
        spreads[i] = y.iloc[i] - beta*x.iloc[i] - alpha

    idx = log_prices.index
    return (
        pd.Series(spreads, index=idx, name="spread"),
        pd.Series(hedge_ratios, index=idx, name="hedge_ratio"),
    )

def compute_zscore(spread: pd.Series, lookback: int) -> pd.Series:
    mean = spread.rolling(lookback, min_periods=lookback).mean()
    std = spread.rolling(lookback, min_periods=lookback).std(ddof=1)
    std = std.replace(0, np.nan)
    return ((spread - mean) / std).rename("zscore")
def generate_signals(
    log_prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    config: SignalConfig,
) -> pd.DataFrame:

    spread, hedge_ratios = compute_rolling_spread(log_prices, ticker_a, ticker_b, config.lookback)
    zscore = compute_zscore(spread, config.lookback)

    position = pd.Series(0.0, index=log_prices.index)
    current = 0.0

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z):
            position.iloc[i] = 0.0
            continue

        if current == 0:
            if z < -config.entry_threshold:
                current = 1.0
            elif z > config.entry_threshold:
                current = -1.0
        elif current == 1.0:
            if z > -config.exit_threshold:
                current = 0.0
        elif current == -1.0:
            if z < config.exit_threshold:
                current = 0.0

        position.iloc[i] = current

    signals = pd.DataFrame({
        "spread": spread,
        "hedge_ratio": hedge_ratios,
        "zscore": zscore,
        "position": position,
    })

    return signals.dropna(subset=["zscore"])