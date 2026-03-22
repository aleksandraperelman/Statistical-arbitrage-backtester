import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    observed_sharpe: float
    null_sharpes: np.ndarray
    pvalue: float
    percentile: float
    n_simulations: int
    block_length: int
    is_significant: bool

    def summary(self):
        sig = "SIGNIFICANT ✓ (p < 0.05)" if self.is_significant else "NOT significant ✗ (p >= 0.05)"
        return (
            f"\n{'='*55}\n"
            f"Monte Carlo Bootstrap Significance Test\n"
            f"{'='*55}\n"
            f"  Observed Sharpe:      {self.observed_sharpe:.4f}\n"
            f"  Null mean Sharpe:     {self.null_sharpes.mean():.4f}\n"
            f"  Null std Sharpe:      {self.null_sharpes.std():.4f}\n"
            f"  Null 95th pctile:     {np.percentile(self.null_sharpes, 95):.4f}\n"
            f"  Null 99th pctile:     {np.percentile(self.null_sharpes, 99):.4f}\n"
            f"  Observed percentile:  {self.percentile:.1f}th\n"
            f"  p-value:              {self.pvalue:.4f}\n"
            f"  Simulations:          {self.n_simulations}\n"
            f"  Block length:         {self.block_length} days\n"
            f"  Result:               {sig}\n"
        )


def _sharpe(returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualized Sharpe ratio. Fast version for simulation loops."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std(ddof=1) * np.sqrt(trading_days)


def _block_bootstrap_sample(
    returns: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Resample the return series by drawing random contiguous blocks.

    Why blocks and not individual days?
    Financial returns have serial dependence — volatility clusters,
    momentum persists for days, mean-reversion takes weeks.
    If we shuffle individual days we destroy that structure, making
    the null distribution too easy to beat (understating the null Sharpe).
    Block bootstrap preserves local dependence by keeping consecutive
    returns together. Block length ≈ sqrt(T) is a standard heuristic.
    """
    T = len(returns)
    n_blocks = int(np.ceil(T / block_length))
    starts = rng.integers(0, T - block_length + 1, size=n_blocks)
    blocks = [returns[s: s + block_length] for s in starts]
    return np.concatenate(blocks)[:T]


def run_bootstrap_test(
    strategy_returns: pd.Series,
    n_simulations: int = 2000,
    block_length: int = None,
    seed: int = 42,
    trading_days: int = 252,
) -> BootstrapResult:
    """
    The key question: is our Sharpe ratio real, or just luck?

    We answer it by building a null distribution — the distribution of
    Sharpe ratios we'd expect to see if the strategy had zero edge.

    Method:
      1. Take our actual return series (which has real statistical properties)
      2. Resample it with replacement (block bootstrap) 2000 times
      3. Compute the Sharpe for each resample
      4. Ask: what fraction of null Sharpes >= our observed Sharpe?
         That fraction is the p-value.

    If p < 0.05: only 5% chance the result is due to chance → significant.
    If p >= 0.05: the result could plausibly be random → not significant.
    """
    returns = strategy_returns.dropna().values
    T = len(returns)

    if block_length is None:
        block_length = max(1, int(np.ceil(np.sqrt(T))))

    observed = _sharpe(returns, trading_days)
    rng = np.random.default_rng(seed)

    null_sharpes = np.array([
        _sharpe(_block_bootstrap_sample(returns, block_length, rng), trading_days)
        for _ in range(n_simulations)
    ])

    pvalue = float(np.mean(null_sharpes >= observed))
    percentile = float(np.mean(null_sharpes < observed) * 100)

    return BootstrapResult(
        observed_sharpe=observed,
        null_sharpes=null_sharpes,
        pvalue=pvalue,
        percentile=percentile,
        n_simulations=n_simulations,
        block_length=block_length,
        is_significant=pvalue < 0.05,
    )