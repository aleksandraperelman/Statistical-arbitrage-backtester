import numpy as np
import pandas as pd
from dataclasses import dataclass
from strategy_signals import SignalConfig, generate_signals
from Backtest.backtest_engine import run_backtest


@dataclass
class SensitivityResult:
    lookback_values: np.ndarray
    entry_values: np.ndarray
    sharpe_grid: np.ndarray
    return_grid: np.ndarray
    drawdown_grid: np.ndarray
    best_sharpe: float
    best_params: dict
    fraction_profitable: float
    fraction_above_half: float
    median_sharpe: float

    def summary(self):
        return (
            f"\n{'='*55}\n"
            f"Parameter Sensitivity Analysis\n"
            f"{'='*55}\n"
            f"  Parameter combinations: {self.sharpe_grid.size}\n"
            f"  Best Sharpe:            {self.best_sharpe:.3f}\n"
            f"  Best params:            lookback={self.best_params['lookback']}, "
            f"entry={self.best_params['entry_threshold']:.2f}\n"
            f"  Median Sharpe:          {self.median_sharpe:.3f}\n"
            f"  Fraction profitable:    {self.fraction_profitable*100:.1f}%\n"
            f"  Fraction Sharpe > 0.5:  {self.fraction_above_half*100:.1f}%\n"
        )


def run_sensitivity_analysis(
    log_prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    lookback_range: tuple = (60, 301, 20),
    entry_range: tuple = (1.0, 3.1, 0.25),
    transaction_cost: float = 0.001,
    verbose: bool = True,
) -> SensitivityResult:
    """
    Sweep over lookback and entry_threshold, running a full backtest
    for each combination.

    This answers the question: is our chosen parameter set (lookback=200,
    entry=2.0) genuinely optimal, or did we just get lucky picking it?

    A robust strategy shows a broad ridge of good Sharpe ratios.
    An overfit strategy shows a single bright spot surrounded by poor results.
    """
    lookbacks = np.arange(*lookback_range)
    entries = np.arange(*entry_range)

    n_lb, n_en = len(lookbacks), len(entries)
    sharpe_grid = np.full((n_lb, n_en), np.nan)
    return_grid = np.full((n_lb, n_en), np.nan)
    dd_grid = np.full((n_lb, n_en), np.nan)

    total = n_lb * n_en
    done = 0

    if verbose:
        print(f"[sensitivity] Running {total} backtests "
              f"({n_lb} lookbacks x {n_en} entry thresholds)...")

    for i, lb in enumerate(lookbacks):
        for j, entry in enumerate(entries):
            try:
                config = SignalConfig(
                    lookback=int(lb),
                    entry_threshold=float(entry),
                    exit_threshold=0.5,
                )
                signals = generate_signals(log_prices, ticker_a, ticker_b, config)

                if signals.empty or (signals["position"] == 0).all():
                    sharpe_grid[i, j] = 0.0
                    return_grid[i, j] = 0.0
                    dd_grid[i, j] = 0.0
                    continue

                result = run_backtest(
                    signals, log_prices, ticker_a, ticker_b,
                    transaction_cost=transaction_cost,
                )
                sharpe_grid[i, j] = result.sharpe_ratio
                return_grid[i, j] = result.total_return
                dd_grid[i, j] = result.max_drawdown

            except Exception:
                sharpe_grid[i, j] = np.nan

            done += 1
            if verbose and done % 20 == 0:
                print(f"  ... {done}/{total}")

    if verbose:
        print("[sensitivity] Done.")

    valid = sharpe_grid[~np.isnan(sharpe_grid)].flatten()
    best_idx = np.unravel_index(np.nanargmax(sharpe_grid), sharpe_grid.shape)

    return SensitivityResult(
        lookback_values=lookbacks,
        entry_values=entries,
        sharpe_grid=sharpe_grid,
        return_grid=return_grid,
        drawdown_grid=dd_grid,
        best_sharpe=float(sharpe_grid[best_idx]),
        best_params={
            "lookback": int(lookbacks[best_idx[0]]),
            "entry_threshold": float(entries[best_idx[1]]),
        },
        fraction_profitable=float(np.mean(valid > 0)),
        fraction_above_half=float(np.mean(valid > 0.5)),
        median_sharpe=float(np.median(valid)),
    )