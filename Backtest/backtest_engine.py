import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestResult:
    returns: pd.Series
    equity_curve: pd.Series
    positions: pd.Series
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    hit_rate: float
    num_trades: int

    def summary(self):
        return (
            f"\n{'='*45}\n"
            f"Backtest Performance\n"
            f"{'='*45}\n"
            f"  Total return:       {self.total_return*100:+.2f}%\n"
            f"  Annualized return:  {self.annualized_return*100:+.2f}%\n"
            f"  Annualized vol:     {self.annualized_volatility*100:.2f}%\n"
            f"  Sharpe ratio:       {self.sharpe_ratio:.3f}\n"
            f"  Max drawdown:       {self.max_drawdown*100:.2f}%\n"
            f"  Calmar ratio:       {self.calmar_ratio:.3f}\n"
            f"  Hit rate:           {self.hit_rate*100:.1f}%\n"
            f"  Number of trades:   {self.num_trades}\n"
        )


def _max_drawdown(equity: pd.Series) -> float:
    """
    Largest peak-to-trough decline in the equity curve.
    Returns a negative number e.g. -0.12 means 12% drawdown.
    Every quant interviewer will ask about this — it matters as
    much as Sharpe because a strategy with Sharpe=1.5 but a 40%
    drawdown is untradeable for most funds.
    """
    rolling_max = equity.cummax()
    return ((equity - rolling_max) / rolling_max).min()


def run_backtest(
    signals: pd.DataFrame,
    log_prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    transaction_cost: float = 0.001,
    trading_days: int = 252,
) -> BacktestResult:
    """
    Simulate the pairs strategy from the signal DataFrame.

    Return calculation:
      Each day we hold the previous day's position (shift(1) — no look-ahead)
      and earn the change in the spread.

      r_t = position_{t-1} * (spread_t - spread_{t-1})

    Transaction costs:
      Deducted whenever position changes. 0.001 = 10bps per trade,
      realistic for liquid large-cap equities.

    Why spread returns and not dollar PnL?
      Cleaner — avoids having to model capital allocation, leverage,
      and margin. The Sharpe ratio is scale-invariant anyway.
    """
    log_a = log_prices[ticker_a].reindex(signals.index)
    log_b = log_prices[ticker_b].reindex(signals.index)

    # Daily spread change using the rolling hedge ratio
    spread_change = (
        log_a.diff()
        - signals["hedge_ratio"].shift(1) * log_b.diff()
    )

    # Lagged position — critical to avoid look-ahead bias
    pos_lagged = signals["position"].shift(1).fillna(0)
    gross = pos_lagged * spread_change

    # Transaction costs on position changes
    costs = signals["position"].diff().abs() * transaction_cost
    net = (gross - costs).dropna()

    equity = (1 + net).cumprod()

    # ── Metrics ──────────────────────────────────────────────
    total_ret = equity.iloc[-1] - 1.0
    n_years = len(net) / trading_days
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = net.std(ddof=1) * np.sqrt(trading_days)
    sharpe = net.mean() / net.std(ddof=1) * np.sqrt(trading_days) if ann_vol > 0 else 0.0
    max_dd = _max_drawdown(equity)
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.inf

    # Hit rate: fraction of active days with positive return
    active = net[pos_lagged.reindex(net.index).abs() > 0]
    hit_rate = (active > 0).mean() if len(active) > 0 else 0.0

    # Count entries (0 → nonzero transitions)
    pos = signals["position"]
    num_trades = int(((pos != 0) & (pos.shift(1) == 0)).sum())

    return BacktestResult(
        returns=net,
        equity_curve=equity,
        positions=signals["position"],
        total_return=total_ret,
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        hit_rate=hit_rate,
        num_trades=num_trades,
    )