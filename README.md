**Statistical Arbitrage Backtester with Monte Carlo Significance Testing**
A pairs trading system built on the XOM/CVX cointegrated pair, with rigorous statistical validation to distinguish genuine alpha from random chance.
**What it does:**
- Identifies cointegrated pairs using the Engle-Granger two-step test (ADF stationarity test on OLS residuals)
- Generates mean-reversion signals via rolling z-score with configurable entry/exit thresholds
- Simulates the strategy with realistic transaction costs and look-ahead bias prevention
- Validates results using block bootstrap — builds a null distribution of Sharpe ratios to compute a p-value for the strategy's performance
- Sweeps 117 parameter combinations (lookback × entry threshold) to test robustness and distinguish a genuine edge from a lucky parameter pick

**Key finding: **The strategy produces a Sharpe of 0.48–0.74 depending on parameters, with 71.8% of parameter combinations profitable. However, the block bootstrap test (p = 0.54) shows the result is not statistically significant — demonstrating that a reasonable-looking backtest can still fail rigorous significance testing. This is the point.
**Stack: ** Python · pandas · numpy · statsmodels · yfinance · matplotlib · scipy
**Concepts demonstrated:** Engle-Granger cointegration · ADF unit root testing · rolling OLS hedge ratio · z-score signal generation · block bootstrap · multiple testing · parameter sensitivity analysis
