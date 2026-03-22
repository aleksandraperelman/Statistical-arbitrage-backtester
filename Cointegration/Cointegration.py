import numpy as np
import pandas as pd
from dataclasses import dataclass
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools import add_constant


@dataclass
class CointegrationResult:
    dependent: str
    independent: str
    hedge_ratio: float
    intercept: float
    spread: pd.Series
    adf_statistic: float
    adf_pvalue: float
    adf_critical_values: dict
    is_cointegrated: bool
    half_life_days: float

    def summary(self):
        result_str = "COINTEGRATED ✓" if self.is_cointegrated else "NOT cointegrated ✗"
        return (
            f"\n{'='*55}\n"
            f"Engle-Granger: {self.dependent} ~ {self.independent}\n"
            f"{'='*55}\n"
            f"  Hedge ratio (beta):   {self.hedge_ratio:.4f}\n"
            f"  Intercept (alpha):    {self.intercept:.4f}\n"
            f"  ADF statistic:        {self.adf_statistic:.4f}\n"
            f"  ADF p-value:          {self.adf_pvalue:.4f}\n"
            f"  Critical value (5%):  {self.adf_critical_values['5%']:.4f}\n"
            f"  Half-life:            {self.half_life_days:.1f} trading days\n"
            f"  Result:               {result_str}\n"
        )


def _estimate_half_life(spread: pd.Series) -> float:

    spread_lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    spread_lag, delta = spread_lag.align(delta, join="inner")

    X = add_constant(spread_lag)
    lam = OLS(delta, X).fit().params.iloc[1]

    if lam >= 0:
        return np.inf  # Diverging, not mean-reverting

    return max(-np.log(2) / np.log(1 + lam), 0.5)


def run_engle_granger(
    log_prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    significance_level: float = 0.05,
) -> CointegrationResult:

    y = log_prices[ticker_a]
    x = log_prices[ticker_b]

    # Step 1: OLS
    ols = OLS(y, add_constant(x)).fit()
    intercept = ols.params.iloc[0]
    hedge_ratio = ols.params.iloc[1]
    spread = ols.resid

    # Step 2: ADF on residuals
    adf_stat, adf_pval, _, _, adf_crit, _ = adfuller(
        spread, autolag="AIC", regression="c"
    )

    return CointegrationResult(
        dependent=ticker_a,
        independent=ticker_b,
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        spread=spread,
        adf_statistic=adf_stat,
        adf_pvalue=adf_pval,
        adf_critical_values=adf_crit,
        is_cointegrated=(adf_pval < significance_level),
        half_life_days=_estimate_half_life(spread),
    )


def run_both_directions(log_prices, ticker_a, ticker_b):
    return (
        run_engle_granger(log_prices, ticker_a, ticker_b),
        run_engle_granger(log_prices, ticker_b, ticker_a),
    )


def select_best_direction(result_ab, result_ba):
    if result_ab.adf_pvalue <= result_ba.adf_pvalue:
        print(f"[cointegration] Using: {result_ab.dependent} ~ {result_ab.independent}")
        return result_ab
    else:
        print(f"[cointegration] Using: {result_ba.dependent} ~ {result_ba.independent}")
        return result_ba