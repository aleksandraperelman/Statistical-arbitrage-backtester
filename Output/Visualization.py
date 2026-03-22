import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from data_fetcher import fetch_prices, to_log_prices
from Cointegration.Cointegration import run_both_directions, select_best_direction
from strategy_signals import SignalConfig, generate_signals
from Backtest.backtest_engine import run_backtest
from montecarlo_bootstrap import run_bootstrap_test
from montecarlo_sensitivity import run_sensitivity_analysis

# ── dark theme ───────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
})
BLUE   = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
AMBER  = "#d29922"
MUTED  = "#8b949e"

OUT = os.path.dirname(os.path.abspath(__file__))

# ── data + pipeline ──────────────────────────────────────
print("Loading data...")
prices    = fetch_prices(["XOM", "CVX"])
log_p     = to_log_prices(prices)

print("Running cointegration test...")
ab, ba    = run_both_directions(log_p, "CVX", "XOM")
coint     = select_best_direction(ab, ba)

config    = SignalConfig(lookback=140, entry_threshold=2.0, exit_threshold=0.5)
signals   = generate_signals(log_p, "CVX", "XOM", config)
result    = run_backtest(signals, log_p, "CVX", "XOM")

print("Running bootstrap (2000 sims)...")
boot      = run_bootstrap_test(result.returns, n_simulations=2000)

print("Running sensitivity (117 backtests)...")
sens      = run_sensitivity_analysis(
    log_p, "CVX", "XOM",
    lookback_range=(60, 301, 20),
    entry_range=(1.0, 3.1, 0.25),
    verbose=False,
)

# ════════════════════════════════════════════════════════
#  FIGURE 1 — Cointegration diagnostics
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(13, 9))
fig.suptitle("Figure 1 — Cointegration Diagnostics: CVX / XOM", fontsize=13)

# Panel A: normalized log prices
ax = axes[0]
lp_cvx = log_p["CVX"] - log_p["CVX"].iloc[0]
lp_xom = log_p["XOM"] - log_p["XOM"].iloc[0]
ax.plot(lp_cvx.index, lp_cvx, color=BLUE,  label="CVX", alpha=0.9)
ax.plot(lp_xom.index, lp_xom, color=AMBER, label="XOM", alpha=0.9)
ax.set_ylabel("Log price (normalized to 0)")
ax.legend(framealpha=0.3)
ax.grid(True, alpha=0.4)
ax.set_title(f"A  Log prices   β = {coint.hedge_ratio:.3f}", loc="left",
             fontsize=10, color=MUTED)

# Panel B: spread
ax = axes[1]
sp = coint.spread
ax.plot(sp.index, sp, color=GREEN, linewidth=0.9, alpha=0.85)
ax.axhline(0, color=MUTED, linewidth=0.7, linestyle="--")
ax.fill_between(sp.index, sp, 0, where=(sp > 0), color=GREEN, alpha=0.08)
ax.fill_between(sp.index, sp, 0, where=(sp < 0), color=RED,   alpha=0.08)
s2 = sp.std() * 2
ax.axhline( s2, color=RED, linewidth=0.7, linestyle=":", alpha=0.6)
ax.axhline(-s2, color=RED, linewidth=0.7, linestyle=":", alpha=0.6)
ax.set_ylabel("Spread (residual)")
ax.grid(True, alpha=0.4)
coint_lbl = "COINTEGRATED ✓" if coint.is_cointegrated else "NOT cointegrated ✗"
coint_col  = GREEN if coint.is_cointegrated else RED
ax.set_title(
    f"B  Spread   ADF p={coint.adf_pvalue:.3f}   "
    f"half-life={coint.half_life_days:.0f}d   {coint_lbl}",
    loc="left", fontsize=10, color=coint_col
)

# Panel C: rolling 30d std
ax = axes[2]
rstd = sp.rolling(30).std()
ax.plot(rstd.index, rstd, color=AMBER, alpha=0.85)
ax.axhline(rstd.mean(), color=MUTED, linewidth=0.8, linestyle="--",
           label=f"Mean = {rstd.mean():.4f}")
ax.set_ylabel("Rolling 30d std")
ax.set_xlabel("Date")
ax.legend(framealpha=0.3, fontsize=9)
ax.grid(True, alpha=0.4)
ax.set_title("C  Spread volatility (stationarity check)", loc="left",
             fontsize=10, color=MUTED)

plt.tight_layout()
path1 = os.path.join(OUT, "fig1_cointegration.png")
fig.savefig(path1, dpi=150, bbox_inches="tight")
print(f"Saved: {path1}")
plt.close()

# ════════════════════════════════════════════════════════
#  FIGURE 2 — Signals
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle(
    f"Figure 2 — Trading Signals: CVX/XOM   "
    f"lookback={config.lookback}d  entry=±{config.entry_threshold}σ",
    fontsize=13
)

ax = axes[0]
z   = signals["zscore"]
pos = signals["position"]
ax.fill_between(z.index, z, 0, where=(pos == 1),  color=GREEN, alpha=0.15, label="Long spread")
ax.fill_between(z.index, z, 0, where=(pos == -1), color=RED,   alpha=0.15, label="Short spread")
ax.plot(z.index, z, color=BLUE, linewidth=0.9, alpha=0.9)
for level, col, ls in [
    ( config.entry_threshold, RED,   "--"),
    (-config.entry_threshold, RED,   "--"),
    ( config.exit_threshold,  GREEN, ":"),
    (-config.exit_threshold,  GREEN, ":"),
]:
    ax.axhline(level, color=col, linewidth=0.8, linestyle=ls, alpha=0.7)
ax.axhline(0, color=MUTED, linewidth=0.5)
ax.set_ylabel("Z-score")
ax.set_ylim(-6, 6)
ax.legend(framealpha=0.3, fontsize=9)
ax.grid(True, alpha=0.35)

ax = axes[1]
ax.fill_between(pos.index, pos, 0, where=(pos > 0), color=GREEN, alpha=0.5, step="post")
ax.fill_between(pos.index, pos, 0, where=(pos < 0), color=RED,   alpha=0.5, step="post")
ax.step(pos.index, pos, color=MUTED, linewidth=0.7, where="post")
ax.axhline(0, color=MUTED, linewidth=0.5)
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels(["Short", "Flat", "Long"], fontsize=9)
ax.set_xlabel("Date")
ax.set_ylabel("Position")
ax.grid(True, alpha=0.35)

plt.tight_layout()
path2 = os.path.join(OUT, "fig2_signals.png")
fig.savefig(path2, dpi=150, bbox_inches="tight")
print(f"Saved: {path2}")
plt.close()

# ════════════════════════════════════════════════════════
#  FIGURE 3 — Equity curve + drawdown
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle(
    f"Figure 3 — Strategy Performance: CVX/XOM   "
    f"Sharpe={result.sharpe_ratio:.3f}   "
    f"MaxDD={result.max_drawdown*100:.1f}%",
    fontsize=13
)

eq = result.equity_curve
ax = axes[0]
ax.plot(eq.index, eq, color=BLUE, linewidth=1.5)
ax.fill_between(eq.index, 1, eq, where=(eq >= 1), color=GREEN, alpha=0.12)
ax.fill_between(eq.index, 1, eq, where=(eq <  1), color=RED,   alpha=0.12)
ax.axhline(1.0, color=MUTED, linewidth=0.7, linestyle="--")
ax.set_ylabel("Portfolio value (start = 1.0)")
ax.grid(True, alpha=0.35)
stats = (
    f"Total return:  {result.total_return*100:+.1f}%\n"
    f"Ann. return:   {result.annualized_return*100:+.1f}%\n"
    f"Ann. vol:      {result.annualized_volatility*100:.1f}%\n"
    f"Sharpe:        {result.sharpe_ratio:.3f}\n"
    f"Calmar:        {result.calmar_ratio:.3f}\n"
    f"Hit rate:      {result.hit_rate*100:.1f}%\n"
    f"Trades:        {result.num_trades}"
)
ax.text(0.02, 0.97, stats, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", family="monospace", color="#c9d1d9",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                  edgecolor="#30363d", alpha=0.9))

ax = axes[1]
dd = (eq - eq.cummax()) / eq.cummax()
ax.fill_between(dd.index, dd, 0, color=RED, alpha=0.5)
ax.plot(dd.index, dd, color=RED, linewidth=0.7)
ax.set_ylabel("Drawdown")
ax.set_xlabel("Date")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax.grid(True, alpha=0.35)

plt.tight_layout()
path3 = os.path.join(OUT, "fig3_equity.png")
fig.savefig(path3, dpi=150, bbox_inches="tight")
print(f"Saved: {path3}")
plt.close()

# ════════════════════════════════════════════════════════
#  FIGURE 4 — Bootstrap null distribution
# ════════════════════════════════════════════════════════
from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle("Figure 4 — Monte Carlo Bootstrap: Is the Sharpe Real?", fontsize=13)

null = boot.null_sharpes
obs  = boot.observed_sharpe

ax.hist(null, bins=60, color=MUTED, alpha=0.4, density=True,
        edgecolor="none", label="Null distribution (block bootstrap)")

xr  = np.linspace(null.min() - 0.2, max(null.max(), obs) + 0.3, 500)
kde = gaussian_kde(null, bw_method=0.15)
ax.plot(xr, kde(xr), color=MUTED, linewidth=1.5, alpha=0.8)
ax.fill_between(xr, kde(xr), 0, where=(xr >= obs),
                color=RED, alpha=0.3,
                label=f"p-value region = {boot.pvalue:.3f}")

obs_color = GREEN if boot.is_significant else AMBER
ax.axvline(obs, color=obs_color, linewidth=2.5,
           label=f"Observed Sharpe = {obs:.3f} ({boot.percentile:.0f}th pctile)")

p95 = np.percentile(null, 95)
ax.axvline(p95, color=RED, linewidth=1.2, linestyle="--", alpha=0.7,
           label=f"95th pctile = {p95:.3f}")

sig_text = ("✓ Statistically significant (p < 0.05)"
            if boot.is_significant else
            "✗ Not significant — Sharpe may be due to chance (p = "
            f"{boot.pvalue:.3f})")
ax.set_title(sig_text, fontsize=11,
             color=GREEN if boot.is_significant else RED, pad=10)
ax.set_xlabel("Sharpe Ratio (annualized)")
ax.set_ylabel("Density")
ax.legend(framealpha=0.3, fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
path4 = os.path.join(OUT, "fig4_bootstrap.png")
fig.savefig(path4, dpi=150, bbox_inches="tight")
print(f"Saved: {path4}")
plt.close()

# ════════════════════════════════════════════════════════
#  FIGURE 5 — Sensitivity heatmap
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(
    f"Figure 5 — Parameter Sensitivity: CVX/XOM   "
    f"median Sharpe={sens.median_sharpe:.2f}   "
    f"{sens.fraction_profitable*100:.0f}% profitable",
    fontsize=13
)

lb = sens.lookback_values
en = sens.entry_values
sg = sens.sharpe_grid

# Panel A: heatmap
ax = axes[0]
vmax = np.nanpercentile(np.abs(sg), 95)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax.imshow(sg, aspect="auto", cmap="RdYlGn", norm=norm, origin="lower")
plt.colorbar(im, ax=ax, label="Sharpe Ratio")
ax.set_xticks(range(len(en)))
ax.set_xticklabels([f"{v:.2f}" for v in en], rotation=45, fontsize=8)
ax.set_yticks(range(len(lb)))
ax.set_yticklabels([str(v) for v in lb], fontsize=8)
ax.set_xlabel("Entry threshold (σ)")
ax.set_ylabel("Lookback (days)")
ax.set_title("A  Sharpe ratio heatmap", loc="left", fontsize=10, color=MUTED)

# Mark best
bi = np.unravel_index(np.nanargmax(sg), sg.shape)
ax.scatter(bi[1], bi[0], marker="*", s=300, color="white", zorder=5,
           label=f"Best: {sens.best_sharpe:.2f} @ lb={sens.best_params['lookback']}")
ax.legend(fontsize=8, framealpha=0.4)

# Panel B: fraction profitable by lookback
ax = axes[1]
frac = np.mean(sg > 0, axis=1)
med  = np.nanmedian(sg, axis=1)

colors = [GREEN if f >= 0.5 else RED for f in frac]
ax.barh(range(len(lb)), frac, color=colors, alpha=0.7, height=0.7)
ax.axvline(0.5, color=MUTED, linewidth=1.5, linestyle="--", alpha=0.8)

ax2 = ax.twiny()
ax2.plot(med, range(len(lb)), color=BLUE, linewidth=1.5,
         marker="o", markersize=4, label="Median Sharpe")
ax2.axvline(0, color=BLUE, linewidth=0.7, linestyle=":", alpha=0.5)
ax2.set_xlabel("Median Sharpe", color=BLUE)
ax2.tick_params(axis="x", colors=BLUE)

ax.set_yticks(range(len(lb)))
ax.set_yticklabels([str(v) for v in lb], fontsize=8)
ax.set_xlabel("Fraction of entry thresholds profitable")
ax.set_ylabel("Lookback (days)")
ax.set_title("B  Robustness by lookback", loc="left", fontsize=10, color=MUTED)
ax.set_xlim(0, 1)

plt.tight_layout()
path5 = os.path.join(OUT, "fig5_sensitivity.png")
fig.savefig(path5, dpi=150, bbox_inches="tight")
print(f"Saved: {path5}")
plt.close()

print("\nAll figures saved. Open them in your project folder.")