import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import random
from scipy.optimize import minimize
from scipy.stats import norm


TICKERS = ['SPY', 'QQQ', 'TLT', 'GLD']
START_DATE = '2010-01-01'
END_DATE = '2025-12-31'
SIMULATIONS = 10000
HORIZON_DAYS = 126                 # 6 months
DT = 1 / 252
INITIAL_CAPITAL = 100_000
RISK_FREE_RATE = 0.05
DRIFT_TYPE = 'physical'            # Use physical drift for optimisation (historical means)
REBAL_FREQ = 21                    # Rebalance every 21 trading days (~monthly)

# Shorting flag
ALLOW_SHORTING = True               # If True, mean-variance can short; others remain long-only

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Using historical data from {START_DATE} to {END_DATE} for calibration.")
logger.info(f"Drift type: {DRIFT_TYPE}")
logger.info(f"Shorting allowed for mean-variance: {ALLOW_SHORTING}")

# data processing

try:
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=False)['Adj Close']
    data = data.dropna()
    if data.empty:
        raise ValueError("No data downloaded.")
    if len(data) < HORIZON_DAYS:
        raise ValueError(f"Only {len(data)} trading days available, need at least {HORIZON_DAYS}.")
    logger.info("Data downloaded successfully.")
except Exception as e:
    logger.error(f"Data download failed: {e}")
    exit(1)

log_returns = np.log(data / data.shift(1)).dropna()

# Daily statistics
daily_mean = log_returns.mean()
daily_cov = log_returns.cov()

# Annualise
annual_vol = np.sqrt(np.diag(daily_cov) * 252)
if hasattr(annual_vol, 'values'):
    annual_vol = annual_vol.values

if DRIFT_TYPE == 'physical':
    annual_mean = daily_mean * 252
    if hasattr(annual_mean, 'values'):
        annual_mean = annual_mean.values
    logger.info("Using physical drift (historical mean returns).")
else:
    annual_mean = np.full(len(TICKERS), RISK_FREE_RATE)
    logger.info("Using risk-neutral drift (all assets grow at risk-free rate).")

# Annual covariance matrix
annual_cov = daily_cov * 252

# Latest prices (starting point)
latest_prices = data.iloc[-1].values

logger.info(f"Latest prices: {dict(zip(TICKERS, latest_prices))}")
logger.info(f"Annualised volatilities: {dict(zip(TICKERS, annual_vol))}")

# optimization functions

def mean_variance_weights(mu, cov, rf, allow_short=False):
    """Maximise Sharpe ratio.
    
    If allow_short is True, weights can be negative but sum to 1 (no leverage limit).
    Otherwise, weights are bounded between 0 and 1.
    """
    n = len(mu)
    def neg_sharpe(w):
        port_return = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return -(port_return - rf) / port_vol
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    if allow_short:
        # Allow short positions up to -100% of capital, long up to +100% (bounds -1 to 1)
        bounds = [(-1, 1) for _ in range(n)]
    else:
        bounds = [(0, 1) for _ in range(n)]
    initial = np.ones(n) / n
    result = minimize(neg_sharpe, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def risk_parity_weights(cov):
    """Equal risk contribution (ERC) – long-only only."""
    n = cov.shape[0]
    def risk_contributions(w):
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        mrc = np.dot(cov, w) / port_vol
        return w * mrc
    def objective(w):
        rc = risk_contributions(w)
        target = rc.mean()
        return np.sum((rc - target)**2)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n)]
    initial = np.ones(n) / n
    result = minimize(objective, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def min_variance_weights(cov):
    """Minimum variance – long-only only."""
    n = cov.shape[0]
    def portfolio_var(w):
        return np.dot(w.T, np.dot(cov, w))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n)]
    initial = np.ones(n) / n
    result = minimize(portfolio_var, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def low_vol_weights(vols):
    """Inverse volatility weighting – positive only."""
    inv_vol = 1 / vols
    return inv_vol / inv_vol.sum()

# set weights for each strategy

strategies = {}

# Baseline: equal weight
strategies['Equal Weight'] = np.ones(len(TICKERS)) / len(TICKERS)

# Mean-Variance (Max Sharpe) – with shorting if allowed
strategies['Mean-Variance'] = mean_variance_weights(annual_mean, annual_cov, RISK_FREE_RATE, allow_short=ALLOW_SHORTING)

# Risk Parity (long-only)
strategies['Risk Parity'] = risk_parity_weights(annual_cov)

# Minimum Variance (long-only)
strategies['Min Variance'] = min_variance_weights(annual_cov)

# Low Volatility Factor (positive only)
strategies['Low Vol'] = low_vol_weights(annual_vol)

# Dynamic Rebalancing (equal weight, rebalanced monthly) – always long-only
strategies['Dynamic Equal'] = np.ones(len(TICKERS)) / len(TICKERS)

# Print weights for inspection
logger.info("\nOptimised weights:")
for name, w in strategies.items():
    logger.info(f"{name}: {dict(zip(TICKERS, w.round(4)))}")

# MC Simulator

# Cholesky decomposition
reg = 1e-8
cov_reg = daily_cov + np.eye(len(TICKERS)) * reg
chol = np.linalg.cholesky(cov_reg)

# Pre‑allocate price paths (simulations x days x assets)
prices = np.zeros((SIMULATIONS, HORIZON_DAYS + 1, len(TICKERS)))
prices[:, 0, :] = latest_prices

# Random innovations (same for all strategies)
rng = np.random.default_rng(42)
random_norm = rng.normal(size=(SIMULATIONS, HORIZON_DAYS, len(TICKERS)))
correlated_innovations = np.dot(random_norm, chol.T)

# Daily drift term
drift = (annual_mean - 0.5 * annual_vol**2) * DT

for t in range(1, HORIZON_DAYS + 1):
    daily_log_returns = drift + correlated_innovations[:, t-1, :]
    prices[:, t, :] = prices[:, t-1, :] * np.exp(daily_log_returns)

# Simulator for each strategy

results = {}

for strategy_name, target_weights in strategies.items():
    if strategy_name == 'Dynamic Equal':
        # Dynamic rebalancing
        # Initial shares (positive because equal weight)
        shares = (INITIAL_CAPITAL * target_weights) / latest_prices
        # Ensure shares is a 2D array (simulations x assets)
        if shares.ndim == 1:
            shares = np.tile(shares, (SIMULATIONS, 1))
        else:
            shares = np.tile(shares, (SIMULATIONS, 1))

        portfolio_values = np.zeros((SIMULATIONS, HORIZON_DAYS + 1))
        portfolio_values[:, 0] = INITIAL_CAPITAL

        for t in range(1, HORIZON_DAYS + 1):
            # Value before rebalance
            portfolio_values[:, t] = np.sum(prices[:, t, :] * shares, axis=1)
            # Rebalance if needed (skip last day as there is no future)
            if t % REBAL_FREQ == 0 and t < HORIZON_DAYS:
                # Compute new shares for each asset
                for i, asset in enumerate(range(len(TICKERS))):
                    shares[:, i] = (portfolio_values[:, t] * target_weights[i]) / prices[:, t, i]
        # final values already stored
        final_values = portfolio_values[:, -1]
        returns = (final_values - INITIAL_CAPITAL) / INITIAL_CAPITAL
    else:
        # Static allocation (hold initial shares)
        shares = (INITIAL_CAPITAL * target_weights) / latest_prices
        if hasattr(shares, 'values'):
            shares = shares.values
        portfolio_values = np.sum(prices * shares, axis=2)
        final_values = portfolio_values[:, -1]
        returns = (final_values - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # Compute metrics
    exp_return = returns.mean()
    std_return = returns.std()
    var_95 = -np.percentile(returns, 5)  # because VaR is positive loss
    cvar_95 = -returns[returns <= -var_95].mean() if np.any(returns <= -var_95) else 0.0
    prob_loss = (returns < 0).mean() * 100
    T_years = HORIZON_DAYS / 252
    annual_return = (1 + exp_return) ** (1 / T_years) - 1
    annual_vol_est = std_return * np.sqrt(1 / T_years)
    sharpe = (annual_return - RISK_FREE_RATE) / annual_vol_est if annual_vol_est > 0 else 0.0

    results[strategy_name] = {
        'Expected Return (%)': exp_return * 100,
        'Volatility (%)': std_return * 100,
        '95% VaR (%)': var_95 * 100,
        '95% CVaR (%)': cvar_95 * 100,
        'Prob Loss (%)': prob_loss,
        'Annualised Sharpe': sharpe,
    }

# displaying results

df_results = pd.DataFrame(results).T
df_results = df_results.round(2)
logger.info("\n" + "="*70)
logger.info("COMPARISON OF PORTFOLIO OPTIMISATION STRATEGIES")
logger.info("="*70)
print(df_results.to_string())

# Plots

sns.set_style("whitegrid")

# Bar chart: Sharpe ratio
plt.figure(figsize=(10, 6))
sharpe_values = df_results['Annualised Sharpe'].sort_values()
sharpe_values.plot(kind='barh', color='steelblue')
plt.xlabel('Annualised Sharpe Ratio')
plt.title('Sharpe Ratio Comparison')
plt.tight_layout()
plt.show()

# Bar chart: VaR (95%)
plt.figure(figsize=(10, 6))
var_values = df_results['95% VaR (%)'].sort_values()
var_values.plot(kind='barh', color='coral')
plt.xlabel('95% Value at Risk (%)')
plt.title('Risk Comparison (Lower is Better)')
plt.tight_layout()
plt.show()

# Histograms of returns for a few selected strategies
selected = ['Equal Weight', 'Mean-Variance', 'Risk Parity', 'Dynamic Equal']
plt.figure(figsize=(12, 8))
for i, strat in enumerate(selected):
    # Recompute returns array for this strategy
    if strat == 'Dynamic Equal':
        shares = (INITIAL_CAPITAL * strategies[strat]) / latest_prices
        if shares.ndim == 1:
            shares = np.tile(shares, (SIMULATIONS, 1))
        else:
            shares = np.tile(shares, (SIMULATIONS, 1))
        portfolio_values = np.zeros((SIMULATIONS, HORIZON_DAYS + 1))
        portfolio_values[:, 0] = INITIAL_CAPITAL
        for t in range(1, HORIZON_DAYS + 1):
            portfolio_values[:, t] = np.sum(prices[:, t, :] * shares, axis=1)
            if t % REBAL_FREQ == 0 and t < HORIZON_DAYS:
                for j, asset in enumerate(range(len(TICKERS))):
                    shares[:, j] = (portfolio_values[:, t] * strategies[strat][j]) / prices[:, t, j]
        final_values = portfolio_values[:, -1]
        rets = (final_values - INITIAL_CAPITAL) / INITIAL_CAPITAL
    else:
        shares = (INITIAL_CAPITAL * strategies[strat]) / latest_prices
        if hasattr(shares, 'values'):
            shares = shares.values
        portfolio_values = np.sum(prices * shares, axis=2)
        final_values = portfolio_values[:, -1]
        rets = (final_values - INITIAL_CAPITAL) / INITIAL_CAPITAL
    plt.subplot(2, 2, i+1)
    plt.hist(rets, bins=50, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    sns.kdeplot(rets, color='darkblue', linewidth=2)
    plt.axvline(x=-var_95, color='red', linestyle='--', label=f'VaR 95%')
    plt.title(strat)
    plt.xlabel('Return')
plt.tight_layout()
plt.show()
