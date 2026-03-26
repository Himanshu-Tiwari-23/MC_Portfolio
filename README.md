# Monte Carlo Portfolio Risk & Optimization

This project performs a Monte Carlo simulation to evaluate the risk and return characteristics of a multi‑asset portfolio, comparing several classical portfolio optimization strategies. It demonstrates the application of quantitative finance techniques without relying on machine learning or neural networks.

## Features

- **Data Download**: Automatically fetches historical adjusted closing prices from Yahoo Finance for user‑specified tickers.
- **Monte Carlo Simulation**: Simulates price paths for multiple assets using a correlated geometric Brownian motion model.
- **Drift Options**: Choose between physical (historical mean) or risk‑neutral (risk‑free rate) drift.
- **Portfolio Strategies**:
  - **Equal Weight**: Baseline static allocation.
  - **Mean‑Variance (Max Sharpe)**: Optimizes for the highest Sharpe ratio; optionally allows short selling.
  - **Risk Parity**: Allocates capital so that each asset contributes equally to portfolio risk.
  - **Minimum Variance**: Minimizes portfolio volatility.
  - **Low Volatility Factor**: Inverse volatility weighting.
  - **Dynamic Equal Weight**: Rebalances the equal‑weight portfolio at a user‑defined frequency.
- **Risk Metrics**: Computes expected return, volatility, Value at Risk (VaR), Conditional VaR (CVaR), probability of loss, and maximum drawdown.
- **Visualizations**: Histograms with kernel density estimates, fan charts, and bar charts comparing Sharpe ratios and VaR across strategies.

## Requirements

Install the required packages with:

```bash
pip install numpy pandas yfinance matplotlib seaborn scipy
