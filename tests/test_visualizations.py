"""
Test script for portfolio visualization module.

This script demonstrates how to create compelling visualizations that
transform complex portfolio analytics into intuitive visual insights.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fetcher_v2 import DataFetcher
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.risk.metrics import RiskAnalyzer
from src.visualization.plots import PortfolioVisualizer
from src.backtesting.engine import BacktestEngine, BacktestConfig, RebalanceFrequency
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def main():
    print("Portfolio Visualization Showcase")
    print("=" * 70)
    
    # Step 1: Prepare data for visualization
    print("\n1. Fetching data for visualization examples...")
    fetcher = DataFetcher()
    
    # Get a diversified set of assets
    tickers = ['SPY', 'TLT', 'GLD', 'QQQ', 'IWM', 'EEM', 'VNQ']
    prices = fetcher.fetch_price_data(tickers, '2019-01-01', '2024-01-01')
    
    print(f"   ✓ Fetched data for {len(tickers)} assets")
    
    # Step 2: Run optimization to get efficient frontier data
    print("\n2. Generating efficient frontier data...")
    optimizer = MeanVarianceOptimizer()
    
    # Generate multiple portfolios for the efficient frontier
    n_portfolios = 5000
    n_assets = len(tickers)
    
    portfolio_returns = []
    portfolio_volatilities = []
    portfolio_sharpe_ratios = []
    
    # Calculate returns and covariance
    returns = np.log(prices / prices.shift(1)).dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Generate random portfolios
    np.random.seed(42)
    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= weights.sum()
        
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (portfolio_return - 0.02) / portfolio_vol
        
        portfolio_returns.append(portfolio_return)
        portfolio_volatilities.append(portfolio_vol)
        portfolio_sharpe_ratios.append(sharpe)
    
    # Find optimal portfolio
    optimal_idx = np.argmax(portfolio_sharpe_ratios)
    optimal_portfolio = {
        'return': portfolio_returns[optimal_idx],
        'volatility': portfolio_volatilities[optimal_idx],
        'sharpe': portfolio_sharpe_ratios[optimal_idx]
    }
    
    print(f"   ✓ Generated {n_portfolios} portfolios")
    print(f"   ✓ Optimal Sharpe ratio: {optimal_portfolio['sharpe']:.2f}")
    
    # Step 3: Create visualizations
    print("\n3. Creating visualizations...")
    visualizer = PortfolioVisualizer(style='professional')
    
    # Create output directory
    os.makedirs('output/visualizations', exist_ok=True)
    
    # Visualization 1: Efficient Frontier
    print("\n   a) Efficient Frontier Plot...")
    fig1 = visualizer.plot_efficient_frontier(
        returns=portfolio_returns,
        volatilities=portfolio_volatilities,
        sharpe_ratios=portfolio_sharpe_ratios,
        optimal_portfolio=optimal_portfolio,
        save_path='output/visualizations/efficient_frontier.png'
    )
    plt.close(fig1)
    print("      ✓ Saved to output/visualizations/efficient_frontier.png")
    
    # Visualization 2: Portfolio Performance
    print("\n   b) Portfolio Performance Dashboard...")
    
    # Run a simple backtest to get performance data
    optimizer_result = optimizer.optimize(prices, objective='max_sharpe')
    
    # Create a simple backtest for performance visualization
    initial_value = 100000
    portfolio_values = pd.Series(index=prices.index, dtype=float)
    portfolio_values.iloc[0] = initial_value
    
    # Calculate portfolio value over time (simple buy-and-hold)
    for i in range(1, len(prices)):
        daily_returns = prices.iloc[i] / prices.iloc[i-1] - 1
        portfolio_return = np.sum(optimizer_result.weights * daily_returns)
        portfolio_values.iloc[i] = portfolio_values.iloc[i-1] * (1 + portfolio_return)
    
    # Define some market events
    events = {
        '2020-03-23': 'COVID-19 Bottom',
        '2021-01-27': 'GameStop Squeeze',
        '2022-01-03': '2022 Bear Market Begins',
        '2023-03-10': 'Banking Crisis'
    }
    
    fig2 = visualizer.plot_portfolio_performance(
        portfolio_values=portfolio_values,
        benchmark_values=prices['SPY'] * initial_value / prices['SPY'].iloc[0],
        events=events,
        save_path='output/visualizations/performance_dashboard.png'
    )
    plt.close(fig2)
    print("      ✓ Saved to output/visualizations/performance_dashboard.png")
    
    # Visualization 3: Asset Allocation
    print("\n   c) Asset Allocation Charts...")
    weights_dict = dict(zip(tickers, optimizer_result.weights))
    
    fig3 = visualizer.plot_asset_allocation(
        weights=weights_dict,
        title="Optimal Portfolio Allocation",
        save_path='output/visualizations/asset_allocation.png'
    )
    plt.close(fig3)
    print("      ✓ Saved to output/visualizations/asset_allocation.png")
    
    # Visualization 4: Risk-Return Scatter
    print("\n   d) Risk-Return Analysis...")
    
    # Calculate risk-return for each asset
    assets_data = {}
    risk_analyzer = RiskAnalyzer()
    
    for ticker in tickers:
        asset_returns = returns[ticker]
        annual_return = asset_returns.mean() * 252
        annual_vol = asset_returns.std() * np.sqrt(252)
        
        assets_data[ticker] = {
            'return': annual_return,
            'volatility': annual_vol
        }
    
    # Add the optimal portfolio
    assets_data['Optimal Portfolio'] = {
        'return': optimal_portfolio['return'],
        'volatility': optimal_portfolio['volatility']
    }
    
    fig4 = visualizer.plot_risk_return_scatter(
        assets_data=assets_data,
        highlight_assets=['Optimal Portfolio', 'SPY'],
        save_path='output/visualizations/risk_return_scatter.png'
    )
    plt.close(fig4)
    print("      ✓ Saved to output/visualizations/risk_return_scatter.png")
    
    # Step 4: Create interactive dashboard
    print("\n   e) Interactive Dashboard...")
    
    # Prepare data for interactive plot
    portfolio_data = pd.DataFrame(index=portfolio_values.index)
    portfolio_data['value'] = portfolio_values
    
    # Calculate drawdown
    cumulative_returns = portfolio_values / portfolio_values.iloc[0]
    running_max = cumulative_returns.expanding().max()
    portfolio_data['drawdown'] = (cumulative_returns - running_max) / running_max
    
    # Calculate rolling volatility
    portfolio_returns = portfolio_values.pct_change()
    portfolio_data['rolling_volatility'] = portfolio_returns.rolling(90).std() * np.sqrt(252)
    
    interactive_fig = visualizer.create_interactive_performance_dashboard(
        portfolio_data=portfolio_data
    )
    
    # Save interactive plot
    interactive_fig.write_html('output/visualizations/interactive_dashboard.html')
    print("      ✓ Saved to output/visualizations/interactive_dashboard.html")
    
    print("\n" + "=" * 70)
    print("Visualization showcase complete!")
    print("\nAll visualizations saved to: output/visualizations/")
    print("\nKey insights from visualizations:")
    print("- The efficient frontier clearly shows the risk-return trade-off")
    print("- Performance dashboard reveals drawdown periods and volatility clusters")
    print("- Asset allocation charts make portfolio composition immediately clear")
    print("- Risk-return scatter identifies which assets offer best risk-adjusted returns")
    print("\nOpen the interactive dashboard in your browser for dynamic exploration!")

if __name__ == "__main__":
    main()