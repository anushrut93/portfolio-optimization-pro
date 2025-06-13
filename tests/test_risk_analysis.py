"""
Test script for risk analysis module.
This demonstrates comprehensive risk analysis of portfolios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fetcher_v2 import DataFetcher
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.risk.metrics import RiskAnalyzer
import numpy as np
import pandas as pd

def main():
    print("Portfolio Risk Analysis Test")
    print("=" * 60)
    
    # Step 1: Fetch data for analysis
    print("\n1. Fetching historical data...")
    fetcher = DataFetcher()
    
    # Use a longer time period for better risk analysis
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']
    prices = fetcher.fetch_price_data(tickers, '2020-01-01', '2024-01-01')
    
    print(f"   ✓ Fetched {len(prices)} days of data for {len(prices.columns)} stocks")
    
    # Step 2: Create optimal portfolio using our optimizer
    print("\n2. Creating optimal portfolio...")
    optimizer = MeanVarianceOptimizer(risk_free_rate=0.04)
    optimization_result = optimizer.optimize(prices, objective='max_sharpe')
    
    print(f"   ✓ Optimal weights: {dict(zip(tickers, optimization_result.weights.round(3)))}")
    
    # Step 3: Analyze portfolio risk
    print("\n3. Analyzing portfolio risk...")
    risk_analyzer = RiskAnalyzer(risk_free_rate=0.04)
    
    # Analyze the optimal portfolio
    portfolio_risk = risk_analyzer.analyze_risk(prices, optimization_result.weights)
    
    print("\n4. Portfolio Risk Report:")
    print(portfolio_risk)
    
    # Step 5: Compare with individual stock risk
    print("\n5. Individual Stock Risk Comparison:")
    print("-" * 60)
    print(f"{'Stock':<10} {'Volatility':<12} {'Max DD':<12} {'Sharpe':<10}")
    print("-" * 60)
    
    for ticker in tickers:
        stock_risk = risk_analyzer.analyze_risk(prices[ticker])
        print(f"{ticker:<10} {stock_risk.volatility:<12.1%} "
              f"{stock_risk.max_drawdown:<12.1%} {stock_risk.sharpe_ratio:<10.2f}")
    
    # Step 6: Risk interpretation
    print("\n6. Risk Interpretation:")
    print("-" * 60)
    
    # VaR interpretation
    portfolio_value = 100000  # Assume $100k portfolio
    daily_var_95 = portfolio_risk.var_95 / np.sqrt(252)  # Convert to daily
    print(f"For a ${portfolio_value:,} portfolio:")
    print(f"- On a typical bad day (1 in 20), you might lose up to ${-daily_var_95 * portfolio_value:,.0f}")
    print(f"- In a really bad day (1 in 100), losses could reach ${-portfolio_risk.var_99 / np.sqrt(252) * portfolio_value:,.0f}")
    print(f"- The worst historical drawdown was {portfolio_risk.max_drawdown:.1%}, "
          f"taking {portfolio_risk.max_drawdown_duration} days to recover")
    
    # Risk-adjusted performance
    if portfolio_risk.sortino_ratio > portfolio_risk.sharpe_ratio:
        print(f"\n✓ Good news: Sortino > Sharpe ({portfolio_risk.sortino_ratio:.2f} > {portfolio_risk.sharpe_ratio:.2f})")
        print("  This means the portfolio's volatility is mostly to the upside!")

if __name__ == "__main__":
    main()