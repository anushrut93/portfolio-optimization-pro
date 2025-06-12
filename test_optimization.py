"""
Test script for portfolio optimization.
This demonstrates how the data fetcher and optimizer work together.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fetcher_v2 import DataFetcher
from src.optimization.mean_variance import MeanVarianceOptimizer
import pandas as pd

def main():
    print("Portfolio Optimization Test")
    print("=" * 50)
    
    # Step 1: Fetch data
    print("\n1. Fetching price data...")
    fetcher = DataFetcher()
    
    # Let's use some well-known stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']
    prices = fetcher.fetch_price_data(tickers, '2022-01-01', '2024-01-01')
    
    print(f"   ✓ Fetched data for {len(prices.columns)} stocks")
    print(f"   ✓ Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # Step 2: Optimize portfolio
    print("\n2. Running portfolio optimization...")
    optimizer = MeanVarianceOptimizer(risk_free_rate=0.04)  # 4% risk-free rate
    
    # Find maximum Sharpe ratio portfolio
    result = optimizer.optimize(prices, objective='max_sharpe')
    
    print("\n3. Optimization Results:")
    print(result)
    
    # Also find minimum volatility portfolio
    print("\n4. Minimum Volatility Portfolio:")
    min_vol_result = optimizer.optimize(prices, objective='min_volatility')
    print(min_vol_result)
    
    # Compare the two portfolios
    print("\n5. Comparison:")
    print(f"   Max Sharpe Portfolio: Return={result.expected_return:.1%}, "
          f"Risk={result.volatility:.1%}, Sharpe={result.sharpe_ratio:.2f}")
    print(f"   Min Volatility Portfolio: Return={min_vol_result.expected_return:.1%}, "
          f"Risk={min_vol_result.volatility:.1%}, Sharpe={min_vol_result.sharpe_ratio:.2f}")

if __name__ == "__main__":
    main()