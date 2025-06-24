"""
Test script to verify portfolio_utils.py integration with existing codebase.
"""

import sys
import os
# Add the parent directory to the path to find the src module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.portfolio_utils import (
    calculate_portfolio_stats,
    maximize_sharpe_ratio,
    minimize_volatility,
    calculate_risk_parity_weights,
    calculate_portfolio_metrics,
    OptimizationResult
)
from src.data.fetcher import DataFetcher
from src.optimization.mean_variance import MeanVarianceOptimizer

def test_portfolio_utils():
    """Test portfolio utility functions with real data."""
    print("="*70)
    print("TESTING PORTFOLIO UTILITIES INTEGRATION")
    print("="*70)
    
    # Check if we're in the right directory structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"\nBase directory: {base_dir}")
    
    # 1. Test data loading
    print("\n1. Loading test data...")
    try:
        # Load sample data - adjust path based on where script is run from
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prices_path = os.path.join(base_dir, 'data', 'processed', 'prices.csv')
        returns_path = os.path.join(base_dir, 'data', 'processed', 'returns.csv')
        
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded data for {len(prices.columns)} assets")
        print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    except Exception as e:
        print(f"✗ Could not load data: {e}")
        print("  Using synthetic data instead...")
        # Create synthetic data for testing
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')  # Business days only
        assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']
        
        # Create price data more carefully
        n_days = len(dates)
        n_assets = len(assets)
        
        # Generate returns first, then prices
        daily_returns = np.random.normal(0.0002, 0.02, (n_days-1, n_assets))
        
        # Initialize prices
        price_data = np.zeros((n_days, n_assets))
        price_data[0, :] = 100  # Starting price of 100
        
        # Calculate prices from returns
        for i in range(1, n_days):
            price_data[i, :] = price_data[i-1, :] * (1 + daily_returns[i-1, :])
        
        # Create DataFrames
        prices = pd.DataFrame(price_data, index=dates, columns=assets)
        returns = pd.DataFrame(daily_returns, index=dates[1:], columns=assets)
    
    # Calculate inputs
    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    risk_free_rate = 0.02
    
    print(f"\nCalculated annualized metrics:")
    print(f"  Expected returns range: {expected_returns.min():.2%} to {expected_returns.max():.2%}")
    print(f"  Covariance matrix shape: {cov_matrix.shape}")
    
    # 2. Test basic portfolio calculations
    print("\n2. Testing basic portfolio calculations...")
    try:
        equal_weights = np.ones(len(expected_returns)) / len(expected_returns)
        port_return, port_vol = calculate_portfolio_stats(equal_weights, expected_returns, cov_matrix)
        print(f"✓ Equal weight portfolio: Return={port_return:.2%}, Vol={port_vol:.2%}")
    except Exception as e:
        print(f"✗ Basic calculation failed: {e}")
        return False
    
    # 3. Test optimization functions
    print("\n3. Testing optimization functions...")
    
    # Maximum Sharpe
    try:
        max_sharpe_weights, sharpe = maximize_sharpe_ratio(expected_returns, cov_matrix, risk_free_rate)
        result = OptimizationResult(
            weights=max_sharpe_weights,
            expected_return=np.dot(max_sharpe_weights, expected_returns),
            volatility=np.sqrt(np.dot(max_sharpe_weights.T, np.dot(cov_matrix, max_sharpe_weights))),
            sharpe_ratio=sharpe,
            asset_names=list(expected_returns.index)
        )
        print(f"✓ Max Sharpe optimization: Sharpe={sharpe:.3f}")
        print(f"  Weights: {result.get_allocation()}")
    except Exception as e:
        print(f"✗ Max Sharpe optimization failed: {e}")
    
    # Minimum Volatility
    try:
        min_vol_weights, vol = minimize_volatility(expected_returns, cov_matrix)
        print(f"✓ Min Volatility optimization: Vol={vol:.2%}")
    except Exception as e:
        print(f"✗ Min Volatility optimization failed: {e}")
    
    # Risk Parity
    try:
        rp_weights = calculate_risk_parity_weights(cov_matrix)
        print(f"✓ Risk Parity optimization: Sum of weights={rp_weights.sum():.3f}")
    except Exception as e:
        print(f"✗ Risk Parity optimization failed: {e}")
    
    # 4. Test compatibility with existing optimizer
    print("\n4. Testing compatibility with existing MeanVarianceOptimizer...")
    try:
        existing_optimizer = MeanVarianceOptimizer(risk_free_rate=risk_free_rate)
        existing_result = existing_optimizer.optimize(prices, objective='max_sharpe')
        
        # Compare results
        print(f"✓ Existing optimizer Sharpe: {existing_result.sharpe_ratio:.3f}")
        print(f"  New utilities Sharpe: {sharpe:.3f}")
        print(f"  Difference: {abs(existing_result.sharpe_ratio - sharpe):.4f}")
        
        if abs(existing_result.sharpe_ratio - sharpe) < 0.1:
            print("  ✓ Results are consistent!")
        else:
            print("  ⚠ Results differ significantly - may need adjustment")
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
    
    # 5. Test performance metrics calculation
    print("\n5. Testing performance metrics calculation...")
    try:
        # Use equal weight portfolio returns for testing
        portfolio_returns = returns.dot(equal_weights)
        metrics = calculate_portfolio_metrics(portfolio_returns)
        
        print("✓ Portfolio metrics calculated:")
        for metric, value in metrics.items():
            if 'ratio' in metric or 'return' in metric or 'volatility' in metric:
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"✗ Metrics calculation failed: {e}")
    
    # 6. Test constraint handling
    print("\n6. Testing constraint handling...")
    try:
        # Test with position limits
        position_limit = 0.3
        bounds = [(0, position_limit) for _ in range(len(expected_returns))]
        
        constrained_weights, constrained_sharpe = maximize_sharpe_ratio(
            expected_returns, cov_matrix, risk_free_rate, bounds=bounds
        )
        
        max_weight = constrained_weights.max()
        print(f"✓ Constrained optimization: Max weight={max_weight:.2%} (limit={position_limit:.0%})")
        print(f"  Sharpe ratio: {constrained_sharpe:.3f}")
        
        if max_weight <= position_limit + 0.001:  # Small tolerance for numerical errors
            print("  ✓ Constraints respected!")
        else:
            print(f"  ✗ Constraint violated: {max_weight:.2%} > {position_limit:.0%}")
    except Exception as e:
        print(f"✗ Constraint test failed: {e}")
    
    print("\n" + "="*70)
    print("INTEGRATION TEST COMPLETE")
    print("="*70)
    
    return True


if __name__ == "__main__":
    test_portfolio_utils()