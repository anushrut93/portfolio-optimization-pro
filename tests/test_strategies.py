"""
Test script for strategies.py module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.strategies import (
    BlackLittermanModel,
    create_market_cap_weights,
    calculate_dynamic_black_litterman_weights,
    create_max_sharpe_strategy,
    create_min_volatility_strategy,
    create_risk_parity_strategy,
    create_equal_weight_strategy,
    create_black_litterman_strategy,
    create_dynamic_black_litterman_strategy,
    create_ml_enhanced_strategy
)
from src.portfolio_utils import calculate_portfolio_stats


def create_test_data():
    """Create synthetic test data."""
    np.random.seed(42)
    
    # Create price data
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='B')
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']
    
    # Generate prices with different characteristics
    price_data = pd.DataFrame(index=dates, columns=assets)
    
    # Starting prices
    start_prices = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 2800, 'AMZN': 3200, 'JPM': 140}
    
    # Generate realistic price movements
    for asset in assets:
        returns = np.random.normal(0.0002, 0.02, len(dates))
        price_series = start_prices[asset] * np.exp(np.cumsum(returns))
        price_data[asset] = price_series
    
    return price_data


def test_black_litterman_model():
    """Test Black-Litterman model implementation."""
    print("\n1. Testing Black-Litterman Model")
    print("-" * 50)
    
    # Create test data
    prices = create_test_data()
    returns = prices.pct_change().dropna()
    cov_matrix = returns.cov() * 252
    
    try:
        # Initialize model
        model = BlackLittermanModel(cov_matrix)
        print("✓ Model initialized successfully")
        
        # Test equilibrium returns
        eq_returns = model.equilibrium_returns
        print(f"✓ Equilibrium returns calculated: mean={eq_returns.mean():.2%}")
        
        # Test adding views
        views = {
            'AAPL': 0.15,  # Bullish on AAPL
            'AMZN': 0.08   # Bearish on AMZN
        }
        
        posterior_returns, posterior_cov = model.add_views(views, confidence=0.8)
        print(f"✓ Views added successfully")
        print(f"  AAPL: Prior={eq_returns['AAPL']:.2%} → Posterior={posterior_returns['AAPL']:.2%}")
        print(f"  AMZN: Prior={eq_returns['AMZN']:.2%} → Posterior={posterior_returns['AMZN']:.2%}")
        
        # Test optimization
        weights = model.get_optimal_weights(views, confidence=0.8)
        print(f"✓ Optimal weights calculated: sum={weights.sum():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Black-Litterman test failed: {e}")
        return False


def test_strategy_factories():
    """Test strategy factory functions."""
    print("\n2. Testing Strategy Factories")
    print("-" * 50)
    
    prices = create_test_data()
    
    strategies = {
        'Max Sharpe': create_max_sharpe_strategy(),
        'Min Volatility': create_min_volatility_strategy(),
        'Risk Parity': create_risk_parity_strategy(),
        'Equal Weight': create_equal_weight_strategy()
    }
    
    all_passed = True
    
    for name, strategy_func in strategies.items():
        try:
            # Test strategy
            weights = strategy_func(prices)
            
            # Verify weights
            if abs(weights.sum() - 1.0) > 0.001:
                print(f"✗ {name}: Weights don't sum to 1 ({weights.sum():.3f})")
                all_passed = False
            else:
                print(f"✓ {name}: Weights sum to {weights.sum():.3f}")
                
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            all_passed = False
    
    return all_passed


def test_dynamic_black_litterman():
    """Test dynamic Black-Litterman strategy."""
    print("\n3. Testing Dynamic Black-Litterman")
    print("-" * 50)
    
    prices = create_test_data()
    
    try:
        # Test direct function
        weights = calculate_dynamic_black_litterman_weights(
            prices,
            lookback_days=60,
            momentum_threshold=0.05
        )
        print(f"✓ Dynamic weights calculated: sum={weights.sum():.3f}")
        
        # Test strategy factory
        dynamic_strategy = create_dynamic_black_litterman_strategy()
        weights2 = dynamic_strategy(prices)
        print(f"✓ Strategy factory works: sum={weights2.sum():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dynamic Black-Litterman failed: {e}")
        return False


def test_ml_enhanced_strategy():
    """Test ML-enhanced strategy."""
    print("\n4. Testing ML-Enhanced Strategy")
    print("-" * 50)
    
    prices = create_test_data()
    
    try:
        # Create fake ML predictions
        assets = prices.columns
        ml_predictions = pd.Series(
            [0.12, 0.10, 0.15, 0.08, 0.09],  # Predicted returns
            index=assets
        )
        
        # Create strategy
        ml_strategy = create_ml_enhanced_strategy(
            ml_predictions,
            blend_factor=0.6
        )
        
        # Test strategy
        weights = ml_strategy(prices)
        print(f"✓ ML-enhanced weights calculated: sum={weights.sum():.3f}")
        print(f"  Weight distribution: {dict(zip(assets, weights.round(3)))}")
        
        return True
        
    except Exception as e:
        print(f"✗ ML-enhanced strategy failed: {e}")
        return False


def test_black_litterman_strategy():
    """Test static Black-Litterman strategy factory."""
    print("\n5. Testing Static Black-Litterman Strategy")
    print("-" * 50)
    
    prices = create_test_data()
    
    try:
        # Create views
        views = {
            'AAPL': 0.20,  # Very bullish on AAPL
            'JPM': 0.06    # Bearish on JPM
        }
        
        # Create strategy
        bl_strategy = create_black_litterman_strategy(views, confidence=1.0)
        
        # Test strategy
        weights = bl_strategy(prices)
        print(f"✓ Black-Litterman strategy weights: sum={weights.sum():.3f}")
        
        # The strategy should overweight AAPL and underweight JPM
        asset_weights = dict(zip(prices.columns, weights))
        print(f"  AAPL weight: {asset_weights['AAPL']:.2%}")
        print(f"  JPM weight: {asset_weights['JPM']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Black-Litterman strategy failed: {e}")
        return False


def test_integration():
    """Test integration with portfolio_utils."""
    print("\n6. Testing Integration with portfolio_utils")
    print("-" * 50)
    
    prices = create_test_data()
    returns = prices.pct_change().dropna()
    expected_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    try:
        # Test that strategies produce valid portfolios
        strategies = {
            'Equal Weight': create_equal_weight_strategy(),
            'Max Sharpe': create_max_sharpe_strategy(),
            'Risk Parity': create_risk_parity_strategy()
        }
        
        for name, strategy in strategies.items():
            weights = strategy(prices)
            
            # Calculate portfolio stats
            ret, vol = calculate_portfolio_stats(weights, expected_returns, cov_matrix)
            sharpe = (ret - 0.02) / vol
            
            print(f"✓ {name}: Return={ret:.2%}, Vol={vol:.2%}, Sharpe={sharpe:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("TESTING STRATEGIES MODULE")
    print("="*70)
    
    tests = [
        test_black_litterman_model,
        test_strategy_factories,
        test_dynamic_black_litterman,
        test_ml_enhanced_strategy,
        test_black_litterman_strategy,
        test_integration
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("="*70)
    
    if passed == len(tests):
        print("\n✓ All tests passed! strategies.py is ready to use.")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()