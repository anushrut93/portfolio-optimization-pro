"""
Test script for the backtesting engine.

This script demonstrates how backtesting reveals the real-world performance
of portfolio strategies, including the impact of transaction costs and
different rebalancing frequencies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.fetcher import DataFetcher
from src.optimization.mean_variance import MeanVarianceOptimizer
from src.backtesting.engine import BacktestEngine, BacktestConfig, RebalanceFrequency
import pandas as pd
import numpy as np

def create_rebalancing_strategy(lookback_period: int = 252):
    """
    Create a strategy that reoptimizes based on recent historical data.
    
    This strategy looks at the past 'lookback_period' days and finds the
    optimal portfolio for that period. It's like driving by looking in
    the rearview mirror - not perfect, but often effective.
    
    Args:
        lookback_period: Number of days to use for optimization
        
    Returns:
        Strategy function compatible with the backtesting engine
    """
    def strategy(prices_up_to_now: pd.DataFrame) -> np.ndarray:
        # Use only recent history for optimization
        if len(prices_up_to_now) < lookback_period:
            # Not enough history - equal weight
            n_assets = len(prices_up_to_now.columns)
            return np.ones(n_assets) / n_assets
        
        recent_prices = prices_up_to_now.iloc[-lookback_period:]
        
        # Optimize based on recent data
        optimizer = MeanVarianceOptimizer()
        try:
            result = optimizer.optimize(recent_prices, objective='max_sharpe')
            return result.weights
        except:
            # If optimization fails, return equal weights
            n_assets = len(prices_up_to_now.columns)
            return np.ones(n_assets) / n_assets
    
    return strategy

def main():
    print("Portfolio Backtesting Analysis")
    print("=" * 70)
    
    # Step 1: Fetch historical data
    print("\n1. Fetching historical data for backtesting...")
    fetcher = DataFetcher()
    
    # Use a diverse set of assets for more realistic testing
    tickers = ['SPY', 'TLT', 'GLD', 'QQQ', 'IWM']  # Stocks, Bonds, Gold, Tech, Small-cap
    prices = fetcher.fetch_price_data(tickers, '2015-01-01', '2024-01-01')
    
    print(f"   ✓ Fetched {len(prices)} days of data for {len(prices.columns)} assets")
    print(f"   ✓ Assets: {', '.join(prices.columns)}")
    
    # Step 2: Create strategies to test
    print("\n2. Creating portfolio strategies...")
    
    # Strategy 1: Rebalancing with 1-year lookback
    rebalancing_strategy = create_rebalancing_strategy(lookback_period=252)
    
    # Strategy 2: Simple buy-and-hold (equal weight)
    def buy_and_hold_strategy(prices: pd.DataFrame) -> np.ndarray:
        n_assets = len(prices.columns)
        return np.ones(n_assets) / n_assets
    
    # Step 3: Run backtests with different configurations
    print("\n3. Running backtests with different configurations...")
    
    # Test different rebalancing frequencies
    frequencies = [
        RebalanceFrequency.MONTHLY,
        RebalanceFrequency.QUARTERLY,
        RebalanceFrequency.YEARLY,
        RebalanceFrequency.NEVER
    ]
    
    results = {}
    
    for freq in frequencies:
        print(f"\n   Testing {freq.value} rebalancing...")
        
        config = BacktestConfig(
            initial_capital=100000,
            rebalance_frequency=freq,
            transaction_cost_pct=0.001,  # 10 basis points
            slippage_pct=0.0005  # 5 basis points
        )
        
        engine = BacktestEngine(config)
        
        # Use appropriate strategy
        if freq == RebalanceFrequency.NEVER:
            strategy = buy_and_hold_strategy
        else:
            strategy = rebalancing_strategy
        
        # Run backtest
        backtest_results = engine.run_backtest(
            prices,
            strategy,
            start_date='2016-01-01'  # Use first year for initial learning
        )
        
        results[freq.value] = backtest_results
        
        # Print summary
        print(f"   → Return: {backtest_results.annualized_return:.1%}")
        print(f"   → Volatility: {backtest_results.volatility:.1%}")
        print(f"   → Sharpe: {backtest_results.sharpe_ratio:.2f}")
        print(f"   → Max Drawdown: {backtest_results.max_drawdown:.1%}")
        print(f"   → Total Costs: ${backtest_results.total_transaction_costs + backtest_results.total_slippage_costs:,.2f}")
    
    # Step 4: Compare results
    print("\n4. Comparative Analysis")
    print("-" * 70)
    print(f"{'Strategy':<15} {'Return':<10} {'Risk':<10} {'Sharpe':<10} {'Max DD':<10} {'Costs':<10}")
    print("-" * 70)
    
    for freq_name, result in results.items():
        total_costs = result.total_transaction_costs + result.total_slippage_costs
        print(f"{freq_name:<15} {result.annualized_return:<10.1%} "
              f"{result.volatility:<10.1%} {result.sharpe_ratio:<10.2f} "
              f"{result.max_drawdown:<10.1%} ${total_costs:<10,.0f}")
    
    # Step 5: Key insights
    print("\n5. Key Insights from Backtesting")
    print("-" * 70)
    
    # Find best and worst strategies
    best_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio)
    highest_return = max(results.items(), key=lambda x: x[1].annualized_return)
    lowest_cost = min(results.items(), key=lambda x: x[1].total_transaction_costs)
    
    print(f"→ Best risk-adjusted performance (Sharpe): {best_sharpe[0]}")
    print(f"→ Highest returns: {highest_return[0]}")
    print(f"→ Lowest costs: {lowest_cost[0]}")
    
    # Calculate cost impact
    monthly_result = results['monthly']
    cost_impact = (monthly_result.total_transaction_costs + 
                   monthly_result.total_slippage_costs) / 100000  # As % of initial capital
    
    print(f"\n→ Transaction costs reduced returns by approximately {cost_impact:.1%} for monthly rebalancing")
    
    # Detailed report for best strategy
    print(f"\n6. Detailed Report - {best_sharpe[0].title()} Rebalancing")
    print("-" * 70)
    print(best_sharpe[1].summary_report())

if __name__ == "__main__":
    main()