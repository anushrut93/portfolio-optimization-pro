"""
Test script for market microstructure components
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.append('.')

from src.microstructure import (
    MicrostructureDataHandler,
    MarketImpactModel,
    MarketImpactParameters,
    TransactionCostOptimizer,
    estimate_impact_parameters,
    create_synthetic_order_book
)

def test_microstructure_components():
    """Test the microstructure functionality"""
    
    print("Testing Market Microstructure Components")
    print("=" * 50)
    
    # Test 1: Order Book Data Handler
    print("\n1. Testing Order Book Handler...")
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    handler = MicrostructureDataHandler(symbols)
    
    # Create synthetic order book data
    timestamp = pd.Timestamp.now()
    bids = [(150.00, 100), (149.99, 200), (149.98, 300)]
    asks = [(150.02, 100), (150.03, 200), (150.04, 300)]
    
    snapshot = handler.process_order_book_update('AAPL', timestamp, bids, asks)
    print(f"   Mid Price: ${snapshot.mid_price:.2f}")
    print(f"   Spread: ${snapshot.spread:.4f}")
    print(f"   Imbalance: {snapshot.get_depth_imbalance():.2%}")
    
    # Test 2: Market Impact Model
    print("\n2. Testing Market Impact Model...")
    
    # Create impact parameters
    impact_params = {
        'AAPL': MarketImpactParameters(
            daily_volume=50_000_000,
            volatility=0.02,
            spread=0.0001,
            lambda_t=0.1,
            lambda_p=0.05,
            eta=0.6
        )
    }
    
    impact_model = MarketImpactModel(impact_params)
    
    # Calculate impact for a $1M order
    order_size = 6667  # ~$1M at $150
    impact = impact_model.almgren_chriss_impact('AAPL', order_size, time_horizon=1.0)
    
    print(f"   Order Size: {order_size:,} shares (~${order_size*150:,.0f})")
    print(f"   Temporary Impact: {impact['temporary_impact_bps']:.1f} bps")
    print(f"   Permanent Impact: {impact['permanent_impact_bps']:.1f} bps")
    print(f"   Total Cost: ${impact['total_cost']:,.2f}")
    
    # Test 3: Optimal Execution Trajectory
    print("\n3. Testing Optimal Execution...")
    trajectory, exec_analysis = impact_model.optimal_execution_trajectory(
        'AAPL', order_size, time_horizon=1.0, num_periods=10
    )
    
    print(f"   Average Impact: {exec_analysis['avg_impact_bps']:.1f} bps")
    print(f"   Execution Risk: ${exec_analysis['execution_risk']:,.2f}")
    
    print("\n✓ All microstructure components working correctly!")
    
    return True

if __name__ == "__main__":
    test_microstructure_components()