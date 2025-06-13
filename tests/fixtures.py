import pandas as pd
import numpy as np

def generate_test_returns(n_assets=3, n_periods=500, seed=42):
    """Generate synthetic return data for testing"""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    
    # Generate returns with GARCH-like properties but less correlation
    returns = {}
    for i in range(n_assets):
        # Simple GARCH(1,1) simulation with different parameters for each asset
        omega = 0.00001 * (1 + i * 0.5)  # Different base volatility
        alpha = 0.1 * (1 + i * 0.1)      # Different alpha
        beta = 0.8 - i * 0.05            # Different beta
        
        returns_i = np.zeros(n_periods)
        sigma2 = np.zeros(n_periods)
        sigma2[0] = omega / (1 - alpha - beta)
        
        # Add some independent noise to reduce correlation
        noise = np.random.normal(0, 0.01, n_periods)
        
        for t in range(1, n_periods):
            sigma2[t] = omega + alpha * returns_i[t-1]**2 + beta * sigma2[t-1]
            returns_i[t] = np.sqrt(sigma2[t]) * np.random.normal() + noise[t]
        
        returns[f'ASSET{i+1}'] = returns_i
    
    return pd.DataFrame(returns, index=dates)