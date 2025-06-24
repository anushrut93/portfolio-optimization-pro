"""
Advanced Market Impact and Execution Cost Models
Implements Almgren-Chriss and other sophisticated models for realistic transaction cost analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy.optimize import minimize
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketImpactParameters:
    """Parameters for market impact models"""
    daily_volume: float  # Average daily volume
    volatility: float    # Daily volatility
    spread: float        # Average bid-ask spread
    lambda_t: float      # Temporary impact coefficient
    lambda_p: float      # Permanent impact coefficient
    eta: float          # Power law exponent (typically 0.5 for square-root model)
    
class MarketImpactModel:
    """
    Advanced market impact modeling for portfolio optimization
    Implements multiple impact models used by sophisticated market makers
    """
    
    def __init__(self, impact_params: Dict[str, MarketImpactParameters]):
        """
        Initialize with market impact parameters for each asset
        """
        self.impact_params = impact_params
        
    # Alternative: Create simple impact calculation functions
    def calculate_almgren_chriss_impact(symbol, shares, time_horizon=1.0):
        """Calculate Almgren-Chriss impact directly from parameters"""
        params = impact_params[symbol]
        current_price = current_prices[symbol]
        
        participation_rate = shares / (params.daily_volume * time_horizon)
        
        # Temporary impact
        temp_impact_bps = params.spread * 10000 + params.lambda_t * np.sqrt(participation_rate) * 10000
        
        # Permanent impact
        perm_impact_bps = params.lambda_p * participation_rate * 10000
        
        # Total impact
        total_impact_bps = temp_impact_bps + perm_impact_bps
        
        return {
            'temporary_impact_bps': temp_impact_bps,
            'permanent_impact_bps': perm_impact_bps,
            'total_impact_bps': total_impact_bps,
            'total_cost': (total_impact_bps / 10000) * shares * current_price
        }

    def calculate_i_star_impact(symbol, shares, urgency=0.5):
        """Calculate I-Star model impact"""
        # First get base impact
        base_impact = calculate_almgren_chriss_impact(symbol, shares)
        
        # Adjust for urgency
        urgency_multiplier = 1 + urgency * 0.5
        
        return {
            'total_impact_bps': base_impact['total_impact_bps'] * urgency_multiplier,
            'total_cost': base_impact['total_cost'] * urgency_multiplier
        }
    
    def optimal_execution_trajectory(self, symbol: str, order_size: float,
                                   time_horizon: float, risk_aversion: float = 1e-6,
                                   num_periods: int = 10) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calculate optimal execution trajectory using Almgren-Chriss framework
        Returns trading trajectory and cost analysis
        """
        if symbol not in self.impact_params:
            raise ValueError(f"No impact parameters for {symbol}")
        
        params = self.impact_params[symbol]
        
        # Time increment
        tau = time_horizon / num_periods
        
        # Almgren-Chriss optimal trajectory parameters
        kappa = np.sqrt(params.lambda_t / (risk_aversion * params.volatility**2))
        
        # Calculate optimal trading rate
        times = np.linspace(0, time_horizon, num_periods + 1)
        
        # Optimal trajectory (linear for zero risk aversion, curved for positive)
        if risk_aversion == 0:
            # VWAP-like execution
            trajectory = order_size * (1 - times / time_horizon)
        else:
            # Risk-adjusted optimal trajectory
            trajectory = order_size * np.sinh(kappa * (time_horizon - times)) / np.sinh(kappa * time_horizon)
        
        # Trading rate at each period
        trade_sizes = -np.diff(trajectory)
        
        # Calculate costs
        temp_costs = []
        perm_costs = []
        
        for i, trade in enumerate(trade_sizes):
            # Instantaneous participation rate
            inst_rate = trade / (params.daily_volume * tau)
            
            # Temporary impact for this slice
            temp_impact = params.spread/2 + params.lambda_t * params.volatility * np.sqrt(inst_rate)
            temp_costs.append(temp_impact * trade)
            
            # Permanent impact accumulation
            perm_impact = params.lambda_p * params.volatility * (trade / params.daily_volume) ** params.eta
            perm_costs.append(perm_impact * (order_size - sum(trade_sizes[:i])))
        
        total_temp_cost = sum(temp_costs)
        total_perm_cost = sum(perm_costs)
        total_cost = total_temp_cost + total_perm_cost
        
        # Risk (variance of execution cost)
        execution_risk = risk_aversion * params.volatility**2 * order_size**2 * time_horizon / 3
        
        return trajectory, {
            'trade_sizes': trade_sizes,
            'temporary_cost': total_temp_cost,
            'permanent_cost': total_perm_cost,
            'total_cost': total_cost,
            'avg_impact_bps': (total_cost / order_size) * 10000,
            'execution_risk': execution_risk,
            'risk_adjusted_cost': total_cost + execution_risk
        }
    
    def portfolio_impact(self, trades: Dict[str, float], 
                        time_horizon: float = 1.0,
                        correlation_matrix: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate market impact for a portfolio trade
        Accounts for cross-asset impact through correlation
        """
        total_impact = 0
        individual_impacts = {}
        
        # Calculate individual impacts
        for symbol, trade_size in trades.items():
            if abs(trade_size) < 1e-6:
                continue
                
            impact = self.almgren_chriss_impact(
                symbol, abs(trade_size), time_horizon
            )
            individual_impacts[symbol] = impact
            total_impact += impact['total_cost'] * np.sign(trade_size)
        
        # Cross-impact adjustment (if correlation matrix provided)
        cross_impact = 0
        if correlation_matrix is not None:
            for sym1, trade1 in trades.items():
                for sym2, trade2 in trades.items():
                    if sym1 != sym2 and sym1 in correlation_matrix and sym2 in correlation_matrix:
                        correlation = correlation_matrix.loc[sym1, sym2]
                        # Cross-impact is proportional to correlation and trade sizes
                        params1 = self.impact_params.get(sym1)
                        params2 = self.impact_params.get(sym2)
                        
                        if params1 and params2:
                            cross_term = 0.5 * correlation * np.sign(trade1) * np.sign(trade2)
                            cross_term *= np.sqrt(abs(trade1) / params1.daily_volume)
                            cross_term *= np.sqrt(abs(trade2) / params2.daily_volume)
                            cross_term *= np.sqrt(params1.volatility * params2.volatility)
                            cross_impact += cross_term
        
        # Portfolio-level metrics
        total_value = sum(abs(trade) for trade in trades.values())
        
        return {
            'individual_impacts': individual_impacts,
            'total_impact': total_impact,
            'cross_impact': cross_impact,
            'net_impact': total_impact + cross_impact,
            'avg_impact_bps': ((total_impact + cross_impact) / total_value) * 10000 if total_value > 0 else 0
        }

class TransactionCostOptimizer:
    """
    Optimize portfolio considering realistic transaction costs
    """
    
    def __init__(self, impact_model: MarketImpactModel):
        self.impact_model = impact_model
        
    def calculate_net_alpha(self, expected_returns: pd.Series,
                          current_weights: pd.Series,
                          target_weights: pd.Series,
                          portfolio_value: float,
                          time_horizon: float = 1.0) -> pd.Series:
        """
        Calculate expected returns net of transaction costs
        """
        trades = {}
        net_returns = expected_returns.copy()
        
        for symbol in expected_returns.index:
            current_pos = current_weights.get(symbol, 0) * portfolio_value
            target_pos = target_weights.get(symbol, 0) * portfolio_value
            trade_size = target_pos - current_pos
            
            if abs(trade_size) > 1e-6:
                trades[symbol] = trade_size
                
                # Calculate impact for this trade
                impact = self.impact_model.almgren_chriss_impact(
                    symbol, abs(trade_size), time_horizon
                )
                
                # Adjust expected return by transaction cost
                cost_drag = impact['total_impact_bps'] / 10000
                net_returns[symbol] -= cost_drag / time_horizon  # Annualized
        
        return net_returns
    
    def optimize_with_impact(self, expected_returns: pd.Series,
                           cov_matrix: pd.DataFrame,
                           current_weights: pd.Series,
                           portfolio_value: float,
                           risk_aversion: float = 1.0,
                           time_horizon: float = 1.0,
                           max_turnover: float = 2.0) -> Dict[str, any]:
        """
        Optimize portfolio considering market impact
        """
        n_assets = len(expected_returns)
        
        def objective(weights):
            # Portfolio return
            portfolio_return = np.dot(weights, expected_returns)
            
            # Portfolio risk
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Transaction costs
            trades = weights - current_weights.values
            impact_cost = 0
            
            for i, symbol in enumerate(expected_returns.index):
                trade_value = trades[i] * portfolio_value
                if abs(trade_value) > 1e-6:
                    impact = self.impact_model.almgren_chriss_impact(
                        symbol, abs(trade_value), time_horizon
                    )
                    impact_cost += impact['total_cost']
            
            # Risk-adjusted return net of transaction costs
            net_return = portfolio_return - (impact_cost / portfolio_value) / time_horizon
            utility = net_return - 0.5 * risk_aversion * portfolio_risk**2
            
            return -utility  # Minimize negative utility
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda w: max_turnover - np.sum(np.abs(w - current_weights.values))}  # Turnover limit
        ]
        
        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = current_weights.values
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = pd.Series(result.x, index=expected_returns.index)
            
            # Calculate final metrics
            trades = optimal_weights - current_weights
            portfolio_impact = self.impact_model.portfolio_impact(
                trades * portfolio_value, time_horizon, cov_matrix
            )
            
            return {
                'optimal_weights': optimal_weights,
                'trades': trades,
                'expected_return': np.dot(optimal_weights, expected_returns),
                'expected_risk': np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))),
                'transaction_cost': portfolio_impact['net_impact'],
                'impact_bps': portfolio_impact['avg_impact_bps'],
                'turnover': np.sum(np.abs(trades))
            }
        else:
            raise ValueError("Optimization failed: " + result.message)

# Helper function to create impact parameters from market data
def estimate_impact_parameters(prices_df: pd.DataFrame, 
                             volumes_df: Optional[pd.DataFrame] = None,
                             spreads_df: Optional[pd.DataFrame] = None) -> Dict[str, MarketImpactParameters]:
    """
    Estimate market impact parameters from historical data
    """
    params = {}
    
    for symbol in prices_df.columns:
        # Calculate returns and volatility
        returns = prices_df[symbol].pct_change().dropna()
        daily_vol = returns.std()
        
        # Estimate ADV (use provided or approximate)
        if volumes_df is not None and symbol in volumes_df:
            adv = volumes_df[symbol].mean()
        else:
            # Approximate based on price level
            adv = 1000000 * (100 / prices_df[symbol].mean())  # Simple heuristic
        
        # Estimate spread (use provided or approximate based on volatility)
        if spreads_df is not None and symbol in spreads_df:
            avg_spread = spreads_df[symbol].mean()
        else:
            # Approximate: spread ≈ 2 * daily_vol / sqrt(252)
            avg_spread = 2 * daily_vol / np.sqrt(252)
        
        # Set impact coefficients (these would be calibrated in practice)
        # Higher volatility assets typically have higher impact
        lambda_t = 0.1 + 0.2 * (daily_vol / 0.02)  # Scale by volatility
        lambda_p = lambda_t * 0.5  # Permanent impact is ~50% of temporary
        
        params[symbol] = MarketImpactParameters(
            daily_volume=adv,
            volatility=daily_vol,
            spread=avg_spread,
            lambda_t=lambda_t,
            lambda_p=lambda_p,
            eta=0.6  # Standard choice
        )
    
    return params